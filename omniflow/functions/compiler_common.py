"""Parameter/binding helpers shared by the compiler and Function authoring.

Moved verbatim from ``omniflow.functions.compiler``; that module re-exports these names.
"""

from __future__ import annotations

import json
import re
from typing import Any
from omniflow.functions.management import parameter_candidates, semantic_parameter_evidence


def _source_parameter_candidates(facts: dict[str, Any]) -> list[dict[str, Any]]:
    fact_steps = [step for step in facts.get("steps") or () if isinstance(step, dict)]
    function_view = {
        "bindings": [],
        "render_bindings": [],
        "steps": [
            {
                "step_index": index,
                "source_step_index": step.get("source_step_index", index),
                "action": step["action"],
                "metadata": step.get("metadata") or {},
            }
            for index, step in enumerate(fact_steps)
        ],
    }
    evidence_by_target = {
        (
            evidence.get("source_step_index"),
            str(evidence.get("arg_name") or "").strip(),
        ): evidence
        for evidence in facts.get("parameter_evidence") or ()
        if isinstance(evidence, dict)
    }
    candidates = []
    for candidate in parameter_candidates(function_view):
        local_step_index = int(candidate["step_index"])
        source_step_index = int(
            fact_steps[local_step_index].get(
                "source_step_index", candidate["step_index"]
            )
        )
        semantic_evidence = semantic_parameter_evidence(
            function_view["steps"][local_step_index],
            str(candidate["arg_name"]),
            candidate["recorded_value"],
            facts,
        )
        if semantic_evidence is None:
            continue
        value = {
            "source_step_index": source_step_index,
            "tool": candidate["tool"],
            "arg_name": candidate["arg_name"],
            "recorded_value": candidate["recorded_value"],
            **semantic_evidence,
        }
        evidence = evidence_by_target.get(
            (source_step_index, candidate["arg_name"])
        )
        if evidence is not None:
            value.update(
                {
                    "task_parameter_name": evidence["task_parameter_name"],
                    "task_parameter_value": evidence["task_parameter_value"],
                    "value_contract": evidence["value_contract"],
                }
            )
        candidates.append(value)
    return candidates


def _agent_owned_function_artifact(
    *,
    function_id: str,
    name: str,
    description: str,
    source_indices: list[int],
    facts: dict[str, Any],
    parameters: list[dict[str, Any]],
    occurrence_index: int,
    agent_visible: bool = True,
) -> dict[str, Any]:
    if not name or not description:
        raise ValueError("function_author_inventory_definition_invalid")
    fact_steps = {
        int(step["source_step_index"]): step
        for step in facts.get("steps") or ()
        if isinstance(step, dict) and isinstance(step.get("source_step_index"), int)
    }
    if any(index not in fact_steps for index in source_indices):
        raise ValueError("function_author_plan_source_steps_invalid")
    local_by_source = {
        source_index: local_index
        for local_index, source_index in enumerate(source_indices)
    }
    properties: dict[str, dict[str, Any]] = {}
    required: list[str] = []
    bindings: list[dict[str, Any]] = []
    render_bindings: list[dict[str, Any]] = []
    for parameter in parameters:
        parameter_name = str(parameter["name"])
        value = parameter["values"][occurrence_index]
        properties[parameter_name] = {
            "type": _json_schema_type(value),
            "description": str(parameter["description"])[:240],
        }
        required.append(parameter_name)
        for binding in parameter["bindings"]:
            if binding["occurrence_index"] != occurrence_index:
                continue
            source_index = int(binding["source_step_index"])
            if source_index not in local_by_source:
                raise ValueError("function_author_parameter_binding_outside_occurrence")
            local_index = local_by_source[source_index]
            if binding["binding_kind"] == "action_arg":
                arg_name = str(binding.get("arg_name") or "").strip()
                action = fact_steps[source_index].get("action")
                args = action.get("args") if isinstance(action, dict) else None
                if not arg_name or not isinstance(args, dict) or arg_name not in args:
                    raise ValueError("function_author_plan_parameter_target_invalid")
                bindings.append(
                    {
                        "source": f"$.arguments.{parameter_name}",
                        "target": f"$.steps[{local_index}].action.args.{arg_name}",
                    }
                )
            else:
                render_bindings.append(
                    {
                        "source": f"$.arguments.{parameter_name}",
                        "step_index": local_index,
                        "node_id": str(binding.get("node_id") or "").strip(),
                        "attribute": str(binding.get("attribute") or "").strip(),
                        "recorded_value": str(binding.get("recorded_value") or ""),
                    }
                )
    return {
        "schema_version": "omniflow.function.v2",
        "function_id": function_id,
        "name": name,
        "description": description,
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        },
        "bindings": bindings,
        "render_bindings": render_bindings,
        "steps": [
            {
                "step_index": local_index,
                "source_state_id": str(fact_steps[source_index]["before_state_id"]),
                "action": json.loads(
                    json.dumps(fact_steps[source_index]["action"], ensure_ascii=False)
                ),
            }
            for local_index, source_index in enumerate(source_indices)
        ],
        "agent_visible": agent_visible,
    }


def _json_schema_type(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    if value is None:
        return "null"
    return "string"


def _state_level_local_authoring_functions(
    facts: dict[str, Any],
    *,
    source_indices: list[int],
    reserved_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Build executable one-transition Functions for uncovered source states."""

    reserved = set(reserved_ids or ())
    by_index = {
        int(step.get("source_step_index")): step
        for step in facts.get("steps") or ()
        if isinstance(step, dict)
        and isinstance(step.get("source_step_index"), int)
    }
    result: list[dict[str, Any]] = []
    for source_index in source_indices:
        step = by_index.get(int(source_index))
        if step is None:
            continue
        action = step.get("action") if isinstance(step.get("action"), dict) else {}
        tool = str(action.get("tool") or "transition").strip() or "transition"
        function_id = f"state_transition_{int(source_index):03d}"
        suffix = 2
        while function_id in reserved:
            function_id = f"state_transition_{int(source_index):03d}_{suffix}"
            suffix += 1
        reserved.add(function_id)
        readable_tool = tool.replace("_", " ")
        result.append(
            {
                "function_id": function_id,
                "name": f"State transition {int(source_index) + 1}: {readable_tool}",
                "description": (
                    f"Perform the recorded {readable_tool} transition from its "
                    "matching GUI state."
                ),
                "source_step_indices": [int(source_index)],
                # The compiler derives eligible input_text bindings from the
                # immutable facts, so synthesized Functions cannot lose args.
                "parameters": [],
            }
        )
    return result


def _compiler_parse_bounds(value: Any) -> tuple[float, float, float, float] | None:
    match = re.fullmatch(
        r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]"
        r"\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]",
        str(value or "").strip(),
    )
    if match is None:
        return None
    values = match.groups()
    return (
        float(values[0]),
        float(values[1]),
        float(values[2]),
        float(values[3]),
    )
