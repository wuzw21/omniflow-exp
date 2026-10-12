"""Model-driven Function authoring: patch protocol, workflow/plan materialization (split out of compiler).

Moved verbatim from ``omniflow.functions.compiler``; that module re-exports these names.
"""

from __future__ import annotations

import copy
import json
import re
from typing import Any, Callable
from omniflow.functions.management import parameter_candidates
from omniflow.functions.compiler_common import (
    _agent_owned_function_artifact,
    _compiler_parse_bounds,
    _source_parameter_candidates,
    _state_level_local_authoring_functions,
)


_PATCH_COORDINATE_ARGS = {"x", "y", "x1", "y1", "x2", "y2"}


_PATCH_FUNCTION_ID = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,63}")


_PATCH_PARAMETER_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,63}")


def _base_authoring_draft(source_indices: list[int]) -> dict[str, Any]:
    return {
        "kinds": {
            index: "online_observation" for index in source_indices
        },
        "bindings": {index: [] for index in source_indices},
        "explicit_online": [],
        "spans": [],
    }


def _judgment_card(facts: dict[str, Any]) -> dict[str, Any]:
    """Show each recorded call without the page's full node list."""

    steps: list[dict[str, Any]] = []
    for step in facts.get("steps") or ():
        if not isinstance(step, dict):
            continue
        action = step.get("action") if isinstance(step.get("action"), dict) else {}
        args = action.get("args") if isinstance(action.get("args"), dict) else {}
        card_step: dict[str, Any] = {
            "source_step_index": step.get("source_step_index"),
            "action_type": str(action.get("tool") or ""),
            "args": {
                key: value
                for key, value in args.items()
                if key not in _PATCH_COORDINATE_ARGS
            },
        }
        hit = _card_hit_node(step)
        if hit:
            card_step["hit"] = hit
        steps.append(card_step)
    return {
        "goal": facts.get("goal") or "",
        "task_parameters": facts.get("task_parameters") or {},
        "steps": steps,
    }


def _card_hit_node(step: dict[str, Any]) -> dict[str, str] | None:
    action = step.get("action") if isinstance(step.get("action"), dict) else {}
    if str(action.get("tool") or "") not in {"click", "long_press"}:
        return None
    args = action.get("args") if isinstance(action.get("args"), dict) else {}
    source_ui = step.get("source_ui") if isinstance(step.get("source_ui"), dict) else {}
    display = source_ui.get("display") if isinstance(source_ui.get("display"), dict) else {}
    try:
        width = float(display.get("width") or 0)
        height = float(display.get("height") or 0)
        point = (
            float(args["x"]) / 1000.0 * width,
            float(args["y"]) / 1000.0 * height,
        )
    except (KeyError, TypeError, ValueError):
        return None
    if width <= 0 or height <= 0:
        return None
    containing: list[tuple[float, dict[str, Any]]] = []
    for node in source_ui.get("nodes") or ():
        if not isinstance(node, dict):
            continue
        bounds = _compiler_parse_bounds(str(node.get("bounds") or ""))
        if bounds is None or not (
            bounds[0] <= point[0] <= bounds[2] and bounds[1] <= point[1] <= bounds[3]
        ):
            continue
        containing.append(
            ((bounds[2] - bounds[0]) * (bounds[3] - bounds[1]), node)
        )
    if not containing:
        return None
    node = min(containing, key=lambda item: item[0])[1]
    hit = {
        key: str(node.get(key) or "")
        for key in ("id", "text", "content-desc")
        if str(node.get(key) or "").strip()
    }
    if "id" in hit:
        hit["node_id"] = hit.pop("id")
    return hit or None


def _draft_source_indices(facts: dict[str, Any]) -> list[int]:
    return [
        int(step["source_step_index"])
        for step in facts.get("steps") or ()
        if isinstance(step, dict) and isinstance(step.get("source_step_index"), int)
    ]


def _span_indices(source_indices: list[int], start: int, end: int) -> list[int]:
    if start not in source_indices or end not in source_indices:
        raise ValueError(
            "authoring_patch_span_invalid:"
            f"start={start}:end={end}:"
            f"allowed_source_step_indices={source_indices}"
        )
    start_at = source_indices.index(start)
    end_at = source_indices.index(end)
    if start_at > end_at:
        raise ValueError(
            "authoring_patch_span_invalid:"
            f"start={start}:end={end}:range_is_reversed"
        )
    return source_indices[start_at : end_at + 1]


def _fact_step(facts: dict[str, Any], source_index: int) -> dict[str, Any]:
    for step in facts.get("steps") or ():
        if isinstance(step, dict) and step.get("source_step_index") == source_index:
            return step
    raise ValueError(
        "authoring_patch_step_invalid:"
        f"unknown_source_step_index={source_index}"
    )


def _recorded_node_value(
    step: dict[str, Any], node_id: str, attribute: str
) -> str:
    source_ui = step.get("source_ui") if isinstance(step.get("source_ui"), dict) else {}
    matches = [
        node
        for node in source_ui.get("nodes") or ()
        if isinstance(node, dict) and str(node.get("id") or "") == node_id
    ]
    if len(matches) != 1 or attribute not in matches[0]:
        available = [
            str(node.get("id") or "")
            for node in source_ui.get("nodes") or ()
            if isinstance(node, dict) and str(node.get("id") or "").strip()
        ]
        raise ValueError(
            "authoring_patch_node_invalid:"
            f"source_step_index={step.get('source_step_index')}:"
            f"node_id={node_id}:attribute={attribute}:"
            f"available_node_ids={available}"
        )
    return str(matches[0].get(attribute) or "")


def _parse_patch_assignments(tail: str) -> dict[str, str]:
    assignments: dict[str, str] = {}
    for token in tail.split():
        if "=" not in token:
            raise ValueError(f"authoring_patch_assignment_invalid:{token}")
        key, value = token.split("=", 1)
        if not key or not value or key in assignments:
            raise ValueError(f"authoring_patch_assignment_invalid:{token}")
        assignments[key] = value
    return assignments


def _parameters_for_occurrences(
    draft: dict[str, Any], occurrences: list[list[int]]
) -> list[dict[str, Any]]:
    names: list[str] = []
    for indices in occurrences:
        for index in indices:
            for binding in draft["bindings"].get(index) or ():
                name = str(binding["name"])
                if name not in names:
                    names.append(name)
    parameters: list[dict[str, Any]] = []
    for name in names:
        raw_bindings: list[dict[str, Any]] = []
        for occurrence_index, indices in enumerate(occurrences):
            for index in indices:
                for binding in draft["bindings"].get(index) or ():
                    if binding["name"] != name:
                        continue
                    item = {
                        "occurrence_index": occurrence_index,
                        "source_step_index": index,
                        "binding_kind": binding["binding_kind"],
                    }
                    if binding["binding_kind"] == "action_arg":
                        item["arg_name"] = binding["arg_name"]
                    else:
                        item["node_id"] = binding["node_id"]
                        item["attribute"] = binding["attribute"]
                        item["recorded_value"] = binding["recorded_value"]
                    raw_bindings.append(item)
        parameters.append(
            {
                "name": name,
                "description": name,
                "bindings": raw_bindings,
            }
        )
    return parameters


def _proposal_from_draft(
    draft: dict[str, Any], facts: dict[str, Any]
) -> dict[str, Any]:
    source_indices = _draft_source_indices(facts)
    grouped: dict[str, list[list[int]]] = {}
    order: list[str] = []
    for span in draft["spans"]:
        function_id = str(span["function_id"])
        if function_id not in grouped:
            order.append(function_id)
            grouped[function_id] = []
        grouped[function_id].append(list(span["indices"]))
    functions: list[dict[str, Any]] = []
    for function_id in order:
        occurrences = grouped[function_id]
        functions.append(
            {
                "function_id": function_id,
                "name": function_id.replace("_", " "),
                "description": f"Replay the recorded {function_id} steps.",
                "occurrences": [
                    {"source_step_indices": indices} for indices in occurrences
                ],
                "parameters": _parameters_for_occurrences(draft, occurrences),
            }
        )
    has_online = any(
        draft["kinds"].get(index) == "online_observation" for index in source_indices
    )
    execution_mode = "planner_handoff" if has_online else "direct_replay"
    complete_parameters = (
        []
        if has_online
        else _parameters_for_occurrences(draft, [source_indices])
    )
    return {
        "binding_owner": "agent",
        "reason": "Applied authoring patches.",
        "semantic_analysis": {
            "steps": [
                {
                    "source_step_index": index,
                    "semantic_kind": draft["kinds"][index],
                    "parameter_names": [
                        str(binding["name"])
                        for binding in draft["bindings"].get(index) or ()
                    ],
                    "reason": "authoring patch",
                }
                for index in source_indices
            ]
        },
        "functions": functions,
        "complete_function": {
            "function_id": "complete_source_workflow",
            "name": "Complete source workflow",
            "description": "Preserve the successful source workflow.",
            "source_step_indices": source_indices,
            "execution_mode": execution_mode,
            "parameters": complete_parameters,
        },
    }


def _apply_authoring_line(
    draft: dict[str, Any],
    line: str,
    facts: dict[str, Any],
) -> dict[str, Any]:
    """Apply one patch line. Reject the line without changing the draft."""

    source_indices = _draft_source_indices(facts)
    text = " ".join(str(line or "").split())
    parts = text.split(" ")
    candidate = copy.deepcopy(draft)
    if len(parts) >= 4 and parts[0] == "span":
        try:
            start = int(parts[1])
            end = int(parts[2])
        except ValueError as error:
            raise ValueError("authoring_patch_span_invalid") from error
        function_id = parts[3]
        if (
            len(parts) != 4
            or _PATCH_FUNCTION_ID.fullmatch(function_id) is None
            or function_id == "complete_source_workflow"
        ):
            raise ValueError(
                "authoring_patch_span_invalid:"
                f"function_id={function_id!r}"
            )
        indices = _span_indices(source_indices, start, end)
        explicit_online = {
            int(index) for index in candidate.get("explicit_online") or ()
        }
        blocked = [index for index in indices if index in explicit_online]
        if blocked:
            raise ValueError(
                "authoring_patch_span_contains_online:"
                f"source_step_indices={blocked}"
            )
        covered = {
            index
            for span in candidate["spans"]
            for index in span["indices"]
        }
        overlap = [index for index in indices if index in covered]
        if overlap:
            raise ValueError(
                "authoring_patch_span_overlap:"
                f"source_step_indices={overlap}"
            )
        for index in indices:
            if candidate["kinds"].get(index) == "online_observation":
                candidate["kinds"][index] = "stable"
        candidate["spans"].append(
            {"function_id": function_id, "indices": indices}
        )
    elif len(parts) >= 3 and parts[0] == "step":
        try:
            source_index = int(parts[1])
        except ValueError as error:
            raise ValueError("authoring_patch_step_invalid") from error
        if source_index not in source_indices:
            raise ValueError(
                "authoring_patch_step_invalid:"
                f"unknown_source_step_index={source_index}:"
                f"allowed_source_step_indices={source_indices}"
            )
        kind = parts[2]
        step = _fact_step(facts, source_index)
        if kind == "online_observation" and len(parts) == 3:
            candidate["kinds"][source_index] = "online_observation"
            candidate["bindings"][source_index] = []
            if source_index not in candidate["explicit_online"]:
                candidate["explicit_online"].append(source_index)
            replacement: list[dict[str, Any]] = []
            for span in candidate["spans"]:
                if source_index not in span["indices"]:
                    replacement.append(span)
                    continue
                left = [index for index in span["indices"] if index < source_index]
                right = [index for index in span["indices"] if index > source_index]
                for part in (left, right):
                    if part:
                        replacement.append(
                            {
                                "function_id": span["function_id"],
                                "indices": part,
                            }
                        )
            candidate["spans"] = replacement
        elif kind == "stable" and len(parts) == 3:
            candidate["kinds"][source_index] = "stable"
            candidate["bindings"][source_index] = []
            candidate["explicit_online"] = [
                index
                for index in candidate["explicit_online"]
                if int(index) != source_index
            ]
        elif kind == "task_parameter":
            assignments = _parse_patch_assignments(" ".join(parts[3:]))
            name = assignments.get("name") or ""
            if (
                set(assignments) not in (
                    {"name", "arg"},
                    {"name", "node", "attr"},
                )
                or _PATCH_PARAMETER_NAME.fullmatch(name) is None
            ):
                raise ValueError(
                    "authoring_patch_parameter_invalid:"
                    f"received={sorted(assignments)}"
                )
            if not any(
                source_index in span["indices"] for span in candidate["spans"]
            ):
                raise ValueError(
                    "authoring_patch_parameter_outside_span:"
                    f"source_step_index={source_index}"
                )
            action = step.get("action") if isinstance(step.get("action"), dict) else {}
            args = action.get("args") if isinstance(action.get("args"), dict) else {}
            if "arg" in assignments:
                arg_name = assignments["arg"]
                if arg_name not in args:
                    raise ValueError(
                        "function_author_plan_parameter_target_invalid:"
                        f"source_step_index={source_index}:arg_name={arg_name!r}:"
                        f"available_action_args={sorted(args)}"
                    )
                binding = {
                    "name": name,
                    "binding_kind": "action_arg",
                    "arg_name": arg_name,
                }
            else:
                attribute = assignments["attr"]
                if attribute not in {"text", "content-desc"}:
                    raise ValueError(
                        "authoring_patch_attribute_invalid:"
                        f"attribute={attribute}"
                    )
                binding = {
                    "name": name,
                    "binding_kind": "render_node",
                    "node_id": assignments["node"],
                    "attribute": attribute,
                    "recorded_value": _recorded_node_value(
                        step, assignments["node"], attribute
                    ),
                }
            existing = [
                item
                for item in candidate["bindings"].get(source_index) or ()
                if item.get("name") != name
            ]
            existing.append(binding)
            candidate["bindings"][source_index] = existing
            candidate["kinds"][source_index] = "task_parameter"
            candidate["explicit_online"] = [
                index
                for index in candidate["explicit_online"]
                if int(index) != source_index
            ]
        else:
            raise ValueError(f"authoring_patch_line_invalid:{text}")
    else:
        raise ValueError(f"authoring_patch_line_invalid:{text}")
    _materialize_agent_owned_workflow(
        _proposal_from_draft(candidate, facts),
        facts,
    )
    return candidate


def _parse_authoring_patches(raw: str) -> list[str]:
    value = json.loads(raw)
    if not isinstance(value, dict) or set(value) != {"patches"}:
        raise ValueError(
            "authoring_patches_required:"
            "expected_top_level_fields=patches:"
            f"received_fields={sorted(value) if isinstance(value, dict) else type(value).__name__}"
        )
    patches = value.get("patches")
    if not isinstance(patches, list) or any(
        not isinstance(patch, str) or not patch.strip() for patch in patches
    ):
        raise ValueError("authoring_patches_required")
    return [" ".join(patch.split()) for patch in patches]


def _author_by_patches(
    facts: dict[str, Any],
    *,
    complete_json: Callable[[str], str] | None,
    client: Any | None,
    selected_model: str | None,
    timeout: float,
    authoring_prompt: str,
    usage: dict[str, int],
    authoring_attempt_trace: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, Exception | None, str]:
    """Ask for patch lines and keep every line the current compiler accepts."""

    source_indices = _draft_source_indices(facts)
    draft = _base_authoring_draft(source_indices)
    applied: list[str] = []
    failed: list[dict[str, str]] = []
    authoring_error: Exception | None = ValueError(
        "authoring_patches_produced_no_executable_function"
    )
    raw_author_response = ""
    card = _judgment_card(facts)
    for attempt in range(1, 4):
        request_payload: dict[str, Any] = {
            "output_format": "json",
            "judgment_card": card,
            "applied_patches": list(applied),
        }
        if failed:
            request_payload["failed_patches"] = list(failed)
            request_payload["instruction"] = (
                "Return JSON with a patches array containing only replacement "
                "lines for the failed patches. Applied patches stay in effect."
            )
        response = None if complete_json is not None else client.chat.completions.create(
            model=selected_model,
            messages=[
                {"role": "system", "content": authoring_prompt},
                {
                    "role": "user",
                    "content": json.dumps(request_payload, ensure_ascii=False),
                },
            ],
            response_format={"type": "json_object"},
            max_tokens=8192,
            temperature=0,
            stream=False,
            reasoning_effort="none",
            extra_body={
                "enable_thinking": False,
                "thinking": {"type": "disabled"},
            },
            timeout=float(timeout),
        )
        response_usage = getattr(response, "usage", None)
        prompt_tokens = int(getattr(response_usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(
            getattr(response_usage, "completion_tokens", 0) or 0
        )
        total_tokens = int(getattr(response_usage, "total_tokens", 0) or 0)
        usage["model_calls"] += 1
        usage["prompt_tokens"] += prompt_tokens
        usage["completion_tokens"] += completion_tokens
        usage["total_tokens"] += total_tokens or prompt_tokens + completion_tokens
        raw_author_response = (
            complete_json(
                authoring_prompt
                + "\n\n"
                + json.dumps(request_payload, ensure_ascii=False)
            )
            if complete_json is not None
            else str(response.choices[0].message.content or "")
        )
        try:
            patches = _parse_authoring_patches(raw_author_response)
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            authoring_error = error
            authoring_attempt_trace.append(
                {
                    "attempt": attempt,
                    "accepted": False,
                    "error": str(error) or type(error).__name__,
                    "raw_response": raw_author_response,
                }
            )
            failed = [{"patch": raw_author_response, "error": str(error)}]
            if applied:
                break
            continue
        attempt_failures: list[dict[str, str]] = []
        for patch in patches:
            if patch in applied:
                continue
            try:
                draft = _apply_authoring_line(draft, patch, facts)
            except (TypeError, ValueError) as error:
                attempt_failures.append(
                    {"patch": patch, "error": str(error) or type(error).__name__}
                )
                continue
            applied.append(patch)
        failed = attempt_failures
        authoring_error = None if applied else ValueError(
            "authoring_patches_produced_no_executable_function"
        )
        authoring_attempt_trace.append(
            {
                "attempt": attempt,
                "accepted": bool(applied) and not failed,
                "error": None
                if applied and not failed
                else (
                    failed[-1]["error"]
                    if failed
                    else "authoring_patches_produced_no_executable_function"
                ),
                "raw_response": raw_author_response,
                "applied_patches": list(applied),
                "failed_patches": list(failed),
            }
        )
        if applied and not failed:
            break
        if applied and attempt >= 2:
            break
        if not applied and attempt >= 3:
            break
    if not applied:
        return None, authoring_error, raw_author_response
    authored = _materialize_agent_owned_workflow(
        _proposal_from_draft(draft, facts),
        facts,
    )
    authored["authoring_workflow"]["applied_patches"] = list(applied)
    authored["authoring_workflow"]["rejected_patches"] = list(failed)
    return authored, None, raw_author_response


def _materialize_authoring_response(
    value: Any,
    facts: dict[str, Any],
    *,
    candidate_map: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Accept the compact Agent proposal and retain legacy plans for callers."""

    if isinstance(value, dict) and value.get("binding_owner") == "agent":
        return _materialize_agent_owned_workflow(value, facts)
    if isinstance(value, dict) and (
        "functions" in value or "complete_function" in value
    ):
        return _materialize_authoring_workflow(
            value,
            facts,
            candidate_map=candidate_map,
        )
    if isinstance(value, dict) and set(value) == {"reason", "plan"}:
        return _materialize_authoring_plan(value, facts)
    raise ValueError(
        "function_author_response_contract_invalid:"
        "expected_top_level_fields=binding_owner,reason,semantic_analysis,functions,complete_function:"
        f"received_fields={sorted(value) if isinstance(value, dict) else type(value).__name__}:"
        "return_the_proposal_directly_without_a_wrapper"
    )


def _materialize_agent_owned_workflow(
    value: Any,
    facts: dict[str, Any],
) -> dict[str, Any]:
    """Map the A-side Agent's complete binding declaration into Store artifacts."""

    if not isinstance(value, dict) or set(value) != {
        "binding_owner",
        "reason",
        "semantic_analysis",
        "functions",
        "complete_function",
    }:
        raise ValueError("function_author_workflow_response_contract_invalid")
    reason = str(value.get("reason") or "").strip()
    if not reason:
        raise ValueError("function_author_reason_must_be_string")
    raw_functions = value.get("functions")
    raw_complete = value.get("complete_function")
    if not isinstance(raw_functions, list) or not isinstance(raw_complete, dict):
        raise ValueError("function_author_inventory_contract_invalid")

    source_indices = [
        int(step["source_step_index"])
        for step in facts.get("steps") or ()
        if isinstance(step, dict) and isinstance(step.get("source_step_index"), int)
    ]
    semantic_steps = _agent_owned_semantic_analysis(
        value.get("semantic_analysis"),
        source_indices=source_indices,
    )
    complete_fields = {
        "function_id",
        "name",
        "description",
        "source_step_indices",
        "execution_mode",
        "parameters",
    }
    if set(raw_complete) != complete_fields:
        raise ValueError("function_author_inventory_complete_invalid")
    if raw_complete.get("source_step_indices") != source_indices:
        raise ValueError("function_author_plan_complete_sequence_required")
    execution_mode = str(raw_complete.get("execution_mode") or "").strip()
    if execution_mode not in {"direct_replay", "planner_handoff"}:
        raise ValueError("function_author_complete_execution_mode_invalid")
    has_online_observation = any(
        item["semantic_kind"] == "online_observation"
        for item in semantic_steps.values()
    )
    if has_online_observation and execution_mode != "planner_handoff":
        raise ValueError(
            "function_author_online_observation_requires_planner_handoff"
        )
    if not has_online_observation and execution_mode != "direct_replay":
        raise ValueError(
            "function_author_complete_execution_mode_invalid:"
            "all_steps_are_stable_or_task_parameter:expected=direct_replay:"
            "task_parameters_are_bound_before_execution_and_do_not_require_planner_handoff"
        )

    function_fields = {
        "function_id",
        "name",
        "description",
        "occurrences",
        "parameters",
    }
    materialized_functions: list[dict[str, Any]] = []
    arguments: dict[str, dict[str, Any]] = {}
    source_calls: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for raw_function in raw_functions:
        if not isinstance(raw_function, dict) or set(raw_function) != function_fields:
            raise ValueError("function_author_inventory_definition_invalid")
        occurrences = raw_function.get("occurrences")
        if not isinstance(occurrences, list) or not occurrences:
            raise ValueError("function_author_inventory_occurrence_invalid")
        spans = [
            _agent_owned_occurrence_span(occurrence, source_indices)
            for occurrence in occurrences
        ]
        if any(
            semantic_steps[source_index]["semantic_kind"]
            == "online_observation"
            for span in spans
            for source_index in span
        ):
            raise ValueError(
                "function_author_local_online_observation_not_replayable"
            )
        function_id = _agent_owned_function_id(raw_function, seen_ids)
        parameters = _agent_owned_parameters(
            raw_function.get("parameters"),
            location=f"functions[{len(materialized_functions)}].parameters",
            occurrence_count=len(spans),
            facts=facts,
            spans=spans,
        )
        _validate_agent_owned_binding_spans(parameters, spans)
        _validate_agent_owned_semantic_bindings(
            parameters,
            spans=spans,
            semantic_steps=semantic_steps,
            require_all=True,
        )
        function = _agent_owned_function_artifact(
            function_id=function_id,
            name=str(raw_function.get("name") or "").strip(),
            description=str(raw_function.get("description") or "").strip(),
            source_indices=spans[0],
            facts=facts,
            parameters=parameters,
            occurrence_index=0,
        )
        materialized_functions.append(function)
        arguments[function_id] = _agent_owned_occurrence_arguments(parameters, 0)
        for occurrence_index in range(len(spans)):
            source_calls.append(
                {
                    "function_id": function_id,
                    "arguments": _agent_owned_occurrence_arguments(
                        parameters, occurrence_index
                    ),
                }
            )

    safe_source_indices = [
        source_index
        for source_index in source_indices
        if semantic_steps[source_index]["semantic_kind"]
        in {"stable", "task_parameter"}
    ]
    if execution_mode == "planner_handoff" and not raw_functions and safe_source_indices:
        raise ValueError(
            "function_author_planner_handoff_safe_local_required:"
            "safe_source_step_indices="
            + ",".join(str(index) for index in safe_source_indices)
        )
    complete_id = _agent_owned_function_id(raw_complete, seen_ids)
    complete_parameters = _agent_owned_parameters(
        raw_complete.get("parameters"),
        location="complete_function.parameters",
        occurrence_count=1,
        facts=facts,
        spans=[source_indices],
    )
    _validate_agent_owned_binding_spans(complete_parameters, [source_indices])
    _validate_agent_owned_semantic_bindings(
        complete_parameters,
        spans=[source_indices],
        semantic_steps=semantic_steps,
        require_all=execution_mode == "direct_replay",
    )
    complete_function = _agent_owned_function_artifact(
        function_id=complete_id,
        name=str(raw_complete.get("name") or "").strip(),
        description=str(raw_complete.get("description") or "").strip(),
        source_indices=source_indices,
        facts=facts,
        parameters=complete_parameters,
        occurrence_index=0,
        agent_visible=execution_mode == "direct_replay",
    )
    materialized_functions.append(complete_function)
    arguments[complete_id] = _agent_owned_occurrence_arguments(
        complete_parameters, 0
    )
    if not source_calls:
        source_calls.append(
            {"function_id": complete_id, "arguments": arguments[complete_id]}
        )

    from omniflow.functions.artifact import bind_function, parse_function_artifact

    for function in materialized_functions:
        parsed = parse_function_artifact(function)
        bind_function(parsed, arguments[function["function_id"]])
    return {
        "reason": reason,
        "bundle": {
            "schema_version": "omniflow.function-bundle.v2",
            "run_id": str(facts["run_id"]),
            "arguments": arguments,
            "checker_rules": [],
            "functions": materialized_functions,
        },
        "source_calls": source_calls,
        "authoring_workflow": {
            "schema_version": "omniflow.function-authoring-workflow.v3",
            "binding_owner": "agent",
            "agent_proposal_accepted": True,
            "fallback_mode": None,
            "semantic_analysis": {
                "counts": {
                    semantic_kind: sum(
                        item["semantic_kind"] == semantic_kind
                        for item in semantic_steps.values()
                    )
                    for semantic_kind in (
                        "stable",
                        "task_parameter",
                        "online_observation",
                    )
                },
                "steps": [semantic_steps[index] for index in source_indices],
            },
            "complete_execution_mode": execution_mode,
            "definition_count": len(raw_functions),
            "invocation_count": len(source_calls),
            "complete_function_id": complete_id,
        },
    }


def _agent_owned_semantic_analysis(
    value: Any,
    *,
    source_indices: list[int],
) -> dict[int, dict[str, Any]]:
    """Validate the Agent's explicit semantic classification without inferring it."""

    if not isinstance(value, dict) or set(value) != {"steps"}:
        raise ValueError("function_author_semantic_analysis_invalid")
    raw_steps = value.get("steps")
    if not isinstance(raw_steps, list):
        raise ValueError("function_author_semantic_analysis_invalid")
    expected_fields = {
        "source_step_index",
        "semantic_kind",
        "parameter_names",
        "reason",
    }
    parsed: dict[int, dict[str, Any]] = {}
    for raw_step in raw_steps:
        if not isinstance(raw_step, dict) or set(raw_step) != expected_fields:
            raise ValueError("function_author_semantic_step_invalid")
        source_step_index = raw_step.get("source_step_index")
        semantic_kind = str(raw_step.get("semantic_kind") or "").strip()
        parameter_names = raw_step.get("parameter_names")
        reason = str(raw_step.get("reason") or "").strip()
        if isinstance(source_step_index, int) and not isinstance(source_step_index, bool):
            if source_step_index not in source_indices:
                raise ValueError(
                    "function_author_semantic_step_invalid:"
                    f"unknown_source_step_index={source_step_index}:"
                    f"allowed_source_step_indices={source_indices}"
                )
            if source_step_index in parsed:
                raise ValueError(
                    "function_author_semantic_step_invalid:"
                    f"duplicate_source_step_index={source_step_index}"
                )
        if (
            isinstance(source_step_index, bool)
            or not isinstance(source_step_index, int)
            or semantic_kind
            not in {"stable", "task_parameter", "online_observation"}
            or not isinstance(parameter_names, list)
            or not reason
        ):
            raise ValueError("function_author_semantic_step_invalid")
        normalized_names = [str(name or "").strip() for name in parameter_names]
        if (
            any(
                re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", name) is None
                for name in normalized_names
            )
            or len(normalized_names) != len(set(normalized_names))
            or (semantic_kind == "task_parameter") != bool(normalized_names)
        ):
            raise ValueError("function_author_semantic_parameter_names_invalid")
        parsed[source_step_index] = {
            "source_step_index": source_step_index,
            "semantic_kind": semantic_kind,
            "parameter_names": normalized_names,
            "reason": reason,
        }
    if set(parsed) != set(source_indices):
        missing = [
            index for index in dict.fromkeys(source_indices) if index not in parsed
        ]
        raise ValueError(
            "function_author_semantic_steps_incomplete:"
            f"missing={','.join(str(index) for index in missing) or '<none>'}:"
            "semantic_analysis_and_complete_function_must_cover="
            + ",".join(str(index) for index in source_indices)
        )
    return parsed


def _agent_owned_occurrence_span(value: Any, source_indices: list[int]) -> list[int]:
    if not isinstance(value, dict) or set(value) != {"source_step_indices"}:
        raise ValueError("function_author_inventory_occurrence_invalid")
    span = value.get("source_step_indices")
    if (
        not isinstance(span, list)
        or not span
        or any(isinstance(index, bool) or not isinstance(index, int) for index in span)
        or span != sorted(set(span))
        or any(index not in source_indices for index in span)
    ):
        raise ValueError("function_author_inventory_occurrence_invalid")
    positions = [source_indices.index(index) for index in span]
    if positions != list(range(positions[0], positions[-1] + 1)):
        raise ValueError("function_author_inventory_occurrence_not_contiguous")
    return list(span)


def _agent_owned_function_id(value: dict[str, Any], seen_ids: set[str]) -> str:
    function_id = str(value.get("function_id") or "").strip()
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", function_id) is None:
        raise ValueError(
            "function_author_inventory_function_id_invalid:"
            f"function_id={function_id!r}:expected_pattern=[A-Za-z_][A-Za-z0-9_]{{0,63}}"
        )
    if function_id in seen_ids:
        raise ValueError(
            "function_author_inventory_function_id_invalid:"
            f"duplicate_function_id={function_id}:"
            "local_functions_and_complete_function_must_have_distinct_ids"
        )
    seen_ids.add(function_id)
    return function_id


def _agent_owned_parameters(
    value: Any,
    *,
    location: str = "parameters",
    occurrence_count: int,
    facts: dict[str, Any],
    spans: list[list[int]],
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError("function_author_parameterization_invalid")
    parameter_fields = {
        "name",
        "description",
        "bindings",
    }
    parsed: list[dict[str, Any]] = []
    names: set[str] = set()
    for parameter_index, parameter in enumerate(value):
        if not isinstance(parameter, dict) or set(parameter) != parameter_fields:
            raise ValueError(
                "function_author_parameter_invalid:"
                f"path={location}[{parameter_index}]:"
                "expected_fields=bindings,description,name:"
                f"received_fields={sorted(parameter) if isinstance(parameter, dict) else type(parameter).__name__}:"
                "each_declared_parameter_requires_its_own_bindings"
            )
        name = str(parameter.get("name") or "").strip()
        description = str(parameter.get("description") or "").strip()
        if (
            re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", name) is None
            or name in names
            or not description
        ):
            raise ValueError("function_author_parameter_invalid")
        raw_bindings = parameter.get("bindings")
        if not isinstance(raw_bindings, list) or not raw_bindings:
            raise ValueError("function_author_parameter_binding_invalid")
        bindings: list[dict[str, Any]] = []
        covered_occurrences: set[int] = set()
        for binding in raw_bindings:
            parsed_binding = _agent_owned_binding(binding, occurrence_count)
            bindings.append(parsed_binding)
            covered_occurrences.add(parsed_binding["occurrence_index"])
        if covered_occurrences != set(range(occurrence_count)):
            raise ValueError(
                "function_author_parameter_occurrence_incomplete:"
                f"parameter={name}:missing_occurrence_indices={sorted(set(range(occurrence_count)) - covered_occurrences)}:"
                f"occurrence_source_steps={spans}:"
                "each_occurrence_is_a_whole_function_invocation_and_requires_every_parameter"
            )
        values = _agent_owned_default_values(
            bindings,
            occurrence_count=occurrence_count,
            facts=facts,
            spans=spans,
        )
        names.add(name)
        parsed.append(
            {
                "name": name,
                "description": description,
                "values": values,
                "bindings": bindings,
            }
        )
    return parsed


def _agent_owned_default_values(
    bindings: list[dict[str, Any]],
    *,
    occurrence_count: int,
    facts: dict[str, Any],
    spans: list[list[int]],
) -> dict[int, Any]:
    fact_steps = {
        int(step["source_step_index"]): step
        for step in facts.get("steps") or ()
        if isinstance(step, dict) and isinstance(step.get("source_step_index"), int)
    }
    values: dict[int, Any] = {}
    for occurrence_index in range(occurrence_count):
        occurrence_values: list[Any] = []
        for binding in bindings:
            if binding["occurrence_index"] != occurrence_index:
                continue
            source_index = int(binding["source_step_index"])
            if source_index not in spans[occurrence_index]:
                raise ValueError("function_author_parameter_binding_outside_occurrence")
            if binding["binding_kind"] == "action_arg":
                step = fact_steps.get(source_index)
                action = step.get("action") if isinstance(step, dict) else None
                args = action.get("args") if isinstance(action, dict) else None
                arg_name = str(binding.get("arg_name") or "")
                if not isinstance(args, dict) or arg_name not in args:
                    raise ValueError(
                        "function_author_plan_parameter_target_invalid:"
                        f"source_step_index={source_index}:arg_name={arg_name!r}:"
                        f"available_action_args={sorted(args) if isinstance(args, dict) else []}:"
                        "arg_name_must_match_a_source_action_argument_not_the_parameter_name"
                    )
                occurrence_values.append(args[arg_name])
            else:
                occurrence_values.append(binding.get("recorded_value"))
        if not occurrence_values:
            raise ValueError("function_author_parameter_occurrence_incomplete")
        default_value = occurrence_values[0]
        if any(item != default_value for item in occurrence_values[1:]):
            raise ValueError("function_author_parameter_default_conflict")
        values[occurrence_index] = json.loads(
            json.dumps(default_value, ensure_ascii=False)
        )
    return values


def _agent_owned_binding(value: Any, occurrence_count: int) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("function_author_parameter_binding_invalid")
    binding_kind = str(value.get("binding_kind") or "").strip()
    common = {"occurrence_index", "source_step_index", "binding_kind"}
    if binding_kind == "action_arg":
        expected = common | {"arg_name"}
        allowed_shapes = {frozenset(expected), frozenset(expected | {"recorded_value"})}
    elif binding_kind == "render_node":
        expected = common | {"node_id", "attribute", "recorded_value"}
        allowed_shapes = {frozenset(expected)}
    else:
        raise ValueError(
            "function_author_parameter_binding_kind_invalid:"
            f"{binding_kind or '<missing>'}"
        )
    actual = set(value)
    if frozenset(actual) not in allowed_shapes:
        raise ValueError(
            f"function_author_{binding_kind}_fields_invalid:"
            f"expected={','.join(sorted(expected))}:"
            f"got={','.join(sorted(actual))}"
        )
    occurrence_index = value.get("occurrence_index")
    source_step_index = value.get("source_step_index")
    if isinstance(occurrence_index, int) and not isinstance(occurrence_index, bool) and occurrence_index not in range(occurrence_count):
        raise ValueError(
            "function_author_parameter_binding_invalid:"
            f"occurrence_index={occurrence_index}:allowed_occurrence_indices={list(range(occurrence_count))}:"
            "occurrence_index_is_not_source_step_index"
        )
    if (
        isinstance(occurrence_index, bool)
        or not isinstance(occurrence_index, int)
        or occurrence_index not in range(occurrence_count)
        or isinstance(source_step_index, bool)
        or not isinstance(source_step_index, int)
    ):
        raise ValueError("function_author_parameter_binding_invalid")
    return json.loads(json.dumps(value, ensure_ascii=False))


def _agent_owned_occurrence_arguments(
    parameters: list[dict[str, Any]], occurrence_index: int
) -> dict[str, Any]:
    return {
        str(parameter["name"]): json.loads(
            json.dumps(parameter["values"][occurrence_index], ensure_ascii=False)
        )
        for parameter in parameters
    }


def _validate_agent_owned_binding_spans(
    parameters: list[dict[str, Any]], spans: list[list[int]]
) -> None:
    for parameter in parameters:
        for binding in parameter["bindings"]:
            occurrence_index = int(binding["occurrence_index"])
            if int(binding["source_step_index"]) not in spans[occurrence_index]:
                raise ValueError("function_author_parameter_binding_outside_occurrence")


def _validate_agent_owned_semantic_bindings(
    parameters: list[dict[str, Any]],
    *,
    spans: list[list[int]],
    semantic_steps: dict[int, dict[str, Any]],
    require_all: bool,
) -> None:
    """Check only consistency between Agent-owned classifications and bindings."""

    bound_by_occurrence: list[dict[int, set[str]]] = [
        {} for _ in range(len(spans))
    ]
    for parameter in parameters:
        parameter_name = str(parameter["name"])
        for binding in parameter["bindings"]:
            occurrence_index = int(binding["occurrence_index"])
            source_step_index = int(binding["source_step_index"])
            semantic = semantic_steps[source_step_index]
            if (
                semantic["semantic_kind"] != "task_parameter"
                or parameter_name not in semantic["parameter_names"]
            ):
                raise ValueError(
                    "function_author_binding_semantic_classification_mismatch"
                )
            bound_by_occurrence[occurrence_index].setdefault(
                source_step_index, set()
            ).add(parameter_name)
    if not require_all:
        return
    for occurrence_index, span in enumerate(spans):
        for source_step_index in span:
            semantic = semantic_steps[source_step_index]
            if semantic["semantic_kind"] != "task_parameter":
                continue
            expected = set(semantic["parameter_names"])
            actual = bound_by_occurrence[occurrence_index].get(
                source_step_index, set()
            )
            if not expected.issubset(actual):
                raise ValueError(
                    "function_author_task_parameter_binding_incomplete:"
                    f"occurrence_index={occurrence_index}:"
                    f"source_step_index={source_step_index}:"
                    f"missing_agent_declared_parameters={','.join(sorted(expected - actual))}"
                )


def _materialize_authoring_workflow(
    value: Any,
    facts: dict[str, Any],
    *,
    candidate_map: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Validate one semantic proposal, then compile and register its artifacts."""

    if not isinstance(value, dict) or set(value) != {
        "reason",
        "functions",
        "complete_function",
    }:
        raise ValueError("function_author_workflow_response_contract_invalid")
    reason = str(value.get("reason") or "").strip()
    if not reason:
        raise ValueError("function_author_reason_must_be_string")

    source_steps = [
        step for step in facts.get("steps") or () if isinstance(step, dict)
    ]
    source_indices = [int(step["source_step_index"]) for step in source_steps]
    source_positions = {
        source_index: position
        for position, source_index in enumerate(source_indices)
    }
    source_tools = {
        int(step["source_step_index"]): str(
            (step.get("action") or {}).get("tool") or ""
        )
        for step in source_steps
    }
    if not source_indices:
        raise ValueError("successful_source_actions_required")

    raw_functions = value.get("functions")
    raw_complete = value.get("complete_function")
    if not isinstance(raw_functions, list):
        raise ValueError("function_author_inventory_contract_invalid")
    raw_definitions: list[dict[str, Any]] = []
    raw_parameterization: list[dict[str, Any]] = []
    proposal_definition_fields = {
        "function_id",
        "name",
        "description",
        "occurrences",
        "parameters",
    }
    for raw_function in raw_functions:
        if (
            not isinstance(raw_function, dict)
            or set(raw_function) != proposal_definition_fields
        ):
            raise ValueError("function_author_inventory_definition_invalid")
        raw_definitions.append(
            {
                key: raw_function[key]
                for key in ("function_id", "name", "description", "occurrences")
            }
        )
        raw_parameterization.append(
            {
                "function_id": raw_function["function_id"],
                "parameters": raw_function["parameters"],
            }
        )
    definition_fields = {
        "function_id",
        "name",
        "description",
        "occurrences",
    }
    occurrence_fields = {"source_step_indices"}
    definitions: dict[str, dict[str, Any]] = {}
    occurrence_spans: dict[str, list[list[int]]] = {}
    representative_occurrence_indices: dict[str, int] = {}
    all_occurrences: list[tuple[int, str, int, list[int]]] = []
    for raw_definition in raw_definitions:
        if not isinstance(raw_definition, dict) or set(raw_definition) != definition_fields:
            raise ValueError("function_author_inventory_definition_invalid")
        function_id = str(raw_definition.get("function_id") or "").strip()
        name = str(raw_definition.get("name") or "").strip()
        description = str(raw_definition.get("description") or "").strip()
        raw_occurrences = raw_definition.get("occurrences")
        if (
            not function_id
            or not name
            or not description
            or function_id in definitions
            or not isinstance(raw_occurrences, list)
            or not raw_occurrences
        ):
            raise ValueError("function_author_inventory_definition_invalid")
        spans: list[list[int]] = []
        tool_shapes: list[tuple[str, ...]] = []
        for occurrence_index, raw_occurrence in enumerate(raw_occurrences):
            if (
                not isinstance(raw_occurrence, dict)
                or set(raw_occurrence) != occurrence_fields
            ):
                raise ValueError("function_author_inventory_occurrence_invalid")
            raw_span = raw_occurrence.get("source_step_indices")
            if (
                not isinstance(raw_span, list)
                or not raw_span
                or any(
                    isinstance(item, bool) or not isinstance(item, int)
                    for item in raw_span
                )
            ):
                raise ValueError("function_author_inventory_occurrence_invalid")
            span = list(raw_span)
            if span != sorted(set(span)) or any(
                item not in source_positions for item in span
            ):
                raise ValueError("function_author_inventory_occurrence_invalid")
            positions = [source_positions[item] for item in span]
            if positions != list(range(positions[0], positions[-1] + 1)):
                raise ValueError("function_author_inventory_occurrence_not_contiguous")
            tool_shapes.append(tuple(source_tools[item] for item in span))
            spans.append(span)
            all_occurrences.append((positions[0], function_id, occurrence_index, span))
        definitions[function_id] = {
            "function_id": function_id,
            "name": name,
            "description": description,
        }
        occurrence_spans[function_id] = spans
        shape_counts = {
            shape: tool_shapes.count(shape)
            for shape in tool_shapes
        }
        representative_occurrence_indices[function_id] = max(
            range(len(tool_shapes)),
            key=lambda index: shape_counts[tool_shapes[index]],
        )

    complete_fields = {
        "function_id",
        "name",
        "description",
        "source_step_indices",
    }
    if isinstance(raw_complete, dict) and "parameters" in raw_complete:
        raise ValueError(
            "function_author_inventory_complete_invalid:"
            "parameters_must_be_declared_on_a_function"
        )
    if not isinstance(raw_complete, dict) or set(raw_complete) != complete_fields:
        raise ValueError("function_author_inventory_complete_invalid")
    complete_id = str(raw_complete.get("function_id") or "").strip()
    complete_indices = raw_complete.get("source_step_indices")
    if (
        not complete_id
        or complete_id in definitions
        or not str(raw_complete.get("name") or "").strip()
        or not str(raw_complete.get("description") or "").strip()
    ):
        raise ValueError("function_author_inventory_complete_invalid")
    complete_indices_normalized = complete_indices != source_indices
    raw_complete = {
        **raw_complete,
        "source_step_indices": source_indices,
    }

    parameterization_fields = {"function_id", "parameters"}
    parameter_fields = {"name", "description", "bindings"}
    legacy_parameter_binding_fields = {"occurrence_index", "candidate_id"}
    semantic_parameter_binding_fields = {
        "occurrence_index",
        "source_step_index",
        "binding_kind",
    }
    parameters_by_function: dict[str, list[dict[str, Any]]] = {}
    selected_candidate_ids: set[str] = set()
    invocation_arguments: dict[tuple[str, int], dict[str, Any]] = {
        (function_id, occurrence_index): {}
        for function_id, spans in occurrence_spans.items()
        for occurrence_index in range(len(spans))
    }
    candidate_parameter_names: dict[str, str] = {}
    for raw_function_parameters in raw_parameterization:
        if (
            not isinstance(raw_function_parameters, dict)
            or set(raw_function_parameters) != parameterization_fields
        ):
            raise ValueError("function_author_parameterization_invalid")
        function_id = str(raw_function_parameters.get("function_id") or "").strip()
        raw_parameters = raw_function_parameters.get("parameters")
        if (
            function_id not in definitions
            or function_id in parameters_by_function
            or not isinstance(raw_parameters, list)
        ):
            raise ValueError("function_author_parameterization_invalid")
        parsed_parameters: list[dict[str, Any]] = []
        parameter_names: set[str] = set()
        for raw_parameter in raw_parameters:
            if not isinstance(raw_parameter, dict) or set(raw_parameter) != parameter_fields:
                raise ValueError("function_author_parameter_invalid")
            name = str(raw_parameter.get("name") or "").strip()
            description = str(raw_parameter.get("description") or "").strip()
            raw_bindings = raw_parameter.get("bindings")
            if (
                re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", name) is None
                or name in parameter_names
                or not description
                or not isinstance(raw_bindings, list)
                or not raw_bindings
            ):
                raise ValueError("function_author_parameter_invalid")
            parameter_names.add(name)
            by_occurrence: dict[int, list[tuple[str, dict[str, Any]]]] = {}
            for raw_binding in raw_bindings:
                if not isinstance(raw_binding, dict):
                    raise ValueError("function_author_parameter_binding_invalid")
                raw_binding_fields = set(raw_binding)
                occurrence_index = raw_binding.get("occurrence_index")
                if (
                    isinstance(occurrence_index, bool)
                    or not isinstance(occurrence_index, int)
                    or occurrence_index not in range(len(occurrence_spans[function_id]))
                ):
                    raise ValueError("function_author_parameter_binding_invalid")
                span = occurrence_spans[function_id][occurrence_index]
                resolved_candidates: list[tuple[str, dict[str, Any]]] = []
                if raw_binding_fields == legacy_parameter_binding_fields:
                    candidate_id = str(
                        raw_binding.get("candidate_id") or ""
                    ).strip()
                    if candidate_id not in candidate_map:
                        raise ValueError(
                            "function_author_parameter_binding_evidence_missing"
                        )
                    resolved_candidates = [(candidate_id, candidate_map[candidate_id])]
                elif raw_binding_fields == semantic_parameter_binding_fields:
                    source_step_index = raw_binding.get("source_step_index")
                    binding_kind = str(
                        raw_binding.get("binding_kind") or ""
                    ).strip()
                    if (
                        isinstance(source_step_index, bool)
                        or not isinstance(source_step_index, int)
                        or binding_kind not in {"action_arg", "render_node"}
                    ):
                        raise ValueError(
                            "function_author_parameter_binding_invalid"
                        )
                    if source_step_index not in span:
                        raise ValueError(
                            "function_author_parameter_binding_outside_occurrence:"
                            f"function={function_id}:parameter={name}:"
                            f"occurrence={occurrence_index}:"
                            f"source_step={source_step_index}:span={span}"
                        )
                    for candidate_id, candidate in candidate_map.items():
                        suggested_name = str(
                            candidate.get("suggested_name") or ""
                        ).strip()
                        parameter_name_matches = bool(suggested_name) and (
                            name == suggested_name
                            or re.fullmatch(
                                re.escape(suggested_name) + r"_[2-9][0-9]*",
                                name,
                            )
                            is not None
                        )
                        if (
                            candidate.get("kind") == binding_kind
                            and candidate.get("source_step_index")
                            == source_step_index
                            and parameter_name_matches
                        ):
                            resolved_candidates.append((candidate_id, candidate))
                    if not resolved_candidates:
                        raise ValueError(
                            "function_author_parameter_binding_evidence_missing:"
                            f"function={function_id}:parameter={name}:"
                            f"occurrence={occurrence_index}:"
                            f"source_step={source_step_index}:kind={binding_kind}"
                        )
                else:
                    raise ValueError("function_author_parameter_binding_invalid")

                for candidate_id, candidate in resolved_candidates:
                    if candidate_id in selected_candidate_ids:
                        raise ValueError(
                            "function_author_parameter_binding_duplicate_evidence"
                        )
                    if candidate.get("source_step_index") not in span:
                        raise ValueError(
                            "function_author_parameter_binding_outside_occurrence"
                        )
                    suggested_name = str(
                        candidate.get("suggested_name") or ""
                    ).strip()
                    if suggested_name and not (
                        name == suggested_name
                        or re.fullmatch(
                            re.escape(suggested_name) + r"_[2-9][0-9]*",
                            name,
                        )
                    ):
                        raise ValueError(
                            "function_author_parameter_name_not_suggested:"
                            f"{candidate_id}:expected={suggested_name}:got={name}"
                        )
                    selected_candidate_ids.add(candidate_id)
                    candidate_parameter_names[candidate_id] = name
                    by_occurrence.setdefault(occurrence_index, []).append(
                        (candidate_id, candidate)
                    )
            if set(by_occurrence) != set(range(len(occurrence_spans[function_id]))):
                raise ValueError("function_author_parameter_occurrence_incomplete")
            for occurrence_index, bindings in sorted(by_occurrence.items()):
                values: set[str] = set()
                for _candidate_id, candidate in bindings:
                    if candidate["kind"] == "action_arg":
                        parameter_value = str(candidate.get("recorded_value") or "")
                    else:
                        parameter_value = str(
                            candidate.get("task_parameter_value")
                            or candidate.get("recorded_value")
                            or ""
                        )
                    values.add(parameter_value)
                if len(values) != 1 or not next(iter(values), ""):
                    raise ValueError("function_author_parameter_value_ambiguous")
                invocation_arguments[(function_id, occurrence_index)][name] = next(
                    iter(values)
                )
            parsed_parameters.append(
                {
                    "name": name,
                    "description": description,
                    "bindings_by_occurrence": by_occurrence,
                }
            )
        parameters_by_function[function_id] = parsed_parameters
    if set(parameters_by_function) != set(definitions):
        raise ValueError("function_author_parameterization_definition_mismatch")
    unselected_candidate_ids = sorted(set(candidate_map) - selected_candidate_ids)

    expected_function_ids = [*definitions, complete_id]
    registered_occurrences = sorted(all_occurrences)
    covered_local_indices = {
        index
        for _position, _function_id, _occurrence_index, span in registered_occurrences
        for index in span
    }
    uncovered_local_indices = [
        index for index in source_indices if index not in covered_local_indices
    ]

    selected_node_evidence: list[dict[str, Any]] = []
    for candidate_id, parameter_name in candidate_parameter_names.items():
        candidate = candidate_map[candidate_id]
        if candidate.get("kind") != "render_node":
            continue
        evidence = {
            key: json.loads(json.dumps(item, ensure_ascii=False))
            for key, item in candidate.items()
            if key != "kind"
        }
        evidence["parameter_name"] = parameter_name
        evidence["suggested_name"] = parameter_name
        selected_node_evidence.append(evidence)
    materialization_facts = json.loads(json.dumps(facts, ensure_ascii=False))
    materialization_facts["node_parameter_evidence"] = selected_node_evidence

    legacy_functions: list[dict[str, Any]] = []
    for function_id, definition in definitions.items():
        representative_occurrence_index = representative_occurrence_indices[
            function_id
        ]
        representative_span = occurrence_spans[function_id][
            representative_occurrence_index
        ]
        legacy_parameters: list[dict[str, Any]] = []
        for parameter in parameters_by_function[function_id]:
            for _candidate_id, candidate in parameter["bindings_by_occurrence"][
                representative_occurrence_index
            ]:
                if candidate.get("kind") != "action_arg":
                    continue
                legacy_parameters.append(
                    {
                        "name": parameter["name"],
                        "description": parameter["description"],
                        "source_step_index": int(candidate["source_step_index"]),
                        "arg_name": str(candidate["arg_name"]),
                    }
                )
        legacy_functions.append(
            {
                **definition,
                "source_step_indices": representative_span,
                "parameters": legacy_parameters,
            }
        )

    complete_parameters: list[dict[str, Any]] = []
    complete_names: dict[str, str] = {}
    for _position, function_id, occurrence_index, _span in registered_occurrences:
        for parameter in parameters_by_function[function_id]:
            parameter_name = parameter["name"]
            parameter_value = invocation_arguments[(function_id, occurrence_index)][
                parameter_name
            ]
            complete_name = parameter_name
            suffix = 2
            while (
                complete_name in complete_names
                and complete_names[complete_name] != parameter_value
            ):
                complete_name = f"{parameter_name}_{suffix}"
                suffix += 1
            complete_names[complete_name] = parameter_value
            for _candidate_id, candidate in parameter["bindings_by_occurrence"][
                occurrence_index
            ]:
                if candidate.get("kind") != "action_arg":
                    continue
                complete_parameters.append(
                    {
                        "name": complete_name,
                        "description": parameter["description"],
                        "source_step_index": int(candidate["source_step_index"]),
                        "arg_name": str(candidate["arg_name"]),
                    }
                )
    legacy_plan = {
        "reason": reason,
        "plan": {
            "functions": legacy_functions,
            "complete_function": {
                **raw_complete,
                "parameters": complete_parameters,
            },
        },
    }
    materialized = _materialize_authoring_plan(
        legacy_plan,
        materialization_facts,
        synthesize_missing_locals=False,
    )
    bundle = materialized["bundle"]
    materialized_ids = {
        str(function.get("function_id") or "")
        for function in bundle.get("functions") or ()
        if isinstance(function, dict)
    }
    redundant_definition_ids = {
        function_id
        for function_id in definitions
        if occurrence_spans[function_id] == [source_indices]
    }
    if (
        complete_id not in materialized_ids
        or set(expected_function_ids) - materialized_ids - redundant_definition_ids
    ):
        raise ValueError("function_author_registration_materialization_mismatch")
    source_calls: list[dict[str, Any]] = []
    for _position, function_id, occurrence_index, _span in registered_occurrences:
        call_function_id = function_id
        call_arguments = invocation_arguments[(function_id, occurrence_index)]
        if function_id not in materialized_ids:
            call_function_id = complete_id
            call_arguments = (bundle.get("arguments") or {}).get(complete_id) or {}
        source_calls.append(
            {
                "function_id": call_function_id,
                "arguments": json.loads(
                    json.dumps(call_arguments, ensure_ascii=False)
                ),
            }
        )
    if not source_calls:
        source_calls.append(
            {
                "function_id": complete_id,
                "arguments": json.loads(
                    json.dumps(
                        (bundle.get("arguments") or {}).get(complete_id) or {},
                        ensure_ascii=False,
                    )
                ),
            }
        )
    functions_by_id = {
        str(function.get("function_id") or ""): function
        for function in bundle.get("functions") or ()
        if isinstance(function, dict)
    }
    from omniflow.functions.artifact import bind_function, parse_function_artifact

    for call in source_calls:
        bind_function(
            parse_function_artifact(functions_by_id[call["function_id"]]),
            call["arguments"],
        )
    return {
        "reason": materialized["reason"],
        "bundle": bundle,
        "source_calls": source_calls,
        "authoring_workflow": {
            "schema_version": "omniflow.function-authoring-workflow.v2",
            "definition_count": len(definitions),
            "invocation_count": len(source_calls),
            "definitions": [
                {
                    "function_id": function_id,
                    "occurrence_count": len(occurrence_spans[function_id]),
                    "representative_source_step_indices": occurrence_spans[
                        function_id
                    ][representative_occurrence_indices[function_id]],
                    "registered": function_id in materialized_ids,
                }
                for function_id in definitions
            ],
            "complete_function_id": complete_id,
            "complete_source_indices_normalized": complete_indices_normalized,
            "uncovered_local_source_step_indices": uncovered_local_indices,
            "unselected_candidate_ids": unselected_candidate_ids,
            "validation_notes": (
                "The Agent authored semantic parameter-to-step binding requests. "
                "Compiler Harness resolved every request against immutable source "
                "evidence, validated each selected segment, selected a representative "
                "occurrence shape, and materialized the complete source sequence."
            ),
        },
    }


def _direct_source_authoring_plan(facts: dict[str, Any]) -> dict[str, Any]:
    source_steps = [
        step for step in facts.get("steps") or () if isinstance(step, dict)
    ]
    if not source_steps:
        raise ValueError("successful_source_actions_required")
    goal = " ".join(str(facts.get("goal") or "").split())
    function_id = "complete_source_workflow"
    name = f"Complete task: {goal}" if goal else "Complete source workflow"
    description = (
        f"Complete the requested task: {goal}. Execute every converted source "
        "action in order."
        if goal
        else "Execute every converted source action in order."
    )
    return _materialize_authoring_plan(
        {
            "reason": "Registered the complete converted source workflow.",
            "plan": {
                "functions": [],
                "complete_function": {
                    "function_id": function_id,
                    "name": name,
                    "description": description,
                    "source_step_indices": [
                        int(step["source_step_index"])
                        for step in source_steps
                    ],
                    "parameters": [],
                },
            },
        },
        facts,
        synthesize_missing_locals=False,
    )


def _materialize_authoring_plan(
    value: Any,
    facts: dict[str, Any],
    *,
    synthesize_missing_locals: bool = True,
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"reason", "plan"}:
        raise ValueError("function_author_plan_response_contract_invalid")
    reason = value.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("function_author_reason_must_be_string")
    plan = value.get("plan")
    if not isinstance(plan, dict) or set(plan) != {
        "functions",
        "complete_function",
    }:
        raise ValueError("function_author_plan_contract_invalid")
    raw_functions = plan.get("functions")
    raw_complete_function = plan.get("complete_function")
    if not isinstance(raw_functions, list):
        raise ValueError("function_author_plan_functions_required")
    fallback_complete_function = raw_complete_function is None or (
        not raw_functions
        and isinstance(raw_complete_function, dict)
        and raw_complete_function.get("source_step_indices") == []
    )
    if fallback_complete_function:
        source_steps = [
            step
            for step in facts.get("steps") or ()
            if isinstance(step, dict)
            and isinstance(step.get("source_step_index"), int)
        ]
        if not source_steps:
            raise ValueError("function_author_plan_complete_function_required")
        goal_summary = " ".join(str(facts.get("goal") or "").split())[:240]
        raw_complete_function = {
            "function_id": "complete_source_workflow",
            "name": f"Complete task: {goal_summary}" if goal_summary else "Complete recorded workflow",
            "description": (
                f"Complete the requested task: {goal_summary}. "
                "Execute the successful source workflow in order."
                if goal_summary
                else "Execute the successful source workflow in order."
            ),
            "source_step_indices": [
                int(step["source_step_index"]) for step in source_steps
            ],
            "parameters": [],
        }
    elif not isinstance(raw_complete_function, dict):
        raise ValueError("function_author_plan_complete_function_required")

    source_steps = list(facts.get("steps") or ())
    source_step_positions = {
        int(step.get("source_step_index", position)): position
        for position, step in enumerate(source_steps)
        if isinstance(step, dict)
    }
    materialization_notes: list[str] = []
    candidates = {
        (candidate["source_step_index"], candidate["arg_name"]): candidate
        for candidate in _source_parameter_candidates(facts)
    }
    functions: list[dict[str, Any]] = []
    materialized_function_ids: set[str] = set()
    arguments: dict[str, dict[str, Any]] = {}
    from omniflow.core.schemas import load_canonical_action_schema

    builtin_tool_names = {
        str(tool.get("name") or "").strip()
        for tool in load_canonical_action_schema().get("tools") or ()
        if isinstance(tool, dict)
    }
    if fallback_complete_function:
        materialization_notes.append(
            "Compiler supplied a complete Function from the successful source action sequence because authoring omitted complete_function."
        )
    function_fields = {
        "function_id",
        "name",
        "description",
        "source_step_indices",
        "parameters",
    }
    parameter_fields = {
        "name",
        "description",
        "source_step_index",
        "arg_name",
    }
    complete_function_id = str(raw_complete_function.get("function_id") or "").strip()
    complete_source_indices = raw_complete_function.get("source_step_indices")
    deduplicated_functions: list[Any] = []
    for raw_function in raw_functions:
        if isinstance(raw_function, dict) and (
            str(raw_function.get("function_id") or "").strip()
            == complete_function_id
            or raw_function.get("source_step_indices") == complete_source_indices
        ):
            materialization_notes.append(
                "Compiler omitted a redundant local Function that duplicated "
                "the complete Function."
            )
            continue
        deduplicated_functions.append(raw_function)

    # Deduplicate before measuring local coverage.  Otherwise an authoring
    # response that repeats the complete source sequence as a "local" Function
    # makes every transition appear covered; the duplicate is then removed and
    # the Store silently loses all state-level local Functions.
    covered_source_indices = {
        int(index)
        for raw_function in deduplicated_functions
        if isinstance(raw_function, dict)
        for index in (raw_function.get("source_step_indices") or [])
        if isinstance(index, int) and not isinstance(index, bool)
    }
    missing_local_indices = [
        int(step["source_step_index"])
        for step in source_steps
        if isinstance(step, dict)
        and isinstance(step.get("source_step_index"), int)
        and int(step["source_step_index"]) not in covered_source_indices
    ]
    if synthesize_missing_locals and missing_local_indices and len(source_steps) > 1:
        reserved_ids = {
            str(item.get("function_id") or "").strip()
            for item in deduplicated_functions
            if isinstance(item, dict)
        }
        raw_functions = [
            *deduplicated_functions,
            *_state_level_local_authoring_functions(
                facts,
                source_indices=missing_local_indices,
                reserved_ids=reserved_ids,
            ),
        ]
        materialization_notes.append(
            "Compiler synthesized missing state-level local Functions so every source transition remains reusable."
        )
    else:
        raw_functions = deduplicated_functions
    planned_functions = [
        *((raw_function, False) for raw_function in raw_functions),
        (raw_complete_function, True),
    ]
    for raw_function, is_complete in planned_functions:
        if not isinstance(raw_function, dict) or set(raw_function) != function_fields:
            raise ValueError("function_author_plan_function_contract_invalid")
        requested_function_id = str(raw_function.get("function_id") or "").strip()
        function_id = requested_function_id
        if function_id in builtin_tool_names:
            base_function_id = (
                "complete_source_workflow"
                if is_complete
                else "reusable_source_segment"
            )
            function_id = base_function_id
            suffix = 2
            while function_id in materialized_function_ids:
                function_id = f"{base_function_id}_{suffix}"
                suffix += 1
            materialization_notes.append(
                "Compiler renamed the reserved Function id "
                f"{requested_function_id} to {function_id}; the source Action "
                "tool and arguments remain unchanged."
            )
        name = str(raw_function.get("name") or "").strip()
        description = str(raw_function.get("description") or "").strip()
        raw_indices = raw_function.get("source_step_indices")
        if (
            not function_id
            or not name
            or not description
            or not isinstance(raw_indices, list)
            or not raw_indices
            or any(isinstance(index, bool) or not isinstance(index, int) for index in raw_indices)
        ):
            raise ValueError("function_author_plan_function_invalid")
        indices = list(raw_indices)
        if indices != sorted(set(indices)) or any(
            index not in source_step_positions for index in indices
        ):
            raise ValueError("function_author_plan_source_steps_invalid")
        source_positions = [source_step_positions[index] for index in indices]
        if source_positions != list(
            range(source_positions[0], source_positions[-1] + 1)
        ):
            raise ValueError("function_author_plan_steps_not_contiguous")
        complete_indices = list(source_step_positions)
        if is_complete and indices != complete_indices:
            raise ValueError("function_author_plan_complete_sequence_required")
        if is_complete:
            materialization_notes.append(
                "Compiler materialized the complete successful source sequence."
            )
        function = {
            "schema_version": "omniflow.function.v2",
            "function_id": function_id,
            "name": name,
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
            "bindings": [],
            "render_bindings": [],
            "steps": [
                {
                    "step_index": local_index,
                    "source_state_id": str(
                        source_steps[source_step_positions[source_index]][
                            "before_state_id"
                        ]
                    ),
                    "action": json.loads(
                        json.dumps(
                            source_steps[source_step_positions[source_index]][
                                "action"
                            ],
                            ensure_ascii=False,
                        )
                    ),
                }
                for local_index, source_index in enumerate(indices)
            ],
            "agent_visible": True,
        }
        raw_parameters = raw_function.get("parameters")
        if not isinstance(raw_parameters, list):
            raise ValueError("function_author_plan_parameters_invalid")
        parameter_proposals: list[dict[str, Any]] = []
        source_arguments: dict[str, Any] = {}
        selected_parameter_targets: set[tuple[int, str]] = set()
        for parameter in raw_parameters:
            if not isinstance(parameter, dict) or set(parameter) != parameter_fields:
                raise ValueError("function_author_plan_parameter_contract_invalid")
            source_index = parameter.get("source_step_index")
            arg_name = str(parameter.get("arg_name") or "").strip()
            candidate = candidates.get((source_index, arg_name))
            if candidate is None:
                raise ValueError("function_author_plan_parameter_target_invalid")
            if source_index not in indices:
                raise ValueError("function_author_plan_parameter_step_not_selected")
            target_key = (int(source_index), arg_name)
            if target_key in selected_parameter_targets:
                continue
            parameter_name = str(parameter.get("name") or "").strip()
            parameter_proposals.append(
                {
                    "name": parameter_name,
                    "description": str(parameter.get("description") or "").strip(),
                    "step_index": indices.index(source_index),
                    "arg_name": arg_name,
                }
            )
            source_arguments[parameter_name] = candidate["recorded_value"]
            selected_parameter_targets.add(target_key)
        selected_targets = {
            (
                indices[int(item["step_index"])]
                if isinstance(item.get("step_index"), int)
                and 0 <= int(item["step_index"]) < len(indices)
                else None,
                str(item.get("arg_name") or ""),
            )
            for item in parameter_proposals
        }
        input_value_names: dict[str, str] = {}
        used_parameter_names = {
            str(item.get("name") or "").strip()
            for item in parameter_proposals
            if str(item.get("name") or "").strip()
        }
        for source_index in indices:
            candidate = next(
                (
                    item
                    for (candidate_source_index, candidate_arg_name), item in candidates.items()
                    if candidate_source_index == source_index
                    and candidate_arg_name == "text"
                    and item.get("tool") == "input_text"
                ),
                None,
            )
            target_key = (source_index, "text")
            if candidate is None or target_key in selected_targets:
                continue
            normalized_value = " ".join(
                str(candidate.get("recorded_value") or "").casefold().split()
            )
            parameter_name = input_value_names.get(normalized_value)
            if parameter_name is None:
                parameter_name = str(candidate.get("suggested_name") or "").strip()
                if not parameter_name or parameter_name in used_parameter_names:
                    suffix = 1
                    parameter_name = "input_text"
                    while parameter_name in used_parameter_names:
                        suffix += 1
                        parameter_name = f"input_text_{suffix}"
                input_value_names[normalized_value] = parameter_name
                used_parameter_names.add(parameter_name)
            parameter_proposals.append(
                {
                    "name": parameter_name,
                    "description": "The requested value for this text field",
                    "step_index": indices.index(source_index),
                    "arg_name": "text",
                }
            )
            source_arguments[parameter_name] = candidate["recorded_value"]
        preferred_value_names = {
            " ".join(str(item.get("recorded_value") or "").casefold().split()): str(
                item["suggested_name"]
            ).strip()
            for item in candidates.values()
            if item.get("evidence") == "task_parameter_exact_value"
            and str(item.get("suggested_name") or "").strip()
        }
        if preferred_value_names:
            for proposal in parameter_proposals:
                proposal_step = proposal.get("step_index")
                proposal_arg = str(proposal.get("arg_name") or "")
                if not isinstance(proposal_step, int) or not proposal_arg:
                    continue
                source_index = indices[proposal_step]
                candidate = candidates.get((source_index, proposal_arg))
                if candidate is None:
                    continue
                preferred_name = preferred_value_names.get(
                    " ".join(
                        str(candidate.get("recorded_value") or "").casefold().split()
                    )
                )
                if preferred_name:
                    proposal["name"] = preferred_name
            source_arguments = {}
            for proposal in parameter_proposals:
                proposal_step = proposal.get("step_index")
                proposal_arg = str(proposal.get("arg_name") or "")
                if not isinstance(proposal_step, int) or not proposal_arg:
                    continue
                candidate = candidates.get((indices[proposal_step], proposal_arg))
                if candidate is not None:
                    source_arguments[str(proposal["name"])] = candidate[
                        "recorded_value"
                    ]
        for proposal in parameter_proposals:
            proposal_step = proposal.get("step_index")
            proposal_arg = str(proposal.get("arg_name") or "")
            if not isinstance(proposal_step, int) or not proposal_arg:
                continue
            candidate = candidates.get((indices[proposal_step], proposal_arg))
            if candidate is None:
                continue
            if candidate.get("evidence") != "task_parameter_filename_stem":
                continue
            proposal["name"] = str(candidate["suggested_name"])
            suffix = str(candidate.get("fixed_suffix") or "")
            proposal["description"] = (
                "The requested file name without the fixed "
                f"{suffix} extension selected by this Function."
            )
        if any(
            candidate.get("evidence") == "task_parameter_filename_stem"
            for candidate in candidates.values()
        ):
            source_arguments = {}
            for proposal in parameter_proposals:
                proposal_step = proposal.get("step_index")
                proposal_arg = str(proposal.get("arg_name") or "")
                if not isinstance(proposal_step, int) or not proposal_arg:
                    continue
                candidate = candidates.get((indices[proposal_step], proposal_arg))
                if candidate is not None:
                    source_arguments[str(proposal["name"])] = candidate[
                        "recorded_value"
                    ]
        parameter_names_by_value: dict[tuple[str, str], str] = {}
        used_materialized_names: set[str] = set()
        for proposal in parameter_proposals:
            proposal_step = proposal.get("step_index")
            proposal_arg = str(proposal.get("arg_name") or "")
            if not isinstance(proposal_step, int) or not proposal_arg:
                continue
            candidate = candidates.get((indices[proposal_step], proposal_arg))
            if candidate is None:
                continue
            base_name = str(proposal.get("name") or "input_value").strip()
            recorded_value = " ".join(
                str(candidate.get("recorded_value") or "").casefold().split()
            )
            value_key = (base_name, recorded_value)
            materialized_name = parameter_names_by_value.get(value_key)
            if materialized_name is None:
                materialized_name = base_name
                suffix = 2
                while materialized_name in used_materialized_names:
                    materialized_name = f"{base_name}_{suffix}"
                    suffix += 1
                parameter_names_by_value[value_key] = materialized_name
                used_materialized_names.add(materialized_name)
            proposal["name"] = materialized_name
        source_arguments = {}
        for proposal in parameter_proposals:
            proposal_step = proposal.get("step_index")
            proposal_arg = str(proposal.get("arg_name") or "")
            if not isinstance(proposal_step, int) or not proposal_arg:
                continue
            candidate = candidates.get((indices[proposal_step], proposal_arg))
            if candidate is not None:
                source_arguments[str(proposal["name"])] = candidate[
                    "recorded_value"
                ]
        _materialize_agent_parameters(function, parameter_proposals)
        source_arguments = {
            name: value
            for name, value in source_arguments.items()
            if name in function["input_schema"]["properties"]
        }
        node_parameter_literals = _materialize_node_parameters(
            function,
            facts.get("node_parameter_evidence") or (),
            source_indices=indices,
            source_arguments=source_arguments,
        )
        parameter_literals = _parameter_literal_map(
            parameter_proposals,
            candidates=candidates,
            source_indices=indices,
        )
        parameter_literals.update(node_parameter_literals)
        function["name"] = _redact_parameter_literals(
            function["name"],
            function,
            parameter_literals=parameter_literals,
        )
        function["description"] = _description_with_action_plan(
            description,
            function=function,
            source_steps=source_steps,
            source_indices=indices,
            source_step_positions=source_step_positions,
            parameter_literals=parameter_literals,
        )
        if function_id in materialized_function_ids:
            if not is_complete:
                raise ValueError("function_author_plan_duplicate_function_id")
            functions = [
                item
                for item in functions
                if item.get("function_id") != function_id
            ]
            arguments.pop(function_id, None)
            materialized_function_ids.remove(function_id)
            materialization_notes.append(
                "Compiler kept the complete Function envelope when a local Function reused its id."
            )
        materialized_function_ids.add(function_id)
        functions.append(function)
        arguments[function_id] = source_arguments

    return {
        "reason": " ".join([reason.strip(), *materialization_notes]),
        "bundle": {
            "schema_version": "omniflow.function-bundle.v2",
            "run_id": str(facts["run_id"]),
            "arguments": arguments,
            "checker_rules": [],
            "functions": functions,
        },
    }


def _description_with_action_plan(
    description: str,
    *,
    function: dict[str, Any],
    source_steps: list[dict[str, Any]],
    source_indices: list[int],
    source_step_positions: dict[int, int],
    parameter_literals: dict[str, str] | None = None,
) -> str:
    parameter_by_target = {
        str(binding["target"]): str(binding["source"]).removeprefix("$.arguments.")
        for binding in function.get("bindings") or ()
        if isinstance(binding, dict)
    }
    parameter_by_render_step = {
        int(binding["step_index"]): str(binding["source"]).removeprefix(
            "$.arguments."
        )
        for binding in function.get("render_bindings") or ()
        if isinstance(binding, dict)
        and isinstance(binding.get("step_index"), int)
        and str(binding.get("source") or "").startswith("$.arguments.")
    }
    plan: list[str] = []
    for local_index, step in enumerate(function.get("steps") or ()):
        action = step.get("action") if isinstance(step, dict) else None
        if not isinstance(action, dict):
            continue
        args = action.get("args")
        semantic_args: dict[str, Any] = {}
        if isinstance(args, dict):
            for arg_name, arg_value in args.items():
                if arg_name in {"x", "y", "x1", "y1", "x2", "y2"}:
                    continue
                target = f"$.steps[{local_index}].action.args.{arg_name}"
                parameter_name = parameter_by_target.get(target)
                semantic_args[arg_name] = (
                    f"<{parameter_name}>" if parameter_name else arg_value
                )
        tool = str(action.get("tool") or "")
        render_parameter = parameter_by_render_step.get(local_index)
        if render_parameter and tool in {"click", "long_press"}:
            semantic_args["target"] = f"<{render_parameter}>"
        encoded_args = ",".join(
            f"{name}={json.dumps(value, ensure_ascii=False, separators=(',', ':'))}"
            for name, value in semantic_args.items()
        )
        entry = f"{local_index + 1}:{tool}"
        if encoded_args:
            entry += f"({encoded_args})"
        source_position = source_step_positions[source_indices[local_index]]
        metadata = source_steps[source_position].get("metadata")
        purpose = ""
        if isinstance(metadata, dict):
            purpose = str(
                metadata.get("purpose")
                or metadata.get("action_description")
                or metadata.get("summary")
                or ""
            ).strip()
        if (
            purpose
            and not purpose.startswith("Manually reviewed recorded ")
            and (tool == "swipe" or not semantic_args)
        ):
            entry += f"[{purpose}]"
        plan.append(entry)
    semantic_description = str(description or "").split(" Action plan:", 1)[0].strip()
    semantic_description = _redact_parameter_literals(
        semantic_description,
        function,
        parameter_literals=parameter_literals,
    )
    encoded_plan = ";".join(plan)
    return f"{semantic_description} Action plan: {encoded_plan}"


def _redact_parameter_literals(
    text: str,
    function: dict[str, Any],
    *,
    parameter_literals: dict[str, str] | None = None,
) -> str:
    """Remove source-instance values from agent-facing Function metadata.

    The authoring model is instructed to describe semantic values rather than
    copy source-instance literals, but the compiler must enforce that contract:
    source values remain in the private source arguments/bindings while the
    Router sees the generated parameter names.  This is deliberately limited
    to values proven by a Function binding and never touches coordinates or
    unbound UI labels.
    """

    literals: dict[str, str] = {
        str(literal): str(parameter_name)
        for parameter_name, literal in (parameter_literals or {}).items()
        if str(literal).strip() and str(parameter_name).strip()
    }
    steps = function.get("steps") or ()
    for raw_binding in function.get("bindings") or ():
        if not isinstance(raw_binding, dict):
            continue
        parameter_name = str(raw_binding.get("source") or "")
        if not parameter_name.startswith("$.arguments."):
            continue
        parameter_name = parameter_name.removeprefix("$.arguments.").strip()
        target = str(raw_binding.get("target") or "")
        match = re.fullmatch(
            r"\$\.steps\[(\d+)\]\.action\.args\.([A-Za-z_][A-Za-z0-9_]*)",
            target,
        )
        if match is None:
            continue
        step_index = int(match.group(1))
        argument_name = match.group(2)
        if step_index < 0 or step_index >= len(steps):
            continue
        step = steps[step_index]
        action = step.get("action") if isinstance(step, dict) else None
        args = action.get("args") if isinstance(action, dict) else None
        value = args.get(argument_name) if isinstance(args, dict) else None
        if isinstance(value, str) and value.strip():
            literals.setdefault(value, parameter_name)
    for raw_binding in function.get("render_bindings") or ():
        if not isinstance(raw_binding, dict):
            continue
        source = str(raw_binding.get("source") or "")
        if not source.startswith("$.arguments."):
            continue
        literal = str(raw_binding.get("recorded_value") or "").strip()
        parameter_name = source.removeprefix("$.arguments.").strip()
        if literal and parameter_name:
            literals.setdefault(literal, parameter_name)

    redacted = str(text or "")
    for literal, parameter_name in sorted(
        literals.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        redacted = redacted.replace(literal, f"<{parameter_name}>")
    return redacted


def _parameter_literal_map(
    parameters: list[dict[str, Any]],
    *,
    candidates: dict[tuple[int, str], dict[str, Any]],
    source_indices: list[int],
) -> dict[str, str]:
    literals: dict[str, str] = {}
    for parameter in parameters:
        if not isinstance(parameter, dict):
            continue
        local_index = parameter.get("step_index")
        if not isinstance(local_index, int) or not 0 <= local_index < len(source_indices):
            continue
        arg_name = str(parameter.get("arg_name") or "").strip()
        candidate = candidates.get((source_indices[local_index], arg_name))
        name = str(parameter.get("name") or "").strip()
        if candidate is None or not name:
            continue
        literal = str(candidate.get("recorded_value") or "").strip()
        if literal:
            literals[name] = literal
    return literals


def _materialize_node_parameters(
    function: dict[str, Any],
    candidates: Any,
    *,
    source_indices: list[int],
    source_arguments: dict[str, Any],
) -> dict[str, str]:
    """Bind task values into the source node rendered for point actions."""

    literals: dict[str, str] = {}
    selected: dict[int, list[dict[str, Any]]] = {}
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        source_index = candidate.get("source_step_index")
        if (
            not isinstance(source_index, int)
            or source_index not in source_indices
        ):
            continue
        selected.setdefault(int(source_index), []).append(candidate)
    render_bindings = function.setdefault("render_bindings", [])
    if not isinstance(render_bindings, list):
        raise ValueError("function_render_bindings_invalid")
    properties = function["input_schema"]["properties"]
    required = function["input_schema"]["required"]
    for local_index, source_index in enumerate(source_indices):
        for candidate in selected.get(int(source_index), ()):
            parameter_name = str(candidate.get("parameter_name") or "").strip()
            task_value = str(candidate.get("task_parameter_value") or "")
            recorded_value = str(candidate.get("recorded_value") or "")
            if not parameter_name or not task_value or not recorded_value:
                continue
            base_parameter_name = parameter_name
            suffix = 2
            while (
                parameter_name in source_arguments
                and str(source_arguments[parameter_name]) != task_value
            ):
                parameter_name = f"{base_parameter_name}_{suffix}"
                suffix += 1
            if parameter_name not in properties:
                properties[parameter_name] = {
                    "type": "string",
                    "description": (
                        f"The requested {parameter_name.replace('_', ' ')} "
                        "visible on the selected GUI item."
                    ),
                }
                required.append(parameter_name)
            source_arguments[parameter_name] = task_value
            literals[parameter_name] = recorded_value
            binding = {
                "source": f"$.arguments.{parameter_name}",
                "step_index": local_index,
                "node_id": str(candidate.get("node_id") or "").strip(),
                "attribute": str(candidate.get("attribute") or "").strip(),
                "recorded_value": recorded_value,
            }
            if not binding["node_id"] or not binding["attribute"]:
                raise ValueError("function_node_render_binding_evidence_invalid")
            duplicate = any(
                isinstance(existing, dict)
                and all(existing.get(key) == binding[key] for key in binding)
                for existing in render_bindings
            )
            if not duplicate:
                render_bindings.append(binding)
    return literals


def _materialize_agent_parameters(
    function: dict[str, Any],
    parameters: list[dict[str, Any]],
) -> None:
    candidates = {
        (candidate["step_index"], candidate["arg_name"]): candidate
        for candidate in parameter_candidates(function)
    }
    values_by_name: dict[str, str] = {}
    for parameter in parameters:
        name = str(parameter.get("name") or "").strip()
        description = str(parameter.get("description") or "").strip()[:240]
        step_index = parameter.get("step_index")
        arg_name = str(parameter.get("arg_name") or "").strip()
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", name) is None:
            raise ValueError("function_author_plan_parameter_name_invalid")
        candidate = candidates.get((step_index, arg_name))
        if candidate is None:
            raise ValueError("function_author_plan_parameter_target_invalid")
        recorded_value = str(candidate["recorded_value"])
        binding_steps = [step_index]
        if (
            arg_name == "text"
            and function["steps"][step_index]["action"].get("tool")
            == "input_text"
        ):
            normalized_recorded_value = " ".join(recorded_value.casefold().split())
            bound_targets = {
                str(binding.get("target") or "")
                for binding in function.get("bindings") or ()
                if isinstance(binding, dict)
            }
            for repeated_step_index, repeated_step in enumerate(function["steps"]):
                if repeated_step_index == step_index:
                    continue
                repeated_action = repeated_step.get("action")
                if not isinstance(repeated_action, dict):
                    continue
                if repeated_action.get("tool") != "input_text":
                    continue
                repeated_target = (
                    f"$.steps[{repeated_step_index}].action.args.text"
                )
                if repeated_target in bound_targets:
                    continue
                repeated_value = repeated_action.get("args", {}).get("text")
                if not isinstance(repeated_value, str):
                    continue
                if " ".join(repeated_value.casefold().split()) != normalized_recorded_value:
                    continue
                binding_steps.append(repeated_step_index)
        unbound_binding_steps = []
        for binding_step_index in binding_steps:
            target = (
                f"$.steps[{binding_step_index}].action.args.{arg_name}"
            )
            if any(
                str(binding.get("target") or "") == target
                for binding in function.get("bindings") or ()
                if isinstance(binding, dict)
            ):
                continue
            unbound_binding_steps.append(binding_step_index)
        if not unbound_binding_steps:
            continue
        if name in values_by_name and values_by_name[name] != recorded_value:
            raise ValueError("function_author_plan_parameter_name_ambiguous")
        if name not in function["input_schema"]["properties"]:
            definition = {"type": "string"}
            if description:
                definition["description"] = description
            function["input_schema"]["properties"][name] = definition
            function["input_schema"]["required"].append(name)
        for binding_step_index in unbound_binding_steps:
            target = (
                f"$.steps[{binding_step_index}].action.args.{arg_name}"
            )
            function["bindings"].append(
                {
                    "source": f"$.arguments.{name}",
                    "target": target,
                }
            )
            function["steps"][binding_step_index]["action"]["args"][arg_name] = ""
        values_by_name[name] = recorded_value
