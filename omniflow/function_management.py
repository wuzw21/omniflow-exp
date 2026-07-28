from __future__ import annotations

import json
import re
from typing import Any, Callable

from omniflow.artifact import parse_function_artifact
from omniflow.schemas import canonicalize_action


_PARAMETER_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
_PARAMETERIZABLE_ACTION_ARGS = {
    "input_text": frozenset({"text"}),
}


def edit_function(
    value: dict[str, Any],
    edits: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    original = parse_function_artifact(value).to_dict()
    updated = json.loads(json.dumps(original, ensure_ascii=False))
    steps = updated["steps"]
    changes: list[dict[str, Any]] = []
    deletes: set[int] = set()
    for edit in edits:
        if not isinstance(edit, dict):
            continue
        index = _integer(edit.get("index"), -1)
        if index not in range(len(steps)):
            continue
        action = steps[index]["action"]
        tool = action["tool"]
        expected_tool = str(edit.get("expected_tool") or "").strip()
        if expected_tool and expected_tool != tool:
            continue
        operation = str(edit.get("op") or "").strip().lower()
        if operation == "delete":
            deletes.add(index)
        elif operation == "replace_args":
            patch = edit.get("args")
            if not isinstance(patch, dict) or not patch:
                continue
            canonical = canonicalize_action(
                {"tool": tool, "args": {**action["args"], **patch}},
                replayable_only=True,
                persisted_only=True,
            )
            if canonical["args"] == action["args"]:
                continue
            action["args"] = canonical["args"]
            changes.append(_change("replace_args", index, tool, edit.get("reason")))
    if deletes and (len(deletes) >= len(steps) or updated["bindings"]):
        deletes.clear()
    for index in sorted(deletes, reverse=True):
        tool = steps[index]["action"]["tool"]
        del steps[index]
        reason = next(
            (
                edit.get("reason")
                for edit in edits
                if isinstance(edit, dict)
                and str(edit.get("op") or "").strip().lower() == "delete"
                and _integer(edit.get("index"), -1) == index
            ),
            None,
        )
        changes.append(_change("delete", index, tool, reason))
    for index, step in enumerate(steps):
        step["step_index"] = index
    return parse_function_artifact(updated).to_dict(), changes


def enhance_function(
    value: dict[str, Any],
    run_log: dict[str, Any],
    complete_json: Callable[[str], str],
) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    original = parse_function_artifact(value).to_dict()
    proposal = _json_object(complete_json(_enhancement_prompt(original, run_log)))
    _require_checker_evidence(proposal, run_log)
    updated = json.loads(json.dumps(original, ensure_ascii=False))
    changes: list[dict[str, Any]] = []
    for field, limit in (("name", 80), ("description", 2000)):
        replacement = str(proposal.get(field) or "").strip()[:limit]
        if replacement and replacement != updated[field]:
            updated[field] = replacement
            changes.append({"part": "function", "field": field})
    if "parameters" in proposal and _apply_parameters(
        updated,
        proposal["parameters"],
        run_log,
    ):
        changes.append({"part": "function", "field": "parameters"})
    if "checker_rules" in proposal:
        candidate = dict(updated)
        candidate["checker_rules"] = proposal["checker_rules"]
        canonical_rules = parse_function_artifact(candidate).to_dict()["checker_rules"]
        if canonical_rules != updated["checker_rules"]:
            updated["checker_rules"] = canonical_rules
            changes.append({"part": "function", "field": "checker_rules"})
    canonical = parse_function_artifact(updated).to_dict()
    return canonical, changes, "enhanced" if changes else "unchanged"


def _enhancement_prompt(function: dict[str, Any], run_log: dict[str, Any]) -> str:
    steps = [
        {
            "index": index,
            "tool": step["action"]["tool"],
            "target": str(step["action"]["args"].get("target_description") or "")[:120],
        }
        for index, step in enumerate(function["steps"])
    ]
    run_log_facts = {
        "run_id": str(run_log.get("run_id") or ""),
        "goal": str(run_log.get("goal") or ""),
        "steps": [
            {
                key: step.get(key)
                for key in (
                    "step_index",
                    "before_state_id",
                    "action",
                    "result",
                    "after_state_id",
                    "metadata",
                )
            }
            for step in run_log.get("steps") or ()
            if isinstance(step, dict)
        ],
    }
    brief = {
        "function_id": function["function_id"],
        "name": function["name"],
        "description": function["description"],
        "steps": steps,
        "parameter_candidates": _parameter_candidates(function),
        "run_log": run_log_facts,
    }
    return f"""
Improve the reusable Android automation Function below for future recall.
Return one JSON object with optional keys: name, description, parameters, and checker_rules.
Describe when to reuse the Function, visible operations, inputs, success signal, and avoid cases.
Never add, remove, reorder, or alter actions, tools, arguments, coordinates, selectors, or function_id.
Do not invent app state. Use the same language as the current name/description.

parameters is an array of semantic input bindings. Each item has exactly:
{{"name":"query","description":"Text to search for","step_index":1,"arg_name":"text"}}.
Only select entries listed in parameter_candidates and copy step_index and arg_name exactly.
Choose a stable identifier name and a concise user-facing description. Return parameters=[]
when the recorded value is intentionally fixed. Do not return input_schema, bindings, or steps;
the runtime derives them and verifies the original successful RunLog evidence.

checker_rules is an ordered array. Each rule has exactly:
{{"schema_version":"omniflow.checker_rule.v1","trigger":"text_contains(\\"跳过广告\\")","source_state_id":"state-id","action":{{"tool":"click","args":{{"x":900,"y":100}}}}}}.
Create a checker only when RunLog metadata explicitly identifies a successful recovery step
(metadata.origin == "checker" and result.success == true). Copy its action and before_state_id.
When metadata.checker_trigger exists, copy it exactly. Otherwise return checker_rules=[].

Function:
{json.dumps(brief, ensure_ascii=False, separators=(",", ":"))}
""".strip()


def _parameter_candidates(function: dict[str, Any]) -> list[dict[str, Any]]:
    bound_targets = {
        str(binding.get("target") or "")
        for binding in function.get("bindings") or ()
        if isinstance(binding, dict)
    }
    candidates: list[dict[str, Any]] = []
    for step_index, step in enumerate(function.get("steps") or ()):
        if not isinstance(step, dict):
            continue
        action = step.get("action")
        if not isinstance(action, dict):
            continue
        tool = str(action.get("tool") or "")
        args = action.get("args")
        if not isinstance(args, dict):
            continue
        for arg_name in _PARAMETERIZABLE_ACTION_ARGS.get(tool, ()):
            target = f"$.steps[{step_index}].action.args.{arg_name}"
            value = args.get(arg_name)
            if target in bound_targets or not isinstance(value, str) or not value:
                continue
            candidates.append(
                {
                    "step_index": step_index,
                    "tool": tool,
                    "arg_name": arg_name,
                    "recorded_value": value,
                }
            )
    return candidates


def _apply_parameters(
    function: dict[str, Any],
    proposal: Any,
    run_log: dict[str, Any],
) -> bool:
    if not isinstance(proposal, list):
        raise ValueError("function_enhancement_parameters_invalid")
    candidates = {
        (candidate["step_index"], candidate["arg_name"]): candidate
        for candidate in _parameter_candidates(function)
    }
    schema = function["input_schema"]
    properties = schema["properties"]
    required = schema["required"]
    bindings = function["bindings"]
    existing_names = set(properties)
    changed = False
    for parameter in proposal:
        if not isinstance(parameter, dict) or set(parameter) - {
            "name",
            "description",
            "step_index",
            "arg_name",
        }:
            raise ValueError("function_enhancement_parameter_contract_invalid")
        name = str(parameter.get("name") or "").strip()
        description = str(parameter.get("description") or "").strip()[:240]
        step_index = _integer(parameter.get("step_index"), -1)
        arg_name = str(parameter.get("arg_name") or "").strip()
        candidate = candidates.get((step_index, arg_name))
        if candidate is None:
            raise ValueError("function_enhancement_parameter_target_invalid")
        if _PARAMETER_NAME.fullmatch(name) is None or name in existing_names:
            raise ValueError("function_enhancement_parameter_name_invalid")
        step = function["steps"][step_index]
        value = step["action"]["args"][arg_name]
        if not _has_parameter_evidence(step, arg_name, value, run_log):
            raise ValueError("function_enhancement_parameter_evidence_missing")
        definition: dict[str, Any] = {"type": "string"}
        if description:
            definition["description"] = description
        properties[name] = definition
        required.append(name)
        bindings.append(
            {
                "source": f"$.arguments.{name}",
                "target": f"$.steps[{step_index}].action.args.{arg_name}",
            }
        )
        step["action"]["args"][arg_name] = ""
        existing_names.add(name)
        changed = True
    return changed


def _has_parameter_evidence(
    function_step: dict[str, Any],
    arg_name: str,
    value: Any,
    run_log: dict[str, Any],
) -> bool:
    for raw_step in run_log.get("steps") or ():
        if not isinstance(raw_step, dict):
            continue
        result = raw_step.get("result")
        action = raw_step.get("action")
        if not isinstance(result, dict) or result.get("success") is not True:
            continue
        if not isinstance(action, dict):
            continue
        args = action.get("args")
        if not isinstance(args, dict):
            continue
        if (
            str(raw_step.get("before_state_id") or "")
            == str(function_step.get("source_state_id") or "")
            and str(action.get("tool") or "")
            == str(function_step.get("action", {}).get("tool") or "")
            and args.get(arg_name) == value
        ):
            return True
    return False


def _require_checker_evidence(
    proposal: dict[str, Any],
    run_log: dict[str, Any],
) -> None:
    if "checker_rules" not in proposal:
        return
    evidence: list[tuple[str, dict[str, Any], str]] = []
    for step in run_log.get("steps") or ():
        if not isinstance(step, dict):
            continue
        metadata = step.get("metadata")
        result = step.get("result")
        if not isinstance(metadata, dict) or not isinstance(result, dict):
            continue
        if metadata.get("origin") != "checker" or result.get("success") is not True:
            continue
        state_id = str(step.get("before_state_id") or "").strip()
        if not state_id:
            continue
        evidence.append(
            (
                state_id,
                canonicalize_action(step.get("action"), replayable_only=True),
                str(metadata.get("checker_trigger") or "").strip(),
            )
        )
    candidate = {
        "schema_version": "omniflow.function.v2",
        "function_id": "EvidenceCheck",
        "name": "Evidence check",
        "description": "Validate proposed checker rules.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        "bindings": [],
        "steps": [
            {
                "step_index": 0,
                "source_state_id": "evidence",
                "action": {"tool": "wait", "args": {"duration_ms": 1}},
            }
        ],
        "checker_rules": proposal.get("checker_rules"),
        "agent_visible": False,
    }
    rules = parse_function_artifact(candidate).to_dict()["checker_rules"]
    for rule in rules:
        matches = [
            item
            for item in evidence
            if item[0] == rule["source_state_id"] and item[1] == rule["action"]
        ]
        if not matches:
            raise ValueError("checker_rule_missing_recovery_evidence")
        captured = {item[2] for item in matches if item[2]}
        if captured and rule["trigger"] not in captured:
            raise ValueError("checker_rule_trigger_mismatch")


def _json_object(raw: str) -> dict[str, Any]:
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("function_enhancement_json_missing")
    value = json.loads(raw[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("function_enhancement_json_invalid")
    return value


def _change(operation: str, index: int, tool: str, reason: Any) -> dict[str, Any]:
    value = {
        "part": "action",
        "field": operation,
        "op": operation,
        "step_index": index,
        "tool": tool,
    }
    text = str(reason or "").strip()
    if text:
        value["reason"] = text
    return value


def _integer(value: Any, default: int) -> int:
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
