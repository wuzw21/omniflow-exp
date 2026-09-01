from __future__ import annotations

import json
import re
from datetime import date
from typing import Any, Callable

from omniflow.core.schemas import canonicalize_action
from omniflow.functions.artifact import parse_function_artifact

_PARAMETER_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
_PARAMETER_TARGET = re.compile(
    r"^\$\.steps\[(?P<step_index>\d+)]\.action\.args\.(?P<arg_name>[A-Za-z_][A-Za-z0-9_]*)$"
)
_PARAMETERIZABLE_ACTION_ARGS = {
    # App launch is part of the public Function API.  The package is a
    # task-level input when a global startup Function is reused for another
    # app; keeping it out of this set silently freezes the source app.
    "open_app": frozenset({"package_name"}),
    "input_text": frozenset({"text"}),
}
_NON_SEMANTIC_TASK_PARAMETER_NAMES = frozenset(
    {
        "seed",
        "source_seed",
        "evaluation_seed",
        "task_random_seed",
    }
)


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
    if deletes and (
        len(deletes) >= len(steps)
        or updated["bindings"]
        or updated.get("render_bindings")
    ):
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
    *,
    instruction: str = "",
) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    original = parse_function_artifact(value).to_dict()
    proposal = _json_object(
        complete_json(_enhancement_prompt(original, run_log, instruction=instruction))
    )
    updated = json.loads(json.dumps(original, ensure_ascii=False))
    changes: list[dict[str, Any]] = []
    for field, limit in (("name", 80), ("description", 2000)):
        replacement = str(proposal.get(field) or "").strip()[:limit]
        if replacement and replacement != updated[field]:
            updated[field] = replacement
            changes.append({"part": "function", "field": field})
    if "parameters" in proposal:
        parameters_changed = apply_parameters(
            updated,
            proposal["parameters"],
            run_log,
        )
        if parameters_changed:
            _generalize_bound_literals(updated, original)
            changed_fields = {
                str(change.get("field") or "")
                for change in changes
                if isinstance(change, dict)
            }
            for field in ("name", "description"):
                if updated[field] != original[field] and field not in changed_fields:
                    changes.append({"part": "function", "field": field})
            changes.append({"part": "function", "field": "parameters"})
    canonical = parse_function_artifact(updated).to_dict()
    return canonical, changes, "enhanced" if changes else "unchanged"


def _enhancement_prompt(
    function: dict[str, Any],
    run_log: dict[str, Any],
    *,
    instruction: str = "",
) -> str:
    steps = [
        {
            "index": index,
            "tool": step["action"]["tool"],
        }
        for index, step in enumerate(function["steps"])
    ]
    run_log_facts = {
        "run_id": str(run_log.get("run_id") or ""),
        "goal": str(run_log.get("goal") or ""),
        "task_parameters": (
            dict(run_log.get("task_parameters") or {})
            if isinstance(run_log.get("task_parameters"), dict)
            else {}
        ),
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
        "parameter_candidates": _eligible_parameter_candidates(function, run_log),
        "run_log": run_log_facts,
        "user_instruction": str(instruction or "").strip()[:2000],
    }
    return f"""
Improve the reusable Android automation Function below for future recall.
Return one JSON object with optional keys: name, description, and parameters.
Describe when to reuse the Function, visible operations, inputs, success signal, and avoid cases.
Never add, remove, reorder, or alter actions, tools, arguments, coordinates, selectors, or function_id.
Do not invent app state. Use the same language as the current name/description.
Treat user_instruction as optional enhancement guidance. It may refine semantic naming,
description and parameter selection, but it cannot override
the action immutability and RunLog evidence requirements above.

parameters is an array of semantic input bindings. Each item has exactly:
{{"name":"query","description":"Text to search for","step_index":1,"arg_name":"text"}}.
parameter_candidates already contains only compiler-approved, goal/task-backed values.
Select only entries from that list; copy step_index and arg_name exactly and use
suggested_name when present. Return parameters=[] when the list is empty. Stable UI
labels, navigation labels, coordinates, repetition counts, and an app fixed by the
Function's capability stay fixed.
For open_app.package_name, use the name package_name and describe an installed Android
package identifier; never use a friendly app label as its value. Choose stable parameter
names and concise user-facing descriptions. When a value becomes a parameter, remove its
recorded instance literal from name and description and describe the requested semantic
value instead. Give distinct semantic values distinct parameter names. Reuse one name
for multiple targets only when those targets intentionally consume the same recorded
semantic value. Do not return input_schema, bindings, or steps; the compiler derives
them and verifies the original successful RunLog evidence.

Function:
{json.dumps(brief, ensure_ascii=False, separators=(",", ":"))}
""".strip()


def parameter_candidates(function: dict[str, Any]) -> list[dict[str, Any]]:
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
        for arg_name in sorted(_PARAMETERIZABLE_ACTION_ARGS.get(tool, ())):
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


def _eligible_parameter_candidates(
    function: dict[str, Any],
    run_log: dict[str, Any],
) -> list[dict[str, Any]]:
    eligible: list[dict[str, Any]] = []
    for candidate in parameter_candidates(function):
        step_index = int(candidate["step_index"])
        evidence = semantic_parameter_evidence(
            function["steps"][step_index],
            str(candidate["arg_name"]),
            candidate["recorded_value"],
            run_log,
        )
        if evidence is None or not _has_parameter_evidence(
            function["steps"][step_index],
            str(candidate["arg_name"]),
            candidate["recorded_value"],
            run_log,
        ):
            continue
        eligible.append({**candidate, **evidence})
    return eligible


def apply_parameters(
    function: dict[str, Any],
    proposal: Any,
    run_log: dict[str, Any],
) -> bool:
    if not isinstance(proposal, list):
        raise ValueError("function_enhancement_parameters_invalid")
    candidates = {
        (candidate["step_index"], candidate["arg_name"]): candidate
        for candidate in parameter_candidates(function)
    }
    schema = function["input_schema"]
    properties = schema["properties"]
    required = schema["required"]
    bindings = function["bindings"]
    proposed_values: dict[str, str] = {}
    changed = False
    for parameter in proposal:
        if not isinstance(parameter, dict) or set(parameter) - {
            "name",
            "description",
            "step_index",
            "arg_name",
        }:
            raise ValueError("function_enhancement_parameter_contract_invalid")
        proposed_name = str(parameter.get("name") or "").strip()
        description = str(parameter.get("description") or "").strip()[:240]
        step_index = _integer(parameter.get("step_index"), -1)
        arg_name = str(parameter.get("arg_name") or "").strip()
        candidate = candidates.get((step_index, arg_name))
        if candidate is None:
            raise ValueError("function_enhancement_parameter_target_invalid")
        step = function["steps"][step_index]
        value = step["action"]["args"][arg_name]
        if not _has_parameter_evidence(step, arg_name, value, run_log):
            raise ValueError("function_enhancement_parameter_evidence_missing")
        semantic_evidence = semantic_parameter_evidence(
            step,
            arg_name,
            value,
            run_log,
        )
        if semantic_evidence is None:
            raise ValueError("function_enhancement_parameter_semantics_missing")
        name = str(
            semantic_evidence.get("suggested_name") or proposed_name
        ).strip()
        if _PARAMETER_NAME.fullmatch(name) is None:
            raise ValueError("function_enhancement_parameter_name_invalid")
        normalized_value = _normalize_parameter_value(value)
        prior_value = proposed_values.get(name)
        if prior_value is not None and prior_value != normalized_value:
            raise ValueError("function_enhancement_parameter_name_ambiguous")
        already_defined = name in properties
        if already_defined and prior_value is None:
            if semantic_evidence.get("suggested_name") != name:
                raise ValueError("function_enhancement_parameter_name_invalid")
            definition = properties.get(name)
            if not isinstance(definition, dict) or definition.get("type") != "string":
                raise ValueError("function_enhancement_parameter_name_invalid")
        if not already_defined:
            definition = {"type": "string"}
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
        proposed_values[name] = normalized_value
        changed = True
    return changed


def semantic_parameter_evidence(
    function_step: dict[str, Any],
    arg_name: str,
    value: Any,
    run_log: dict[str, Any],
) -> dict[str, str] | None:
    action = (
        function_step.get("action")
        if isinstance(function_step.get("action"), dict)
        else {}
    )
    tool = str(action.get("tool") or "").strip()
    normalized_value = _normalize_parameter_value(value)
    if not normalized_value:
        return None
    if tool == "open_app" and arg_name == "package_name":
        for evidence in run_log.get("parameter_evidence") or ():
            if not isinstance(evidence, dict):
                continue
            if (
                str(evidence.get("arg_name") or "") == "package_name"
                and _normalize_parameter_value(evidence.get("recorded_value"))
                == normalized_value
                and evidence.get("task_parameter_name") == "app_name"
                and evidence.get("value_contract") == "android_package_name"
            ):
                return {
                    "evidence": "task_parameter_app_name",
                    "suggested_name": "package_name",
                }
        return None

    task_matches = _matching_task_parameter_names(
        run_log,
        normalized_value,
        function_step=function_step,
    )
    if task_matches:
        evidence = {"evidence": "task_parameter_exact_value"}
        if len(task_matches) == 1:
            evidence["suggested_name"] = task_matches[0]
        return evidence

    goal = " ".join(str(run_log.get("goal") or "").casefold().split())
    if tool == "input_text" and arg_name == "text":
        return {
            "evidence": (
                "goal_input_literal"
                if _contains_parameter_literal(goal, normalized_value)
                else "recorded_input_text"
            )
        }
    return None


def _matching_task_parameter_names(
    run_log: dict[str, Any],
    normalized_value: str,
    *,
    function_step: dict[str, Any] | None = None,
) -> list[str]:
    task_parameters = run_log.get("task_parameters")
    if not isinstance(task_parameters, dict):
        return []
    names = [
        name
        for raw_name, raw_value in task_parameters.items()
        if (name := str(raw_name or "").strip())
        and name not in _NON_SEMANTIC_TASK_PARAMETER_NAMES
        and _PARAMETER_NAME.fullmatch(name) is not None
        and _normalize_parameter_value(raw_value) == normalized_value
    ]
    derived = _derived_task_parameter_name(
        run_log,
        normalized_value,
        function_step=function_step,
    )
    if derived and derived not in names:
        names.append(derived)
    return names


def _derived_task_parameter_name(
    run_log: dict[str, Any],
    normalized_value: str,
    *,
    function_step: dict[str, Any] | None,
) -> str:
    task_parameters = run_log.get("task_parameters")
    if not isinstance(task_parameters, dict):
        return ""
    try:
        day = int(task_parameters.get("day"))
        month = int(task_parameters.get("month"))
        year = int(task_parameters.get("year"))
    except (TypeError, ValueError):
        day = month = year = 0
    try:
        hour = int(task_parameters.get("hour"))
    except (TypeError, ValueError):
        hour = -1
    try:
        duration_mins = int(task_parameters.get("duration_mins"))
    except (TypeError, ValueError):
        duration_mins = -1

    if day > 0 and month > 0 and year > 0:
        try:
            event_date = date(year, month, day)
        except ValueError:
            event_date = None
        if event_date is not None:
            full_date = f"{day} {event_date.strftime('%B')} {year}".casefold()
            if normalized_value == full_date:
                return "event_date"

    if 0 <= hour <= 23:
        if normalized_value == f"{hour:02d}:00".casefold():
            return "event_start_time"
        if normalized_value == f"{hour} hours".casefold():
            return "event_start_hour"

    if duration_mins >= 0 and 0 <= hour <= 23:
        end_total = hour * 60 + duration_mins
        end_hour = (end_total // 60) % 24
        end_minute = end_total % 60
        if normalized_value == f"{end_hour} hours".casefold():
            return "event_end_hour"
    return ""


def _normalize_parameter_value(value: Any) -> str:
    return " ".join(str(value or "").casefold().split())


def _contains_parameter_literal(text: str, literal: str) -> bool:
    if not literal:
        return False
    pattern = re.escape(literal)
    if literal[0].isalnum():
        pattern = rf"(?<!\w){pattern}"
    if literal[-1].isalnum():
        pattern = rf"{pattern}(?!\w)"
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def _generalize_bound_literals(
    function: dict[str, Any],
    source_function: dict[str, Any],
) -> None:
    for binding in function.get("bindings") or ():
        if not isinstance(binding, dict):
            continue
        source = str(binding.get("source") or "")
        parameter_name = source.removeprefix("$.arguments.")
        target = _PARAMETER_TARGET.fullmatch(str(binding.get("target") or ""))
        if not parameter_name or parameter_name == source or target is None:
            continue
        step_index = int(target.group("step_index"))
        arg_name = target.group("arg_name")
        try:
            recorded_value = source_function["steps"][step_index]["action"]["args"][
                arg_name
            ]
        except (IndexError, KeyError, TypeError):
            continue
        value = str(recorded_value or "").strip()
        if not value:
            continue
        replacement = f"requested {parameter_name.replace('_', ' ')}"
        for field in ("name", "description"):
            function[field] = _replace_parameter_literal(
                str(function.get(field) or ""),
                value,
                replacement,
            )


def _replace_parameter_literal(text: str, value: str, replacement: str) -> str:
    pattern = re.escape(value)
    if value[0].isalnum():
        pattern = rf"(?<!\w){pattern}"
    if value[-1].isalnum():
        pattern = rf"{pattern}(?!\w)"
    return re.sub(pattern, replacement, text, flags=re.IGNORECASE)


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
