from __future__ import annotations

import hashlib
from itertools import pairwise
import json
import os
from pathlib import Path
import re
from typing import Any

from omniflow.core.trajectory import canonicalize_run_log, state_id
from omniflow.functions.management import apply_parameters, parameter_candidates
from omniflow.runlog import project_androidworld_step_actions
from omniflow.runtime.checker import CheckerLibrary, validate_checker_rule


def compile_runlog_to_store(
    run_log: str | Path | dict[str, Any],
    output_root: str | Path,
    *,
    function_bundle: dict[str, Any] | None = None,
    model: str | None = None,
    client: Any | None = None,
    prompt: str | None = None,
    timeout: float = 120.0,
    source_states: str | Path | dict[str, Any] | None = None,
    state_loader: Any | None = None,
) -> dict[str, Any]:
    """Register strict v2 Functions and their referenced source states."""
    from omniflow.functions.artifact import bind_function, parse_function_artifact
    from omniflow.functions.store import FunctionStore
    from omniflow.transfer.runtime import (
        TRANSFER_STATE_CATALOG_FILENAME,
        TRANSFER_STATE_CATALOG_VERSION,
        load_transfer_state_catalog,
    )

    root = Path(output_root).expanduser().resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"immutable output version already exists: {root}")

    if isinstance(run_log, dict):
        raw = dict(run_log)
    else:
        value = json.loads(Path(run_log).expanduser().resolve().read_text())
        if not isinstance(value, dict):
            raise ValueError("source_runlog_must_be_object")
        raw = value
    payload = canonicalize_run_log(raw)
    goal = str(payload.get("goal") or "").strip()
    if not goal:
        raise ValueError("successful_source_goal_required")

    steps: list[dict[str, Any]] = []
    recovery_examples: list[dict[str, Any]] = []
    for step in payload["steps"]:
        if not isinstance(step, dict):
            continue
        metadata = step.get("metadata") if isinstance(step.get("metadata"), dict) else {}
        result = step.get("result") if isinstance(step.get("result"), dict) else {}
        if result.get("success") is not True:
            continue
        observation = step["observation"]
        before_state_id = state_id(observation)
        next_observation = step.get("next_observation")
        after_state_id = state_id(
            next_observation
            if isinstance(next_observation, dict)
            else observation
        )
        action_type = str(step.get("action", {}).get("action_type") or "")
        if action_type in {"answer", "status", "unknown"}:
            continue
        projected_actions = project_androidworld_step_actions(step)
        if metadata.get("origin") == "checker":
            for action in projected_actions:
                example = {
                    "source_state_id": before_state_id,
                    "action": action,
                    "metadata": {
                        key: metadata[key]
                        for key in ("thinking", "summary")
                        if str(metadata.get(key) or "").strip()
                    },
                }
                trigger = str(metadata.get("checker_trigger") or "").strip()
                if trigger:
                    example["trigger"] = trigger
                recovery_examples.append(example)
            continue
        action_metadata = {
            key: metadata[key]
            for key in ("summary", "thinking", "action_description")
            if str(metadata.get(key) or "").strip()
        }
        for action in projected_actions:
            steps.append(
                {
                    "source_step_index": len(steps),
                    "before_state_id": before_state_id,
                    "action": action,
                    "result": {"success": True},
                    "after_state_id": after_state_id,
                    "metadata": action_metadata,
                }
            )
    if not steps:
        raise ValueError("successful_source_actions_required")
    facts = {
        "schema_version": "omniflow.function-compilation-facts.v2",
        "run_id": str(payload.get("run_id") or "successful-source"),
        "goal": goal,
        "status": "succeeded",
        "success": True,
        "steps": steps,
    }
    default_bundle = _default_bundle(facts, recovery_examples)
    source_parameter_candidates = _source_parameter_candidates(facts)
    authoring_prompt = prompt or """Convert successful GUI source facts into a reusable Function plan.
Return exactly:
{"reason":"account for every source step: kept, grouped, or omitted and why","plan":{"functions":[{"function_id":"enter_requested_name","name":"Enter requested name","description":"Enter the name requested by the user.","source_step_indices":[6,7],"parameters":[{"name":"name","description":"Name requested by the user","source_step_index":6,"arg_name":"text"}]}]}}

Do not output input_schema, bindings, steps, actions, coordinates, checker rules,
agent_visible, schema_version, arguments, or source_state_id. The compiler owns
all of them and materializes canonical omniflow.function.v2 artifacts from the
selected immutable source actions.

Inspect source_run in source_step_index order. The reason must account for every
source index. Within one Function, source_step_indices must be strictly increasing
and contiguous. Never omit a click immediately following input_text when that click
commits, submits, confirms, or advances the form; keep both in one Function.

Create Functions only for meaningful actions or tightly coupled contiguous groups.
Do not classify Functions as semantic, full-flow, complete-task, root, or child.
A Function call is atomic: the Planner observes only after its last step. Never
encode repetition count when the task requires reading changing UI after each
repeat. Keep one representative action as a one-step Function and let the Planner
call it repeatedly.

Do not reinterpret onboarding, installers, permissions, ads, errors, waits, or
navigation accidents as standalone capabilities. Omit unsafe or unclear actions
and explain each omission. Preserve the successful source order.

Parameterize only entries copied exactly from parameter_candidates, and only when
the same Function selects that source_step_index. Choose a stable identifier name
and concise description. Use parameters=[] for fixed recorded values. Coordinates
never appear in candidates and can never become Function inputs.
"""
    selected_model = str(model or "").strip() or None
    usage = {
        "model_calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    if function_bundle is not None:
        if selected_model is not None or client is not None or prompt is not None:
            raise ValueError("function_bundle_cannot_use_author_model_options")
        authored = {
            "reason": "Registered offline Codex-authored Functions.",
            "bundle": json.loads(json.dumps(function_bundle, ensure_ascii=False)),
        }
    elif selected_model is None:
        if client is not None or prompt is not None:
            raise ValueError("author_model_required_for_author_options")
        if default_bundle is None:
            raise ValueError("default_bundle_actions_required")
        authored = {
            "reason": "Registered one complete recorded Function.",
            "bundle": default_bundle,
        }
    else:
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError("Install omniflow[llm] to compile RunLogs") from exc
            options: dict[str, Any] = {
                "api_key": os.getenv("OPENAI_API_KEY") or "not-required"
            }
            if os.getenv("OPENAI_BASE_URL"):
                options["base_url"] = os.environ["OPENAI_BASE_URL"]
            client = OpenAI(**options)
        response = client.chat.completions.create(
            model=selected_model,
            messages=[
                {"role": "system", "content": authoring_prompt},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "source_run": facts,
                            "parameter_candidates": source_parameter_candidates,
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            response_format={"type": "json_object"},
            max_tokens=4096,
            temperature=0,
            timeout=float(timeout),
        )
        response_usage = getattr(response, "usage", None)
        prompt_tokens = int(getattr(response_usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(
            getattr(response_usage, "completion_tokens", 0) or 0
        )
        total_tokens = int(getattr(response_usage, "total_tokens", 0) or 0)
        usage = {
            "model_calls": 1,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens
            or prompt_tokens + completion_tokens,
        }
        raw_author_response = str(response.choices[0].message.content or "")
        try:
            proposal = json.loads(raw_author_response)
            authored = _materialize_authoring_plan(proposal, facts)
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            _write_authoring_failure(
                root,
                error=error,
                model=selected_model,
                prompt=authoring_prompt,
                response=raw_author_response,
                usage=usage,
            )
            raise
    if not isinstance(authored, dict) or set(authored) != {"reason", "bundle"}:
        raise ValueError("function_author_response_contract_invalid")
    if not isinstance(authored["reason"], str):
        raise ValueError("function_author_reason_must_be_string")

    bundle = authored["bundle"]
    if bundle is None:
        raise ValueError("functions_required")
    if not isinstance(bundle, dict):
        raise ValueError("function_author_bundle_must_be_object_or_null")
    if set(bundle) != {
        "schema_version",
        "run_id",
        "arguments",
        "functions",
        "checker_rules",
    }:
        raise ValueError("function_bundle_contract_invalid")
    if bundle.get("schema_version") != "omniflow.function-bundle.v2":
        raise ValueError("unsupported_function_bundle_version")
    if str(bundle.get("run_id") or "") != facts["run_id"]:
        raise ValueError("function_bundle_run_id_mismatch")
    raw_functions = bundle.get("functions")
    arguments_by_function = bundle.get("arguments")
    raw_checker_rules = bundle.get("checker_rules")
    if not isinstance(raw_functions, list) or not raw_functions:
        raise ValueError("function_bundle_functions_required")
    if not isinstance(arguments_by_function, dict):
        raise ValueError("function_bundle_source_arguments_invalid")
    if not isinstance(raw_checker_rules, list):
        raise ValueError("function_bundle_checker_rules_invalid")
    restored_commit_steps = _restore_post_input_commit_steps(
        raw_functions,
        facts["steps"],
    )
    if restored_commit_steps:
        authored["reason"] = (
            f"{authored['reason']} Compiler restored {restored_commit_steps} "
            "successful post-input commit action(s)."
        )
    functions = [parse_function_artifact(value) for value in raw_functions]
    checker_rules = [validate_checker_rule(rule) for rule in raw_checker_rules]
    checker_ids = [rule["id"] for rule in checker_rules]
    if len(checker_ids) != len(set(checker_ids)):
        raise ValueError("function_bundle_duplicate_checker_id")
    function_ids = [function.id for function in functions]
    if len(function_ids) != len(set(function_ids)):
        raise ValueError("function_bundle_duplicate_function_id")
    if set(arguments_by_function) - set(function_ids):
        raise ValueError("function_bundle_source_arguments_unknown_function")
    normalized_arguments: dict[str, dict[str, Any]] = {}
    for function in functions:
        arguments = arguments_by_function.get(function.id, {})
        if not isinstance(arguments, dict):
            raise ValueError("function_bundle_source_arguments_invalid")
        bind_function(function, arguments)
        normalized_arguments[function.id] = dict(arguments)
    arguments_by_function = normalized_arguments

    if source_states is not None and state_loader is not None:
        raise ValueError("function_source_state_provider_ambiguous")
    referenced_state_ids = _referenced_source_state_ids(functions)
    states: dict[str, dict[str, Any]]
    source_catalog_run_id = facts["run_id"]
    if source_states is not None:
        if isinstance(source_states, (str, Path)):
            source_catalog_path = Path(source_states).expanduser().resolve()
            raw_catalog = json.loads(source_catalog_path.read_text(encoding="utf-8"))
            if not isinstance(raw_catalog, dict):
                raise ValueError("function_source_state_catalog_invalid")
            source_catalog_run_id = str(raw_catalog.get("run_id") or "").strip()
            states = load_transfer_state_catalog(source_catalog_path)
        elif isinstance(source_states, dict):
            raw_states = source_states.get("states")
            if raw_states is None:
                raw_states = source_states
            elif source_states.get("schema_version") != (
                TRANSFER_STATE_CATALOG_VERSION
            ):
                raise ValueError("function_source_state_catalog_invalid")
            if not isinstance(raw_states, dict):
                raise ValueError("function_source_state_catalog_invalid")
            source_catalog_run_id = str(
                source_states.get("run_id") or facts["run_id"]
            ).strip()
            states = {
                str(state_id): _normalize_source_state(value, str(state_id))
                for state_id, value in raw_states.items()
            }
        else:
            raise ValueError("function_source_state_catalog_invalid")
    elif callable(state_loader):
        states = {
            state_id: _normalize_source_state(state_loader(state_id), state_id)
            for state_id in referenced_state_ids
        }
    else:
        raise ValueError("function_source_states_required")
    if source_catalog_run_id != facts["run_id"]:
        raise ValueError("function_source_state_run_id_mismatch")
    missing_state_ids = [
        state_id for state_id in referenced_state_ids if state_id not in states
    ]
    if missing_state_ids:
        raise ValueError(
            "function_source_states_missing:" + ",".join(missing_state_ids)
        )
    frozen_states = {
        state_id: states[state_id] for state_id in referenced_state_ids
    }

    root.mkdir(parents=True, exist_ok=True)
    store_path = root / "store.json"
    store = FunctionStore(store_path)
    for function in functions:
        store.put_function(function)
    checker_store_path = root / "checker_store.json"
    CheckerLibrary(tuple(checker_rules)).save(checker_store_path)
    transfer_state_catalog_path = root / TRANSFER_STATE_CATALOG_FILENAME
    transfer_state_catalog_path.write_text(
        json.dumps(
            {
                "schema_version": TRANSFER_STATE_CATALOG_VERSION,
                "run_id": facts["run_id"],
                "states": frozen_states,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    report = {
        "schema_version": "omniflow.androidworld.function-gate.v2",
        "success": True,
        "live_probe_allowed": True,
        "classification": "ready_for_live_probe",
        "reason": authored["reason"],
        "model": selected_model,
        "prompt_sha256": (
            hashlib.sha256(authoring_prompt.encode()).hexdigest()
            if selected_model is not None
            else None
        ),
        "store_path": str(store_path),
        "checker_store_path": str(checker_store_path),
        "checker_count": len(checker_rules),
        "transfer_state_catalog": str(transfer_state_catalog_path),
        "transfer_state_count": len(frozen_states),
        "function_ids": function_ids,
        "function_count": len(function_ids),
        "source_calls": [
            {
                "function_id": function_id,
                "arguments": json.loads(
                    json.dumps(arguments_by_function[function_id], ensure_ascii=False)
                ),
            }
            for function_id in function_ids
        ],
        "source_arguments": json.loads(
            json.dumps(arguments_by_function, ensure_ascii=False)
        ),
        **usage,
    }
    (root / "compile_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    return report


def _source_parameter_candidates(facts: dict[str, Any]) -> list[dict[str, Any]]:
    function_view = {
        "bindings": [],
        "steps": [
            {
                "step_index": index,
                "action": step["action"],
            }
            for index, step in enumerate(facts.get("steps") or ())
        ],
    }
    return [
        {
            "source_step_index": candidate["step_index"],
            "tool": candidate["tool"],
            "arg_name": candidate["arg_name"],
            "recorded_value": candidate["recorded_value"],
        }
        for candidate in parameter_candidates(function_view)
    ]


def _materialize_authoring_plan(
    value: Any,
    facts: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"reason", "plan"}:
        raise ValueError("function_author_plan_response_contract_invalid")
    reason = value.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("function_author_reason_must_be_string")
    plan = value.get("plan")
    if not isinstance(plan, dict) or set(plan) != {"functions"}:
        raise ValueError("function_author_plan_contract_invalid")
    raw_functions = plan.get("functions")
    if not isinstance(raw_functions, list) or not raw_functions:
        raise ValueError("function_author_plan_functions_required")

    source_steps = list(facts.get("steps") or ())
    candidates = {
        (candidate["source_step_index"], candidate["arg_name"]): candidate
        for candidate in _source_parameter_candidates(facts)
    }
    functions: list[dict[str, Any]] = []
    arguments: dict[str, dict[str, Any]] = {}
    selected_source_indices: set[int] = set()
    materialization_notes: list[str] = []
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
    for raw_function in raw_functions:
        if not isinstance(raw_function, dict) or set(raw_function) != function_fields:
            raise ValueError("function_author_plan_function_contract_invalid")
        function_id = str(raw_function.get("function_id") or "").strip()
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
        if (
            indices != sorted(set(indices))
            or indices[0] < 0
            or indices[-1] >= len(source_steps)
            or indices != list(range(indices[0], indices[-1] + 1))
        ):
            raise ValueError("function_author_plan_source_steps_invalid")
        (
            indices,
            function_id,
            name,
            description,
            atomicized_count,
        ) = _atomicize_repeated_click_function(
            indices,
            source_steps,
            function_id=function_id,
            name=name,
            description=description,
        )
        if atomicized_count:
            materialization_notes.append(
                f"Compiler reduced {atomicized_count} identical clicks in "
                f"{function_id} to one atomic step so the Planner observes "
                "after every click."
            )
        if selected_source_indices.intersection(indices):
            raise ValueError("function_author_plan_source_step_reused")
        selected_source_indices.update(indices)

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
            "steps": [
                {
                    "step_index": local_index,
                    "source_state_id": str(
                        source_steps[source_index]["before_state_id"]
                    ),
                    "action": json.loads(
                        json.dumps(
                            source_steps[source_index]["action"],
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
        for parameter in raw_parameters:
            if not isinstance(parameter, dict) or set(parameter) != parameter_fields:
                raise ValueError("function_author_plan_parameter_contract_invalid")
            source_index = parameter.get("source_step_index")
            arg_name = str(parameter.get("arg_name") or "").strip()
            candidate = candidates.get((source_index, arg_name))
            if candidate is None or source_index not in indices:
                raise ValueError("function_author_plan_parameter_target_invalid")
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
        apply_parameters(function, parameter_proposals, facts)
        functions.append(function)
        arguments[function_id] = source_arguments

    normalized_reason = reason.strip()
    if materialization_notes:
        normalized_reason = f"{normalized_reason} {' '.join(materialization_notes)}"
    return {
        "reason": normalized_reason,
        "bundle": {
            "schema_version": "omniflow.function-bundle.v2",
            "run_id": str(facts["run_id"]),
            "arguments": arguments,
            "checker_rules": [],
            "functions": functions,
        },
    }


def _atomicize_repeated_click_function(
    indices: list[int],
    source_steps: list[dict[str, Any]],
    *,
    function_id: str,
    name: str,
    description: str,
) -> tuple[list[int], str, str, str, int]:
    actions = [source_steps[index].get("action") for index in indices]
    repeated_click = any(
        current == following
        and isinstance(current, dict)
        and current.get("tool") == "click"
        for current, following in pairwise(actions)
    )
    if not repeated_click:
        return indices, function_id, name, description, 0
    if not actions or any(action != actions[0] for action in actions[1:]):
        raise ValueError("function_author_plan_repeated_click_must_be_atomic")

    atomic_id = re.sub(r"_(?:\d+_)?times?$", "", function_id, flags=re.IGNORECASE)
    if atomic_id == "click":
        atomic_id = "click_recorded_button"
    atomic_name = re.sub(
        r"\s+\d+\s+times?\b",
        "",
        name,
        flags=re.IGNORECASE,
    ).strip()
    atomic_description = re.sub(
        r"\s+\d+\s+times?\b",
        "",
        description,
        flags=re.IGNORECASE,
    ).strip()
    return (
        [indices[0]],
        atomic_id or "click_recorded_button",
        atomic_name or "Click recorded button",
        atomic_description
        or "Click the recorded button once, then return to the Planner.",
        len(indices),
    )


def _write_authoring_failure(
    root: Path,
    *,
    error: Exception,
    model: str,
    prompt: str,
    response: str,
    usage: dict[str, int],
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": "omniflow.function-authoring-failure.v1",
        "success": False,
        "classification": "authoring_rejected",
        "error": str(error) or type(error).__name__,
        "model": model,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "raw_response": response,
        **usage,
    }
    (root / "authoring_failure.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _referenced_source_state_ids(functions: list[Any]) -> list[str]:
    state_ids: list[str] = []
    for function in functions:
        for item in function.steps:
            if isinstance(item, dict):
                state_id = str(item.get("source_state_id") or "").strip()
            else:
                state_id = str(getattr(item, "source_state_id", "") or "").strip()
            if state_id and state_id not in state_ids:
                state_ids.append(state_id)
    if not state_ids:
        raise ValueError("function_source_state_references_required")
    return state_ids


def _normalize_source_state(value: Any, expected_state_id: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"function_source_state_invalid:{expected_state_id}")
    extra = value.get("extra") if isinstance(value.get("extra"), dict) else {}
    state_id = str(
        value.get("state_id") or extra.get("state_id") or expected_state_id
    ).strip()
    if state_id != expected_state_id:
        raise ValueError(f"function_source_state_id_mismatch:{expected_state_id}")
    state: dict[str, Any] = {"state_id": state_id}
    aliases = {
        "xml": ("xml", "page", "observation_xml"),
        "package_name": ("package_name", "packageName"),
        "activity_name": ("activity_name", "activityName"),
        "screenshot_path": ("screenshot_path",),
    }
    for output, names in aliases.items():
        item = next(
            (
                source[name]
                for source in (value, extra)
                for name in names
                if source.get(name) is not None
            ),
            None,
        )
        if item is not None:
            if not isinstance(item, str):
                raise ValueError(
                    f"function_source_state_{output}_invalid:{state_id}"
                )
            state[output] = item
    display = value.get("display") or extra.get("display")
    if not isinstance(display, dict):
        width = value.get("width") or value.get("display_width")
        height = value.get("height") or value.get("display_height")
        if width is None:
            width = extra.get("width") or extra.get("display_width")
        if height is None:
            height = extra.get("height") or extra.get("display_height")
        if width is not None or height is not None:
            display = {"width": width, "height": height}
    if display is not None:
        if not isinstance(display, dict) or set(display) != {"width", "height"}:
            raise ValueError(f"function_source_state_display_invalid:{state_id}")
        try:
            width = int(display["width"])
            height = int(display["height"])
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"function_source_state_display_invalid:{state_id}"
            ) from error
        if width <= 0 or height <= 0:
            raise ValueError(f"function_source_state_display_invalid:{state_id}")
        state["display"] = {"width": width, "height": height}
    return state


def _restore_post_input_commit_steps(
    raw_functions: list[Any],
    source_steps: list[dict[str, Any]],
) -> int:
    """Restore an authored-away source click that immediately follows input."""
    restored = 0
    for source_step, commit_step in pairwise(source_steps):
        source_action = source_step.get("action")
        commit_action = commit_step.get("action")
        if not (
            isinstance(source_action, dict)
            and source_action.get("tool") == "input_text"
            and isinstance(commit_action, dict)
            and commit_action.get("tool") == "click"
        ):
            continue
        commit_state_id = str(commit_step.get("before_state_id") or "").strip()
        if not commit_state_id:
            continue
        if any(
            isinstance(step, dict)
            and step.get("source_state_id") == commit_state_id
            and step.get("action") == commit_action
            for function in raw_functions
            if isinstance(function, dict)
            for step in (
                function.get("steps")
                if isinstance(function.get("steps"), list)
                else ()
            )
        ):
            continue

        input_state_id = str(source_step.get("before_state_id") or "").strip()
        owner_steps: list[Any] | None = None
        for function in raw_functions:
            if not isinstance(function, dict):
                continue
            candidate_steps = function.get("steps")
            if not isinstance(candidate_steps, list) or not candidate_steps:
                continue
            last_step = candidate_steps[-1]
            if (
                isinstance(last_step, dict)
                and last_step.get("source_state_id") == input_state_id
                and isinstance(last_step.get("action"), dict)
                and last_step["action"].get("tool") == "input_text"
            ):
                owner_steps = candidate_steps
                break
        if owner_steps is None:
            continue
        owner_steps.append(
            {
                "step_index": len(owner_steps),
                "source_state_id": commit_state_id,
                "action": json.loads(json.dumps(commit_action, ensure_ascii=False)),
            }
        )
        restored += 1
    return restored


def _default_bundle(
    facts: dict[str, Any],
    recovery_examples: list[dict[str, Any]],
) -> dict[str, Any] | None:
    source_steps = list(facts.get("steps") or ())
    if len(source_steps) < 2:
        return None
    steps = [
        {
            "step_index": index,
            "source_state_id": str(step["before_state_id"]),
            "action": json.loads(json.dumps(step["action"], ensure_ascii=False)),
        }
        for index, step in enumerate(source_steps)
    ]
    digest = hashlib.sha256(
        json.dumps(
            {"goal": facts["goal"], "steps": steps},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()[:12]
    function_id = f"recorded_{digest}"
    return {
        "schema_version": "omniflow.function-bundle.v2",
        "run_id": facts["run_id"],
        "arguments": {function_id: {}},
        "checker_rules": [],
        "functions": [
            {
                "schema_version": "omniflow.function.v2",
                "function_id": function_id,
                "name": str(facts["goal"])[:120],
                "description": f"Replay the recorded workflow: {facts['goal']}",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False,
                },
                "bindings": [],
                "steps": steps,
                "agent_visible": True,
            }
        ],
    }
