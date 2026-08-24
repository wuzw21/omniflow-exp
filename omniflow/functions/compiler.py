from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from omniflow.core.trajectory import canonicalize_run_log, state_id
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
                    "step_index": len(steps),
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
        "schema_version": "omniflow.function-compilation-facts.v1",
        "run_id": str(payload.get("run_id") or "successful-source"),
        "goal": goal,
        "status": "succeeded",
        "success": True,
        "steps": steps,
    }
    default_bundle = _default_bundle(facts, recovery_examples)
    authoring_prompt = (
        prompt
        or """Convert the successful replayable GUI RunLog facts into reusable OmniFlow Functions.
Return exactly {"reason": string, "bundle": object|null}.

The bundle must use schema_version "omniflow.function-bundle.v2" and contain
run_id, arguments, checker_rules, and one or more ordinary
"omniflow.function.v2" Functions. Every Function contains exactly
schema_version, function_id, name, description, input_schema, bindings, steps,
and agent_visible. Checker rules belong only to the bundle-level shared library.

Copy this exact JSON shape. Replace values but never move, rename, or omit keys:
{
  "reason": "step-by-step keep, group, parameterize, or omit decisions",
  "bundle": {
    "schema_version": "omniflow.function-bundle.v2",
    "run_id": "copy the supplied run_id exactly",
    "checker_rules": [],
    "arguments": {
      "enter_requested_name": {"name": "Alice"}
    },
    "functions": [
      {
        "schema_version": "omniflow.function.v2",
        "function_id": "enter_requested_name",
        "name": "Enter requested name",
        "description": "Enter the name requested by the user.",
        "input_schema": {
          "type": "object",
          "properties": {"name": {"type": "string"}},
          "required": ["name"],
          "additionalProperties": false
        },
        "bindings": [
          {
            "source": "$.arguments.name",
            "target": "$.steps[0].action.args.text"
          }
        ],
        "steps": [
          {
            "step_index": 0,
            "source_state_id": "copy the matching before_state_id",
            "action": {
              "tool": "input_text",
              "args": {"target_description": "Name field", "text": ""}
            }
          }
        ],
        "agent_visible": true
      }
    ]
  }
}

`arguments` is an object keyed by function_id. `bindings` is always an
array of {"source", "target"} objects. `steps` is always a non-empty array and
each step contains only step_index, source_state_id, and action. Every action
contains only tool and args. Every Function repeats schema_version
"omniflow.function.v2". Never place a JSON path or template in an action value;
bound action values use empty type-correct placeholders.

Treat this as action-grounded compilation of a raw human Record.
Inspect every supplied run_log step in step_index order before authoring. The
top-level reason must account for every source step index and say whether it was
kept, grouped with neighboring steps, parameterized, or omitted, with a brief
evidence-based explanation.

Actions and args are execution truth. Explicitly preserve meaningful values such
as input_text.text, open_app.package_name, press_key.key, and wait.duration_ms.
Every input_text action must also preserve its non-empty source
target_description so the runtime can derive the source anchor for OmniTransfer.
Never omit a successful click immediately following input_text when that click
commits, submits, confirms, or advances the form; keep it in the same Function.
Use the original RunLog goal plus step metadata.summary, metadata.thinking, and
metadata.action_description only to explain the work represented by those
actions. Never replace or contradict the recorded Action with prose.

Create reusable Functions for meaningful actions or tightly coupled contiguous
action groups. Every retained replayable action must appear in at least one
Function; every omitted action must be explained in reason. Do not label or
classify Functions as semantic, full-flow, complete-task, root, or child.

A Function call is atomic: the Planner receives only the observation after its
last step. Never encode a repetition count in a Function name, description, or
multi-step body when the task requires reading the changing UI after each
repetition (for example, click five times and remember every displayed number).
In that case, retain exactly one representative action as a one-step reusable
Function, omit the later repeated source actions with an explicit explanation,
and let the Planner call that one-step Function repeatedly so every fresh
observation remains visible.

Do not create a Function merely because one recorded action exists. A coordinate
click without supporting goal, metadata, or neighboring-action evidence is not a
named capability. Do not reinterpret an accidental installer, permission page,
advertisement, error page, or other side effect as the intended task. Mechanical
waits and navigation scaffolding should stay inside the workflow they support,
not become misleading standalone Functions. If the complete task needs fresh UI
discovery, a dynamic loop, visual transcription, or a hidden runtime answer,
omit that complete Function but keep safe understandable subsequences. Never
emit kind, parent, Root, Child, recovery, task name, or routing metadata.

input_schema values are strict JSON Schema objects with additionalProperties=false.
Parameterize only action-ready values inferable from the fresh goal and consumed
by Function actions. Every required parameter must have direct bindings from
$.arguments.NAME or a fixed array index to an existing
$.steps[INDEX].action.args.FIELD. Put exact successful values in
arguments. Use empty type-correct placeholders in bound action fields.
Never bind coordinates (x/y/x1/y1/x2/y2); they are source transfer evidence,
not caller-supplied Function arguments. Use the fixed recorded coordinates and
let OmniTransfer map them against the current page.

Preserve selected source actions in order and do not invent actions or UI
evidence. Coordinate fields in the supplied facts are already normalized to
0..1000. Copy each supplied canonical action without adding fields. Return
bundle=null only when no safe reusable action-grounded Function exists.

This compilation prompt does not author recovery behavior. Set the bundle-level
checker_rules=[] unless an explicit independently reusable checker rule is supplied.
"""
    )
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
                        {"run_log": facts, "recovery_examples": recovery_examples},
                        ensure_ascii=False,
                    ),
                },
            ],
            response_format={"type": "json_object"},
            max_tokens=16384,
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
        authored = json.loads(str(response.choices[0].message.content or ""))
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
    for source_step, commit_step in zip(source_steps, source_steps[1:]):
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
