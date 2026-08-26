from __future__ import annotations

import hashlib
from itertools import pairwise
import json
import os
from pathlib import Path
import re
from typing import Any
import xml.etree.ElementTree as ET

from omniflow.core.trajectory import require_complete_source_run_log, state_id
from omniflow.functions.management import (
    apply_parameters,
    parameter_candidates,
    semantic_parameter_evidence,
)
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
    payload = require_complete_source_run_log(raw)
    goal = str(payload.get("goal") or "").strip()
    if not goal:
        raise ValueError("successful_source_goal_required")

    steps: list[dict[str, Any]] = []
    parameter_evidence: list[dict[str, Any]] = []
    recovery_examples: list[dict[str, Any]] = []
    omitted_action_types: set[str] = set()
    previous_successful_step: dict[str, Any] | None = None
    source_steps = payload["steps"]
    execution_trace = _execution_trace_by_step_index(payload)
    for source_step_index, step in enumerate(source_steps):
        if not isinstance(step, dict):
            continue
        metadata = step.get("metadata") if isinstance(step.get("metadata"), dict) else {}
        result = step.get("result") if isinstance(step.get("result"), dict) else {}
        if result.get("success") is not True:
            continue
        if _is_transient_system_action(step):
            # Source collection can legitimately dismiss an incidental Android
            # crash/permission dialog before continuing.  That click is part
            # of collection recovery, not of the task's reusable semantics;
            # retaining it would make a clean target page fail transfer before
            # the Planner gets a chance to act.
            omitted_action_types.add("transient_system_dialog")
            previous_successful_step = step
            continue
        observation = step["observation"]
        before_state_id = state_id(observation)
        next_observation = step.get("next_observation")
        if (
            isinstance(next_observation, dict)
            and state_id(next_observation) == before_state_id
            and source_step_index + 1 < len(source_steps)
        ):
            following_observation = source_steps[source_step_index + 1].get(
                "observation"
            )
            if (
                isinstance(following_observation, dict)
                and state_id(following_observation) != before_state_id
            ):
                next_observation = following_observation
        after_state_id = state_id(
            next_observation
            if isinstance(next_observation, dict)
            else observation
        )
        action_type = str(step.get("action", {}).get("action_type") or "")
        if action_type in {"answer", "status", "unknown"}:
            omitted_action_types.add(action_type)
            continue
        if (
            action_type in {"click", "double_tap", "long_press", "swipe"}
            and isinstance(next_observation, dict)
            and before_state_id == after_state_id
        ):
            omitted_action_types.add(f"noop_{action_type}")
            previous_successful_step = step
            continue
        execution_action = _source_execution_action(
            step,
            source_step_index=source_step_index,
            execution_trace=execution_trace,
        )
        projected_actions = project_androidworld_step_actions(
            step,
            previous_step=previous_successful_step,
            execution_action=execution_action,
        )
        promoted_launcher_entry = False
        if not steps and projected_actions:
            original_entry_action = projected_actions[0]
            projected_actions[0] = _promote_launcher_app_entry(
                projected_actions[0],
                observation=observation,
                next_observation=next_observation,
            )
            promoted_launcher_entry = projected_actions[0] != original_entry_action
        previous_successful_step = step
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
            for key in ("summary", "action_description")
            if str(metadata.get(key) or "").strip()
        }
        purpose = str(
            metadata.get("action_description")
            or metadata.get("summary")
            or metadata.get("reasoning")
            or ""
        ).strip()
        if purpose:
            action_metadata["purpose"] = purpose
        if promoted_launcher_entry:
            action_metadata["fixed_open_app_package"] = True
        for action in projected_actions:
            action = _canonicalize_open_app_action(
                action,
                next_observation=next_observation,
            )
            source_step_index = len(steps)
            parameter_evidence.extend(
                _source_action_parameter_evidence(
                    source_step=step,
                    action=action,
                    source_step_index=source_step_index,
                    task_parameters=payload.get("task_parameters"),
                )
            )
            steps.append(
                {
                    "source_step_index": source_step_index,
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
        "task_parameters": json.loads(
            json.dumps(payload.get("task_parameters") or {}, ensure_ascii=False)
        ),
        "status": "succeeded",
        "success": True,
        "steps": steps,
        "parameter_evidence": parameter_evidence,
        "omitted_action_types": sorted(omitted_action_types),
    }
    facts["observation_dependent_handoff_indices"] = sorted(
        _generic_coordinate_surface_indices(facts)
    )
    default_bundle = _default_bundle(facts, recovery_examples)
    source_parameter_candidates = _source_parameter_candidates(facts)
    authoring_prompt = prompt or """Convert successful GUI source facts into a reusable Function plan.
Return exactly one object with this shape:
{"reason":"account for every source step and explain the composition","plan":{"functions":[{"function_id":"enter_requested_name","name":"Enter requested name","description":"Fill the requested name and submit it so the form reaches its completed state.","source_step_indices":[6,7],"parameters":[{"name":"name","description":"Name requested by the user","source_step_index":6,"arg_name":"text"}]}],"complete_function":{"function_id":"complete_form","name":"Complete form","description":"Complete the form by entering the requested name and submitting it; the final submit action produces the requested completed form.","source_step_indices":[6,7],"parameters":[{"name":"name","description":"Name requested by the user","source_step_index":6,"arg_name":"text"}]}}}

Do not output input_schema, bindings, steps, actions, coordinates, checker rules,
agent_visible, schema_version, arguments, or source_state_id. The compiler owns
all of them and materializes canonical omniflow.function.v2 artifacts from the
selected immutable source actions.

Inspect source_run in source_step_index order. The reason must account for every
source index. Actions already marked origin=checker were removed before this plan;
do not reconstruct them in the main flow. A semantic Function's source_step_indices
must be strictly increasing and contiguous. The complete Function may omit unsafe
middle actions but must preserve source order. Never omit a click immediately
following input_text when that click commits, submits, confirms, or advances the
form; keep both in one Function.
Each source step's metadata.purpose explains what that action accomplishes. Preserve
those purposes in the Function's ordered action description and core effect.
The authored name must identify the concrete user-visible result, not merely a page
or navigation verb. The authored description must name the resulting state, the
completion condition, and the causal role of the final action. Never use vague text
such as "perform the workflow", "navigate as needed", or "advance the task". The
compiler will append immutable, exact tool signatures and ordered metadata.purpose
entries from the RunLog; the model must not invent or rewrite those Action calls.

If source_run.omitted_action_types contains answer or status, those terminal
outputs are intentionally not Function actions. The complete Function is only a
reusable prefix; its name and description must say that the Planner must observe
the returned page, follow the current goal's requested attributes, locate the
requested item, read the requested value or completion state, and provide the
answer/status afterward. Do not claim that the Function itself answered the task,
and do not copy source-instance task parameter values into this handoff.
Actions listed as noop_* were successful gestures whose before and after native
observations were identical; keep them omitted and explain the omission rather
than reconstructing them as Function actions.

In functions, return zero or more reusable semantic actions or tightly coupled
contiguous groups. Then author exactly one complete_function as an ordinary Function.
When the successful RunLog starts with open_app, complete_function must start with
the first open_app and end with the terminal successful task action. It may omit
unsafe internal retries, setup noise, checker actions, and observation-dependent
repetitions while preserving that complete task envelope. A Function is atomic: if
an action result must be observed to calculate a later value, stop the reusable
Function before that action. Do not expose that later value as a Function
parameter. The Planner must observe the returned page and continue with ordinary
native tools or a separately recalled Function.
The complete Function must lift its goal-dependent values into parameters, merge the
selected meanings into one coherent name and description, and never merely hard-code
the successful instance values. Its description must state the task-level core effect,
explain why the final selected action achieves that effect, and distinguish a completed
task from a navigation-only handoff. Do not invent a nesting or parent/child schema.
A Function call is atomic: the Planner observes only after its last step. Never
encode repetition count when the task requires reading changing UI after each
repeat. Keep one representative action as a one-step Function and let the Planner
call it repeatedly.

Do not put a coordinate-only action on a generic canvas, background, grid cell,
map surface, drawing surface, or other reusable container into a cross-instance
Function when the coordinate is what identifies the date, item, or value. A
resource id or label such as `month_view_background` is not enough semantic
grounding for the selected cell. Treat that action as an observation-dependent
handoff boundary: stop the Function before it, and let the Planner inspect the
current page and choose the visible target with native tools. Never preserve the
source coordinate merely because the source RunLog succeeded.

Do not reinterpret onboarding, installers, permissions, ads, errors, waits, or
navigation accidents as standalone capabilities. Omit unsafe or unclear actions
and explain each omission. Preserve the successful source order.

Parameterize only entries copied exactly from parameter_candidates, and only when
the same Function selects that source_step_index. Never parameterize an
observation-dependent input: such a step is a runtime handoff boundary and any
Function containing it is omitted by the compiler. Choose a stable identifier name
and precise description. Every (source_step_index, arg_name) pair must occur
literally in parameter_candidates; never invent file, folder, click-count, integer,
coordinate, or other parameters absent from that list. Usually target_description
is a stable UI label, not a goal-dependent value. The complete_function must repeat
every parameter target selected by a semantic Function. Use parameters=[] for fixed
recorded values. Coordinates never appear in candidates and can never become
Function inputs. Keep reason under 40 words, each description under 120 words, and
return no prose outside the JSON object.
When a value is selected as a parameter, remove its recorded instance literal from
the Function name and description. Describe the requested semantic value and let the
generated input schema carry the concrete value at call time.
Use suggested_name exactly when a candidate provides it. Give distinct semantic
values distinct parameter names. Reuse one parameter name across multiple targets
only when those targets intentionally consume the same recorded semantic value.

For a global Function whose first action is open_app, keep the canonical recorded
package fixed when the goal identifies a concrete app such as Joplin or Settings.
Only expose package_name when its parameter_candidate includes the compiler-backed
task_parameter_name="app_name" and value_contract="android_package_name" evidence.
In that case name the parameter package_name, describe it as the installed Android
package for the requested app, and make the Function name and description generic to
the requested app rather than the recorded source app. A model must never put a
friendly app label such as "clock" into package_name.
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
            "reason": (
                "Registered the safe reusable prefix; the Planner resumes at the "
                "observation-dependent handoff."
                if _observation_dependent_input_indices(facts)
                else "Registered one complete recorded Function."
            ),
            "bundle": default_bundle,
        }
    else:
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError("Install omniflow[llm] to compile RunLogs") from exc
            options: dict[str, Any] = {
                "api_key": os.getenv("OPENAI_API_KEY") or "not-required",
                "max_retries": 0,
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
            max_tokens=1024,
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
            if default_bundle is None:
                raise
            authored = {
                "reason": (
                    "Authoring proposal was unusable; registered the complete "
                    f"schema-valid recorded Function instead ({type(error).__name__})."
                ),
                "bundle": default_bundle,
            }
    if not isinstance(authored, dict) or set(authored) != {"reason", "bundle"}:
        raise ValueError("function_author_response_contract_invalid")
    if not isinstance(authored["reason"], str):
        raise ValueError("function_author_reason_must_be_string")

    bundle = authored["bundle"]
    if bundle is None:
        raise ValueError("functions_required")
    if not isinstance(bundle, dict):
        raise ValueError("function_author_bundle_must_be_object_or_null")
    if (
        selected_model is not None
        and function_bundle is None
        and default_bundle is not None
        and isinstance(bundle.get("functions"), list)
        and not bundle["functions"]
    ):
        authored = {
            "reason": (
                "The authored plan contained no safe tool Function at the runtime "
                "handoff; registered the safe reusable prefix instead."
            ),
            "bundle": default_bundle,
        }
        bundle = default_bundle
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
    try:
        functions = [parse_function_artifact(value) for value in raw_functions]
    except ValueError as error:
        if selected_model is None or function_bundle is not None or default_bundle is None:
            raise
        authored = {
            "reason": (
                "Authored Functions did not satisfy the runtime schema; registered "
                f"the complete schema-valid recorded Function instead ({type(error).__name__})."
            ),
            "bundle": default_bundle,
        }
        bundle = default_bundle
        raw_functions = list(bundle["functions"])
        arguments_by_function = dict(bundle["arguments"])
        raw_checker_rules = list(bundle["checker_rules"])
        functions = [parse_function_artifact(value) for value in raw_functions]
    checker_rules = [validate_checker_rule(rule) for rule in raw_checker_rules]
    checker_ids = [rule["id"] for rule in checker_rules]
    if len(checker_ids) != len(set(checker_ids)):
        raise ValueError("function_bundle_duplicate_checker_id")
    authored_function_ids = [function.id for function in functions]
    if len(authored_function_ids) != len(set(authored_function_ids)):
        raise ValueError("function_bundle_duplicate_function_id")
    if set(arguments_by_function) - set(authored_function_ids):
        raise ValueError("function_bundle_source_arguments_unknown_function")
    normalized_arguments: dict[str, dict[str, Any]] = {}
    for function in functions:
        arguments = arguments_by_function.get(function.id, {})
        if not isinstance(arguments, dict):
            raise ValueError("function_bundle_source_arguments_invalid")
        bind_function(function, arguments)
        normalized_arguments[function.id] = dict(arguments)
    arguments_by_function = normalized_arguments

    function_ids = [function.id for function in functions]

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


def _execution_trace_by_step_index(
    run_log: dict[str, Any],
) -> dict[int, dict[str, Any]]:
    diagnostics = run_log.get("diagnostics")
    trace = diagnostics.get("execution_trace") if isinstance(diagnostics, dict) else None
    if not isinstance(trace, list):
        return {}
    indexed: dict[int, dict[str, Any]] = {}
    for fallback_index, item in enumerate(trace):
        if not isinstance(item, dict):
            continue
        raw_index = item.get("step_index", fallback_index)
        if isinstance(raw_index, bool) or not isinstance(raw_index, int):
            continue
        if raw_index in indexed:
            raise ValueError(f"source_execution_trace_step_duplicate:{raw_index}")
        indexed[raw_index] = item
    return indexed


def _source_execution_action(
    source_step: dict[str, Any],
    *,
    source_step_index: int,
    execution_trace: dict[int, dict[str, Any]],
) -> dict[str, Any] | None:
    source_action = source_step.get("action")
    if not isinstance(source_action, dict):
        return None
    trace_step = execution_trace.get(source_step_index)
    trace_action = trace_step.get("action") if isinstance(trace_step, dict) else None
    trace_result = trace_step.get("result") if isinstance(trace_step, dict) else None
    trace_args = trace_action.get("args") if isinstance(trace_action, dict) else None
    if source_action.get("action_type") == "open_app":
        if (
            isinstance(trace_action, dict)
            and trace_action.get("tool") == "open_app"
            and isinstance(trace_args, dict)
            and str(trace_args.get("package_name") or "").strip()
        ):
            return trace_action
        return None
    if source_action.get("action_type") not in {"scroll", "swipe"}:
        return None
    coordinate_keys = ("x1", "y1", "x2", "y2")
    if (
        isinstance(trace_action, dict)
        and trace_action.get("tool") == "swipe"
        and isinstance(trace_result, dict)
        and trace_result.get("success") is True
        and isinstance(trace_args, dict)
        and all(trace_args.get(key) is not None for key in coordinate_keys)
    ):
        return trace_action
    if all(source_action.get(key) is not None for key in coordinate_keys):
        return None
    raise ValueError(
        f"source_swipe_execution_evidence_required:{source_step_index}"
    )


def _source_parameter_candidates(facts: dict[str, Any]) -> list[dict[str, Any]]:
    function_view = {
        "bindings": [],
        "steps": [
            {
                "step_index": index,
                "action": step["action"],
                "metadata": step.get("metadata") or {},
            }
            for index, step in enumerate(facts.get("steps") or ())
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
        evidence = semantic_parameter_evidence(
            function_view["steps"][int(candidate["step_index"])],
            str(candidate["arg_name"]),
            candidate["recorded_value"],
            facts,
        )
        if evidence is None:
            continue
        value = {
            "source_step_index": candidate["step_index"],
            "tool": candidate["tool"],
            "arg_name": candidate["arg_name"],
            "recorded_value": candidate["recorded_value"],
            **evidence,
        }
        evidence = evidence_by_target.get(
            (candidate["step_index"], candidate["arg_name"])
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
    fixed_open_app_steps = {
        int(step["step_index"])
        for step in function_view["steps"]
        if isinstance(step.get("metadata"), dict)
        and step["metadata"].get("fixed_open_app_package") is True
    }
    return [
        candidate
        for candidate in candidates
        if not (
            candidate["tool"] == "open_app"
            and candidate["arg_name"] == "package_name"
            and candidate["source_step_index"] in fixed_open_app_steps
        )
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
    if not isinstance(plan, dict) or set(plan) != {
        "functions",
        "complete_function",
    }:
        raise ValueError("function_author_plan_contract_invalid")
    raw_functions = plan.get("functions")
    raw_complete_function = plan.get("complete_function")
    if not isinstance(raw_functions, list):
        raise ValueError("function_author_plan_functions_required")
    if not isinstance(raw_complete_function, dict):
        raise ValueError("function_author_plan_complete_function_required")

    source_steps = list(facts.get("steps") or ())
    candidates = {
        (candidate["source_step_index"], candidate["arg_name"]): candidate
        for candidate in _source_parameter_candidates(facts)
    }
    observation_dependent_input_indices = _observation_dependent_input_indices(facts)
    functions: list[dict[str, Any]] = []
    materialized_function_ids: set[str] = set()
    arguments: dict[str, dict[str, Any]] = {}
    selected_source_indices: set[int] = set()
    semantic_parameter_targets: set[tuple[int, str]] = set()
    semantic_parameter_specs: dict[tuple[int, str], dict[str, Any]] = {}
    complete_parameter_targets: set[tuple[int, str]] = set()
    complete_source_indices: set[int] = set()
    materialization_notes: list[str] = []
    preserved_complete_sequence = False
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
    planned_functions = [
        *((raw_function, False) for raw_function in raw_functions),
        (raw_complete_function, True),
    ]
    for raw_function, is_complete in planned_functions:
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
        skip_function = False
        if (
            indices != sorted(set(indices))
            or indices[0] < 0
            or indices[-1] >= len(source_steps)
        ):
            raise ValueError("function_author_plan_source_steps_invalid")
        source_starts_with_open_app = bool(source_steps) and (
            source_steps[0].get("action", {}).get("tool") == "open_app"
        )
        if not is_complete and indices != list(range(indices[0], indices[-1] + 1)):
            raise ValueError("function_author_plan_local_steps_not_contiguous")
        if is_complete and indices[-1] != len(source_steps) - 1:
            raise ValueError("function_author_plan_complete_terminal_step_required")
        if is_complete:
            indices = list(range(len(source_steps)))
            preserved_complete_sequence = True
            materialization_notes.append(
                "Compiler preserved every successful main-flow action in source order."
            )
        if is_complete and observation_dependent_input_indices:
            dynamic_indices = [
                index
                for index in indices
                if index in observation_dependent_input_indices
            ]
            if dynamic_indices:
                boundary = dynamic_indices[0]
                safe_indices = [index for index in indices if index < boundary]
                if safe_indices:
                    indices = safe_indices
                    description = (
                        f"{description} Stop before the value-dependent input "
                        "so the Planner can observe and compute its value."
                    )
                    materialization_notes.append(
                        "Compiler split the global Function at an "
                        f"observation-dependent input (source step {boundary})."
                    )
                else:
                    skip_function = True
                    materialization_notes.append(
                        "Compiler omitted a global Function with no safe prefix; "
                        "the Planner owns the runtime handoff."
                    )
        elif (
            not is_complete
            and observation_dependent_input_indices
            and any(
                index in observation_dependent_input_indices for index in indices
            )
        ):
            skip_function = True
            materialization_notes.append(
                "Compiler omitted a semantic Function containing an "
                "observation-dependent input; the Planner owns that runtime handoff."
            )
        if skip_function:
            continue
        atomicized_count = 0
        if not (is_complete and source_starts_with_open_app):
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
        if not is_complete:
            if selected_source_indices.intersection(indices):
                raise ValueError("function_author_plan_source_step_reused")
            selected_source_indices.update(indices)
        else:
            complete_source_indices.update(indices)

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
        if is_complete:
            # The complete Function is the public API envelope.  Authoring
            # models sometimes select a parameter on a semantic sub-function
            # but forget to repeat it on complete_function.  Repair that
            # omission from the already validated semantic proposal instead of
            # falling back to a hard-coded recorded Function.
            explicit_targets = {
                (
                    parameter.get("source_step_index"),
                    str(parameter.get("arg_name") or "").strip(),
                )
                for parameter in raw_parameters
                if isinstance(parameter, dict)
            }
            raw_parameters = [*raw_parameters]
            for target, parameter in semantic_parameter_specs.items():
                if target[0] in indices and target not in explicit_targets:
                    raw_parameters.append(dict(parameter))
                    materialization_notes.append(
                        "Compiler copied a validated semantic parameter onto "
                        "the complete Function API."
                    )
            for automatic_parameter in _goal_semantic_parameters(
                facts,
                indices=indices,
                candidates=candidates,
                existing_targets=explicit_targets
                | set(semantic_parameter_specs),
            ):
                raw_parameters.append(automatic_parameter)
                materialization_notes.append(
                    "Compiler exposed a source label explicitly named by the "
                    "goal as a public Function API parameter."
                )
        parameter_proposals: list[dict[str, Any]] = []
        source_arguments: dict[str, Any] = {}
        for parameter in raw_parameters:
            if not isinstance(parameter, dict) or set(parameter) != parameter_fields:
                raise ValueError("function_author_plan_parameter_contract_invalid")
            source_index = parameter.get("source_step_index")
            arg_name = str(parameter.get("arg_name") or "").strip()
            candidate = candidates.get((source_index, arg_name))
            if candidate is None:
                raise ValueError("function_author_plan_parameter_target_invalid")
            if arg_name == "package_name" and not _goal_allows_dynamic_app_package(
                facts.get("goal"),
                candidate=candidate,
            ):
                materialization_notes.append(
                    "Compiler kept the concrete app package fixed instead of "
                    "exposing package_name as a public Function input."
                )
                continue
            if source_index not in indices:
                # A complete Function may be truncated at an observation
                # boundary. Its parameter remains owned by the later
                # semantic Function selected after the observation.
                if is_complete:
                    continue
                # Authoring models occasionally copy a public parameter onto
                # a semantic sub-function without including the source step
                # in that sub-function.  The complete Function still owns
                # the public API, so discard only this misplaced sub-function
                # proposal instead of rejecting the entire authoring plan.
                materialization_notes.append(
                    "Compiler dropped a parameter target not selected by its "
                    f"semantic Function (source step {source_index})."
                )
                continue
            parameter_name = str(
                candidate.get("suggested_name")
                or parameter.get("name")
                or ""
            ).strip()
            if (
                arg_name == "package_name"
                and candidate.get("task_parameter_name") == "app_name"
            ):
                parameter_name = "package_name"
            parameter_target = (int(source_index), arg_name)
            if is_complete:
                complete_parameter_targets.add(parameter_target)
            else:
                semantic_parameter_targets.add(parameter_target)
                semantic_parameter_specs.setdefault(
                    parameter_target,
                    {
                        "name": parameter_name,
                        "description": str(parameter.get("description") or "").strip(),
                        "source_step_index": int(source_index),
                        "arg_name": arg_name,
                    },
                )
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
        _generalize_parameterized_semantics(
            function,
            source_arguments=source_arguments,
        )
        _normalize_dynamic_open_app_semantics(
            function,
            parameter_proposals=parameter_proposals,
        )
        function["description"] = _append_registered_action_plan(
            function["description"],
            function=function,
            source_steps=source_steps,
            source_indices=indices,
            source_arguments=source_arguments,
        )
        if is_complete:
            function["name"] = _generalize_task_parameter_literals(
                function["name"],
                facts.get("task_parameters"),
            )
            function["description"] = _append_terminal_handoff_description(
                function["description"],
                facts,
            )
        if function_id in materialized_function_ids:
            if not is_complete:
                raise ValueError("function_author_plan_duplicate_function_id")
            # The complete Function is the public API envelope.  Authoring
            # models may repeat the same id for a semantic prefix and its
            # complete envelope; keep the envelope rather than discarding a
            # usable plan and falling back to a goal-literal recorded Function.
            functions = [
                item
                for item in functions
                if item.get("function_id") != function_id
            ]
            arguments.pop(function_id, None)
            materialized_function_ids.remove(function_id)
            materialization_notes.append(
                "Compiler deduplicated a semantic Function repeated by the "
                "complete Function envelope."
            )
        materialized_function_ids.add(function_id)
        functions.append(function)
        arguments[function_id] = source_arguments

    required_complete_parameters = {
        target
        for target in semantic_parameter_targets
        if target[0] in complete_source_indices
    }
    if required_complete_parameters - complete_parameter_targets:
        raise ValueError("function_author_plan_complete_parameters_missing")

    normalized_reason = (
        "Compiler preserved every successful main-flow action in source order; "
        "only explicit observation-dependent boundaries hand off to the Planner."
        if preserved_complete_sequence
        else reason.strip()
    )
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


def _observation_dependent_input_indices(facts: dict[str, Any]) -> frozenset[int]:
    """Find input values that cannot be supplied before the task is observed."""

    goal = " ".join(str(facts.get("goal") or "").casefold().split())
    indices: set[int] = set()
    for index, step in enumerate(facts.get("steps") or ()):
        action = step.get("action") if isinstance(step, dict) else None
        if not isinstance(action, dict) or action.get("tool") != "input_text":
            continue
        value = " ".join(str(action.get("args", {}).get("text") or "").casefold().split())
        if value and value not in goal:
            indices.add(index)
    return frozenset(indices)


def _goal_semantic_parameters(
    facts: dict[str, Any],
    *,
    indices: list[int],
    candidates: dict[tuple[int, str], dict[str, Any]],
    existing_targets: set[tuple[int, str]],
) -> list[dict[str, Any]]:
    """Expose compiler-approved task inputs on the complete Function API."""
    result: list[dict[str, Any]] = []
    goal = " ".join(str(facts.get("goal") or "").casefold().split())

    # AndroidWorld already records the public task API alongside the goal.
    # Prefer that contract when a recorded string action contains the exact
    # source value.  This covers values such as coordinates whose semantic
    # slot (for example ``location``) is not literally named in the sentence.
    task_parameters = facts.get("task_parameters")
    if isinstance(task_parameters, dict):
        for candidate in sorted(
            candidates.values(),
            key=lambda value: (
                int(value.get("source_step_index", -1)),
                str(value.get("arg_name") or ""),
            ),
        ):
            source_index = int(candidate.get("source_step_index", -1))
            arg_name = str(candidate.get("arg_name") or "").strip()
            target = (source_index, arg_name)
            if source_index not in indices or target in existing_targets:
                continue
            recorded_value = " ".join(
                str(candidate.get("recorded_value") or "").casefold().split()
            )
            if not recorded_value:
                continue
            suggested_name = str(candidate.get("suggested_name") or "").strip()
            if not suggested_name and recorded_value in goal:
                suggested_name = _labeled_input_parameter_name(
                    facts,
                    source_index=source_index,
                )
            if suggested_name:
                description = (
                    "Installed Android package for the app requested by the task."
                    if suggested_name == "package_name"
                    else f"{suggested_name.replace('_', ' ').capitalize()} supplied by the task."
                )
                result.append(
                    {
                        "name": suggested_name,
                        "description": description,
                        "source_step_index": source_index,
                        "arg_name": arg_name,
                    }
                )
                existing_targets.add(target)
                continue
    return result


def _labeled_input_parameter_name(
    facts: dict[str, Any],
    *,
    source_index: int,
) -> str:
    steps = list(facts.get("steps") or ())
    if source_index not in range(len(steps)):
        return ""
    action = steps[source_index].get("action")
    if not isinstance(action, dict) or action.get("tool") != "input_text":
        return ""
    args = action.get("args") if isinstance(action.get("args"), dict) else {}
    label = " ".join(
        str(args.get("target_description") or "").casefold().split()
    )
    return {
        "first name": "first_name",
        "given name": "first_name",
        "last name": "last_name",
        "family name": "last_name",
    }.get(label, "")


def _task_parameter_proposals(
    function: dict[str, Any],
    facts: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build deterministic bindings for exact RunLog task-parameter values."""

    task_parameters = facts.get("task_parameters")
    if not isinstance(task_parameters, dict):
        return []
    proposals: list[dict[str, Any]] = []
    source_candidates = _source_parameter_candidates(facts)
    for candidate in parameter_candidates(function):
        recorded = " ".join(
            str(candidate.get("recorded_value") or "").casefold().split()
        )
        if not recorded:
            continue
        source_candidate = next(
            (
                item
                for item in source_candidates
                if int(item.get("source_step_index", -1))
                == int(candidate["step_index"])
                and str(item.get("arg_name") or "") == candidate["arg_name"]
                and " ".join(
                    str(item.get("recorded_value") or "").casefold().split()
                )
                == recorded
            ),
            None,
        )
        if source_candidate is None:
            continue
        name = str(source_candidate.get("suggested_name") or "").strip()
        if not name:
            continue
        proposals.append(
            {
                "name": name,
                "description": (
                    "Installed Android package for the app requested by the task."
                    if name == "package_name"
                    else f"{name.replace('_', ' ').capitalize()} supplied by the task."
                ),
                "step_index": int(candidate["step_index"]),
                "arg_name": str(candidate["arg_name"]),
                "recorded_value": str(candidate["recorded_value"]),
            }
        )
    return proposals


_GENERIC_COORDINATE_SURFACE_MARKERS = frozenset(
    {
        "background",
        "canvas",
        "calendar background",
        "calendar canvas",
        "calendar grid",
        "drawing canvas",
        "drawing surface",
        "grid cell",
        "map surface",
        "map view",
        "month view background",
        "month view grid",
        "webview",
        "web view",
    }
)


def _generic_coordinate_surface_indices(
    facts: dict[str, Any],
) -> frozenset[int]:
    """Find coordinate actions whose meaning is only the current surface."""

    indices: set[int] = set()
    for index, step in enumerate(facts.get("steps") or ()):
        action = step.get("action") if isinstance(step, dict) else None
        if not isinstance(action, dict) or action.get("tool") not in {
            "click",
            "double_click",
            "long_press",
        }:
            continue
        args = action.get("args")
        if not isinstance(args, dict) or args.get("x") is None or args.get("y") is None:
            continue
        target = " ".join(
            str(args.get("target_description") or "")
            .replace("_", " ")
            .casefold()
            .split()
        )
        if target in _GENERIC_COORDINATE_SURFACE_MARKERS:
            indices.add(index)
    return frozenset(indices)


def _observation_dependent_handoff_indices(
    facts: dict[str, Any],
) -> frozenset[int]:
    raw = facts.get("observation_dependent_handoff_indices")
    if not isinstance(raw, (list, tuple, set, frozenset)):
        return _generic_coordinate_surface_indices(facts)
    return frozenset(
        int(index)
        for index in raw
        if isinstance(index, int) and not isinstance(index, bool)
    )


def _goal_allows_dynamic_app_package(
    _goal: Any,
    *,
    candidate: dict[str, Any] | None = None,
) -> bool:
    if (
        isinstance(candidate, dict)
        and candidate.get("task_parameter_name") == "app_name"
        and candidate.get("value_contract") == "android_package_name"
    ):
        return True
    return False


def _source_action_parameter_evidence(
    *,
    source_step: dict[str, Any],
    action: dict[str, Any],
    source_step_index: int,
    task_parameters: Any,
) -> list[dict[str, Any]]:
    raw_action = source_step.get("action")
    if not isinstance(raw_action, dict) or not isinstance(task_parameters, dict):
        return []
    if str(action.get("tool") or "") != "open_app":
        return []
    if str(raw_action.get("action_type") or "") != "open_app":
        return []
    raw_app_name = " ".join(str(raw_action.get("app_name") or "").casefold().split())
    task_app_name = " ".join(
        str(task_parameters.get("app_name") or "").casefold().split()
    )
    package_name = str(action.get("args", {}).get("package_name") or "").strip()
    if not raw_app_name or raw_app_name != task_app_name or not package_name:
        return []
    return [
        {
            "source_step_index": int(source_step_index),
            "arg_name": "package_name",
            "task_parameter_name": "app_name",
            "task_parameter_value": str(task_parameters["app_name"]),
            "recorded_value": package_name,
            "value_contract": "android_package_name",
        }
    ]


def _normalize_dynamic_open_app_semantics(
    function: dict[str, Any],
    *,
    parameter_proposals: list[dict[str, Any]],
) -> None:
    if len(function.get("steps") or ()) != 1:
        return
    step = function["steps"][0]
    if step.get("action", {}).get("tool") != "open_app":
        return
    dynamic_package = any(
        proposal.get("arg_name") == "package_name"
        and proposal.get("name") == "package_name"
        for proposal in parameter_proposals
    )
    if not dynamic_package:
        return
    function["name"] = "Open requested app"
    function["description"] = (
        "Open the app requested by the task using its installed Android package. "
        "The Function completes after dispatching that app; the Planner handles "
        "any permission pop-up visible afterward."
    )


def _generalize_parameterized_semantics(
    function: dict[str, Any],
    *,
    source_arguments: dict[str, Any],
) -> None:
    for parameter_name, recorded_value in sorted(
        source_arguments.items(),
        key=lambda item: len(str(item[1] or "")),
        reverse=True,
    ):
        value = str(recorded_value or "").strip()
        if not value:
            continue
        label = " ".join(str(parameter_name).replace("_", " ").split())
        replacement = f"requested {label}" if label else "requested value"
        for field in ("name", "description"):
            function[field] = _replace_recorded_literal(
                str(function.get(field) or ""),
                value,
                replacement,
            )


def _promote_launcher_app_entry(
    action: dict[str, Any],
    *,
    observation: dict[str, Any],
    next_observation: Any,
) -> dict[str, Any]:
    """Turn a recorded launcher app-icon click into the global open_app entry."""

    if str(action.get("tool") or "") != "click":
        return action
    before_package = _primary_observation_package(observation)
    after_package = _primary_observation_package(next_observation)
    if not before_package or not after_package:
        return action
    if "launcher" not in before_package.casefold():
        return action
    if after_package == before_package or "systemui" in after_package.casefold():
        return action
    return {
        "tool": "open_app",
        "args": {"package_name": after_package},
    }


def _canonicalize_open_app_action(
    action: dict[str, Any],
    *,
    next_observation: Any,
) -> dict[str, Any]:
    """Persist the native package observed after a recorded app launch."""

    if str(action.get("tool") or "") != "open_app":
        return action
    package_name = str(action.get("args", {}).get("package_name") or "").strip()
    if "." in package_name and " " not in package_name:
        return action
    after_package = _primary_observation_package(next_observation)
    if not after_package:
        return action
    updated = json.loads(json.dumps(action, ensure_ascii=False))
    updated.setdefault("args", {})["package_name"] = after_package
    return updated


def _primary_observation_package(observation: Any) -> str:
    if not isinstance(observation, dict):
        return ""
    auxiliaries = observation.get("auxiliaries")
    if isinstance(auxiliaries, dict):
        package_name = str(auxiliaries.get("package_name") or "").strip()
        if package_name:
            return package_name
    xml = str(observation.get("xml") or observation.get("forest") or "").strip()
    if not xml:
        return ""
    try:
        packages = [
            str(node.get("package") or "").strip()
            for node in ET.fromstring(xml).iter()
            if str(node.get("package") or "").strip()
        ]
    except ET.ParseError:
        packages = re.findall(r'\bpackage="([^"]+)"', xml)
    for package in packages:
        if "systemui" not in package.casefold():
            return package
    return ""


def _atomicize_repeated_click_function(
    indices: list[int],
    source_steps: list[dict[str, Any]],
    *,
    function_id: str,
    name: str,
    description: str,
) -> tuple[list[int], str, str, str, int]:
    actions = [source_steps[index].get("action") for index in indices]
    observation_dependent_repeat = False
    run_start = 0
    while run_start < len(actions):
        run_end = run_start + 1
        while run_end < len(actions) and actions[run_end] == actions[run_start]:
            run_end += 1
        action = actions[run_start]
        if (
            run_end - run_start > 1
            and isinstance(action, dict)
            and action.get("tool") == "click"
        ):
            state_ids = {
                str(source_steps[indices[position]].get("before_state_id") or "")
                for position in range(run_start, run_end)
            }
            if len(state_ids) > 1:
                observation_dependent_repeat = True
                break
        run_start = run_end
    if not observation_dependent_repeat:
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
            and _is_post_input_commit_action(commit_step)
        ):
            continue
        commit_state_id = str(commit_step.get("before_state_id") or "").strip()
        if not commit_state_id:
            continue
        input_state_id = str(source_step.get("before_state_id") or "").strip()
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
                candidate_steps.append(
                    {
                        "step_index": len(candidate_steps),
                        "source_state_id": commit_state_id,
                        "action": json.loads(
                            json.dumps(commit_action, ensure_ascii=False)
                        ),
                    }
                )
                restored += 1
    return restored


def _is_post_input_commit_action(step: dict[str, Any]) -> bool:
    action = step.get("action")
    if not isinstance(action, dict) or action.get("tool") != "click":
        return False
    args = action.get("args") if isinstance(action.get("args"), dict) else {}
    metadata = step.get("metadata") if isinstance(step.get("metadata"), dict) else {}
    target_tokens = set(
        re.findall(
            r"[a-z0-9]+",
            str(args.get("target_description") or "")
            .replace("_", " ")
            .casefold(),
        )
    )
    purpose_tokens = set(
        re.findall(
            r"[a-z0-9]+",
            str(metadata.get("purpose") or "").replace("_", " ").casefold(),
        )
    )
    target_markers = {
        "advance",
        "apply",
        "commit",
        "confirm",
        "continue",
        "create",
        "done",
        "finish",
        "next",
        "ok",
        "save",
        "send",
        "submit",
    }
    purpose_markers = target_markers - {"next"}
    return bool(
        target_tokens.intersection(target_markers)
        or purpose_tokens.intersection(purpose_markers)
    )


def _default_bundle(
    facts: dict[str, Any],
    recovery_examples: list[dict[str, Any]],
) -> dict[str, Any] | None:
    source_steps = list(facts.get("steps") or ())
    if not source_steps:
        return None
    dynamic_indices = _observation_dependent_input_indices(facts)
    boundary_indices = dynamic_indices
    if boundary_indices:
        boundary = min(boundary_indices)
        safe_prefix = source_steps[:boundary]
        if not safe_prefix:
            safe_prefix = [
                step
                for index, step in enumerate(source_steps)
                if index not in boundary_indices
            ][:1]
        if not safe_prefix:
            return None
        prefix_facts = dict(facts)
        prefix_facts["steps"] = safe_prefix
        function = _complete_function_artifact(prefix_facts)
        function_source_steps = safe_prefix
        if boundary in dynamic_indices:
            handoff_description = (
                "Stop before the value-dependent input and let the Planner "
                "continue from the observed page."
            )
        else:
            handoff_description = (
                "Stop before the generic surface and let the Planner inspect "
                "the current page and choose the target."
            )
        function["description"] = f"{function['description']} {handoff_description}"
    else:
        function = _complete_function_artifact(facts)
        function_source_steps = source_steps
    task_parameter_proposals = _task_parameter_proposals(function, facts)
    task_parameter_values = {
        str(proposal["name"]): str(proposal["recorded_value"])
        for proposal in task_parameter_proposals
    }
    if task_parameter_proposals:
        apply_parameters(
            function,
            [
                {
                    key: value
                    for key, value in proposal.items()
                    if key != "recorded_value"
                }
                for proposal in task_parameter_proposals
            ],
            facts,
        )
        _generalize_parameterized_semantics(
            function,
            source_arguments=task_parameter_values,
        )
        _normalize_dynamic_open_app_semantics(
            function,
            parameter_proposals=task_parameter_proposals,
        )
    function["description"] = _append_registered_action_plan(
        function["description"],
        function=function,
        source_steps=function_source_steps,
        source_indices=list(range(len(function.get("steps") or ()))),
        source_arguments=task_parameter_values,
    )
    function["name"] = _generalize_task_parameter_literals(
        function["name"],
        facts.get("task_parameters"),
    )
    function["description"] = _append_terminal_handoff_description(
        function["description"],
        facts,
    )
    function_id = function["function_id"]
    return {
        "schema_version": "omniflow.function-bundle.v2",
        "run_id": facts["run_id"],
        "arguments": {function_id: task_parameter_values},
        "checker_rules": [],
        "functions": [function],
    }


def _complete_function_artifact(
    facts: dict[str, Any],
    *,
    existing_function_ids: set[str] | None = None,
) -> dict[str, Any]:
    source_steps = list(facts.get("steps") or ())
    if not source_steps:
        raise ValueError("complete_function_source_actions_required")
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
    base_function_id = f"complete_recorded_{digest}"
    used_ids = existing_function_ids or set()
    function_id = base_function_id
    suffix = 2
    while function_id in used_ids:
        function_id = f"{base_function_id}_{suffix}"
        suffix += 1
    goal = str(facts["goal"])
    return {
        "schema_version": "omniflow.function.v2",
        "function_id": function_id,
        "name": goal[:120],
        "description": goal,
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


def _append_registered_action_plan(
    description: str,
    *,
    function: dict[str, Any],
    source_steps: list[dict[str, Any]],
    source_indices: list[int],
    source_arguments: dict[str, Any],
) -> str:
    core_effect = " ".join(str(description or "").split()).strip().rstrip(".")
    core = (
        core_effect
        if core_effect.casefold().startswith("core effect:")
        else f"Core effect: {core_effect}"
    )
    plan: list[str] = []
    for local_index, function_step in enumerate(function.get("steps") or ()):
        action = (
            function_step.get("action")
            if isinstance(function_step, dict)
            else None
        )
        if not isinstance(action, dict):
            continue
        source_index = (
            source_indices[local_index]
            if local_index < len(source_indices)
            else local_index
        )
        source_step = (
            source_steps[source_index]
            if source_index in range(len(source_steps))
            else {}
        )
        metadata = (
            source_step.get("metadata")
            if isinstance(source_step, dict)
            and isinstance(source_step.get("metadata"), dict)
            else {}
        )
        purpose = str(
            metadata.get("purpose")
            or metadata.get("action_description")
            or metadata.get("summary")
            or ""
        ).strip().rstrip(".")
        purpose = _generalize_text_with_arguments(purpose, source_arguments)
        action_detail = _registered_action_detail(
            action,
            function=function,
            step_index=local_index,
        )
        detail = (
            f"Purpose: {purpose}. Action: {action_detail}."
            if purpose
            else f"Action: {action_detail}."
        )
        plan.append(f"{local_index + 1}) {detail}")
    if not plan:
        return f"{core}."
    return f"{core}. Action plan: {' '.join(plan)}"


def _generalize_text_with_arguments(
    text: str,
    source_arguments: dict[str, Any],
) -> str:
    result = str(text or "")
    for parameter_name, recorded_value in sorted(
        source_arguments.items(),
        key=lambda item: len(str(item[1] or "")),
        reverse=True,
    ):
        value = str(recorded_value or "").strip()
        if not value:
            continue
        replacement = f"${parameter_name}"
        result = _replace_recorded_literal(result, value, replacement)
    return result


def _replace_recorded_literal(text: str, value: str, replacement: str) -> str:
    pattern = re.escape(value)
    if value[0].isalnum():
        pattern = rf"(?<!\w){pattern}"
    if value[-1].isalnum():
        pattern = rf"{pattern}(?!\w)"
    return re.sub(pattern, replacement, text, flags=re.IGNORECASE)


def _registered_action_detail(
    action: dict[str, Any],
    *,
    function: dict[str, Any],
    step_index: int,
) -> str:
    tool = str(action.get("tool") or "action").strip()
    args = action.get("args") if isinstance(action.get("args"), dict) else {}
    semantic_arg_names = {
        "open_app": ("package_name",),
        "click": ("target_description",),
        "double_click": ("target_description",),
        "long_press": ("target_description",),
        "input_text": ("target_description", "text"),
        "swipe": ("direction",),
        "press_key": ("key",),
    }.get(tool, tuple(sorted(args)))
    signature_args = [
        (
            f"{arg_name}="
            f"{_registered_action_arg(function, step_index, arg_name, args[arg_name])}"
        )
        for arg_name in semantic_arg_names
        if arg_name in args
    ]
    if not signature_args and tool in {"click", "double_click", "long_press"}:
        signature_args.append(
            'target="OmniTransfer mapping from the registered source state"'
        )
    return f"{tool}({', '.join(signature_args)})"


def _registered_action_arg(
    function: dict[str, Any],
    step_index: int,
    arg_name: str,
    recorded_value: Any,
) -> str:
    target = f"$.steps[{step_index}].action.args.{arg_name}"
    for binding in function.get("bindings") or ():
        if not isinstance(binding, dict) or binding.get("target") != target:
            continue
        source = str(binding.get("source") or "").strip()
        parameter = source.removeprefix("$.arguments.")
        if parameter and parameter != source:
            return f"${parameter}"
    return json.dumps(recorded_value, ensure_ascii=False, separators=(",", ":"))


def _append_terminal_handoff_description(
    description: str,
    facts: dict[str, Any],
) -> str:
    omitted = {
        str(value).strip().lower()
        for value in facts.get("omitted_action_types") or ()
    }
    if not omitted.intersection({"answer", "status"}):
        return description
    description = _generalize_task_parameter_literals(
        description,
        facts.get("task_parameters"),
    )
    suffix = (
        " This Function has completed only its registered prefix, not the task. "
        "Continue from the current observed page and follow the current goal: use "
        "its requested attributes to locate the requested item, read the requested "
        "value or completion state, then return the required answer or status."
    )
    return description if description.endswith(suffix) else f"{description}{suffix}"


def _generalize_task_parameter_literals(text: str, task_parameters: Any) -> str:
    if not isinstance(task_parameters, dict):
        return str(text or "")
    result = str(text or "")
    literals: list[tuple[str, str]] = []
    for raw_name, raw_value in task_parameters.items():
        name = str(raw_name or "").strip()
        if name in {"app_name", "package_name", "seed"}:
            continue
        if isinstance(raw_value, bool) or not isinstance(
            raw_value, (str, int, float)
        ):
            continue
        value = str(raw_value).strip()
        if len(value) < 2:
            continue
        label = " ".join(name.replace("_", " ").split()) or "task value"
        literals.append((value, f"current-goal {label}"))
    for value, replacement in sorted(
        literals,
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        tokens = re.findall(r"[^\W_]+", value, flags=re.UNICODE)
        if len(tokens) > 1:
            pattern = r"(?<!\w)" + r"(?:[\W_]+)".join(
                re.escape(token) for token in tokens
            ) + r"(?!\w)"
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
        else:
            result = _replace_recorded_literal(result, value, replacement)
    return result


def _is_transient_system_action(step: dict[str, Any]) -> bool:
    """Return whether a successful source click only dismisses system noise."""
    action = step.get("action") if isinstance(step.get("action"), dict) else {}
    if str(action.get("action_type") or "") != "click":
        return False
    index = action.get("index")
    if isinstance(index, bool) or not isinstance(index, int):
        return False
    observation = step.get("observation")
    if not isinstance(observation, dict):
        return False
    xml = observation.get("xml")
    if not isinstance(xml, str) or not xml.strip():
        return False
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return False
    nodes = root.iter("node")
    for node in nodes:
        if str(node.get("id") or "") != str(index):
            continue
        resource_id = str(node.get("resource-id") or "")
        package = str(node.get("package") or "")
        text = str(node.get("text") or "").strip().lower()
        if package == "android" and (
            resource_id.startswith("android:id/aerr_")
            or text in {"close app", "app info", "wait"}
        ):
            return True
        break
    return False
