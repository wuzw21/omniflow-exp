"""AutoDroid native Memory policy running inside the official AndroidWorld loop.

This module deliberately does not import or start DroidBot.  It consumes the
four converted native Memory tables, renders the live AndroidWorld observation
with the same native description grammar, asks the configured model for the
native ``id/action/input_text`` decision, and dispatches that decision through
the shared AndroidWorld Host (and therefore the canonical OOB physical layer).
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any

import numpy as np

from src.integrations.android_world.host import AndroidWorldHost, make_agent_result
from src.integrations.autodroid_memory import render_androidworld_observation


class AutoDroidMemoryAgent:
    """One native AutoDroid TaskPolicy decision loop without the native runner."""

    def __init__(
        self,
        *,
        env: Any,
        memory_root: str | Path,
        llm: Any,
        adb_serial: str,
        adb_path: str = "",
        max_steps: int = 20,
        evidence_root: str | Path | None = None,
        performance_metrics: Any | None = None,
    ) -> None:
        self.name = "autodroid"
        self.env = env
        self.host = AndroidWorldHost(
            env,
            adb_serial=adb_serial,
            adb_path=adb_path,
            evidence_root=evidence_root,
            performance_metrics=performance_metrics,
            control_backend="oob",
        )
        self.memory_root = Path(memory_root).expanduser().resolve()
        self.llm = llm
        # run_episode's common result writer reads this exact tracker seam.
        self._omniflow_llm_usage_tracker = llm
        self.max_steps = max(1, int(max_steps))
        self.actions_executed = 0
        self.embedding_calls = 0
        self.memory_lookup_count = 0
        self.memory_hit_count = 0
        self.step_index = 0
        self.goal = ""
        self.task_name = ""
        self.action_history: list[str] = []
        self.memory_guidance: dict[str, Any] = {}
        self.screen_size = (1, 1)
        self._render_dir = Path(
            tempfile.mkdtemp(prefix="omniflow-autodroid-render-")
        )
        self._load_memory()

    def _load_memory(self) -> None:
        report_path = self.memory_root / "memory_build_report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        self.memory_report = report
        self.app_package = str(report.get("app_package") or "").strip()
        self.app_key = str(
            report.get("official_app_key")
            or report.get("app_name")
            or self.memory_root.name
        ).strip()
        self.node_table = self._load_table("node_filtered_elements.json")
        self.description_table = self._load_table("element_description.json")
        self.embedding_table = self._load_table("embedded_elements_desc.json")

    def _load_table(self, filename: str) -> dict[str, Any]:
        payload = json.loads((self.memory_root / filename).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"autodroid_memory_table_invalid:{filename}")
        if self.app_key not in payload:
            # Official published tables use a stable app key, while some
            # converted bundles retain a directory name.  This is a key
            # normalization only; it never selects another Memory asset.
            candidates = [key for key in payload if str(key).casefold() == self.app_key.casefold()]
            if len(candidates) != 1:
                raise KeyError(f"autodroid_memory_app_key_missing:{filename}:{self.app_key}")
            self.app_key = str(candidates[0])
        value = payload[self.app_key]
        if not isinstance(value, dict):
            raise ValueError(f"autodroid_memory_app_table_invalid:{filename}")
        return value

    def reset(self, go_home: bool = False) -> None:
        self.host.reset(go_home=go_home)
        self.step_index = 0
        self.actions_executed = 0
        self.action_history = [f"- launchApp {self.app_key}"]
        # AndroidWorld resets the agent after initialize_task().  Keep the
        # task-level native retrieval across that lifecycle boundary; only
        # episode-local action history is reset.

    def set_max_steps(self, step_budget: int) -> None:
        self.max_steps = min(self.max_steps, max(1, int(step_budget)))

    def update_current_task_context(self, task: Any) -> dict[str, Any]:
        params = getattr(task, "params", {})
        return {
            "task_parameters": dict(params) if isinstance(params, dict) else {},
            "memory_root": str(self.memory_root),
            "memory_app_key": self.app_key,
        }

    def set_current_task(
        self,
        task_name: str,
        goal: str = "",
        context: dict[str, Any] | None = None,
    ) -> None:
        del context
        self.task_name = str(task_name or "").strip()
        self.goal = str(goal or self.task_name).strip()
        self.action_history = [f"- launchApp {self.app_key}"]
        self.memory_guidance = self._retrieve_memory_guidance(self.goal)

    def _retrieve_memory_guidance(self, goal: str) -> dict[str, Any]:
        """Perform the native TaskPolicy global task-to-element retrieval."""

        self.memory_lookup_count += 1
        self.embedding_calls += 1
        model = self._load_embedding_model()
        query = np.asarray(model.encode("task: " + str(goal)), dtype=float).reshape(-1)
        query_norm = float(np.linalg.norm(query))
        if query_norm <= 0:
            raise ValueError("autodroid_memory_task_embedding_empty")
        best: tuple[float, str, int, str] | None = None
        for state_id, values in self.embedding_table.items():
            if not isinstance(values, list):
                continue
            for index, vector in enumerate(values):
                if not isinstance(vector, list) or not vector:
                    continue
                candidate = np.asarray(vector, dtype=float).reshape(-1)
                if candidate.shape != query.shape:
                    continue
                candidate_norm = float(np.linalg.norm(candidate))
                if candidate_norm <= 0:
                    continue
                similarity = float(np.dot(query, candidate) / (query_norm * candidate_norm))
                if best is None or similarity > best[0]:
                    best = (similarity, str(state_id), index, self.app_key)
        del model
        if best is None:
            raise ValueError("autodroid_memory_no_embedded_element")
        self.memory_hit_count += 1
        similarity, state_id, element_index, app_key = best
        node = self.node_table.get(state_id) or {}
        descriptions = self.description_table.get(state_id) or []
        elements = node.get("elements") or []
        description = descriptions[element_index] if element_index < len(descriptions) else ""
        statement = elements[element_index] if element_index < len(elements) else None
        return {
            "app_key": app_key,
            "state_id": state_id,
            "element_index": int(element_index),
            "statement": statement,
            "description": str(description or ""),
            "path": list(node.get("path") or []),
            "similarity": round(similarity, 8),
            "retrieval": "native_task_policy_global_cosine",
        }

    @staticmethod
    def _load_embedding_model() -> Any:
        from InstructorEmbedding import INSTRUCTOR

        # InstructorEmbedding 1.x and current sentence-transformers disagree
        # on this private hook.  Keep the compatibility shim local to the
        # official Memory consumer and preserve the native encoder itself.
        import inspect

        loader = getattr(INSTRUCTOR, "_load_sbert_model", None)
        if loader is not None and "token" not in inspect.signature(loader).parameters:
            original = loader

            def compatible(self: Any, model_path: Any, token: Any = None, **kwargs: Any) -> Any:
                del token, kwargs
                return original(self, model_path)

            INSTRUCTOR._load_sbert_model = compatible
        configured = str(os.environ.get("AUTODROID_INSTRUCTOR_MODEL") or "").strip()
        candidates = [
            Path(configured).expanduser() if configured else None,
            Path.home() / "models" / "instructor-xl",
            Path("/models/instructor-xl"),
        ]
        model_ref = next(
            (candidate for candidate in candidates if candidate is not None and candidate.is_dir()),
            Path("hkunlp/instructor-xl"),
        )
        return INSTRUCTOR(str(model_ref), device="cpu")

    @staticmethod
    def _strip_ids(value: str) -> str:
        return re.sub(r"\s+id=\d+", "", str(value or "")).strip()

    def _apply_memory_guidance(self, state_prompt: str) -> tuple[str, dict[str, Any]]:
        path = self.memory_guidance.get("path") or []
        path_index = max(0, len(self.action_history) - 1)
        if path_index >= len(path):
            return state_prompt, {"path_index": path_index, "applied": False}
        target = path[path_index]
        if isinstance(target, list):
            target = target[-1] if target else ""
        target = self._strip_ids(str(target or ""))
        description = str(self.memory_guidance.get("description") or "").strip()
        if not target or not description:
            return state_prompt, {"path_index": path_index, "applied": False}
        lines: list[str] = []
        applied = False
        for line in str(state_prompt).splitlines():
            comparable = self._strip_ids(line)
            if not applied and comparable == target:
                marker = line.find(">")
                if marker >= 0:
                    line = (
                        line[:marker]
                        + f" onclick='go to complete the {description}'"
                        + line[marker:]
                    )
                    applied = True
            lines.append(line)
        return "\n".join(lines), {
            "path_index": path_index,
            "applied": applied,
            "target_statement": target,
            "description": description,
        }

    def _prompt(self, state_prompt: str, guidance: dict[str, Any]) -> str:
        return (
            "You are a smartphone assistant to help users complete tasks by interacting with mobile apps."
            "Given a task, the previous UI actions, and the content of current UI state, your job is to decide "
            "whether the task is already finished by the previous actions, and if not, decide which UI element "
            "in current UI state should be interacted.\n"
            f"Task: {self.goal}\n"
            "Previous UI actions: \n"
            + "\n".join(self.action_history)
            + "\nCurrent UI state: \n"
            + state_prompt
            + "\n"
            "Your answer should always use the following format: { \"Steps\": \"...<steps usually involved to complete the above task on a smartphone>\", \"Analyses\": \"...<Analyses of the relations between the task, and relations between the previous UI actions and current UI state>\", \"Finished\": \"Yes/No\", \"Next step\": \"None or a <high level description of the next step>\", \"id\": \"an integer or -1 (if the task has been completed by previous actions)\", \"action\": \"tap or input\", \"input_text\": \"N/A or ...<input text>\" } \n\n"
            "Note that id is the id number of the UI element to interact with. If you think the task has been completed by previous actions, id should be -1. If Finished is Yes, Next step is None, otherwise it is a high level description of the next step. If action is tap, input_text is N/A, otherwise it is the input text. Output only the JSON format."
            + ("\nMemory guidance was applied to the next candidate." if guidance.get("applied") else "")
        )

    @staticmethod
    def _parse_response(value: Any) -> dict[str, Any]:
        text = str(value or "").strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
            text = re.sub(r"\s*```$", "", text).strip()
        try:
            decoded = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if match is None:
                raise ValueError("autodroid_llm_json_invalid")
            decoded = json.loads(match.group(0))
        if not isinstance(decoded, dict):
            raise ValueError("autodroid_llm_response_object_required")
        return decoded

    def _current_render(self) -> tuple[Any, dict[str, Any]]:
        observation = self.host.observe(xml=True, screenshot=False, app_info=True)
        state = observation.extra.get("androidworld_state")
        if not isinstance(state, dict):
            raise ValueError("autodroid_androidworld_state_snapshot_missing")
        auxiliaries = state.get("auxiliaries")
        display = auxiliaries.get("display") if isinstance(auxiliaries, dict) else None
        if isinstance(display, dict):
            self.screen_size = (
                max(1, int(display.get("width") or 1)),
                max(1, int(display.get("height") or 1)),
            )
        rendered = render_androidworld_observation(state, temp_output=self._render_dir)
        return observation, rendered

    @staticmethod
    def _center(view: dict[str, Any]) -> tuple[float, float, int, int]:
        raw = view.get("bounds")
        if not isinstance(raw, list) or len(raw) != 2:
            raise ValueError("autodroid_selected_view_bounds_missing")
        left, top = raw[0]
        right, bottom = raw[1]
        width = max(1, int(right) - int(left))
        height = max(1, int(bottom) - int(top))
        return (
            (int(left) + int(right)) / 2.0,
            (int(top) + int(bottom)) / 2.0,
            max(1, int(right)),
            max(1, int(bottom)),
        )

    def _dispatch(self, action_type: str, view: dict[str, Any], payload: dict[str, Any]) -> Any:
        if action_type == "KeyEvent":
            return self.host.act({"tool": "press_key", "args": {"key": "back"}})
        x, y, _right, _bottom = self._center(view)
        width, height = self.screen_size
        normalized = {"x": x / width * 1000.0, "y": y / height * 1000.0}
        action_name = str(payload.get("action") or "tap").casefold()
        if action_type == "SetTextEvent" or "input" in action_name:
            normalized["text"] = str(payload.get("input_text") or "")
            normalized["clear_text"] = True
            return self.host.act({"tool": "input_text", "args": normalized})
        if action_type == "LongTouchEvent":
            normalized["duration_ms"] = 2000
            return self.host.act({"tool": "long_press", "args": normalized})
        return self.host.act({"tool": "click", "args": normalized})

    def step(self, goal: str) -> Any:
        if goal and not self.goal:
            self.set_current_task(self.task_name or "androidworld", goal, {})
        if self.step_index >= self.max_steps:
            return make_agent_result(True, {"summary": "autodroid_step_budget_reached", "error": "step_budget"})
        self.step_index += 1
        try:
            observation, rendered = self._current_render()
            package = str(observation.package_name or "").strip()
            if (
                self.step_index == 1
                and self.app_package
                and package
                and package != self.app_package
                and "permissioncontroller" not in package.casefold()
            ):
                launch = self.host.act({"tool": "open_app", "args": {"package_name": self.app_package}})
                if not launch.success:
                    return make_agent_result(True, {"summary": "autodroid_app_launch_failed", "error": launch.error})
                self.actions_executed += 1
                self.action_history.append(f"- launchApp {self.app_key}")
                return make_agent_result(False, {"summary": "autodroid_app_launched", "actions_executed": self.actions_executed})
            state_prompt, guidance = self._apply_memory_guidance(rendered["html"])
            response, _, _ = self.llm.predict(self._prompt(state_prompt, guidance))
            decoded = self._parse_response(response)
            finished = decoded.get("Finished", decoded.get("finished", "No"))
            if isinstance(finished, bool):
                is_finished = finished
            else:
                is_finished = str(finished or "").strip().casefold() in {"yes", "y", "true"}
            try:
                selected_id = int(decoded.get("id", -1))
            except (TypeError, ValueError):
                selected_id = -1
            if is_finished or selected_id < 0:
                return make_agent_result(
                    True,
                    {
                        "summary": "autodroid_model_finished",
                        "actions_executed": self.actions_executed,
                        "memory_guidance": guidance,
                    },
                )
            refs = list(rendered.get("action_view_refs") or [])
            types = list(rendered.get("action_types") or [])
            selected_type = types[selected_id] if selected_id < len(types) else "TouchEvent"
            if selected_id >= len(refs) or (
                selected_type != "KeyEvent"
                and not isinstance(refs[selected_id], dict)
            ):
                raise ValueError(f"autodroid_action_id_out_of_range:{selected_id}")
            result = self._dispatch(selected_type, refs[selected_id] or {}, decoded)
            if not result.success:
                return make_agent_result(True, {"summary": "autodroid_action_failed", "error": result.error})
            self.actions_executed += 1
            chosen = rendered["elements"][selected_id] if selected_id < len(rendered["elements"]) else str(selected_id)
            self.action_history.append(f"- {decoded.get('action', 'tap')}: {chosen}")
            self.memory_guidance["last_step"] = guidance
            return make_agent_result(False, {"summary": "autodroid_action_executed", "actions_executed": self.actions_executed, "memory_guidance": guidance})
        except Exception as error:
            return make_agent_result(True, {"summary": "autodroid_policy_error", "error": str(error), "actions_executed": self.actions_executed})

    def get_execution_summary(self) -> dict[str, Any]:
        usage = self.llm.get_usage_summary() if hasattr(self.llm, "get_usage_summary") else {}
        return {
            **dict(usage or {}),
            "actions_executed": int(self.actions_executed),
            "embedding_calls": int(self.embedding_calls),
            "memory_lookup_count": int(self.memory_lookup_count),
            "memory_hit_count": int(self.memory_hit_count),
            "execution_backend": "autodroid_native_memory_official_androidworld",
            "memory_root": str(self.memory_root),
            "memory_app_key": self.app_key,
            "memory_retrieval": self.memory_guidance.get("retrieval"),
        }
