"""AppAgent-native demonstration adaptation and provenance helpers."""

from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any
import xml.etree.ElementTree as ET

from PIL import Image

from src.integrations.android_world.accessibility import androidworld_forest_xml
from src.integrations.android_world.apps import resolve_androidworld_app_name
from src.integrations.android_world.host import (
    androidworld_elements_xml,
    make_agent_result,
)
from src.integrations.runlog import extract_canonical_step_actions

APPAGENT_OFFICIAL_REVISION = "2c1900422caf6f9e94e96d5dd984b530e5a5fbf8"
APPAGENT_SOURCE_SEED = 111
APPAGENT_TEACHER_SOURCE_SCHEMA = "omniflow.appagent-teacher-source.v2"
APPAGENT_DEMO_MEMORY_SCHEMA = "omniflow.appagent-demo-memory.v2"
APPAGENT_DEMO_MANIFEST = "appagent_demo_manifest.json"
APPAGENT_DEMO_ACTION_TYPES = {
    "click",
    "input_text",
    "long_press",
    "swipe",
}
APPAGENT_SUPPORTED_SOURCE_TYPES = APPAGENT_DEMO_ACTION_TYPES | {"open_app"}

_NON_PRIMITIVE_SOURCE_TYPES = {
    "done",
    "finish",
    "finished",
    "status",
    "wait",
}
_SOURCE_COORDINATE_FIELDS = {
    "bounds",
    "display_height",
    "display_width",
    "end_x",
    "end_y",
    "height",
    "index",
    "node_id",
    "page_pixels",
    "source_node_id",
    "start_x",
    "start_y",
    "width",
    "x",
    "x1",
    "x2",
    "y",
    "y1",
    "y2",
}


def _native_appagent_observation(env: Any) -> tuple[str, Any]:
    state = env.get_state()
    pixels = getattr(state, "pixels", None)
    if pixels is None:
        raise ValueError("appagent_androidworld_screenshot_missing")
    xml_text = str(getattr(state, "xml", "") or "").strip()
    if not xml_text and getattr(state, "forest", None) is not None:
        if isinstance(pixels, Image.Image):
            screen_size = pixels.size
        else:
            shape = tuple(getattr(pixels, "shape", ()) or ())
            screen_size = (
                (int(shape[1]), int(shape[0]))
                if len(shape) >= 2
                else (1, 1)
            )
        xml_text = androidworld_forest_xml(
            state.forest,
            screen_size=screen_size,
        )
    if not xml_text:
        xml_text = androidworld_elements_xml(
            list(getattr(state, "ui_elements", ()) or ())
        )
    if not xml_text:
        raise ValueError("appagent_androidworld_xml_missing")
    return xml_text, pixels


@dataclass(frozen=True)
class AppAgentElement:
    uid: str
    bbox: tuple[tuple[int, int], tuple[int, int]]
    attrib: str


@dataclass(frozen=True)
class GroundedAppAgentAction:
    tag: int
    uid: str
    bbox: tuple[tuple[int, int], tuple[int, int]]
    match_reason: str


class OfficialAppAgentRuntime:
    """Pinned official AppAgent prompt/parser/labeling implementation."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        revision = subprocess.run(
            ["git", "-C", str(self.root), "rev-parse", "HEAD"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout.strip()
        if revision != APPAGENT_OFFICIAL_REVISION:
            raise ValueError(
                "appagent_official_revision_mismatch:"
                f"expected={APPAGENT_OFFICIAL_REVISION}:actual={revision}"
            )
        scripts_dir = self.root / "scripts"
        if not scripts_dir.is_dir():
            raise FileNotFoundError(f"appagent_official_scripts_missing:{scripts_dir}")
        sys.path.insert(0, str(scripts_dir))
        try:
            self._prompts = importlib.import_module("prompts")
            self._model = importlib.import_module("model")
            self._utils = importlib.import_module("utils")
        finally:
            try:
                sys.path.remove(str(scripts_dir))
            except ValueError:
                pass
        for module in (self._prompts, self._model, self._utils):
            module_path = Path(str(getattr(module, "__file__", ""))).resolve()
            if self.root not in module_path.parents:
                raise RuntimeError(f"appagent_official_module_shadowed:{module_path}")
        config = _read_appagent_config(self.root / "config.yaml")
        self.min_dist = float(config.get("MIN_DIST") or 30)
        self.request_interval = max(
            0.0,
            float(
                os.environ.get("APPAGENT_REQUEST_INTERVAL")
                or config.get("REQUEST_INTERVAL")
                or 10
            ),
        )
        self.dark_mode = bool(config.get("DARK_MODE"))

    def draw_elements(
        self,
        source: Path,
        target: Path,
        elements: list[AppAgentElement],
        *,
        record_mode: bool,
    ) -> None:
        self._utils.draw_bbox_multi(
            str(source),
            str(target),
            elements,
            record_mode=record_mode,
            dark_mode=self.dark_mode,
        )

    def draw_grid(self, source: Path, target: Path) -> tuple[int, int]:
        rows, columns = self._utils.draw_grid(str(source), str(target))
        return int(rows), int(columns)

    def build_task_prompt(
        self,
        *,
        goal: str,
        last_action: str,
        ui_document: str,
        grid: bool,
    ) -> str:
        if grid:
            template = self._prompts.task_template_grid
        else:
            template = self._prompts.task_template
            if ui_document:
                documentation = """
            You also have access to the following documentations that describes the functionalities of UI
            elements you can interact on the screen. These docs are crucial for you to determine the target of your
            next action. You should always prioritize these documented elements for interaction:""" + ui_document
                template = re.sub(r"<ui_document>", documentation, template)
            else:
                template = re.sub(r"<ui_document>", "", template)
        prompt = re.sub(r"<task_description>", str(goal), template)
        return re.sub(r"<last_act>", str(last_action), prompt)

    def parse_response(self, response: str, *, grid: bool) -> list[Any]:
        parser = self._model.parse_grid_rsp if grid else self._model.parse_explore_rsp
        return list(parser(response))


class AppAgentAndroidWorldAgent:
    """Run AppAgent's native deployment policy inside one AndroidWorld episode."""

    def __init__(
        self,
        *,
        env: Any,
        official_runtime: Any,
        llm: Any,
        output_root: str | Path,
        docs_root: str | Path | None = None,
        action_source: str | Path | None,
        action_factory: Any | None = None,
    ) -> None:
        self.env = env
        self.official_runtime = official_runtime
        self.llm = llm
        self.output_root = Path(output_root).expanduser().resolve()
        if self.output_root.exists() and any(self.output_root.iterdir()):
            raise FileExistsError(
                f"immutable_appagent_output_exists:{self.output_root}"
            )
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.docs_root = (
            Path(docs_root).expanduser().resolve() if docs_root is not None else None
        )
        if self.docs_root is not None and not self.docs_root.is_dir():
            raise FileNotFoundError(f"appagent_docs_missing:{self.docs_root}")
        if action_source is None or not str(action_source).strip():
            raise ValueError("appagent_action_source_required")
        self.action_source_path = Path(action_source).expanduser().resolve()
        self.startup_actions: list[dict[str, Any]] = []
        action_source_payload = load_appagent_teacher_source(self.action_source_path)
        demo_action_seen = False
        for record in action_source_payload["actions"]:
            action = dict(record.get("action") or {})
            action_type = str(action.get("type") or "").strip()
            if action_type == "open_app":
                if demo_action_seen:
                    raise ValueError(
                        "appagent_runtime_open_app_must_precede_demo_actions"
                    )
                self.startup_actions.append(action)
            else:
                demo_action_seen = True
        self._action_factory = action_factory
        self.name = "appagent"
        self._omniflow_llm_usage_tracker = llm
        self.task_name = ""
        self.goal = ""
        self.task_context: dict[str, Any] = {}
        self.app_name = ""
        self.round_count = 0
        self.actions_executed = 0
        self.last_action = "None"
        self.grid_on = False
        self.grid_rows = 0
        self.grid_columns = 0
        self._startup_actions_executed = False
        self._max_steps = 20
        self._log_path = self.output_root / "appagent_task_log.jsonl"

    def set_max_steps(self, max_steps: int) -> None:
        self._max_steps = max(1, int(max_steps))

    def reset(self, go_home: bool = False) -> None:
        reset = getattr(self.env, "reset", None)
        if callable(reset):
            reset(go_home=go_home)
        self.round_count = 0
        self.actions_executed = 0
        self.last_action = "None"
        self.grid_on = False
        self.grid_rows = 0
        self.grid_columns = 0
        self._startup_actions_executed = False

    def update_current_task_context(self, task: Any) -> dict[str, Any]:
        app_names = [
            str(value).strip()
            for value in tuple(getattr(task, "app_names", ()) or ())
            if str(value).strip()
        ]
        return {"app_names": app_names}

    def set_current_task(
        self,
        task_name: str,
        goal: str,
        task_context: dict[str, Any] | None = None,
    ) -> None:
        self.task_name = str(task_name or "").strip()
        self.goal = str(goal or "").strip()
        self.task_context = dict(task_context or {})
        app_names = [
            str(value).strip()
            for value in self.task_context.get("app_names") or ()
            if str(value).strip()
        ]
        if len(app_names) > 1:
            raise ValueError("appagent_multi_app_task_unsupported")
        self.app_name = app_names[0] if app_names else ""

    def step(self, goal: str) -> Any:
        self.goal = str(goal or self.goal or "").strip()
        if self.round_count >= self._max_steps:
            return make_agent_result(
                done=True,
                data=self._result_data(error="appagent_max_steps_reached"),
            )
        try:
            self._execute_startup_actions()
            self.round_count += 1
            xml_text, raw_image_path = self._capture_round(self.round_count)
            elements = appagent_elements_from_xml(
                xml_text,
                min_dist=float(self.official_runtime.min_dist),
            )
            if self.grid_on:
                image_path = self.output_root / f"round_{self.round_count:03d}_grid.png"
                self.grid_rows, self.grid_columns = self.official_runtime.draw_grid(
                    raw_image_path,
                    image_path,
                )
                ui_document = ""
            else:
                image_path = (
                    self.output_root / f"round_{self.round_count:03d}_labeled.png"
                )
                self.official_runtime.draw_elements(
                    raw_image_path,
                    image_path,
                    elements,
                    record_mode=False,
                )
                ui_document = self._visible_ui_document(elements)
            prompt = self.official_runtime.build_task_prompt(
                goal=self.goal,
                last_action=self.last_action,
                ui_document=ui_document,
                grid=self.grid_on,
            )
            image = Image.open(image_path).convert("RGB")
            response, _, response_metadata = self.llm.predict_mm(prompt, [image])
            parsed = self.official_runtime.parse_response(
                str(response or ""),
                grid=self.grid_on,
            )
            self._append_log(
                {
                    "round": self.round_count,
                    "prompt": prompt,
                    "image": str(image_path),
                    "response": str(response or ""),
                    "response_metadata": response_metadata,
                    "visible_document_uids": [
                        element.uid
                        for element in elements
                        if self.docs_root is not None
                        and (self.docs_root / f"{element.uid}.txt").is_file()
                    ],
                }
            )
            if not parsed or parsed[0] == "ERROR":
                return make_agent_result(
                    done=True,
                    data=self._result_data(error="appagent_response_parse_failed"),
                )
            if parsed[0] == "FINISH":
                return make_agent_result(done=True, data=self._result_data())
            self._execute_parsed_action(parsed, elements)
            if float(self.official_runtime.request_interval) > 0:
                time.sleep(float(self.official_runtime.request_interval))
            return make_agent_result(done=False, data=self._result_data())
        except Exception as exc:
            self._append_log(
                {
                    "round": self.round_count,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                }
            )
            return make_agent_result(done=True, data=self._result_data(error=str(exc)))

    def _execute_startup_actions(self) -> None:
        if self._startup_actions_executed:
            return
        for action in self.startup_actions:
            package_name = str(
                (action.get("params") or {}).get("package_name") or ""
            ).strip()
            app_name = resolve_androidworld_app_name(
                package_name,
                getattr(self.env, "controller", None),
            )
            if not app_name:
                raise ValueError("appagent_runtime_open_app_name_missing")
            native_action = self._new_action(
                action_type="open_app",
                app_name=app_name,
            )
            self.env.execute_action(native_action)
            self.actions_executed += 1
            self._append_log(
                {
                    "event": "startup_action",
                    "action_type": "open_app",
                    "package_name": package_name,
                    "androidworld_app_name": app_name,
                    "execution_backend": "androidworld_native",
                }
            )
        self._startup_actions_executed = True

    def _capture_round(self, round_index: int) -> tuple[str, Path]:
        xml_text, pixels = _native_appagent_observation(self.env)
        xml_path = self.output_root / f"round_{round_index:03d}.xml"
        xml_path.write_text(xml_text, encoding="utf-8")
        image = pixels if isinstance(pixels, Image.Image) else Image.fromarray(pixels)
        image_path = self.output_root / f"round_{round_index:03d}.png"
        image.convert("RGB").save(image_path)
        return xml_text, image_path

    def _visible_ui_document(self, elements: list[AppAgentElement]) -> str:
        if self.docs_root is None:
            return ""
        output = ""
        for index, element in enumerate(elements, 1):
            path = self.docs_root / f"{element.uid}.txt"
            if not path.is_file():
                continue
            content = ast.literal_eval(path.read_text(encoding="utf-8"))
            if not isinstance(content, dict):
                raise ValueError(f"appagent_doc_invalid:{path}")
            output += (
                f"Documentation of UI element labeled with the numeric tag '{index}':\n"
            )
            if content.get("tap"):
                output += f"This UI element is clickable. {content['tap']}\n\n"
            if content.get("text"):
                output += (
                    "This UI element can receive text input. The text input is used "
                    f"for the following purposes: {content['text']}\n\n"
                )
            if content.get("long_press"):
                output += (
                    "This UI element is long clickable. "
                    f"{content['long_press']}\n\n"
                )
            if content.get("v_swipe"):
                output += (
                    "This element can be swiped directly without tapping. You can "
                    "swipe vertically on this UI element. "
                    f"{content['v_swipe']}\n\n"
                )
            if content.get("h_swipe"):
                output += (
                    "This element can be swiped directly without tapping. You can "
                    "swipe horizontally on this UI element. "
                    f"{content['h_swipe']}\n\n"
                )
        return output

    def _execute_parsed_action(
        self,
        parsed: list[Any],
        elements: list[AppAgentElement],
    ) -> None:
        action_name = str(parsed[0])
        action = None
        if action_name == "tap":
            _, tag, last_action = parsed
            x, y = _tag_center(elements, int(tag))
            action = self._new_action(action_type="click", x=x, y=y)
            self.last_action = str(last_action)
        elif action_name == "text":
            _, text_input, last_action = parsed
            action = self._new_action(
                action_type="input_text",
                text=str(text_input),
            )
            self.last_action = str(last_action)
        elif action_name == "long_press":
            _, tag, last_action = parsed
            x, y = _tag_center(elements, int(tag))
            action = self._new_action(action_type="long_press", x=x, y=y)
            self.last_action = str(last_action)
        elif action_name == "swipe":
            _, tag, direction, _distance, last_action = parsed
            _tag_center(elements, int(tag))
            action = self._new_action(
                action_type="swipe",
                direction=str(direction),
            )
            self.last_action = str(last_action)
        elif action_name == "grid":
            self.grid_on = True
            return
        elif action_name in {"tap_grid", "long_press_grid"}:
            _, area, subarea, last_action = parsed
            x, y = _grid_point(
                int(area),
                str(subarea),
                rows=self.grid_rows,
                columns=self.grid_columns,
                width=int(self.env.logical_screen_size[0]),
                height=int(self.env.logical_screen_size[1]),
            )
            action = self._new_action(
                action_type=("click" if action_name == "tap_grid" else "long_press"),
                x=x,
                y=y,
            )
            self.last_action = str(last_action)
        elif action_name == "swipe_grid":
            _, start_area, start_subarea, end_area, end_subarea, last_action = parsed
            start = _grid_point(
                int(start_area),
                str(start_subarea),
                rows=self.grid_rows,
                columns=self.grid_columns,
                width=int(self.env.logical_screen_size[0]),
                height=int(self.env.logical_screen_size[1]),
            )
            end = _grid_point(
                int(end_area),
                str(end_subarea),
                rows=self.grid_rows,
                columns=self.grid_columns,
                width=int(self.env.logical_screen_size[0]),
                height=int(self.env.logical_screen_size[1]),
            )
            direction = _direction_from_points(start, end)
            action = self._new_action(action_type="swipe", direction=direction)
            self.last_action = str(last_action)
        else:
            raise ValueError(f"appagent_action_unsupported:{action_name}")
        self.env.execute_action(action)
        self.actions_executed += 1
        self.grid_on = False

    def _new_action(self, **kwargs: Any) -> Any:
        if self._action_factory is not None:
            return self._action_factory(**kwargs)
        json_action = importlib.import_module("android_world.env.json_action")
        return json_action.JSONAction(**kwargs)

    def _append_log(self, payload: dict[str, Any]) -> None:
        with self._log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")

    def _result_data(self, *, error: str = "") -> dict[str, Any]:
        return {
            "summary": "AppAgent native deployment episode",
            "source": "appagent_official",
            "round_count": self.round_count,
            "actions_executed": self.actions_executed,
            "uses_demo_docs": self.docs_root is not None,
            "docs_root": str(self.docs_root or ""),
            "action_source": str(self.action_source_path or ""),
            "startup_actions_executed": len(self.startup_actions)
            if self._startup_actions_executed
            else 0,
            "error": str(error or "") or None,
        }


class AppAgentTeacherAgent:
    """Capture one AppAgent-native human demonstration from source primitives."""

    def __init__(
        self,
        *,
        env: Any,
        official_runtime: Any,
        teacher_source: str | Path,
        workspace_root: str | Path,
        demo_name: str,
        action_factory: Any | None = None,
    ) -> None:
        self.env = env
        self.official_runtime = official_runtime
        self.teacher_source_path = Path(teacher_source).expanduser().resolve()
        self.teacher_source = load_appagent_teacher_source(self.teacher_source_path)
        self.actions = [dict(item) for item in self.teacher_source["actions"]]
        self.workspace_root = Path(workspace_root).expanduser().resolve()
        self.demo_name = _safe_appagent_name(demo_name)
        self._action_factory = action_factory
        self.name = "appagent_teacher"
        self.task_name = ""
        self.goal = ""
        self.app_name = ""
        self.task_context: dict[str, Any] = {}
        self.demo_root: Path | None = None
        self.teacher_actions_consumed = 0
        self.demo_actions_consumed = 0
        self._stop_written = False
        self._max_steps = len(self.actions) + 1

    def set_max_steps(self, max_steps: int) -> None:
        self._max_steps = max(int(max_steps), len(self.actions) + 1)

    def reset(self, go_home: bool = False) -> None:
        reset = getattr(self.env, "reset", None)
        if callable(reset):
            reset(go_home=go_home)
        self.teacher_actions_consumed = 0
        self.demo_actions_consumed = 0
        self._stop_written = False

    def update_current_task_context(self, task: Any) -> dict[str, Any]:
        app_names = [
            str(value).strip()
            for value in tuple(getattr(task, "app_names", ()) or ())
            if str(value).strip()
        ]
        return {"app_names": app_names}

    def set_current_task(
        self,
        task_name: str,
        goal: str,
        task_context: dict[str, Any] | None = None,
    ) -> None:
        self.task_name = str(task_name or "").strip()
        self.goal = str(goal or "").strip()
        self.task_context = dict(task_context or {})
        app_names = [
            str(value).strip()
            for value in self.task_context.get("app_names") or ()
            if str(value).strip()
        ]
        if len(app_names) > 1:
            raise ValueError("appagent_multi_app_task_unsupported")
        if app_names:
            self.app_name = app_names[0]
            self._prepare_demo_root()

    def step(self, goal: str) -> Any:
        self.goal = str(goal or self.goal or "").strip()
        try:
            self._prepare_demo_root()
            if self.teacher_actions_consumed >= len(self.actions):
                self._capture_demo_state(self.demo_actions_consumed + 1)
                self._append_record("stop")
                self._stop_written = True
                return make_agent_result(done=True, data=self._result_data())
            record = self.actions[self.teacher_actions_consumed]
            action = dict(record.get("action") or {})
            if str(action.get("type") or "").strip() == "open_app":
                package_name = str(
                    (action.get("params") or {}).get("package_name") or ""
                ).strip()
                app_name = resolve_androidworld_app_name(
                    package_name,
                    getattr(self.env, "controller", None),
                )
                if not app_name:
                    raise ValueError("appagent_teacher_open_app_name_missing")
                self.env.execute_action(
                    self._new_action(action_type="open_app", app_name=app_name)
                )
                self.teacher_actions_consumed += 1
                self._append_trace(
                    {
                        "teacher_cursor": self.teacher_actions_consumed,
                        "source_step_index": record.get("source_step_index"),
                        "source_action_index": record.get("source_action_index"),
                        "action_type": "open_app",
                        "package_name": package_name,
                        "androidworld_app_name": app_name,
                        "execution_backend": "androidworld_native",
                        "source_coordinates_used": False,
                    }
                )
                return make_agent_result(done=False, data=self._result_data())
            xml_text, elements = self._capture_demo_state(
                self.demo_actions_consumed + 1
            )
            grounded = ground_appagent_teacher_action(
                xml_text,
                action,
                min_dist=float(self.official_runtime.min_dist),
            )
            self._execute_teacher_action(action, grounded)
            self.teacher_actions_consumed += 1
            self.demo_actions_consumed += 1
            self._append_trace(
                {
                    "teacher_cursor": self.teacher_actions_consumed,
                    "source_step_index": record.get("source_step_index"),
                    "source_action_index": record.get("source_action_index"),
                    "action_type": action.get("type"),
                    "current_tag": grounded.tag,
                    "current_uid": grounded.uid,
                    "match_reason": grounded.match_reason,
                    "current_element_count": len(elements),
                    "source_coordinates_used": False,
                }
            )
            if float(self.official_runtime.request_interval) > 0:
                time.sleep(float(self.official_runtime.request_interval))
            return make_agent_result(done=False, data=self._result_data())
        except Exception as exc:
            self._append_trace(
                {
                    "teacher_cursor": self.teacher_actions_consumed,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                }
            )
            return make_agent_result(done=True, data=self._result_data(error=str(exc)))

    def _prepare_demo_root(self) -> None:
        if self.demo_root is not None:
            return
        if not self.app_name:
            raise ValueError("appagent_single_task_app_required")
        app_dir = self.workspace_root / "apps" / _safe_appagent_name(self.app_name)
        demo_root = app_dir / "demos" / self.demo_name
        if demo_root.exists():
            raise FileExistsError(f"immutable_appagent_demo_exists:{demo_root}")
        for path in (
            demo_root / "raw_screenshots",
            demo_root / "xml",
            demo_root / "labeled_screenshots",
        ):
            path.mkdir(parents=True, exist_ok=False)
        (demo_root / "task_desc.txt").write_text(self.goal, encoding="utf-8")
        (demo_root / "record.txt").touch()
        self.demo_root = demo_root

    def _capture_demo_state(
        self,
        step_index: int,
    ) -> tuple[str, list[AppAgentElement]]:
        if self.demo_root is None:
            raise RuntimeError("appagent_demo_root_not_prepared")
        xml_text, pixels = _native_appagent_observation(self.env)
        base_name = f"{self.demo_name}_{step_index}"
        xml_path = self.demo_root / "xml" / f"{base_name}.xml"
        xml_path.write_text(xml_text, encoding="utf-8")
        image = pixels if isinstance(pixels, Image.Image) else Image.fromarray(pixels)
        raw_path = self.demo_root / "raw_screenshots" / f"{base_name}.png"
        image.convert("RGB").save(raw_path)
        elements = appagent_elements_from_xml(
            xml_text,
            min_dist=float(self.official_runtime.min_dist),
        )
        labeled_path = self.demo_root / "labeled_screenshots" / f"{base_name}.png"
        self.official_runtime.draw_elements(
            raw_path,
            labeled_path,
            elements,
            record_mode=True,
        )
        return xml_text, elements

    def _execute_teacher_action(
        self,
        action: dict[str, Any],
        grounded: GroundedAppAgentAction,
    ) -> None:
        action_type = str(action.get("type") or "")
        params = dict(action.get("params") or {})
        x, y = _bbox_center(grounded.bbox)
        if action_type == "click":
            native_action = self._new_action(action_type="click", x=x, y=y)
            record = f"tap({grounded.tag}):::{grounded.uid}"
        elif action_type == "input_text":
            text = re.sub(r"[\r\n]+", " ", str(params.get("text") or ""))
            native_action = self._new_action(
                action_type="input_text",
                text=text,
                clear_text=True,
            )
            record = f'text({grounded.tag}:sep:"{text}"):::{grounded.uid}'
        elif action_type == "long_press":
            native_action = self._new_action(action_type="long_press", x=x, y=y)
            record = f"long_press({grounded.tag}):::{grounded.uid}"
        elif action_type == "swipe":
            direction = str(params.get("direction") or "")
            native_action = self._new_action(
                action_type="swipe",
                direction=direction,
            )
            record = f"swipe({grounded.tag}:sep:{direction}):::{grounded.uid}"
        else:
            raise ValueError(f"appagent_teacher_action_unsupported:{action_type}")
        self.env.execute_action(native_action)
        self._append_record(record)

    def _new_action(self, **kwargs: Any) -> Any:
        if self._action_factory is not None:
            return self._action_factory(**kwargs)
        json_action = importlib.import_module("android_world.env.json_action")
        return json_action.JSONAction(**kwargs)

    def _append_record(self, line: str) -> None:
        if self.demo_root is None:
            raise RuntimeError("appagent_demo_root_not_prepared")
        if line == "stop" and self._stop_written:
            return
        with (self.demo_root / "record.txt").open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def _append_trace(self, payload: dict[str, Any]) -> None:
        if self.demo_root is None:
            return
        with (self.demo_root / "teacher_trace.jsonl").open(
            "a", encoding="utf-8"
        ) as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _result_data(self, *, error: str = "") -> dict[str, Any]:
        return {
            "summary": "AppAgent source human-demonstration capture",
            "source": "appagent_teacher",
            "actions_executed": self.teacher_actions_consumed,
            "teacher_action_count": len(self.actions),
            "teacher_actions_consumed": self.teacher_actions_consumed,
            "demo_action_count": int(
                self.teacher_source.get("demo_action_count") or 0
            ),
            "demo_actions_consumed": self.demo_actions_consumed,
            "teacher_complete": self.teacher_actions_consumed == len(self.actions),
            "demo_root": str(self.demo_root or ""),
            "error": str(error or "") or None,
        }


def build_appagent_teacher_source(
    source_run_log: str | Path,
    *,
    task_name: str,
    source_seed: int = APPAGENT_SOURCE_SEED,
    provenance_source_run_log: str | Path | None = None,
) -> dict[str, Any]:
    """Build a coordinate-free teacher stream for AppAgent human-demo capture."""

    path = Path(source_run_log).expanduser().resolve()
    if int(source_seed) != APPAGENT_SOURCE_SEED:
        raise ValueError(f"appagent_source_seed_must_be_{APPAGENT_SOURCE_SEED}")
    normalized_task_name = str(task_name or "").strip()
    if not normalized_task_name:
        raise ValueError("appagent_task_name_required")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("appagent_source_run_log_object_required")
    provenance_path = (
        Path(provenance_source_run_log).expanduser().resolve()
        if provenance_source_run_log is not None
        else path
    )
    if not provenance_path.is_file():
        raise FileNotFoundError(
            f"appagent_source_run_log_missing:{provenance_path}"
        )

    source_app_packages: set[str] = set()
    actions: list[dict[str, Any]] = []
    for step_index, step in enumerate(_source_run_log_steps(payload)):
        result = step.get("result")
        if isinstance(result, dict) and result.get("success") is not True:
            continue
        if result is None and step.get("success") is False:
            continue
        for action_index, canonical_action in enumerate(_source_actions(step)):
            action_type = str(canonical_action.get("tool") or "").strip()
            params = dict(canonical_action.get("args") or {})
            if action_type == "open_app":
                package_name = str(
                    params.get("package_name") or params.get("app_name") or ""
                ).strip()
                if package_name:
                    source_app_packages.add(package_name)
            if action_type in _NON_PRIMITIVE_SOURCE_TYPES:
                continue
            if action_type not in APPAGENT_SUPPORTED_SOURCE_TYPES:
                raise ValueError(
                    "appagent_official_action_unsupported:"
                    f"{step_index}:{action_index}:{action_type or 'missing'}"
                )
            if action_type == "input_text" and not str(params.get("text") or ""):
                continue
            if action_type == "swipe":
                direction = _source_swipe_direction(params)
                if direction not in {"up", "down", "left", "right"}:
                    raise ValueError(
                        "appagent_official_swipe_direction_unsupported:"
                        f"{step_index}:{action_index}:{direction or 'missing'}"
                    )
                params["direction"] = direction
            params = _source_semantic_params(step, params)
            actions.append(
                {
                    "source_step_index": step_index,
                    "source_action_index": action_index,
                    "action": {
                        "type": action_type,
                        "params": _adapter_params(action_type, params),
                    },
                }
            )
    if not actions:
        raise ValueError("appagent_official_teacher_actions_required")
    if len(source_app_packages) > 1:
        raise ValueError("appagent_multi_app_demonstration_unsupported")

    return {
        "schema_version": APPAGENT_TEACHER_SOURCE_SCHEMA,
        "task_name": normalized_task_name,
        "source_seed": int(source_seed),
        "source_run_id": str(payload.get("run_id") or ""),
        "source_run_log": str(provenance_path),
        "source_run_log_sha256": hashlib.sha256(
            provenance_path.read_bytes()
        ).hexdigest(),
        "official_appagent_revision": APPAGENT_OFFICIAL_REVISION,
        "actions": actions,
        "action_count": len(actions),
        "demo_action_count": sum(
            str(record["action"].get("type") or "") in APPAGENT_DEMO_ACTION_TYPES
            for record in actions
        ),
        "consumer": "appagent_official_human_demonstration",
        "adapter_scope": "native_androidworld_action_sequence",
        "uses_omniflow_function": False,
        "writes_appagent_docs": False,
        "requires_native_source_episode": True,
        "target_inputs_read": False,
        "coordinate_replay": False,
    }


def load_appagent_teacher_source(path: str | Path) -> dict[str, Any]:
    """Load and validate one immutable AppAgent teacher-source artifact."""

    resolved = Path(path).expanduser().resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("appagent_teacher_source_object_required")
    if payload.get("schema_version") != APPAGENT_TEACHER_SOURCE_SCHEMA:
        raise ValueError("appagent_teacher_source_schema_invalid")
    if payload.get("source_seed") != APPAGENT_SOURCE_SEED:
        raise ValueError(
            f"appagent_teacher_source_seed_must_be_{APPAGENT_SOURCE_SEED}"
        )
    if payload.get("official_appagent_revision") != APPAGENT_OFFICIAL_REVISION:
        raise ValueError("appagent_teacher_source_revision_invalid")
    if payload.get("uses_omniflow_function") is not False:
        raise ValueError("appagent_teacher_source_function_input_forbidden")
    if payload.get("target_inputs_read") is not False:
        raise ValueError("appagent_teacher_source_target_input_forbidden")
    if payload.get("coordinate_replay") is not False:
        raise ValueError("appagent_teacher_source_coordinate_replay_forbidden")
    raw_actions = payload.get("actions")
    if not isinstance(raw_actions, list) or not raw_actions:
        raise ValueError("appagent_teacher_source_actions_required")
    if int(payload.get("action_count") or 0) != len(raw_actions):
        raise ValueError("appagent_teacher_source_action_count_mismatch")
    demo_action_count = 0
    for record in raw_actions:
        action = record.get("action") if isinstance(record, dict) else None
        if not isinstance(action, dict):
            raise ValueError("appagent_teacher_source_action_invalid")
        action_type = str(action.get("type") or "").strip()
        params = action.get("params")
        if action_type not in APPAGENT_SUPPORTED_SOURCE_TYPES:
            raise ValueError(
                f"appagent_teacher_source_action_unsupported:{action_type}"
            )
        if not isinstance(params, dict) or _contains_source_coordinates(params):
            raise ValueError("appagent_teacher_source_coordinates_forbidden")
        if action_type == "open_app" and not str(
            params.get("package_name") or ""
        ).strip():
            raise ValueError("appagent_teacher_source_open_app_package_required")
        if action_type in APPAGENT_DEMO_ACTION_TYPES:
            demo_action_count += 1
    if int(payload.get("demo_action_count") or 0) != demo_action_count:
        raise ValueError("appagent_teacher_source_demo_action_count_mismatch")
    return payload


def seal_appagent_demo_memory(
    *,
    memory_root: str | Path,
    app_name: str,
    demo_name: str,
    teacher_source: str | Path,
    source_result: str | Path,
    document_generation_log: str | Path,
    document_generation_usage: str | Path,
    task_name: str,
    source_episode_wall_sec: float,
    document_generation_wall_sec: float,
    prep_wall_sec: float,
    source_method: str,
    document_generation_model: str,
    source_environment_repair_reason: str = "",
) -> dict[str, Any]:
    """Seal official AppAgent demo docs after one successful source episode."""

    root = Path(memory_root).expanduser().resolve()
    normalized_app = _safe_appagent_name(app_name)
    normalized_demo = _safe_appagent_name(demo_name)
    normalized_task = str(task_name or "").strip()
    normalized_source_method = str(source_method or "").strip()
    normalized_document_model = str(document_generation_model or "").strip()
    if not normalized_task:
        raise ValueError("appagent_memory_task_name_required")
    if not normalized_source_method:
        raise ValueError("appagent_memory_source_method_required")
    if not normalized_document_model:
        raise ValueError("appagent_memory_document_model_required")
    teacher_path = Path(teacher_source).expanduser().resolve()
    teacher = load_appagent_teacher_source(teacher_path)
    if teacher.get("task_name") != normalized_task:
        raise ValueError("appagent_memory_teacher_task_mismatch")
    source_run_log = Path(str(teacher.get("source_run_log") or "")).expanduser()
    if not source_run_log.is_file():
        raise FileNotFoundError(f"appagent_source_run_log_missing:{source_run_log}")
    if _file_sha256(source_run_log) != teacher.get("source_run_log_sha256"):
        raise ValueError("appagent_source_run_log_sha256_mismatch")

    demo_root = root / "apps" / normalized_app / "demos" / normalized_demo
    docs_root = root / "apps" / normalized_app / "demo_docs"
    _validate_demo_artifacts(
        demo_root,
        expected_teacher_action_count=int(teacher.get("action_count") or 0),
        expected_demo_action_count=int(teacher.get("demo_action_count") or 0),
    )
    docs_file_count = _validate_demo_docs(docs_root)
    source_result_path = Path(source_result).expanduser().resolve()
    source_result_row = _official_source_result(
        source_result_path,
        task_name=normalized_task,
    )
    usage_path = Path(document_generation_usage).expanduser().resolve()
    usage_rows = _jsonl_objects(usage_path)
    usage = {
        "model_calls": len(usage_rows),
        "prompt_tokens": sum(int(row.get("prompt_tokens") or 0) for row in usage_rows),
        "completion_tokens": sum(
            int(row.get("completion_tokens") or 0) for row in usage_rows
        ),
        "total_tokens": sum(int(row.get("total_tokens") or 0) for row in usage_rows),
        "models": sorted(
            {
                str(row.get("model") or "").strip()
                for row in usage_rows
                if str(row.get("model") or "").strip()
            }
        ),
    }
    if usage["total_tokens"] <= 0:
        usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
    if set(usage["models"]) != {normalized_document_model}:
        raise ValueError(
            "appagent_document_generation_model_mismatch:"
            f"expected={normalized_document_model}:actual={usage['models']}"
        )
    usage["wall_sec"] = round(float(document_generation_wall_sec), 6)
    source_model_calls = int(source_result_row.get("model_calls") or 0)
    source_prompt_tokens = int(source_result_row.get("prompt_tokens") or 0)
    source_completion_tokens = int(source_result_row.get("completion_tokens") or 0)
    source_total_tokens = int(source_result_row.get("total_tokens") or 0)
    if source_total_tokens <= 0:
        source_total_tokens = source_prompt_tokens + source_completion_tokens
    source_episode_metrics = {
        "duration_sec": round(
            float(source_result_row.get("duration_ms") or 0.0) / 1000.0,
            6,
        ),
        "wall_sec": round(float(source_episode_wall_sec), 6),
        "model_calls": source_model_calls,
        "prompt_tokens": source_prompt_tokens,
        "completion_tokens": source_completion_tokens,
        "total_tokens": source_total_tokens,
        "token_usage_status": str(
            source_result_row.get("token_usage_status")
            or ("tracked" if source_model_calls > 0 else "not_applicable")
        ),
    }
    document_log_path = Path(document_generation_log).expanduser().resolve()
    if not document_log_path.is_file():
        raise FileNotFoundError(
            f"appagent_document_generation_log_missing:{document_log_path}"
        )
    if not usage_path.is_file():
        raise FileNotFoundError(f"appagent_document_usage_missing:{usage_path}")
    manifest_path = root / APPAGENT_DEMO_MANIFEST
    manifest = {
        "schema_version": APPAGENT_DEMO_MEMORY_SCHEMA,
        "official_appagent_revision": APPAGENT_OFFICIAL_REVISION,
        "task_name": normalized_task,
        "app_name": normalized_app,
        "demo_name": normalized_demo,
        "source_seed": APPAGENT_SOURCE_SEED,
        "source_method": normalized_source_method,
        "source_environment_repair_reason": str(
            source_environment_repair_reason or ""
        ).strip(),
        "source_run_id": str(teacher.get("source_run_id") or ""),
        "source_run_log": str(source_run_log.resolve()),
        "source_run_log_sha256": _file_sha256(source_run_log),
        "teacher_source": str(teacher_path),
        "teacher_source_sha256": _file_sha256(teacher_path),
        "teacher_action_count": int(teacher.get("action_count") or 0),
        "teacher_actions_consumed": int(teacher.get("action_count") or 0),
        "demo_action_count": int(teacher.get("demo_action_count") or 0),
        "teacher_complete": True,
        "demo_root": str(demo_root),
        "demo_sha256": _tree_sha256(demo_root),
        "source_result": str(source_result_path),
        "source_result_sha256": _file_sha256(source_result_path),
        "official_source_success": True,
        "official_source_reward": (
            source_result_row.get("androidworld_validator_result") or {}
        ).get("reward"),
        "source_episode_metrics": source_episode_metrics,
        "document_generation_log": str(document_log_path),
        "document_generation_log_sha256": _file_sha256(document_log_path),
        "document_generation_usage_path": str(usage_path),
        "document_generation_usage_sha256": _file_sha256(usage_path),
        "doc_generation_usage": usage,
        "document_generation_model": normalized_document_model,
        "prep_wall_sec": round(float(prep_wall_sec), 6),
        "demo_docs_root": str(docs_root),
        "demo_docs_sha256": _tree_sha256(docs_root),
        "demo_docs_file_count": docs_file_count,
        "native_format": "appagent.demo_docs",
        "uses_omniflow_function": False,
        "target_inputs_read": False,
        "target_observations_read": False,
        "validator_state_read_for_memory": False,
    }
    root.mkdir(parents=True, exist_ok=True)
    try:
        with manifest_path.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    except FileExistsError as exc:
        raise FileExistsError(
            f"immutable_appagent_memory_manifest_exists:{manifest_path}"
        ) from exc
    return manifest


def validate_appagent_source_demo(
    *,
    memory_root: str | Path,
    app_name: str,
    demo_name: str,
    source_result: str | Path,
    task_name: str,
    expected_teacher_action_count: int,
    expected_demo_action_count: int,
) -> dict[str, Any]:
    """Require one complete official-success source demo before doc generation."""

    root = Path(memory_root).expanduser().resolve()
    demo_root = (
        root
        / "apps"
        / _safe_appagent_name(app_name)
        / "demos"
        / _safe_appagent_name(demo_name)
    )
    source_result_row = _official_source_result(
        Path(source_result).expanduser().resolve(),
        task_name=str(task_name or "").strip(),
    )
    _validate_demo_artifacts(
        demo_root,
        expected_teacher_action_count=int(expected_teacher_action_count),
        expected_demo_action_count=int(expected_demo_action_count),
    )
    return source_result_row


def validate_appagent_demo_memory(
    memory_root: str | Path,
    *,
    task_name: str,
    source_run_log: str | Path | None = None,
) -> dict[str, Any]:
    """Validate a sealed AppAgent demo memory before warm evaluation."""

    root = Path(memory_root).expanduser().resolve()
    manifest_path = root / APPAGENT_DEMO_MANIFEST
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != (
        APPAGENT_DEMO_MEMORY_SCHEMA
    ):
        raise ValueError("appagent_demo_memory_schema_invalid")
    if payload.get("official_appagent_revision") != APPAGENT_OFFICIAL_REVISION:
        raise ValueError("appagent_demo_memory_revision_invalid")
    if payload.get("task_name") != str(task_name or "").strip():
        raise ValueError("appagent_demo_memory_task_mismatch")
    if payload.get("source_seed") != APPAGENT_SOURCE_SEED:
        raise ValueError("appagent_demo_memory_source_seed_invalid")
    if payload.get("official_source_success") is not True:
        raise ValueError("appagent_demo_memory_source_success_required")
    source_metrics = payload.get("source_episode_metrics")
    if not isinstance(source_metrics, dict):
        raise ValueError("appagent_demo_memory_source_metrics_missing")
    if float(source_metrics.get("duration_sec") or 0.0) <= 0:
        raise ValueError("appagent_demo_memory_source_duration_missing")
    if float(source_metrics.get("wall_sec") or 0.0) <= 0:
        raise ValueError("appagent_demo_memory_source_wall_time_missing")
    if int(source_metrics.get("total_tokens") or 0) != int(
        source_metrics.get("prompt_tokens") or 0
    ) + int(source_metrics.get("completion_tokens") or 0):
        raise ValueError("appagent_demo_memory_source_tokens_inconsistent")
    doc_usage = payload.get("doc_generation_usage")
    if not isinstance(doc_usage, dict):
        raise ValueError("appagent_demo_memory_doc_usage_missing")
    if int(doc_usage.get("model_calls") or 0) <= 0:
        raise ValueError("appagent_demo_memory_doc_model_calls_missing")
    if int(doc_usage.get("total_tokens") or 0) != int(
        doc_usage.get("prompt_tokens") or 0
    ) + int(doc_usage.get("completion_tokens") or 0):
        raise ValueError("appagent_demo_memory_doc_tokens_inconsistent")
    if float(doc_usage.get("wall_sec") or 0.0) <= 0:
        raise ValueError("appagent_demo_memory_doc_wall_time_missing")
    if float(payload.get("prep_wall_sec") or 0.0) <= 0:
        raise ValueError("appagent_demo_memory_prep_wall_time_missing")
    if payload.get("teacher_complete") is not True or int(
        payload.get("teacher_actions_consumed") or 0
    ) != int(payload.get("teacher_action_count") or -1):
        raise ValueError("appagent_demo_memory_teacher_incomplete")
    teacher_action_count = int(payload.get("teacher_action_count") or 0)
    demo_action_count = int(payload.get("demo_action_count") or 0)
    if (
        teacher_action_count <= 0
        or demo_action_count <= 0
        or demo_action_count > teacher_action_count
    ):
        raise ValueError("appagent_demo_memory_action_count_invalid")
    for key in (
        "target_inputs_read",
        "target_observations_read",
        "validator_state_read_for_memory",
    ):
        if payload.get(key) is not False:
            raise ValueError(f"appagent_demo_memory_leakage:{key}")
    _require_hash(payload, "teacher_source", "teacher_source_sha256")
    teacher_source = load_appagent_teacher_source(payload["teacher_source"])
    if teacher_source.get("task_name") != payload.get("task_name"):
        raise ValueError("appagent_demo_memory_teacher_task_mismatch")
    if int(teacher_source.get("action_count") or 0) != teacher_action_count:
        raise ValueError("appagent_demo_memory_teacher_action_count_mismatch")
    if int(teacher_source.get("demo_action_count") or 0) != demo_action_count:
        raise ValueError("appagent_demo_memory_demo_action_count_mismatch")
    if teacher_source.get("source_run_log_sha256") != payload.get(
        "source_run_log_sha256"
    ):
        raise ValueError("appagent_demo_memory_teacher_source_mismatch")
    _require_hash(payload, "source_result", "source_result_sha256")
    _require_hash(
        payload,
        "document_generation_log",
        "document_generation_log_sha256",
    )
    _require_hash(
        payload,
        "document_generation_usage_path",
        "document_generation_usage_sha256",
    )
    demo_root = Path(str(payload.get("demo_root") or ""))
    docs_root = Path(str(payload.get("demo_docs_root") or ""))
    if _tree_sha256(demo_root) != payload.get("demo_sha256"):
        raise ValueError("appagent_demo_memory_demo_sha256_mismatch")
    if _tree_sha256(docs_root) != payload.get("demo_docs_sha256"):
        raise ValueError("appagent_demo_memory_docs_sha256_mismatch")
    _validate_demo_artifacts(
        demo_root,
        expected_teacher_action_count=teacher_action_count,
        expected_demo_action_count=demo_action_count,
    )
    if _validate_demo_docs(docs_root) != int(payload.get("demo_docs_file_count") or 0):
        raise ValueError("appagent_demo_memory_docs_count_mismatch")
    expected_source = (
        Path(source_run_log).expanduser().resolve()
        if source_run_log is not None
        else Path(str(payload.get("source_run_log") or "")).expanduser().resolve()
    )
    if _file_sha256(expected_source) != payload.get("source_run_log_sha256"):
        raise ValueError("appagent_demo_memory_source_run_log_mismatch")
    return dict(payload)


def appagent_elements_from_xml(
    xml_text: str,
    *,
    min_dist: float,
) -> list[AppAgentElement]:
    """Return AppAgent's official clickable-then-focusable tag ordering."""

    clickable = _traverse_appagent_elements(
        xml_text,
        attribute="clickable",
        min_dist=min_dist,
    )
    focusable = _traverse_appagent_elements(
        xml_text,
        attribute="focusable",
        min_dist=min_dist,
    )
    elements = list(clickable)
    for element in focusable:
        center = _bbox_center(element.bbox)
        if any(
            _point_distance(center, _bbox_center(existing.bbox)) <= min_dist
            for existing in clickable
        ):
            continue
        elements.append(element)
    return elements


def ground_appagent_teacher_action(
    xml_text: str,
    action: dict[str, Any],
    *,
    min_dist: float,
) -> GroundedAppAgentAction:
    """Ground one coordinate-free teacher primitive on the current source XML."""

    action_type = str(action.get("type") or "").strip()
    params = action.get("params")
    if action_type not in APPAGENT_SUPPORTED_SOURCE_TYPES or not isinstance(
        params, dict
    ):
        raise ValueError("appagent_teacher_action_invalid")
    elements = appagent_elements_from_xml(xml_text, min_dist=min_dist)
    if not elements:
        raise ValueError("appagent_current_screen_has_no_interactive_elements")
    root = ET.fromstring(xml_text)
    matching_nodes = _identity_nodes(root, params)
    match_reason = "exact_visible_identity"
    if not matching_nodes and action_type == "input_text":
        matching_nodes = [
            node
            for node in root.iter()
            if str(node.attrib.get("editable") or "").lower() == "true"
        ]
        match_reason = "unique_current_editable"
    if not matching_nodes and action_type == "swipe":
        matching_nodes = [
            node
            for node in root.iter()
            if str(node.attrib.get("scrollable") or "").lower() == "true"
        ]
        match_reason = "unique_current_scrollable"
    if not matching_nodes:
        raise ValueError("appagent_teacher_target_not_found")

    grounded_indexes: set[int] = set()
    for node in matching_nodes:
        node_bbox = _element_bbox(node)
        if node_bbox is None:
            continue
        center = _bbox_center(node_bbox)
        containing = [
            (index, _bbox_area(element.bbox))
            for index, element in enumerate(elements)
            if _bbox_contains(element.bbox, center)
        ]
        if containing:
            minimum_area = min(area for _, area in containing)
            grounded_indexes.update(
                index for index, area in containing if area == minimum_area
            )
    if len(grounded_indexes) != 1:
        raise ValueError(
            "appagent_teacher_target_ambiguous:"
            + str(len(grounded_indexes))
        )
    index = next(iter(grounded_indexes))
    element = elements[index]
    return GroundedAppAgentAction(
        tag=index + 1,
        uid=element.uid,
        bbox=element.bbox,
        match_reason=match_reason,
    )


def _source_run_log_steps(payload: dict[str, Any]) -> list[dict[str, Any]]:
    steps = payload.get("steps")
    if isinstance(steps, list):
        return [dict(step) for step in steps if isinstance(step, dict)]
    wrapped = payload.get("payload")
    if isinstance(wrapped, dict) and isinstance(wrapped.get("steps"), list):
        return [
            dict(step) for step in wrapped["steps"] if isinstance(step, dict)
        ]
    return []


def _traverse_appagent_elements(
    xml_text: str,
    *,
    attribute: str,
    min_dist: float,
) -> list[AppAgentElement]:
    elements: list[AppAgentElement] = []
    path: list[ET.Element] = []
    for event, element in ET.iterparse(
        io.StringIO(xml_text),
        events=("start", "end"),
    ):
        if event == "start":
            path.append(element)
            if str(element.attrib.get(attribute) or "").lower() != "true":
                continue
            bbox = _element_bbox(element)
            if bbox is None:
                continue
            uid = _appagent_element_id(element)
            if len(path) > 1:
                uid = _appagent_element_id(path[-2]) + "_" + uid
            uid += f"_{element.attrib.get('index', '')}"
            center = _bbox_center(bbox)
            if any(
                _point_distance(center, _bbox_center(existing.bbox)) <= min_dist
                for existing in elements
            ):
                continue
            elements.append(
                AppAgentElement(
                    uid=uid,
                    bbox=bbox,
                    attrib=attribute,
                )
            )
        else:
            path.pop()
    return elements


def _appagent_element_id(element: ET.Element) -> str:
    bbox = _element_bbox(element)
    if bbox is None:
        raise ValueError("appagent_element_bounds_missing")
    width = bbox[1][0] - bbox[0][0]
    height = bbox[1][1] - bbox[0][1]
    resource_id = str(element.attrib.get("resource-id") or "")
    if resource_id:
        element_id = resource_id.replace(":", ".").replace("/", "_")
    else:
        element_id = f"{element.attrib.get('class', '')}_{width}_{height}"
    content_desc = str(element.attrib.get("content-desc") or "")
    if content_desc and len(content_desc) < 20:
        cleaned = content_desc.replace("/", "_").replace(" ", "").replace(":", "_")
        element_id += f"_{cleaned}"
    return element_id


def _identity_nodes(root: ET.Element, params: dict[str, Any]) -> list[ET.Element]:
    target_evidence = (
        dict(params.get("target_evidence") or {})
        if isinstance(params.get("target_evidence"), dict)
        else {}
    )
    source_element = {}
    source_context = params.get("source_context")
    if isinstance(source_context, dict) and isinstance(
        source_context.get("element"), dict
    ):
        source_element = dict(source_context["element"])
    resource_ids = {
        _normalized_identity(value)
        for value in (
            target_evidence.get("resource_id"),
            target_evidence.get("resource-id"),
            source_element.get("resource_id"),
            source_element.get("resource-id"),
        )
        if _normalized_identity(value)
    }
    if resource_ids:
        resource_matches = [
            node
            for node in root.iter()
            if _normalized_identity(node.attrib.get("resource-id")) in resource_ids
        ]
        if len(resource_matches) == 1:
            return resource_matches
    labels = {
        _normalized_identity(value)
        for value in (
            params.get("target_description"),
            target_evidence.get("label"),
            source_element.get("text"),
            source_element.get("label"),
            source_element.get("description"),
            source_element.get("content_desc"),
            source_element.get("content-desc"),
        )
        if _normalized_identity(value)
    }
    exact = [
        node
        for node in root.iter()
        if labels
        and any(
            _normalized_identity(node.attrib.get(key)) in labels
            for key in ("text", "content-desc")
        )
    ]
    if exact:
        return exact
    return []


def _normalized_identity(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()


def _element_bbox(
    element: ET.Element,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    match = re.fullmatch(
        r"\[(-?\d+),(-?\d+)\]\[(-?\d+),(-?\d+)\]",
        str(element.attrib.get("bounds") or "").strip(),
    )
    if match is None:
        return None
    x1, y1, x2, y2 = (int(value) for value in match.groups())
    if x2 <= x1 or y2 <= y1:
        return None
    return ((x1, y1), (x2, y2))


def _bbox_center(
    bbox: tuple[tuple[int, int], tuple[int, int]],
) -> tuple[int, int]:
    return (
        (bbox[0][0] + bbox[1][0]) // 2,
        (bbox[0][1] + bbox[1][1]) // 2,
    )


def _bbox_area(bbox: tuple[tuple[int, int], tuple[int, int]]) -> int:
    return (bbox[1][0] - bbox[0][0]) * (bbox[1][1] - bbox[0][1])


def _bbox_contains(
    bbox: tuple[tuple[int, int], tuple[int, int]],
    point: tuple[int, int],
) -> bool:
    return (
        bbox[0][0] <= point[0] <= bbox[1][0]
        and bbox[0][1] <= point[1] <= bbox[1][1]
    )


def _point_distance(left: tuple[int, int], right: tuple[int, int]) -> float:
    return ((left[0] - right[0]) ** 2 + (left[1] - right[1]) ** 2) ** 0.5


def _source_actions(step: dict[str, Any]) -> list[dict[str, Any]]:
    canonical_action = step.get("action")
    if isinstance(canonical_action, dict):
        tool = str(
            canonical_action.get("tool") or canonical_action.get("type") or ""
        ).strip()
        args = canonical_action.get("args")
        if not isinstance(args, dict):
            args = canonical_action.get("params")
        return [{"tool": tool, "args": dict(args or {})}]
    for key in ("actions", "executed_actions"):
        raw_actions = step.get(key)
        if not isinstance(raw_actions, list):
            continue
        actions = []
        for raw_action in raw_actions:
            if not isinstance(raw_action, dict) or raw_action.get("success") is False:
                continue
            tool = str(raw_action.get("tool") or raw_action.get("type") or "").strip()
            args = raw_action.get("args")
            if not isinstance(args, dict):
                args = raw_action.get("params")
            actions.append({"tool": tool, "args": dict(args or {})})
        if actions:
            return actions
    return extract_canonical_step_actions(step)


def _adapter_params(action_type: str, params: dict[str, Any]) -> dict[str, Any]:
    adapted: dict[str, Any] = {}
    target_description = str(params.get("target_description") or "").strip()
    if target_description:
        adapted["target_description"] = target_description
    target_evidence = params.get("target_evidence")
    if isinstance(target_evidence, dict):
        evidence = {
            key: target_evidence[key]
            for key in ("label", "resource_id", "resource-id")
            if str(target_evidence.get(key) or "").strip()
        }
        if evidence:
            adapted["target_evidence"] = evidence
    source_context = params.get("source_context")
    if isinstance(source_context, dict) and isinstance(
        source_context.get("element"), dict
    ):
        element = source_context["element"]
        identity = {
            key: element[key]
            for key in (
                "text",
                "label",
                "description",
                "content_desc",
                "content-desc",
                "resource_id",
                "resource-id",
            )
            if str(element.get(key) or "").strip()
        }
        if identity:
            adapted["source_context"] = {"element": identity}
    if action_type == "input_text":
        adapted["text"] = str(params.get("text") or "")
    elif action_type == "open_app":
        package_name = str(
            params.get("package_name") or params.get("app_name") or ""
        ).strip()
        if package_name:
            adapted["package_name"] = package_name
    elif action_type == "swipe":
        adapted["direction"] = str(params.get("direction") or "")
    return _without_source_coordinates(adapted)


def _source_semantic_params(
    step: dict[str, Any],
    params: dict[str, Any],
) -> dict[str, Any]:
    enriched = dict(params)
    if _adapter_params("click", enriched):
        return enriched
    try:
        x = float(enriched.get("x"))
        y = float(enriched.get("y"))
    except (TypeError, ValueError):
        return enriched
    observation = step.get("observation_before_act")
    if not isinstance(observation, dict):
        observation = step.get("observation")
    xml_text = str(
        observation.get("xml") if isinstance(observation, dict) else ""
    ).strip()
    if not xml_text:
        return enriched
    identity = _source_identity_at_point(xml_text, x=x, y=y)
    if not identity:
        return enriched
    target_description = str(
        identity.get("text") or identity.get("content_desc") or ""
    ).strip()
    if target_description:
        enriched["target_description"] = target_description
    enriched["source_context"] = {"element": identity}
    return enriched


def _source_identity_at_point(
    xml_text: str,
    *,
    x: float,
    y: float,
) -> dict[str, str]:
    root = ET.fromstring(xml_text)
    containing = [
        node
        for node in root.iter()
        if (bbox := _element_bbox(node)) is not None
        and _bbox_contains(bbox, (int(round(x)), int(round(y))))
    ]
    descriptive = [node for node in containing if _node_description(node)]
    if descriptive:
        return _source_node_identity(min(descriptive, key=_node_area))
    interactive = [
        node
        for node in containing
        if str(node.attrib.get("clickable") or "").lower() == "true"
        or str(node.attrib.get("long-clickable") or "").lower() == "true"
    ]
    for node in sorted(interactive, key=_node_area):
        descendants = [
            descendant
            for descendant in node.iter()
            if descendant is not node and _node_description(descendant)
        ]
        if descendants:
            descendants.sort(
                key=lambda descendant: (
                    0
                    if str(descendant.attrib.get("resource-id") or "").endswith(
                        ("/title", "/switch_text")
                    )
                    else 1,
                    _node_area(descendant),
                )
            )
            return _source_node_identity(descendants[0])
    identified = [
        node
        for node in containing
        if str(node.attrib.get("resource-id") or "").strip()
    ]
    return _source_node_identity(min(identified, key=_node_area)) if identified else {}


def _node_description(node: ET.Element) -> str:
    return str(
        node.attrib.get("text") or node.attrib.get("content-desc") or ""
    ).strip()


def _node_area(node: ET.Element) -> int:
    bbox = _element_bbox(node)
    return _bbox_area(bbox) if bbox is not None else 2**63 - 1


def _source_node_identity(node: ET.Element) -> dict[str, str]:
    aliases = {
        "text": "text",
        "content_desc": "content-desc",
        "resource_id": "resource-id",
    }
    return {
        output: value
        for output, source in aliases.items()
        if (value := str(node.attrib.get(source) or "").strip())
    }


def _source_swipe_direction(params: dict[str, Any]) -> str:
    direction = str(params.get("direction") or "").strip().lower()
    if direction:
        return direction
    try:
        x1 = float(params.get("x1"))
        y1 = float(params.get("y1"))
        x2 = float(params.get("x2"))
        y2 = float(params.get("y2"))
    except (TypeError, ValueError):
        return ""
    if abs(x2 - x1) > abs(y2 - y1):
        return "right" if x2 > x1 else "left"
    return "down" if y2 > y1 else "up"


def _without_source_coordinates(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _without_source_coordinates(item)
            for key, item in value.items()
            if str(key) not in _SOURCE_COORDINATE_FIELDS
        }
    if isinstance(value, list):
        return [_without_source_coordinates(item) for item in value]
    return value


def _contains_source_coordinates(value: Any) -> bool:
    if isinstance(value, dict):
        return any(
            str(key) in _SOURCE_COORDINATE_FIELDS
            or _contains_source_coordinates(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_source_coordinates(item) for item in value)
    return False


def _read_appagent_config(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("AppAgent runtime requires PyYAML") from exc
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"appagent_config_object_required:{path}")
    return dict(payload)


def _file_sha256(path: str | Path) -> str:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"appagent_provenance_file_missing:{resolved}")
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_sha256(root: str | Path) -> str:
    resolved = Path(root).expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"appagent_provenance_tree_missing:{resolved}")
    digest = hashlib.sha256()
    files = sorted(path for path in resolved.rglob("*") if path.is_file())
    for path in files:
        digest.update(path.relative_to(resolved).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _jsonl_objects(path: str | Path) -> list[dict[str, Any]]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"appagent_jsonl_missing:{resolved}")
    rows: list[dict[str, Any]] = []
    for line in resolved.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise ValueError(f"appagent_jsonl_object_required:{resolved}")
        rows.append(payload)
    return rows


def _official_source_result(path: Path, *, task_name: str) -> dict[str, Any]:
    matches = [
        row
        for row in _jsonl_objects(path)
        if str(row.get("task_name") or "").strip() == task_name
    ]
    if len(matches) != 1:
        raise ValueError(f"appagent_source_result_count_invalid:{len(matches)}")
    row = matches[0]
    validator = row.get("androidworld_validator_result")
    if row.get("official_validator_used") is not True or not isinstance(
        validator, dict
    ) or validator.get("success") is not True:
        raise ValueError("appagent_official_source_success_required")
    if int(row.get("task_random_seed") or 0) != APPAGENT_SOURCE_SEED:
        raise ValueError("appagent_official_source_seed_invalid")
    return row


def _validate_demo_artifacts(
    demo_root: Path,
    *,
    expected_teacher_action_count: int,
    expected_demo_action_count: int,
) -> None:
    if expected_teacher_action_count <= 0 or expected_demo_action_count <= 0:
        raise ValueError("appagent_demo_action_count_invalid")
    required_dirs = (
        demo_root / "raw_screenshots",
        demo_root / "xml",
        demo_root / "labeled_screenshots",
    )
    for directory in required_dirs:
        if not directory.is_dir():
            raise FileNotFoundError(f"appagent_demo_artifact_dir_missing:{directory}")
        file_count = len([path for path in directory.iterdir() if path.is_file()])
        if file_count != expected_demo_action_count + 1:
            raise ValueError(
                "appagent_demo_artifact_count_mismatch:"
                f"{directory.name}:{file_count}:{expected_demo_action_count + 1}"
            )
    record_path = demo_root / "record.txt"
    lines = [
        line.strip()
        for line in record_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(lines) != expected_demo_action_count + 1 or lines[-1] != "stop":
        raise ValueError("appagent_demo_record_incomplete")
    trace_rows = _jsonl_objects(demo_root / "teacher_trace.jsonl")
    if len(trace_rows) != expected_teacher_action_count:
        raise ValueError("appagent_demo_teacher_trace_count_mismatch")
    if int(trace_rows[-1].get("teacher_cursor") or 0) != (
        expected_teacher_action_count
    ):
        raise ValueError("appagent_demo_teacher_trace_incomplete")
    if any(row.get("source_coordinates_used") is not False for row in trace_rows):
        raise ValueError("appagent_demo_source_coordinate_replay_detected")


def _validate_demo_docs(docs_root: Path) -> int:
    if not docs_root.is_dir():
        raise FileNotFoundError(f"appagent_demo_docs_missing:{docs_root}")
    paths = sorted(path for path in docs_root.glob("*.txt") if path.is_file())
    if not paths:
        raise ValueError("appagent_demo_docs_empty")
    expected_keys = {"tap", "text", "v_swipe", "h_swipe", "long_press"}
    for path in paths:
        payload = ast.literal_eval(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or set(payload) != expected_keys:
            raise ValueError(f"appagent_demo_doc_schema_invalid:{path}")
        if not any(str(value or "").strip() for value in payload.values()):
            raise ValueError(f"appagent_demo_doc_empty:{path}")
    return len(paths)


def _require_hash(payload: dict[str, Any], path_key: str, hash_key: str) -> None:
    path = Path(str(payload.get(path_key) or ""))
    if _file_sha256(path) != payload.get(hash_key):
        raise ValueError(f"appagent_demo_memory_hash_mismatch:{path_key}")


def _tag_center(
    elements: list[AppAgentElement],
    tag: int,
) -> tuple[int, int]:
    if tag < 1 or tag > len(elements):
        raise ValueError(f"appagent_tag_out_of_range:{tag}:{len(elements)}")
    return _bbox_center(elements[tag - 1].bbox)


def _grid_point(
    area: int,
    subarea: str,
    *,
    rows: int,
    columns: int,
    width: int,
    height: int,
) -> tuple[int, int]:
    if rows <= 0 or columns <= 0 or area < 1 or area > rows * columns:
        raise ValueError("appagent_grid_area_invalid")
    row, column = divmod(area - 1, columns)
    unit_width = width // columns
    unit_height = height // rows
    origin_x = column * unit_width
    origin_y = row * unit_height
    offsets = {
        "top-left": (1, 1),
        "top": (2, 1),
        "top-right": (3, 1),
        "left": (1, 2),
        "center": (2, 2),
        "right": (3, 2),
        "bottom-left": (1, 3),
        "bottom": (2, 3),
        "bottom-right": (3, 3),
    }
    horizontal, vertical = offsets.get(str(subarea), offsets["center"])
    return (
        origin_x + unit_width * horizontal // 4,
        origin_y + unit_height * vertical // 4,
    )


def _direction_from_points(
    start: tuple[int, int],
    end: tuple[int, int],
) -> str:
    delta_x = end[0] - start[0]
    delta_y = end[1] - start[1]
    if abs(delta_x) > abs(delta_y):
        return "right" if delta_x > 0 else "left"
    return "down" if delta_y > 0 else "up"


def _safe_appagent_name(value: Any) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "")).strip("._-")
    if not normalized:
        raise ValueError("appagent_name_required")
    return normalized


__all__ = [
    "APPAGENT_DEMO_MANIFEST",
    "APPAGENT_DEMO_MEMORY_SCHEMA",
    "APPAGENT_OFFICIAL_REVISION",
    "APPAGENT_SOURCE_SEED",
    "APPAGENT_TEACHER_SOURCE_SCHEMA",
    "AppAgentAndroidWorldAgent",
    "AppAgentElement",
    "AppAgentTeacherAgent",
    "GroundedAppAgentAction",
    "OfficialAppAgentRuntime",
    "appagent_elements_from_xml",
    "build_appagent_teacher_source",
    "ground_appagent_teacher_action",
    "load_appagent_teacher_source",
    "seal_appagent_demo_memory",
    "validate_appagent_demo_memory",
]
