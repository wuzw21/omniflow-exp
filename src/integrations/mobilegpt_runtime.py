from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any
import xml.etree.ElementTree as ET

_ACTION_NAME_ALIASES = {
    "enter_text": "input",
    "go_back": "back",
    "input_text": "input",
    "long_click": "long-click",
    "long_press": "long-click",
    "navigate_back": "back",
    "press_back": "back",
    "repeat_click": "repeat-click",
    "swipe": "scroll",
    "tap": "click",
    "type_text": "input",
}
_MOBILEGPT_SUPPORTED_ACTIONS = {
    "ask",
    "back",
    "click",
    "go-back",
    "input",
    "long-click",
    "repeat-click",
    "scroll",
    "speak",
    "__omniflow_launch_package",
}
_LATEST_MOBILEGPT_XML = ""


def install_mobilegpt_package_app_resolution(app_agent_class: type) -> None:
    if bool(
        getattr(app_agent_class, "_omniflow_package_app_resolution_installed", False)
    ):
        return
    original_get_package_name = app_agent_class.get_package_name

    def _get_package_name(self, app):
        package_name = str(original_get_package_name(self, app) or "").strip()
        if package_name:
            return package_name
        candidate = str(app or "").strip()
        if "." in candidate and not any(character.isspace() for character in candidate):
            return candidate
        return ""

    app_agent_class.get_package_name = _get_package_name
    app_agent_class._omniflow_package_app_resolution_installed = True


def install_mobilegpt_answer_event(mobilegpt_class: type) -> None:
    if bool(getattr(mobilegpt_class, "_omniflow_answer_event_installed", False)):
        return
    method_name = f"_{mobilegpt_class.__name__}__handle_primitive_subtask"
    original_handle = getattr(mobilegpt_class, method_name)

    def _handle(self, next_subtask):
        if isinstance(next_subtask, dict) and next_subtask.get("name") == "speak":
            parameters = next_subtask.get("parameters")
            message = (
                str(parameters.get("message") or "").strip()
                if isinstance(parameters, dict)
                else ""
            )
            if message:
                _write_stats_event(
                    {
                        "event": "agent_answer",
                        "text": message,
                        "instruction": str(getattr(self, "instruction", "") or ""),
                    }
                )
        return original_handle(self, next_subtask)

    setattr(mobilegpt_class, method_name, _handle)
    mobilegpt_class._omniflow_answer_event_installed = True


def install_mobilegpt_action_error_recovery(mobilegpt_class: type) -> None:
    if bool(
        getattr(mobilegpt_class, "_omniflow_action_error_recovery_installed", False)
    ):
        return

    finish_name = f"_{mobilegpt_class.__name__}__finish_task"

    def _handle_action_error(self, error_message):
        error_text = str(error_message or "unknown action error").strip()
        if getattr(self, "memory", None) is None:
            _write_stats_event(
                {
                    "event": "mobilegpt_action_error_ignored_uninitialized",
                    "error": error_text,
                }
            )
            return None
        error_count = int(getattr(self, "_omniflow_action_error_count", 0)) + 1
        self._omniflow_action_error_count = error_count
        error_limit = max(
            1,
            int(os.getenv("MOBILEGPT_MAX_ACTION_ERRORS") or "4"),
        )
        current_subtask_data = getattr(self, "current_subtask_data", None)
        if isinstance(current_subtask_data, dict):
            actions = current_subtask_data.get("actions")
            if isinstance(actions, list) and actions:
                actions.pop()
        derive_agent = getattr(self, "derive_agent", None)
        response_history = getattr(derive_agent, "response_history", None)
        if isinstance(response_history, list) and response_history:
            response_history.pop()
        action_history = getattr(derive_agent, "action_history", None)
        if isinstance(action_history, list):
            if action_history:
                action_history.pop()
            action_history.append(
                "The previous action was not executed successfully. "
                f"Device error: {error_text}. Choose a different low-level "
                "action for the same subtask and do not repeat the failed action."
            )
        failed_subtask = getattr(self, "current_subtask", None)
        status_type = type(getattr(self, "subtask_status", None))
        learn_status = getattr(status_type, "LEARN", None)
        if learn_status is not None:
            self.subtask_status = learn_status
        _write_stats_event(
            {
                "event": "mobilegpt_action_error_recovery",
                "error": error_text,
                "failed_subtask": failed_subtask,
                "error_count": error_count,
                "error_limit": error_limit,
            }
        )
        if error_count >= error_limit:
            _write_stats_event(
                {
                    "event": "mobilegpt_action_error_limit",
                    "error": error_text,
                    "error_count": error_count,
                    "error_limit": error_limit,
                }
            )
            return getattr(self, finish_name)()
        return self.get_next_action()

    mobilegpt_class.handle_action_error = _handle_action_error
    mobilegpt_class._omniflow_action_error_recovery_installed = True


def normalize_mobilegpt_action(action: Any) -> Any:
    if not isinstance(action, dict):
        return action
    normalized = dict(action)
    name = str(normalized.get("name") or "").strip()
    normalized["name"] = _ACTION_NAME_ALIASES.get(name, name)
    parameters = {}
    legacy_parameters = normalized.pop("parameter", None)
    if isinstance(legacy_parameters, dict):
        parameters.update(legacy_parameters)
    if isinstance(normalized.get("parameters"), dict):
        parameters.update(normalized.get("parameters"))
    for key in (
        "index",
        "input_text",
        "direction",
        "number",
        "message",
        "info_name",
        "question",
        "scroll_ui_index",
        "target_info",
    ):
        if key in normalized and key not in parameters:
            parameters[key] = normalized.pop(key)
    index = parameters.get("index")
    if isinstance(index, list) and len(index) == 1:
        parameters["index"] = index[0]
    normalized["parameters"] = parameters
    return normalized


def install_mobilegpt_action_schema_adapter(server_module: Any) -> None:
    if bool(getattr(server_module, "_omniflow_action_schema_installed", False)):
        return
    original_send = server_module._omniflow_send_action

    def _send(client_socket, action):
        normalized = normalize_mobilegpt_action(action)
        action_name = (
            str(normalized.get("name") or "").strip()
            if isinstance(normalized, dict)
            else ""
        )
        if action_name not in _MOBILEGPT_SUPPORTED_ACTIONS:
            _write_stats_event(
                {
                    "event": "mobilegpt_unsupported_action",
                    "action": normalized,
                }
            )
            normalized = {
                "name": "click",
                "parameters": {"index": -1},
            }
        return original_send(client_socket, normalized)

    server_module._omniflow_send_action = _send
    server_module._omniflow_action_schema_installed = True


def install_mobilegpt_select_schema_repair(select_agent_class: type) -> None:
    if bool(getattr(select_agent_class, "_omniflow_schema_repair_installed", False)):
        return
    method_name = f"_{select_agent_class.__name__}__check_response_validity"
    original_check = getattr(select_agent_class, method_name)

    def _check(self, response, available_subtasks):
        if isinstance(response, dict):
            response.setdefault("completion_rate", 0)
            response.setdefault("speak", "Continuing the task.")
            action = response.get("action")
            if isinstance(action, dict) and action.get("name") == "speak":
                parameters = action.get("parameters")
                if isinstance(parameters, dict):
                    parameters.setdefault(
                        "completion_rate",
                        response["completion_rate"],
                    )
        try:
            return bool(original_check(self, response, available_subtasks))
        except (KeyError, TypeError):
            return False

    setattr(select_agent_class, method_name, _check)
    select_agent_class._omniflow_schema_repair_installed = True


def _extract_json_response(content: str, *, is_list: bool) -> Any:
    pattern = r"\[.*\]" if is_list else r"\{.*\}"
    match = re.search(pattern, str(content or ""), re.DOTALL)
    if not match:
        return str(content or "")
    return json.loads(match.group(0))


def _mobilegpt_response_kind(
    messages: list[dict[str, Any]],
    *,
    is_list: bool,
) -> str:
    if is_list:
        return "list"
    prompt = "\n".join(str(message.get("content") or "") for message in messages)
    if "Subtask given to you:" in prompt:
        return "action"
    if "MobileGPT Select JSON object" in prompt:
        return "action"
    if "List of available actions:" in prompt and '"action"' in prompt:
        return "action"
    if "known APIs" in prompt and '"api"' in prompt:
        return "task"
    if "Respond using the JSON format" in prompt or "Response Format:" in prompt:
        return "object"
    return "text"


def _parse_mobilegpt_model_response(
    content: str,
    *,
    messages: list[dict[str, Any]],
    is_list: bool,
) -> tuple[bool, Any, str]:
    kind = _mobilegpt_response_kind(messages, is_list=is_list)
    if kind == "text":
        return True, str(content or ""), ""
    try:
        parsed = _extract_json_response(content, is_list=is_list)
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        return False, None, f"invalid_json:{error}"
    if kind == "list":
        if isinstance(parsed, list):
            return True, parsed, ""
        return False, None, "expected_list"
    if not isinstance(parsed, dict):
        return False, None, "expected_object"
    if kind == "task":
        api = parsed.get("api")
        if not isinstance(api, dict):
            return False, None, "missing_api"
        parameters = api.get("parameters")
        if not str(api.get("app") or "").strip() and isinstance(parameters, dict):
            nested_app = str(parameters.get("app") or "").strip()
            forced_app = str(os.getenv("MOBILEGPT_TARGET_APP") or "").strip()
            if forced_app or nested_app:
                api["parameters"] = dict(parameters)
                api["parameters"].pop("app", None)
                api["app"] = forced_app or nested_app
        for field in ("name", "description", "app"):
            if not str(api.get(field) or "").strip():
                return False, None, f"missing_api_{field}"
        if not isinstance(api.get("parameters"), dict):
            return False, None, "invalid_api_parameters"
        if "found_match" not in parsed:
            return False, None, "missing_found_match"
    if kind == "action":
        action = parsed.get("action")
        if not isinstance(action, dict) or not str(action.get("name") or "").strip():
            return False, None, "missing_action"
        parsed["action"] = normalize_mobilegpt_action(action)
        parsed.setdefault("completion_rate", 0)
    return True, parsed, ""


def prepare_mobilegpt_chat_messages(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    prepared = [dict(message) for message in messages]
    prompt = "\n".join(str(message.get("content") or "") for message in prepared)
    for message in prepared:
        content = str(message.get("content") or "")
        if (
            str(message.get("role") or "") == "system"
            and "complete the user's request" in content
            and "List of available actions:" in content
        ):
            message["content"] = (
                "***Object-operation invariant***:\n"
                "When the request names a specific object, exact visible text "
                "identity is required. Prefixes, suffixes, dates, extensions, "
                "or other extra characters make a different object unless the "
                "request explicitly allows a partial match. If the exact "
                "object is not visible, scroll or search instead of selecting "
                "a similar-looking distractor.\n"
                "When a dialog contains editable fields and action buttons, "
                "fill every requested field before selecting the action button "
                "that commits the operation. Treat a bottom dialog button such "
                "as OK, Save, Create, Add, Folder, or File as an immediate "
                "commit action unless the current screen visibly marks it as a "
                "toggle or selector. Do not click a commit action merely to "
                "configure a mode.\n"
                "If the user asks a question or explicitly requests a response, "
                "use the speak action with the exact answer before finish. "
                "Finishing without a speak answer does not answer the user, "
                "even when the requested information is visible or was found.\n"
                "When the request applies an operation to one specific visible "
                "object, select that object before opening a global or overflow "
                "operations menu. Do not infer selection merely because a past "
                "action attempted it. Continue only when the current screen "
                "visibly confirms that the intended object, and no unintended "
                "objects, is selected. If it is not selected, choose or create "
                "an object-specific selection action first.\n"
                "Treat toolbars, tabs, navigation bars, shortcuts, and menu "
                "controls as app chrome, not as user data objects merely "
                "because their labels resemble the requested object type. "
                "Use the element hierarchy, role, attributes, and surrounding "
                "content to distinguish controls from data. If the requested "
                "objects are no longer present and past events show that the "
                "requested mutations were performed, select finish instead of "
                "inventing work from unrelated controls. An element marked "
                "[selected] is a selected tab or navigation destination; its "
                "adjacent peer controls are alternative destinations, not "
                "content objects. Never treat a control as a remaining data "
                "object when it lacks the interaction attributes shown by the "
                "previous real objects, such as long-clickable selection for "
                "items that were deleted through a context action. An empty "
                "current content list is direct completion evidence when no "
                "target object and no visible continuation, pagination, or "
                "scrollable content affordance remains. In that case finish; "
                "do not search unrelated tabs, parent directories, hidden app "
                "folders, or settings unless the user requested those scopes.\n\n"
                + content
            )
            content = str(message["content"])
        if (
            str(message.get("role") or "") == "system"
            and "complete the given subtask" in content
            and "Subtask given to you:" in prompt
        ):
            message["content"] = (
                "***Semantic UI and no-op invariant***:\n"
                "For a specifically named target, act only on an exact text "
                "match; a prefixed, suffixed, dated, or otherwise extended "
                "label is a different object. If the exact target is absent, "
                "scroll, search, or finish this subtask for reassessment.\n"
                "In a form or dialog, input the requested values before "
                "clicking the button that commits the operation. Bottom action "
                "buttons such as OK, Save, Create, Add, Folder, and File act "
                "immediately unless the UI visibly identifies them as toggles. "
                "Never use a commit button as a speculative mode selector.\n"
                "A label does not make an element the target data object. "
                "Treat toolbars, tabs, navigation bars, shortcuts, and menu "
                "controls as app chrome unless their hierarchy and attributes "
                "show that they are the requested content item. Never repeat "
                "the same low-level action on an unchanged screen. If the "
                "subtask premise is no longer true or its target is absent, "
                "finish this subtask so the task-level selector can reassess "
                "the current screen. Otherwise choose a different action.\n\n"
                + content
            )
            content = str(message["content"])
        if (
            str(message.get("role") or "") == "system"
            and "list out high-level functions" in content
            and "trigger UIs" in content
        ):
            message["content"] = (
                "***Semantic UI invariant***:\n"
                "Classify functions from element roles and interaction "
                "attributes, not labels alone. Toolbars, tabs, navigation "
                "bars, shortcuts, and menu controls are app chrome, not user "
                "data objects. An element marked [selected] is the active tab "
                "or navigation destination; adjacent peer controls are other "
                "destinations. Describe those controls only as navigation "
                "functions. Never expose them as files, notes, contacts, "
                "messages, or other content objects. A function that selects "
                "a data item for a context action must be backed by a semantic "
                "content element with the corresponding interaction attribute, "
                "such as long-clickable.\n\n"
                + content
            )
            content = str(message["content"])
        if not content.startswith(
            "Error: The selected action is not in the available actions list."
        ):
            continue
        message["content"] = (
            "Error: The selected action is not an available semantic action. "
            "Do not choose an unrelated listed action merely to satisfy the "
            "list. If the intended device operation is a low-level click, "
            "long-click, input, scroll, or back action, create and select a "
            "new semantic action describing its user-visible purpose. The "
            "new_action name must not itself be a low-level device action."
        )
    return prepared


def _mobilegpt_chat_model(requested_model: Any = None) -> str:
    return str(
        os.getenv("MOBILEGPT_CHAT_MODEL") or requested_model or "qwen-plus"
    ).strip()


def install_mobilegpt_openai_runtime(
    *,
    preserve_original_prompts: bool = False,
) -> None:
    utils_module = importlib.import_module("utils.utils")

    def _query(messages, model=None, is_list=False, agent_name="unknown"):
        from openai import OpenAI

        selected_model = _mobilegpt_chat_model(model)
        timeout_sec = max(
            1.0,
            float(os.getenv("MOBILEGPT_CHAT_TIMEOUT_SEC") or "90"),
        )
        max_attempts = max(
            1,
            int(os.getenv("MOBILEGPT_CHAT_MAX_ATTEMPTS") or "3"),
        )
        client = OpenAI(timeout=timeout_sec, max_retries=0)
        last_error: Exception | None = None
        active_messages = (
            [dict(message) for message in messages]
            if preserve_original_prompts
            else prepare_mobilegpt_chat_messages(messages)
        )
        for attempt in range(1, max_attempts + 1):
            started = time.monotonic()
            try:
                response = client.chat.completions.create(
                    model=selected_model,
                    messages=active_messages,
                    temperature=0,
                    max_tokens=900,
                    top_p=0,
                    frequency_penalty=0,
                    presence_penalty=0,
                )
                usage = getattr(response, "usage", None)
                content = str(response.choices[0].message.content or "")
                _write_stats_event(
                    {
                        "event": "chat_call",
                        "agent_name": str(agent_name or "unknown"),
                        "model": selected_model,
                        "attempt": attempt,
                        "latency_sec": round(time.monotonic() - started, 6),
                        "prompt_tokens": int(
                            getattr(usage, "prompt_tokens", 0) or 0
                        ),
                        "completion_tokens": int(
                            getattr(usage, "completion_tokens", 0) or 0
                        ),
                        "total_tokens": int(
                            getattr(usage, "total_tokens", 0) or 0
                        ),
                        "response_content": content,
                    }
                )
                valid, parsed, schema_error = _parse_mobilegpt_model_response(
                    content,
                    messages=messages,
                    is_list=is_list,
                )
                if valid:
                    return parsed
                last_error = ValueError(schema_error)
                _write_stats_event(
                    {
                        "event": "chat_schema_error",
                        "agent_name": str(agent_name or "unknown"),
                        "model": selected_model,
                        "attempt": attempt,
                        "error": schema_error,
                        "response_content": content,
                    }
                )
                active_messages.extend(
                    [
                        {"role": "assistant", "content": content},
                        {
                            "role": "user",
                            "content": (
                                "The previous response was incomplete or did not "
                                "match the required JSON schema. Return one complete "
                                "valid JSON response only."
                            ),
                        },
                    ]
                )
            except Exception as error:
                last_error = error
                _write_stats_event(
                    {
                        "event": "chat_error",
                        "agent_name": str(agent_name or "unknown"),
                        "model": selected_model,
                        "attempt": attempt,
                        "latency_sec": round(time.monotonic() - started, 6),
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
        assert last_error is not None
        raise last_error

    utils_module.query = _query
    module_names = (
        "agents.action_summarize_agent",
        "agents.app_agent",
        "agents.derive_agent",
        "agents.explore_agent",
        "agents.param_fill_agent",
        "agents.select_agent",
        "agents.subtask_merge_agent",
        "agents.task_agent",
        "memory.node_manager",
    )
    for module_name in module_names:
        module = importlib.import_module(module_name)
        if hasattr(module, "query"):
            module.query = _query


def install_mobilegpt_android_action_prompt(derive_prompt_module: Any) -> None:
    actions = getattr(derive_prompt_module, "default_actions", [])
    actions[:] = [
        action
        for action in actions
        if not isinstance(action, dict) or action.get("name") != "repeat-click"
    ]
    if not any(
        isinstance(action, dict) and action.get("name") == "back"
        for action in actions
    ):
        actions.append(
            {
                "name": "back",
                "description": (
                    "Press the Android system Back action to dismiss a dialog, "
                    "close an overlay, or return to the previous screen."
                ),
                "parameters": {},
            }
        )
    for action in actions:
        if not isinstance(action, dict) or action.get("name") != "long-click":
            continue
        action["description"] = (
            "Long-click a UI to select an item or open its context actions. "
            "When a later operation such as move, delete, rename, or share "
            "requires selecting an item, prefer long-click because an ordinary "
            "click usually opens the item. Use this only for a semantic data "
            "item whose element is explicitly long-clickable. Never long-click "
            "a toolbar, tab, navigation control, shortcut, menu command, or an "
            "element that lacks the long-clickable attribute."
        )


def install_mobilegpt_androidworld_layout_encoding(encoder_class: type) -> None:
    if bool(
        getattr(encoder_class, "_omniflow_androidworld_layout_installed", False)
    ):
        return
    original_encode = encoder_class.encode

    def _encode(self, raw_xml, index):
        parsed_xml, hierarchy_xml, encoded_xml = original_encode(
            self,
            raw_xml,
            index,
        )
        backend = str(
            os.environ.get("MOBILEGPT_RUNTIME_OBSERVE_BACKEND") or "androidworld"
        ).strip().lower()
        if backend != "androidworld":
            raise RuntimeError(
                f"mobilegpt_native_observe_backend_required:{backend or 'missing'}"
            )
        tree = ET.fromstring(parsed_xml)
        for element in tree.iter():
            element.attrib.pop("important", None)
            element.attrib.pop("class", None)
        encoded_xml = ET.tostring(tree, encoding="unicode")
        xml_directory = Path(str(getattr(self, "xml_directory", "")))
        if str(xml_directory):
            xml_directory.mkdir(parents=True, exist_ok=True)
            (xml_directory / f"{index}_encoded.xml").write_text(
                encoded_xml,
                encoding="utf-8",
            )
        return parsed_xml, hierarchy_xml, encoded_xml

    encoder_class.encode = _encode
    encoder_class._omniflow_androidworld_layout_installed = True


def _write_stats_event(event: dict[str, Any]) -> None:
    path = str(os.environ.get("MOBILEGPT_STATS_JSONL") or "").strip()
    if not path:
        return
    try:
        output = Path(path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(event)
        payload.setdefault("ts", datetime.now(timezone.utc).isoformat())
        with output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _mobilegpt_class_name(element: ET.Element) -> str:
    attributes = element.attrib
    if str(attributes.get("editable") or "").lower() == "true":
        return "android.widget.EditText"
    if str(attributes.get("checkable") or "").lower() == "true":
        return "android.widget.CheckBox"
    if str(attributes.get("clickable") or "").lower() == "true":
        return "android.widget.Button"
    if str(attributes.get("scrollable") or "").lower() == "true":
        return "androidx.recyclerview.widget.RecyclerView"
    if len(element):
        return "android.view.ViewGroup"
    if str(attributes.get("text") or attributes.get("content-desc") or ""):
        return "android.widget.TextView"
    return "android.view.View"


def mobilegpt_compatible_xml(xml_text: str) -> str:
    root = ET.fromstring(str(xml_text or "").strip())
    next_index = 0
    next_structural_index = -1
    for element in root.iter():
        attributes = element.attrib
        action_candidate = element.tag == "node" and any(
            str(attributes.get(key) or "").strip()
            for key in (
                "id",
                "resource-id",
                "text",
                "content-desc",
                "clickable",
                "editable",
                "scrollable",
                "long-clickable",
            )
        )
        if action_candidate:
            attributes["index"] = str(next_index)
            next_index += 1
        elif element is not root:
            attributes["index"] = str(next_structural_index)
            next_structural_index -= 1
        if not str(attributes.get("resource-id") or "").strip():
            resource_id = str(attributes.get("resource_id") or "").strip()
            if resource_id:
                attributes["resource-id"] = resource_id
        if not str(attributes.get("content-desc") or "").strip():
            description = str(
                attributes.get("content_description")
                or attributes.get("description")
                or ""
            ).strip()
            if description:
                attributes["content-desc"] = description
        if str(attributes.get("selected") or "").strip().lower() == "true":
            description = str(attributes.get("content-desc") or "").strip()
            if description and "selected" not in description.lower():
                attributes["content-desc"] = f"{description} [selected]"
        if not str(attributes.get("class") or "").strip():
            attributes["class"] = _mobilegpt_class_name(element)
    return ET.tostring(root, encoding="unicode")


def install_mobilegpt_androidworld_observe(server_class: type) -> None:
    if bool(
        getattr(server_class, "_omniflow_androidworld_observe_installed", False)
    ):
        return
    method_name = f"_{server_class.__name__}__recv_xml"
    original_receive = getattr(server_class, method_name)

    def _receive_xml(self, client_socket, screen_count, log_directory):
        global _LATEST_MOBILEGPT_XML
        client_xml = str(
            original_receive(self, client_socket, screen_count, log_directory)
            or ""
        )
        backend = str(
            os.environ.get("MOBILEGPT_RUNTIME_OBSERVE_BACKEND") or "androidworld"
        ).strip().lower()
        if backend != "androidworld":
            raise RuntimeError(
                f"mobilegpt_native_observe_backend_required:{backend or 'missing'}"
            )
        androidworld_xml = mobilegpt_compatible_xml(client_xml)
        xml_directory = Path(log_directory).expanduser() / "xmls"
        xml_directory.mkdir(parents=True, exist_ok=True)
        (xml_directory / f"{screen_count}.xml").write_text(
            androidworld_xml,
            encoding="utf-8",
        )
        _LATEST_MOBILEGPT_XML = androidworld_xml
        _write_stats_event(
            {
                "event": "mobilegpt_runtime_observe",
                "backend": "androidworld",
                "client_xml_chars": len(client_xml),
                "androidworld_xml_chars": len(androidworld_xml),
                "screen_index": int(screen_count),
            }
        )
        return androidworld_xml

    setattr(server_class, method_name, _receive_xml)
    server_class._omniflow_androidworld_observe_installed = True


def run_mobilegpt_server(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run MobileGPT with native client runtime observations."
    )
    parser.add_argument("--mobilegpt-root", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=12345)
    parser.add_argument("--buffer-size", type=int, default=4096)
    args = parser.parse_args(argv)

    root = Path(args.mobilegpt_root).expanduser().resolve()
    server_root = root / "Server"
    if not server_root.exists():
        raise FileNotFoundError(f"MobileGPT Server directory not found: {server_root}")
    if str(server_root) not in sys.path:
        sys.path.insert(0, str(server_root))
    os.chdir(server_root)

    from agents.app_agent import AppAgent
    from agents.prompts import derive_agent_prompt
    from agents.select_agent import SelectAgent
    import main as mobilegpt_main  # noqa: F401
    import mobilegpt as mobilegpt_module
    from screenParser.Encoder import xmlEncoder
    import server as mobilegpt_server

    Server = mobilegpt_server.Server
    MobileGPT = mobilegpt_module.MobileGPT

    install_mobilegpt_openai_runtime()
    install_mobilegpt_package_app_resolution(AppAgent)
    install_mobilegpt_android_action_prompt(derive_agent_prompt)
    install_mobilegpt_androidworld_layout_encoding(xmlEncoder)
    install_mobilegpt_action_schema_adapter(mobilegpt_server)
    install_mobilegpt_select_schema_repair(SelectAgent)
    install_mobilegpt_action_error_recovery(MobileGPT)
    install_mobilegpt_answer_event(MobileGPT)
    install_mobilegpt_androidworld_observe(Server)
    _write_stats_event(
        {
            "event": "mobilegpt_server_started",
            "host": args.host,
            "port": int(args.port),
            "runtime_observe_backend": str(
                os.environ.get("MOBILEGPT_RUNTIME_OBSERVE_BACKEND") or "androidworld"
            ),
        }
    )
    Server(
        host=args.host,
        port=int(args.port),
        buffer_size=int(args.buffer_size),
    ).open()
    return 0


if __name__ == "__main__":
    raise SystemExit(run_mobilegpt_server())
