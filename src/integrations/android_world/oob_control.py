from __future__ import annotations

import base64
import binascii
import io
import json
import os
import subprocess
import time
from types import SimpleNamespace
from typing import Any, Callable
import uuid
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

CONTROL_ACTION = "cn.com.omnimind.bot.debug.CONTROL_OMNIFLOW"
CONTROL_PACKAGE = "cn.com.omnimind.bot"
EXPERIMENTAL_CONTROL_PACKAGES = (
    CONTROL_PACKAGE,
    "cn.com.omnimind.bot.debug",
)
CONTROL_ACCESSIBILITY_SERVICE = (
    f"{CONTROL_PACKAGE}/"
    "cn.com.omnimind.accessibility.service.AssistsService"
)
CONTROL_RECEIVER = (
    f"{CONTROL_PACKAGE}/cn.com.omnimind.bot.debug.DebugOmniFlowControlReceiver"
)
CONTROL_RESULT_PREFIX = "files/debug-omniflow-control-result-"
LEGACY_CONTROL_RESULT_PATH = "files/debug-omniflow-control-result.json"
OBSERVE_ACTION = "cn.com.omnimind.bot.debug.OBSERVE_OMNIFLOW"
OBSERVE_RECEIVER = (
    f"{CONTROL_PACKAGE}/cn.com.omnimind.bot.debug.DebugOmniFlowObserveReceiver"
)
OBSERVE_RESULT_PREFIX = "files/debug-omniflow-observe-result-"
LEGACY_OBSERVE_RESULT_PATH = "files/debug-omniflow-observe-result.json"
OBSERVE_XML_ATTEMPTS = 4
OBSERVE_XML_RETRY_DELAY_SECONDS = 0.25


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return str(value).strip().lower() not in {"0", "false", "no", "off"}


def oob_control_accessibility_service(package_name: str = CONTROL_PACKAGE) -> str:
    """Return the service component for the installed experimental APK."""

    return (
        f"{str(package_name or CONTROL_PACKAGE).strip()}/"
        "cn.com.omnimind.accessibility.service.AssistsService"
    )


def oob_control_receiver(
    package_name: str = CONTROL_PACKAGE, *, observe: bool = False
) -> str:
    """Return the receiver component for release or debug package identity."""

    receiver_name = (
        "cn.com.omnimind.bot.debug.DebugOmniFlowObserveReceiver"
        if observe
        else "cn.com.omnimind.bot.debug.DebugOmniFlowControlReceiver"
    )
    return f"{str(package_name or CONTROL_PACKAGE).strip()}/{receiver_name}"


class OobControlClient:
    def __init__(
        self,
        env: Any,
        *,
        adb_serial: str = "",
        adb_path: str = "",
        package_name: str = CONTROL_PACKAGE,
        receiver: str = CONTROL_RECEIVER,
        timeout_seconds: float = 30.0,
        run: Callable[..., subprocess.CompletedProcess[str]] | None = None,
    ) -> None:
        self.env = env
        self.adb_serial = self._resolve_serial(adb_serial)
        self.adb_path = str(adb_path or os.environ.get("ADB_PATH") or "adb")
        self.package_name = str(package_name or CONTROL_PACKAGE).strip()
        self.receiver = str(receiver or CONTROL_RECEIVER).strip()
        self.timeout_seconds = max(1.0, float(timeout_seconds))
        self._run_command = run or subprocess.run
        # Keep the last complete OOB state on the host side.  The resident
        # Android dispatcher normally retains this state too, but that state
        # can be lost when an action crosses into another package (notably
        # DocumentsUI).  Sending the observed state with the next action
        # preserves the normal Observe -> Action contract without falling
        # back to coordinates or native AndroidWorld actions.
        self._last_state: dict[str, Any] | None = None

    def observe(self, *, wait_to_stabilize: bool = False) -> dict[str, Any]:
        for attempt in range(OBSERVE_XML_ATTEMPTS):
            result = self._observe_request(wait_to_stabilize=wait_to_stabilize)
            if not isinstance(result, dict):
                raise RuntimeError("oob_control_observe_result_invalid")
            if str(result.get("xml") or "").strip():
                return result
            if attempt < OBSERVE_XML_ATTEMPTS - 1:
                time.sleep(OBSERVE_XML_RETRY_DELAY_SECONDS)
        raise RuntimeError("oob_control_observe_xml_missing")

    def act(self, action: dict[str, Any]) -> dict[str, Any]:
        # The caller has just observed the state used for transfer.  The OOB
        # control path keeps that state as its current state, and the Android
        # side performs its own pre-dispatch fingerprint plus post-dispatch
        # stabilization.  Re-observing here duplicated that wait on every
        # action without changing the action or the transfer decision.  The
        # default remains fail-safe; the controlled lightweight benchmark can
        # disable the Android-side sampler and rely on the recorder's single
        # fast after-observation plus the runtime's bounded stable retry.
        await_stabilization = _env_bool(
            "OMNIFLOW_ANDROIDWORLD_ACT_AWAIT_STABILIZATION", True
        )
        payload: dict[str, Any] = {
            # The resident OOB executor otherwise returns immediately after
            # dispatch.  That leaves the next Function step and the official
            # validator observing the pre-action UI state.  Stabilization is
            # part of the OOB act contract; wait actions remain cheap because
            # the Android side handles them as already completed.
            "action": action,
            "await_stabilization": await_stabilization,
        }
        # The resident Kotlin dispatcher normally keeps the last observed
        # State, but a package transition (notably DocumentsUI) can recreate
        # that dispatcher-side cache.  Send the identity/display portion of
        # the exact observed State with the action.  The full XML is already
        # recorded by Observe and is intentionally omitted here because an
        # intent extra containing a large DocumentsUI tree can exceed Android's
        # transaction limit.  State.fromMap only needs these fields to verify
        # the Observe -> Action pair; physical execution remains OOB-only.
        if self._last_state is not None:
            state_id = str(self._last_state.get("state_id") or "").strip()
            display = self._last_state.get("display")
            if state_id and isinstance(display, dict):
                payload["state"] = {
                    "state_id": state_id,
                    "package_name": str(
                        self._last_state.get("package_name") or ""
                    ),
                    "activity_name": str(
                        self._last_state.get("activity_name") or ""
                    ),
                    "display": {
                        "width": display.get("width"),
                        "height": display.get("height"),
                    },
                }
        result = self._request("act", payload)
        if not isinstance(result, dict):
            raise RuntimeError("oob_control_act_result_invalid")
        return result

    def _observe_request(self, *, wait_to_stabilize: bool) -> dict[str, Any]:
        request_id = uuid.uuid4().hex
        result_path = f"{OBSERVE_RESULT_PREFIX}{request_id}.json"
        for candidate_path in self._result_paths(
            "observe", request_id, result_path
        ):
            self._remove_result(candidate_path)
        # The resident APK may be installed under the release package or the
        # debug package identity.  OBSERVE_RECEIVER is kept for the release
        # default, but it must not be used verbatim for a debug installation:
        # Android resolves the receiver by package identity as well as class.
        component = oob_control_receiver(self.package_name, observe=True)
        broadcast, broadcast_timed_out = self._run_broadcast(
            [
                "shell",
                "am",
                "broadcast",
                "-a",
                OBSERVE_ACTION,
                "-n",
                component,
                "--ez",
                "includeScreenshot",
                "true",
                "--ez",
                "waitToStabilize",
                "true" if wait_to_stabilize else "false",
                "--es",
                "requestId",
                request_id,
                "--es",
                "resultFile",
                result_path.removeprefix("files/"),
            ],
        )
        broadcast_output = (broadcast.stderr or broadcast.stdout or "").strip()
        # ``am broadcast`` may report ``error: closed`` when the receiver has
        # already accepted the request and the shell-side pipe closes before
        # the receiver writes its result file.  The Android receiver logs a
        # successful completion in that case, so keep polling the result
        # rather than losing an otherwise valid Observe.
        if (
            not broadcast_timed_out
            and broadcast.returncode != 0
            and "error: closed" not in broadcast_output.lower()
        ):
            raise RuntimeError(
                "oob_observe_broadcast_failed:"
                + broadcast_output
            )
        deadline = time.monotonic() + self.timeout_seconds
        last_error = ""
        while time.monotonic() < deadline:
            for candidate_path in self._result_paths(
                "observe", request_id, result_path
            ):
                result = self._read_result(candidate_path)
                text = str(result.stdout or "").strip()
                if result.returncode != 0 or not text:
                    last_error = (result.stderr or result.stdout or "").strip()
                    continue
                response = self._decode_result(text, "oob_observe")
                response_request_id = str(response.get("request_id") or "").strip()
                if response_request_id and response_request_id != request_id:
                    last_error = "oob_observe_result_request_id_mismatch"
                    continue
                if not response_request_id and candidate_path != LEGACY_OBSERVE_RESULT_PATH:
                    last_error = "oob_observe_result_request_id_missing"
                    continue
                if response.get("success") is not True:
                    raise RuntimeError(
                        "oob_observe_failed:" + str(response.get("error") or "unknown")
                    )
                state = response.get("state")
                if isinstance(state, dict):
                    # ``includeScreenshot=true`` adds full PNG bytes to the
                    # host map.  They are evidence for the collector, but
                    # are not part of State.fromMap and make an intent extra
                    # unnecessarily large.  Keep only the state contract
                    # needed by the Android action dispatcher.
                    self._last_state = {
                        key: value
                        for key, value in state.items()
                        if key not in {"image_base64", "extra"}
                    }
                return {
                    **response,
                    **(state if isinstance(state, dict) else {}),
                }
            else:
                last_error = (result.stderr or result.stdout or "").strip()
            time.sleep(0.05)
        raise RuntimeError("oob_observe_result_timeout:" + last_error[-500:])

    def reset(self) -> None:
        self._last_state = None
        self._request("reset", {})

    def _request(self, operation: str, payload: dict[str, Any]) -> Any:
        request_id = uuid.uuid4().hex
        result_path = f"{CONTROL_RESULT_PREFIX}{request_id}.json"
        encoded = base64.b64encode(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(
                "utf-8"
            )
        ).decode("ascii")
        for candidate_path in self._result_paths(
            "control", request_id, result_path
        ):
            self._remove_result(candidate_path)
        component = self.receiver
        if component.startswith("."):
            component = f"{self.package_name}/{component}"
        elif "/" not in component:
            component = f"{self.package_name}/.{component}"
        broadcast, broadcast_timed_out = self._run_broadcast(
            [
                "shell",
                "am",
                "broadcast",
                "-a",
                CONTROL_ACTION,
                "-n",
                component,
                "--es",
                "requestId",
                request_id,
                "--es",
                "operation",
                operation,
                "--es",
                "requestBase64",
                encoded,
                "--es",
                "resultFile",
                result_path.removeprefix("files/"),
            ],
        )
        broadcast_output = (broadcast.stderr or broadcast.stdout or "").strip()
        # See the matching observe path above.  A closed adb pipe is
        # recoverable as long as the receiver publishes the request result.
        if (
            not broadcast_timed_out
            and broadcast.returncode != 0
            and "error: closed" not in broadcast_output.lower()
        ):
            raise RuntimeError(
                "oob_control_broadcast_failed:"
                + broadcast_output
            )
        deadline = time.monotonic() + self.timeout_seconds
        last_error = ""
        while time.monotonic() < deadline:
            for candidate_path in self._result_paths(
                "control", request_id, result_path
            ):
                result = self._read_result(candidate_path)
                text = str(result.stdout or "").strip()
                if result.returncode != 0 or not text:
                    last_error = (result.stderr or result.stdout or "").strip()
                    continue
                response = self._decode_result(text, "oob_control")
                response_request_id = str(response.get("request_id") or "").strip()
                if response_request_id and response_request_id != request_id:
                    last_error = "oob_control_result_request_id_mismatch"
                elif not response_request_id and candidate_path != LEGACY_CONTROL_RESULT_PATH:
                    last_error = "oob_control_result_request_id_missing"
                elif response.get("success") is not True:
                    raise RuntimeError(
                        "oob_control_failed:" + str(response.get("error") or "unknown")
                    )
                else:
                    return response.get("result")
            time.sleep(0.05)
        raise RuntimeError("oob_control_result_timeout:" + last_error[-500:])

    def _result_paths(
        self, operation: str, request_id: str, request_path: str
    ) -> tuple[str, ...]:
        """Return the new request-scoped path and the old fixed path.

        The old debug APK ignores ``resultFile`` and serializes one request per
        device into a fixed file.  The host remains request-scoped by clearing
        both paths before dispatching and accepting a fixed-file response only
        while this client has one outstanding request.
        """

        legacy_path = (
            LEGACY_OBSERVE_RESULT_PATH
            if operation == "observe"
            else LEGACY_CONTROL_RESULT_PATH
        )
        if request_path == legacy_path:
            return (request_path,)
        return (request_path, legacy_path)

    def _remove_result(self, path: str) -> None:
        self._run(
            [
                "shell",
                "run-as",
                self.package_name,
                "rm",
                "-f",
                path,
            ],
            timeout=10.0,
        )

    def _read_result(self, path: str) -> subprocess.CompletedProcess[str]:
        return self._run(
            [
                "shell",
                "run-as",
                self.package_name,
                "cat",
                path,
            ],
            timeout=10.0,
        )

    @staticmethod
    def _decode_result(text: str, prefix: str) -> dict[str, Any]:
        try:
            response = json.loads(text)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"{prefix}_result_invalid_json") from error
        if not isinstance(response, dict):
            raise RuntimeError(f"{prefix}_result_not_object")
        return response

    def _run(self, args: list[str], *, timeout: float) -> subprocess.CompletedProcess[str]:
        command = [self.adb_path]
        if self.adb_serial:
            command.extend(["-s", self.adb_serial])
        command.extend(args)
        return self._run_command(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout,
        )

    def _run_broadcast(
        self, args: list[str]
    ) -> tuple[subprocess.CompletedProcess[str], bool]:
        """Run a broadcast while allowing the legacy receiver to finish asynchronously.

        The old APK can write its fixed result file successfully while
        ``am broadcast`` remains blocked on the shell-side reply.  The result
        polling below is the authoritative completion signal in that mode.
        """

        command = [self.adb_path]
        if self.adb_serial:
            command.extend(["-s", self.adb_serial])
        command.extend(args)
        try:
            return self._run(args, timeout=self.timeout_seconds), False
        except subprocess.TimeoutExpired as error:
            stdout = error.stdout or ""
            stderr = error.stderr or ""
            if isinstance(stdout, bytes):
                stdout = stdout.decode("utf-8", errors="replace")
            if isinstance(stderr, bytes):
                stderr = stderr.decode("utf-8", errors="replace")
            return (
                subprocess.CompletedProcess(
                    command,
                    124,
                    stdout=str(stdout),
                    stderr=str(stderr),
                ),
                True,
            )

    def _resolve_serial(self, value: str) -> str:
        serial = str(value or os.environ.get("ANDROID_SERIAL") or "").strip()
        if serial:
            return serial
        controller = getattr(self.env, "controller", None)
        try:
            port = controller.env._coordinator._simulator._config.emulator_launcher.emulator_console_port
            if port:
                return f"emulator-{int(port)}"
        except Exception:
            pass
        return ""


def oob_state_from_payload(
    payload: dict[str, Any],
    *,
    fallback_screen_size: tuple[int, int] = (1, 1),
) -> Any:
    nested_state = payload.get("state")
    if isinstance(nested_state, dict):
        payload = {**payload, **nested_state}
    xml = str(payload.get("xml") or "")
    if not xml.strip():
        raise ValueError("oob_control_xml_missing")
    display = payload.get("display")
    if isinstance(display, dict):
        width = _positive_int(display.get("width"), fallback_screen_size[0])
        height = _positive_int(display.get("height"), fallback_screen_size[1])
    else:
        width, height = fallback_screen_size
    pixels = _decode_pixels(payload.get("image_base64"), width, height)
    representation_utils = __import__(
        "android_world.env.representation_utils",
        fromlist=["xml_dump_to_ui_elements"],
    )
    ui_elements = representation_utils.xml_dump_to_ui_elements(xml)
    forest = _xml_to_accessibility_forest(xml, width=width, height=height)
    package_name = str(payload.get("package_name") or "")
    activity_name = str(payload.get("activity_name") or "")
    auxiliaries = {
        "observe_backend": "oob_control",
        "package_name": package_name,
        "activity_name": activity_name,
        "display": {"width": width, "height": height},
        # Keep the exact OOB XML alongside the validator-compatible proto
        # forest.  The XML is the evidence source; the proto is only the
        # AndroidWorld validator compatibility representation.
        "xml": xml,
    }
    stabilization = payload.get("stabilization")
    if isinstance(stabilization, dict):
        auxiliaries["stabilization"] = dict(stabilization)
    return SimpleNamespace(
        pixels=pixels,
        forest=forest,
        ui_elements=ui_elements,
        auxiliaries=auxiliaries,
    )


def _xml_to_accessibility_forest(
    xml: str,
    *,
    width: int,
    height: int,
) -> Any:
    """Convert OOB's complete XML into AndroidWorld's validator forest type.

    OOB owns the physical observation and returns the canonical XML.  Some
    AndroidWorld validators, however, still call ``forest_to_ui_elements``
    and require the protobuf forest object rather than XML.  Building this
    compatibility view locally preserves the OOB XML while allowing those
    official validators to inspect the same observed tree.
    """

    from android_env.proto.a11y import android_accessibility_forest_pb2

    root = ET.fromstring(xml)
    nodes = list(root.iter("node"))
    forest = android_accessibility_forest_pb2.AndroidAccessibilityForest()
    window = forest.windows.add()
    window.id = 0
    window.is_active = True
    window.is_focused = True
    window.bounds_in_screen.left = 0
    window.bounds_in_screen.top = 0
    window.bounds_in_screen.right = int(width)
    window.bounds_in_screen.bottom = int(height)

    node_ids = {id(node): index for index, node in enumerate(nodes)}
    for index, element in enumerate(nodes):
        node = window.tree.nodes.add()
        attrs = element.attrib
        node.unique_id = index
        node.window_id = 0
        node.depth = len(list(element.iterancestors())) if hasattr(element, "iterancestors") else 0
        node.class_name = str(attrs.get("class") or "")
        node.content_description = str(attrs.get("content-desc") or "")
        node.hint_text = str(attrs.get("hint-text") or "")
        node.package_name = str(attrs.get("package") or "")
        node.text = str(attrs.get("text") or "")
        node.view_id_resource_name = str(attrs.get("resource-id") or "")
        bounds = str(attrs.get("bounds") or "").strip()
        if bounds.startswith("[") and "][" in bounds and bounds.endswith("]"):
            left_top, right_bottom = bounds[1:-1].split("][", 1)
            left, top = (int(value) for value in left_top.split(",", 1))
            right, bottom = (int(value) for value in right_bottom.split(",", 1))
            node.bounds_in_screen.left = left
            node.bounds_in_screen.top = top
            node.bounds_in_screen.right = right
            node.bounds_in_screen.bottom = bottom
        for field in (
            "is_checkable", "is_checked", "is_clickable", "is_editable",
            "is_enabled", "is_focusable", "is_focused", "is_long_clickable",
            "is_password", "is_scrollable", "is_selected", "is_visible_to_user",
        ):
            source = {
                "is_checkable": "checkable",
                "is_checked": "checked",
                "is_clickable": "clickable",
                "is_editable": "editable",
                "is_enabled": "enabled",
                "is_focusable": "focusable",
                "is_focused": "focused",
                "is_long_clickable": "long-clickable",
                "is_password": "password",
                "is_scrollable": "scrollable",
                "is_selected": "selected",
                "is_visible_to_user": "visible-to-user",
            }[field]
            setattr(node, field, str(attrs.get(source) or "false").lower() == "true")
        node.child_ids.extend(
            node_ids[id(child)] for child in list(element) if child.tag == "node"
        )
    return forest


def _decode_pixels(value: Any, width: int, height: int) -> np.ndarray:
    encoded = str(value or "").strip()
    if encoded.startswith("data:image/") and "," in encoded:
        encoded = encoded.split(",", 1)[1]
    if encoded:
        try:
            image = Image.open(io.BytesIO(base64.b64decode(encoded, validate=False))).convert(
                "RGB"
            )
            return np.asarray(image).copy()
        except (binascii.Error, OSError, ValueError) as error:
            raise ValueError("oob_control_image_invalid") from error
    return np.zeros((max(1, height), max(1, width), 3), dtype=np.uint8)


def _positive_int(value: Any, fallback: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = 0
    return number if number > 0 else max(1, int(fallback))


__all__ = ["OobControlClient", "oob_state_from_payload"]
