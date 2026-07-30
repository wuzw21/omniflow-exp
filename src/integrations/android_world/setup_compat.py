from __future__ import annotations

import json
import logging
import time
from typing import Any

_EQUIVALENT_SETUP_LABELS: dict[str, tuple[str, ...]] = {
    "Accept & continue": ("Use without an account",),
    "Don't allow": ("Don’t allow",),
    "No thanks": ("Keep Google",),
    "Skip": ("SKIP",),
}


def _visible_setup_elements(controller: Any) -> list[dict[str, str]]:
    try:
        elements = tuple(controller._env.get_ui_elements() or ())
    except Exception:  # noqa: BLE001 - diagnostics must not mask setup retries
        return []
    return [
        {
            "package": str(getattr(element, "package_name", "") or ""),
            "text": str(getattr(element, "text", "") or ""),
            "content_description": str(
                getattr(element, "content_description", "") or ""
            ),
        }
        for element in elements
    ]


def _visible_setup_strings(controller: Any) -> set[str]:
    return {
        value.strip()
        for element in _visible_setup_elements(controller)
        for value in (element["text"], element["content_description"])
        if value.strip()
    }


def _setup_click_is_already_complete(
    visible: set[str],
    target_text: str,
) -> bool:
    if "Search or type web address" in visible:
        return target_text in {"Accept & continue", "No thanks"}
    if target_text == "Accept & continue":
        return "No thanks" in visible
    return False


def patch_androidworld_setup_fail_closed(
    setup_module: Any,
    *,
    attempts: int = 2,
    delay_seconds: float = 1.0,
) -> None:
    """Retry official app setup and save snapshots only after success."""

    if getattr(setup_module, "_omniflow_setup_fail_closed_patch", False):
        return

    def setup_app_with_retry(app: Any, env: Any) -> None:
        attempt_count = max(1, int(attempts))
        for attempt in range(1, attempt_count + 1):
            try:
                app.setup(env)
            except ValueError as error:
                if attempt >= attempt_count:
                    raise
                logging.warning(
                    "AndroidWorld app setup failed; retrying app=%s attempt=%d/%d error=%s",
                    app.app_name,
                    attempt,
                    attempt_count,
                    error,
                )
                time.sleep(max(0.0, float(delay_seconds)))
                continue
            setup_module.app_snapshot.save_snapshot(app.app_name, env.controller)
            return

    setup_module.setup_app = setup_app_with_retry
    setup_module._omniflow_setup_fail_closed_patch = True


def patch_androidworld_setup_click_retry(
    tools_module: Any,
    *,
    attempts: int = 15,
    delay_seconds: float = 1.0,
) -> None:
    """Make official AndroidWorld app setup tolerate equivalent onboarding UI.

    AndroidWorld remains the owner of app setup.  This patch only retries its
    requested text click and maps labels whose semantics changed in newer
    system-app builds (notably Chrome's current first-run screen).
    """

    if getattr(tools_module, "_omniflow_setup_click_retry_patch", False):
        return
    controller_type = tools_module.AndroidToolController
    original_click_element = controller_type.click_element

    def click_label(
        controller: Any,
        label: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        if args:
            return original_click_element(controller, label, *args[1:], **kwargs)
        call_kwargs = dict(kwargs)
        call_kwargs["element_text"] = label
        return original_click_element(controller, **call_kwargs)

    def click_element_with_retry(
        controller: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        target_text = str(args[0] if args else kwargs.get("element_text") or "")
        last_error: ValueError | None = None
        for attempt in range(1, max(1, int(attempts)) + 1):
            labels = (target_text, *_EQUIVALENT_SETUP_LABELS.get(target_text, ()))
            visible = _visible_setup_strings(controller)
            if _setup_click_is_already_complete(visible, target_text):
                return None
            visible_casefold = {value.casefold() for value in visible}
            for label in (
                candidate
                for candidate in labels
                if candidate.casefold() in visible_casefold
            ):
                try:
                    return click_label(controller, label, args, kwargs)
                except ValueError as error:
                    message = str(error or "")
                    if not (
                        ("Target text" in message and "not found" in message)
                        or "Invalid element index" in message
                    ):
                        raise
                    last_error = error
            if attempt < max(1, int(attempts)):
                time.sleep(max(0.0, float(delay_seconds)))

        details = json.dumps(
            _visible_setup_elements(controller)[:80],
            ensure_ascii=False,
        )
        if last_error is None:
            raise ValueError(
                f'AndroidWorld setup target "{target_text}" not found; '
                f"visible_setup_elements={details}"
            )
        raise ValueError(
            f"{last_error}; visible_setup_elements={details}"
        ) from last_error

    controller_type.click_element = click_element_with_retry
    tools_module._omniflow_setup_click_retry_patch = True
