from __future__ import annotations

import json
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


def _setup_click_is_already_complete(controller: Any, target_text: str) -> bool:
    if target_text != "No thanks":
        return False
    return "Search or type web address" in _visible_setup_strings(controller)


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
            for label in labels:
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
            if _setup_click_is_already_complete(controller, target_text):
                return None
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
