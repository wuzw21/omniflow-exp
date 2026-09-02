from __future__ import annotations

import sys
from types import SimpleNamespace

from src.integrations.android_world.run_episode import (
    _patch_androidworld_expense_setup_timeout,
)


def test_resource_setup_dismisses_legacy_dialog_before_vlc_onboarding(
    monkeypatch,
) -> None:
    class Controller:
        dismissed = False

        def __init__(self) -> None:
            self._env = SimpleNamespace(
                foreground_activity_name="org.videolan.vlc/.StartActivity",
                get_ui_elements=lambda: [
                    SimpleNamespace(
                        text=(
                            "SKIP"
                            if self.dismissed
                            else "This app was built for an older version of Android "
                            "and may not work properly."
                        ),
                        content_description=None,
                        package_name="org.videolan.vlc",
                    ),
                    SimpleNamespace(
                        text=None if self.dismissed else "OK",
                        content_description=None,
                        package_name="android",
                    ),
                ],
            )

        def click_element(self, label: str) -> None:
            assert label == "OK"
            self.dismissed = True

        def click_resource_id(
            self,
            resource_ids: str | tuple[str, ...],
            timeout_sec: float = 10.0,
        ) -> str:
            del timeout_sec
            if not self.dismissed:
                raise ValueError(f"Target resource ID not found: {resource_ids}.")
            return "clicked"

    fake_tools = SimpleNamespace(AndroidToolController=Controller)
    monkeypatch.setitem(sys.modules, "android_world.env.tools", fake_tools)

    patched = _patch_androidworld_expense_setup_timeout()
    assert patched is not None
    controller = Controller()
    try:
        assert (
            controller.click_resource_id("org.videolan.vlc:id/skip_button")
            == "clicked"
        )
        assert controller.dismissed is True
    finally:
        controller_type, original = patched
        controller_type.click_resource_id = original
