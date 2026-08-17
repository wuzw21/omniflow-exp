from __future__ import annotations

from omniflow import Observation
from omniflow.runtime.checker import transient_obstruction_recovery


def _page(label: str) -> Observation:
    return Observation(
        xml=(
            '<hierarchy rotation="0">'
            '<node text="" class="android.widget.FrameLayout" '
            'package="com.example" clickable="false" enabled="true" '
            'bounds="[0,0][1080,2400]">'
            f'<node text="{label}" resource-id="com.example:id/action" '
            'class="android.widget.Button" package="com.example" '
            'clickable="true" enabled="true" bounds="[100,1800][500,1900]" />'
            "</node></hierarchy>"
        ),
        package_name="com.example",
    )


def test_explicit_transient_dismiss_is_recovered_without_model() -> None:
    action = transient_obstruction_recovery(_page("Not now"))

    assert action is not None
    assert action.tool == "click"
    assert action.args["target_description"] == "关闭临时遮挡"
    assert 0.0 <= action.args["x"] <= 1000.0
    assert 0.0 <= action.args["y"] <= 1000.0


def test_transient_dismiss_uses_host_display_instead_of_xml_content_height() -> None:
    observation = Observation(
        xml=(
            '<hierarchy rotation="0" width="1080" height="2148">'
            '<node text="No Thanks" resource-id="com.example:id/negative_button" '
            'class="android.widget.Button" package="com.example" '
            'clickable="true" enabled="true" bounds="[44,2005][299,2104]" />'
            "</hierarchy>"
        ),
        package_name="com.example",
        extra={"display": {"width": 1080, "height": 2280}},
    )

    action = transient_obstruction_recovery(observation)

    assert action is not None
    assert action.args["x"] == 171.5 / 1080 * 1000
    assert action.args["y"] == 2054.5 / 2280 * 1000


def test_exact_ok_is_not_treated_as_transient_obstruction() -> None:
    assert transient_obstruction_recovery(_page("OK")) is None


def test_ok_prefix_is_not_treated_as_confirmation() -> None:
    assert transient_obstruction_recovery(_page("OKAY")) is None


def test_ambiguous_cancel_is_not_dismissed_automatically() -> None:
    assert transient_obstruction_recovery(_page("取消")) is None
