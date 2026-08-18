from __future__ import annotations

import pytest

from omniflow.core.model import Action, Observation
from omniflow.core.schemas import vlm_action_tools
from omniflow.runtime.semantic_grounding import (
    resolve_semantic_action,
    semantic_target_at_point,
)


def observation(xml: str) -> Observation:
    return Observation(xml=xml, extra={"display": {"width": 1080, "height": 2400}})


def test_vlm_click_schema_exposes_semantic_target() -> None:
    click = next(
        tool["function"]
        for tool in vlm_action_tools()
        if tool["function"]["name"] == "click"
    )

    assert set(click["parameters"]["properties"]) == {
        "target_description",
        "x",
        "y",
    }
    assert set(click["parameters"]["required"]) == {"x", "y"}


def test_unique_text_match_resolves_to_accessibility_center() -> None:
    result = resolve_semantic_action(
        Action("click", {"target_description": "Turn off", "x": 572, "y": 572}),
        observation(
            '<hierarchy><node id="64:5" class="android.widget.Button" '
            'text="Turn off" content-desc="" resource-id="android:id/button1" '
            'bounds="[730,1310][926,1436]" clickable="true" enabled="true"/>'
            "</hierarchy>"
        ),
    )

    assert result.action.args["x"] == pytest.approx(828 / 1080 * 1000)
    assert result.action.args["y"] == pytest.approx(1373 / 2400 * 1000)
    assert result.action.args["node_id"] == "64:5"
    assert result.action.args["node_resource_id"] == "android:id/button1"
    assert result.detail and result.detail["status"] == "resolved"


def test_unique_content_description_match_resolves_clickable_parent() -> None:
    result = resolve_semantic_action(
        Action("click", {"target_description": "Add contact", "x": 1, "y": 1}),
        observation(
            '<hierarchy><node id="parent" bounds="[800,1800][1000,2000]" '
            'clickable="true"><node content-desc="Add contact" bounds="[820,1820][980,1980]" '
            'clickable="false"/></node></hierarchy>'
        ),
    )

    assert result.action.args["x"] == pytest.approx(900 / 1080 * 1000)
    assert result.action.args["y"] == pytest.approx(1900 / 2400 * 1000)
    assert result.action.args["node_id"] == "parent"


def test_unique_virtual_target_resolves_to_its_current_bounds() -> None:
    result = resolve_semantic_action(
        Action("click", {"target_description": "30", "x": 1, "y": 1}),
        observation(
            '<hierarchy><node id="minute-30" content-desc="30" '
            'bounds="[485,1201][595,1311]" clickable="false" '
            'enabled="true" visible="true"/></hierarchy>'
        ),
    )

    assert result.action.args["x"] == pytest.approx(540 / 1080 * 1000)
    assert result.action.args["y"] == pytest.approx(1256 / 2400 * 1000)
    assert result.action.args["node_id"] == "minute-30"
    assert result.detail and result.detail["status"] == "resolved"


def test_source_point_reads_a_visible_virtual_target_label() -> None:
    xml = (
        '<hierarchy><node content-desc="30" bounds="[485,1201][595,1311]" '
        'clickable="false" enabled="true" visible="true"/></hierarchy>'
    )

    assert semantic_target_at_point(xml, 540, 1256) == "30"


def test_ambiguous_match_preserves_model_coordinates() -> None:
    action = Action("click", {"target_description": "More", "x": 400, "y": 500})
    result = resolve_semantic_action(
        action,
        observation(
            '<hierarchy><node text="More" bounds="[0,0][100,100]" clickable="true"/>'
            '<node text="More" bounds="[200,0][300,100]" clickable="true"/></hierarchy>'
        ),
    )

    assert result.action == action
    assert result.detail and result.detail["reason"] == "target_ambiguous"


def test_missing_match_preserves_model_coordinates_without_oracle_leakage() -> None:
    action = Action("click", {"target_description": "Unknown", "x": 400, "y": 500})
    result = resolve_semantic_action(
        action,
        observation(
            '<hierarchy><node text="Turn off" bounds="[730,1310][926,1436]" '
            'clickable="true"/></hierarchy>'
        ),
    )

    assert result.action == action
    assert result.detail and result.detail["reason"] == "target_missing"
