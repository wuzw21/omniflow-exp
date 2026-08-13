from __future__ import annotations

from omniflow.vlm.planner import build_model_turn_request, project_ui


def _node(
    *,
    text: str = "",
    description: str = "",
    resource_id: str = "",
    bounds: str,
    clickable: bool = False,
) -> str:
    return (
        f'<node text="{text}" content-desc="{description}" '
        f'resource-id="{resource_id}" bounds="{bounds}" '
        f'clickable="{str(clickable).lower()}" />'
    )


def _merchant_menu_xml() -> str:
    nodes = [
        _node(
            text=f"拿铁咖啡 {index}",
            bounds=f"[40,{300 + index * 40}][800,{330 + index * 40}]",
        )
        for index in range(40)
    ]
    nodes.extend(
        (
            _node(
                bounds="[910,340][990,420]",
                clickable=True,
            ),
            _node(
                description="搜索",
                resource_id="com.example:id/search",
                bounds="[930,80][1030,180]",
                clickable=True,
            ),
        )
    )
    return f"<hierarchy>{''.join(nodes)}</hierarchy>"


def test_projection_reserves_global_and_unlabeled_visual_controls() -> None:
    projection = project_ui(_merchant_menu_xml(), "点一杯拿铁咖啡")

    assert projection.selected_count <= 30
    assert projection.visual_candidate_count == 1
    assert projection.text.index("[global_controls]") < projection.text.index(
        "[goal_matches]"
    )
    assert projection.text.index('"d":"搜索"') < projection.text.index(
        '"t":"拿铁咖啡 0"'
    )
    assert "[goal_controls]" in projection.text
    assert '"v":"A' in projection.text
    assert '"b":"[910,340][990,420]"' in projection.text


def test_projection_promotes_bottom_control_near_goal_text() -> None:
    generic_controls = "".join(
        _node(
            bounds=f"[20,{220 + index * 35}][100,{280 + index * 35}]",
            clickable=True,
        )
        for index in range(35)
    )
    xml = (
        f"<hierarchy>{generic_controls}"
        '<node text="摩卡拿铁" bounds="[113,1933][359,2039]" />'
        '<node id="add-latte" bounds="[299,2041][359,2101]" '
        'clickable="true" /></hierarchy>'
    )

    projection = project_ui(xml, "点一杯拿铁")

    assert projection.selected_count == 30
    assert "[goal_controls]" in projection.text
    assert '"i":"add-latte"' in projection.text
    assert projection.text.index('"t":"摩卡拿铁"') < projection.text.index(
        '"i":"add-latte"'
    )


def test_vlm_turn_keeps_visual_evidence_when_xml_matches_goal() -> None:
    request = build_model_turn_request(
        goal="点一杯拿铁咖啡",
        model="test-model",
        state={
            "xml": _merchant_menu_xml(),
            "image_base64": "current-screen",
            "display": {"width": 1080, "height": 2376},
        },
        max_steps=20,
        turn_index=6,
    )

    content = request["messages"][1]["content"]
    assert [item["type"] for item in content] == ["text", "image_url"]
    assert content[1]["image_url"]["url"] == (
        "data:image/jpeg;base64,current-screen"
    )
    assert "[global_controls]" in content[0]["text"]
    assert content[0]["text"].index('"d":"搜索"') < content[0]["text"].index(
        '"t":"拿铁咖啡 0"'
    )
