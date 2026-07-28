from __future__ import annotations

from omniflow import Action, Observation
from omniflow.runtime import execution

SOURCE_XML = """\
<hierarchy width="720" height="1280">
  <node package="com.android.settings" bounds="[0,0][720,1280]">
    <node class="android.widget.LinearLayout" clickable="true"
          bounds="[0,490][720,608]">
      <node class="android.widget.TextView" resource-id="android:id/title"
            text="Bluetooth" bounds="[20,500][500,570]" />
    </node>
  </node>
</hierarchy>
"""

TARGET_XML = """\
<hierarchy width="2208" height="1840">
  <node package="com.android.settings" bounds="[802,0][2208,1840]">
    <node class="android.widget.LinearLayout" clickable="true"
          bounds="[802,544][2208,699]">
      <node class="android.widget.TextView" resource-id="android:id/title"
            text="Bluetooth" bounds="[900,570][1400,640]" />
    </node>
    <node class="android.widget.LinearLayout" clickable="true"
          bounds="[802,699][2208,905]">
      <node class="android.widget.TextView" resource-id="android:id/title"
            text="Cast" bounds="[900,730][1400,800]" />
    </node>
  </node>
</hierarchy>
"""


def test_transfer_rejects_semantically_conflicting_row_match(monkeypatch) -> None:
    monkeypatch.setattr(
        execution,
        "transfer_action",
        lambda **_kwargs: {
            "mapped": True,
            "mapping_mode": "mutual_graph_matcher_no_null_v3",
            "new_x": 1505.0,
            "new_y": 802.0,
            "target_bbox": [802.0, 699.0, 2208.0, 905.0],
            "score": 1.0,
            "margin": 0.1,
        },
    )

    result = execution.default_transfer(
        Action("click", {"x": 500.0, "y": 428.90625}),
        Observation(
            xml=TARGET_XML,
            package_name="com.android.settings",
            extra={"display": {"width": 2208, "height": 1840}},
        ),
        Observation(
            xml=SOURCE_XML,
            package_name="com.android.settings",
            extra={"display": {"width": 720, "height": 1280}},
        ),
    )

    assert result.action is None
    assert result.reason == "omnitransfer_semantic_conflict"
    assert result.detail["source_title"] == "bluetooth"
    assert result.detail["target_title"] == "cast"


def test_transfer_keeps_semantically_consistent_row_match(monkeypatch) -> None:
    request = {}

    def transfer_action(**kwargs):
        request.update(kwargs)
        return {
            "mapped": True,
            "mapping_mode": "mutual_graph_matcher_no_null_v3",
            "new_x": 1505.0,
            "new_y": 621.5,
            "target_bbox": [802.0, 544.0, 2208.0, 699.0],
            "score": 1.0,
            "margin": 0.1,
        }

    monkeypatch.setattr(execution, "transfer_action", transfer_action)

    result = execution.default_transfer(
        Action("click", {"x": 500.0, "y": 428.90625}),
        Observation(
            xml=TARGET_XML,
            package_name="com.android.settings",
            extra={"display": {"width": 2208, "height": 1840}},
        ),
        Observation(
            xml=SOURCE_XML,
            package_name="com.android.settings",
            extra={"display": {"width": 720, "height": 1280}},
        ),
    )

    assert result.action is not None
    assert result.action.tool == "click"
    assert result.action.args["x"] == 681.6123188405797
    assert abs(result.action.args["y"] - 337.77173913043475) < 1e-9
    assert request["source_element"] == {"text": "bluetooth"}
