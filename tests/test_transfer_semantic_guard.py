from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

from omniflow import Action, Observation
from omniflow.runtime import execution
from omniflow.transfer import runtime as transfer_runtime
from omniflow.transfer.runtime import audit_transfer_action_sources, load_omnitransfer

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

GENERIC_SOURCE_ROW_XML = """\
<hierarchy width="720" height="1280">
  <node package="com.android.settings" bounds="[0,0][720,1280]">
    <node id="15" clickable="true" focusable="true"
          bounds="[0,406][720,562]">
      <node text="Internet" bounds="[144,420][380,458]" />
      <node text="Networks available" bounds="[144,492][380,530]" />
    </node>
  </node>
</hierarchy>
"""

FOLD_NETWORK_XML = """\
<hierarchy width="2208" height="1840">
  <node package="com.android.settings" bounds="[0,0][2208,1840]">
    <node class="android.widget.LinearLayout" clickable="true"
          bounds="[802,544][2208,750]">
      <node class="android.widget.TextView" resource-id="android:id/title"
            text="Internet" bounds="[991,586][1171,657]" />
      <node class="android.widget.TextView" resource-id="android:id/summary"
            text="Networks available" bounds="[991,657][1304,708]" />
    </node>
    <node class="android.widget.LinearLayout" clickable="true"
          bounds="[802,750][2208,956]">
      <node class="android.widget.TextView" resource-id="android:id/title"
            text="Calls &amp; SMS" bounds="[991,792][1275,863]" />
    </node>
  </node>
</hierarchy>
"""

COMPACT_ANDROIDWORLD_XML = """\
<hierarchy>
  <node id="0" bounds="[0,0][720,1280]">
    <node id="11" clickable="true" focusable="true"
          bounds="[222,1040][498,1136]" />
  </node>
</hierarchy>
"""


def test_omnitransfer_loads_only_from_configured_canonical_root(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = tmp_path / "OmniTransfer"
    package = root / "src" / "omnitransfer"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        "CANONICAL_TEST_MARKER = 'configured-root'\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("OMNITRANSFER_ROOT", str(root))
    try:
        module = load_omnitransfer()
        assert module.CANONICAL_TEST_MARKER == "configured-root"
        assert Path(module.__file__).resolve().is_relative_to(package.resolve())
    finally:
        for name in tuple(sys.modules):
            if name == "omnitransfer" or name.startswith("omnitransfer."):
                sys.modules.pop(name, None)


def test_source_audit_accepts_native_compact_androidworld_semantics(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        transfer_runtime,
        "transfer_action",
        lambda **_kwargs: {
            "mapped": True,
            "src_element": {
                "bounds": [222, 1040, 498, 1136],
            },
        },
    )
    function = SimpleNamespace(
        id="start_recorder",
        steps=(
            SimpleNamespace(
                step_index=0,
                source_state_id="source-state",
                action=Action("click", {"x": 500, "y": 851.5625}),
            ),
        ),
    )

    audit = audit_transfer_action_sources(
        (function,),
        {
            "source-state": {
                "state_id": "source-state",
                "xml": COMPACT_ANDROIDWORLD_XML,
                "display": {"width": 720, "height": 1280},
            }
        },
    )

    assert audit["source_target_audit_complete"] is True
    assert audit["source_target_count"] == 1


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


def test_transfer_recovers_generic_source_row_title_and_rejects_wrong_fold_row(
    monkeypatch,
) -> None:
    request = {}

    def transfer_action(**kwargs):
        request.update(kwargs)
        return {
            "mapped": True,
            "mapping_mode": "mutual_graph_matcher_no_null_v3",
            "new_x": 1505.0,
            "new_y": 853.0,
            "target_bbox": [802.0, 750.0, 2208.0, 956.0],
            "score": 0.9699,
            "margin": 0.1,
        }

    monkeypatch.setattr(execution, "transfer_action", transfer_action)

    result = execution.default_transfer(
        Action(
            "click",
            {
                "x": 363.88888888888886,
                "y": 399.21875,
            },
        ),
        Observation(
            xml=FOLD_NETWORK_XML,
            package_name="com.android.settings",
            extra={"display": {"width": 2208, "height": 1840}},
        ),
        Observation(
            xml=GENERIC_SOURCE_ROW_XML,
            package_name="com.android.settings",
            extra={"display": {"width": 720, "height": 1280}},
        ),
    )

    assert result.action is None
    assert result.reason == "omnitransfer_semantic_conflict"
    assert result.detail["source_title"] == "internet"
    assert result.detail["target_title"] == "calls & sms"
    assert request["source_element"] == {"text": "internet"}


def test_transfer_rejects_incomplete_fold_graph_before_matching(monkeypatch) -> None:
    def transfer_action(**_kwargs):
        raise AssertionError("incomplete target graph must not reach OmniTransfer")

    monkeypatch.setattr(execution, "transfer_action", transfer_action)

    result = execution.default_transfer(
        Action(
            "click",
            {
                "x": 363.88888888888886,
                "y": 399.21875,
            },
        ),
        Observation(
            xml="""\
<hierarchy width="2208" height="1840" bounds="[0,0][2208,1840]">
  <node package="com.android.settings" bounds="[0,0][802,1840]" />
</hierarchy>
""",
            package_name="com.android.settings",
            extra={
                "display": {"width": 2208, "height": 1840},
                "ui_graph_source": "uiautomator_partial",
            },
        ),
        Observation(
            xml=GENERIC_SOURCE_ROW_XML,
            package_name="com.android.settings",
            extra={"display": {"width": 720, "height": 1280}},
        ),
    )

    assert result.action is None
    assert result.reason == "omnitransfer_target_graph_incomplete"


def test_transfer_rejects_full_screen_root_candidate(monkeypatch) -> None:
    monkeypatch.setattr(
        execution,
        "transfer_action",
        lambda **_kwargs: {
            "mapped": True,
            "mapping_mode": "mutual_graph_matcher_no_null_v3",
            "new_x": 1104.0,
            "new_y": 920.0,
            "target_bbox": [0.0, 0.0, 2208.0, 1840.0],
            "score": 0.99999,
            "margin": 0.98,
        },
    )

    result = execution.default_transfer(
        Action(
            "click",
            {
                "x": 363.88888888888886,
                "y": 399.21875,
            },
        ),
        Observation(
            xml=FOLD_NETWORK_XML,
            package_name="com.android.settings",
            extra={"display": {"width": 2208, "height": 1840}},
        ),
        Observation(
            xml=GENERIC_SOURCE_ROW_XML,
            package_name="com.android.settings",
            extra={"display": {"width": 720, "height": 1280}},
        ),
    )

    assert result.action is None
    assert result.reason == "omnitransfer_invalid_root_candidate"


def test_transfer_falls_back_when_source_title_is_missing_from_target(
    monkeypatch,
) -> None:
    def transfer_action(**_kwargs):
        raise AssertionError("missing source semantics must not use structural match")

    monkeypatch.setattr(execution, "transfer_action", transfer_action)
    calls_only_xml = FOLD_NETWORK_XML.replace(
        """\
    <node class="android.widget.LinearLayout" clickable="true"
          bounds="[802,544][2208,750]">
      <node class="android.widget.TextView" resource-id="android:id/title"
            text="Internet" bounds="[991,586][1171,657]" />
      <node class="android.widget.TextView" resource-id="android:id/summary"
            text="Networks available" bounds="[991,657][1304,708]" />
    </node>
""",
        "",
    )

    result = execution.default_transfer(
        Action(
            "click",
            {
                "x": 363.88888888888886,
                "y": 399.21875,
            },
        ),
        Observation(
            xml=calls_only_xml,
            package_name="com.android.settings",
            extra={"display": {"width": 2208, "height": 1840}},
        ),
        Observation(
            xml=GENERIC_SOURCE_ROW_XML,
            package_name="com.android.settings",
            extra={"display": {"width": 720, "height": 1280}},
        ),
    )

    assert result.action is None
    assert result.reason == "omnitransfer_target_semantic_missing"
    assert result.detail["source_title"] == "internet"
