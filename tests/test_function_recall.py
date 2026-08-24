from __future__ import annotations

import asyncio

from omniflow import Action, ActionResult, Function, Observation, OmniFlow, ToolCall
from omniflow.core.config import OmniFlowConfig, PluginSet
from omniflow.core.model import FunctionStep, TransferResult
from omniflow.functions.artifact import FUNCTION_ARTIFACT_VERSION
from omniflow.functions.recall import (
    FUNCTION_PAGE_SIMILARITY_THRESHOLD,
    GOAL_LEXICAL_WEIGHT,
    PAGE_SIMILARITY_WEIGHT,
    recall_functions,
)
from omniflow.functions.store import FunctionStore


def _page(
    text: str,
    resource_id: str,
    *,
    package: str = "com.example",
    variant: str = "form",
) -> Observation:
    if variant == "camera":
        xml = (
            '<hierarchy><node package="%s" class="android.widget.FrameLayout" '
            'bounds="[0,0][1000,1000]"><node package="%s" '
            'class="android.view.SurfaceView" bounds="[0,0][1000,850]" />'
            '<node package="%s" class="android.widget.ImageButton" '
            'content-desc="%s" resource-id="%s" bounds="[420,860][580,1000]" '
            'clickable="true" /></node></hierarchy>'
        ) % (package, package, package, text, resource_id)
    else:
        xml = (
            '<hierarchy><node package="%s" class="android.widget.LinearLayout" '
            'bounds="[0,0][1000,1000]"><node package="%s" '
            'class="android.widget.TextView" text="%s" '
            'bounds="[50,50][950,180]" /><node package="%s" '
            'class="android.widget.EditText" resource-id="%s" '
            'bounds="[50,240][950,360]" editable="true" />'
            '<node package="%s" class="android.widget.Button" text="Continue" '
            'bounds="[650,800][950,920]" clickable="true" /></node></hierarchy>'
        ) % (package, package, text, package, resource_id, package)
    return Observation(
        xml=xml,
        package_name=package,
        activity_name="MainActivity",
        extra={"display": {"width": 1000, "height": 1000}},
    )


def _function(
    function_id: str,
    name: str,
    description: str,
    source_state_id: str,
    *,
    action: Action | None = None,
) -> Function:
    return Function(
        function_id=function_id,
        name=name,
        description=description,
        steps=(
            FunctionStep(
                step_index=0,
                source_state_id=source_state_id,
                action=action or Action("click", {"x": 500.0, "y": 500.0}),
            ),
        ),
        schema_version=FUNCTION_ARTIFACT_VERSION,
        input_schema={
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        agent_visible=True,
    )


def test_function_recall_uses_one_lexical_and_page_score() -> None:
    current = _page("Account details", "account_form")
    matching = _function(
        "submit_current_form",
        "Tap continue",
        "Advance from the current form.",
        "account_page",
    )
    lexical_but_wrong_page = _function(
        "finish_account_setup",
        "Finish account setup",
        "Finish account setup from this screen.",
        "camera_page",
    )

    async def transfer(
        action: Action,
        _observation: Observation,
        source_state: Observation | None,
    ) -> TransferResult:
        if source_state is not None and "Camera shutter" in str(source_state.xml):
            return TransferResult(None, reason="omnitransfer_null_target")
        return TransferResult(
            Action(action.tool, {"x": 800.0, "y": 860.0}),
            reason="omnitransfer_unified_association_v1",
            detail={"absolute_contextual_confidence": 0.95},
        )

    result = asyncio.run(
        recall_functions(
            "Finish account setup",
            observation=current,
            functions=(matching, lexical_but_wrong_page),
            source_states={
                "account_page": current,
                "camera_page": _page(
                    "Camera shutter",
                    "camera_shutter",
                    variant="camera",
                ),
            },
            transfer=transfer,
        )
    )

    assert [function.id for function in result.functions] == [matching.id]
    decisions = {
        item["function_id"]: item for item in result.audit["decisions"]
    }
    assert decisions[matching.id]["goal_lexical_score"] == 0.0
    assert decisions[matching.id]["page_similarity"] == 1.0
    assert decisions[matching.id]["score"] == PAGE_SIMILARITY_WEIGHT
    assert decisions[matching.id]["page_match"] is True
    assert decisions[lexical_but_wrong_page.id]["score"] > decisions[matching.id][
        "score"
    ]
    assert decisions[lexical_but_wrong_page.id]["page_match"] is False
    assert decisions[lexical_but_wrong_page.id]["rejection_reason"] == (
        "omnitransfer_null_target"
    )
    assert result.audit["ranking_weights"] == {
        "page_similarity": PAGE_SIMILARITY_WEIGHT,
        "goal_lexical": GOAL_LEXICAL_WEIGHT,
    }
    assert result.audit["encoder"]["dimension"] == 1024
    assert result.audit["encoder"]["architecture"] == (
        "omnitransfer_point_conditioned_sparse_graph_v10"
    )
    assert result.audit["page_similarity_threshold"] == 0.8
    assert FUNCTION_PAGE_SIMILARITY_THRESHOLD == 0.8
    assert GOAL_LEXICAL_WEIGHT > PAGE_SIMILARITY_WEIGHT


def test_recall_uses_page_embedding_only_for_ranking_and_hard_gates_on_transfer() -> None:
    current = _page("Account details", "account_form")
    function = _function(
        "finish_account_setup",
        "Finish account setup",
        "Finish account setup from this screen.",
        "camera_page",
        action=Action("click", {"x": 500.0, "y": 930.0}),
    )

    async def transfer(
        action: Action,
        _observation: Observation,
        _source_state: Observation | None,
    ) -> TransferResult:
        return TransferResult(
            Action(action.tool, {"x": 800.0, "y": 860.0}),
            reason="omnitransfer_unified_association_v1",
            detail={"absolute_contextual_confidence": 0.93},
        )

    result = asyncio.run(
        recall_functions(
            "Finish account setup",
            observation=current,
            functions=(function,),
            source_states={
                "camera_page": _page(
                    "Camera shutter",
                    "camera_shutter",
                    variant="camera",
                ),
            },
            transfer=transfer,
        )
    )

    assert result.functions == (function,)
    decision = result.audit["decisions"][0]
    assert decision["page_match"] is False
    assert decision["mapping_confidence"] == 0.93
    assert decision["selected"] is True


def test_recall_hides_function_when_first_step_mapping_is_below_point_eight() -> None:
    current = _page("Account details", "account_form")
    function = _function(
        "continue_form",
        "Continue form",
        "Continue the current form.",
        "source_page",
    )

    async def transfer(
        action: Action,
        _observation: Observation,
        _source_state: Observation | None,
    ) -> TransferResult:
        return TransferResult(
            Action(action.tool, {"x": 800.0, "y": 860.0}),
            reason="omnitransfer_unified_association_v1",
            detail={"absolute_contextual_confidence": 0.79},
        )

    result = asyncio.run(
        recall_functions(
            "Continue form",
            observation=current,
            functions=(function,),
            source_states={"source_page": current},
            transfer=transfer,
        )
    )

    assert result.functions == ()
    decision = result.audit["decisions"][0]
    assert decision["mapping_confidence"] == 0.79
    assert decision["rejection_reason"] == "omnitransfer_low_confidence"


def test_package_mismatch_is_diagnostic_and_does_not_override_transfer() -> None:
    source = _page("Account details", "account_form", package="com.example")
    current = _page("Account details", "account_form", package="com.other")
    function = _function(
        "submit_current_form",
        "Tap continue",
        "Advance from the current form.",
        "source_page",
    )

    async def transfer(
        action: Action,
        _observation: Observation,
        _source_state: Observation | None,
    ) -> TransferResult:
        return TransferResult(
            Action(action.tool, {"x": 800.0, "y": 860.0}),
            detail={"absolute_contextual_confidence": 0.95},
        )

    result = asyncio.run(
        recall_functions(
            "Continue",
            observation=Observation(
                xml=current.xml,
                extra={"display": {"width": 1000, "height": 1000}},
            ),
            functions=(function,),
            source_states={"source_page": Observation(xml=source.xml)},
            transfer=transfer,
        )
    )

    assert result.functions == (function,)
    decision = result.audit["decisions"][0]
    assert decision["page_match"] is False
    assert decision["mapping_confidence"] == 0.95


def test_open_app_function_uses_default_entry_page_score() -> None:
    current = _page("Bluetooth settings", "bluetooth_switch")
    open_app_function = _function(
        "z_open_settings",
        "Continue",
        "Continue from this page.",
        "unrelated_source_page",
        action=Action("open_app", {"package_name": "com.android.settings"}),
    )
    click_function = _function(
        "a_click_continue",
        "Continue",
        "Continue from this page.",
        "other_page",
    )

    result = asyncio.run(
        recall_functions(
            "Continue",
            observation=current,
            functions=(click_function, open_app_function),
            source_states={
                "unrelated_source_page": _page(
                    "Camera", "shutter", variant="camera"
                ),
                "other_page": _page("Camera", "shutter", variant="camera"),
            },
            limit=1,
        )
    )

    assert result.functions == (open_app_function,)
    decisions = {item["function_id"]: item for item in result.audit["decisions"]}
    assert decisions[open_app_function.id]["page_similarity"] == 1.0
    assert decisions[open_app_function.id]["observed_page_similarity"] < 1.0
    assert decisions[open_app_function.id]["entry_page_override"] == "open_app"
    assert decisions[open_app_function.id]["score"] > decisions[click_function.id][
        "score"
    ]


def test_recall_can_select_local_function_after_global_is_excluded() -> None:
    current = _page("Account details", "account_form")
    global_function = _function(
        "complete_task",
        "Complete task",
        "Complete the whole task.",
        "task_start",
        action=Action("open_app", {"package_name": "com.example"}),
    )
    local_function = _function(
        "submit_current_form",
        "Submit current form",
        "Submit the visible form.",
        "current_form",
    )

    async def transfer(
        action: Action,
        _observation: Observation,
        _source_state: Observation | None,
    ) -> TransferResult:
        return TransferResult(
            Action(action.tool, {"x": 800.0, "y": 860.0}),
            detail={"absolute_contextual_confidence": 0.95},
        )

    result = asyncio.run(
        recall_functions(
            "Complete the task",
            observation=current,
            functions=(global_function, local_function),
            source_states={
                "task_start": _page("Start", "start"),
                "current_form": current,
            },
            transfer=transfer,
            exclude_function_ids=frozenset({global_function.id}),
        )
    )

    assert result.functions == (local_function,)


def test_missing_page_evidence_prevents_recall() -> None:
    function = _function(
        "tap_continue",
        "Tap continue",
        "Advance from the current form.",
        "source_page",
    )

    result = asyncio.run(
        recall_functions(
            "Continue",
            observation=Observation(package_name="com.example"),
            functions=(function,),
            source_states={"source_page": _page("Form", "form")},
        )
    )

    assert result.functions == ()
    decision = result.audit["decisions"][0]
    assert decision["page_similarity"] == 0.0
    assert decision["score"] == GOAL_LEXICAL_WEIGHT * decision["goal_lexical_score"]
    assert decision["page_match"] is False
    assert decision["rejection_reason"] == "function_transfer_unavailable"


def test_recall_excludes_functions_without_source_page_evidence() -> None:
    function = _function(
        "tap_settings_control",
        "Turn bluetooth on",
        "Use the Settings control to turn bluetooth on.",
        "settings_page",
    )

    second = _function("z_second", "Unrelated", "No shared terms.", "missing")
    first = _function("a_first", "Different", "Still unrelated.", "missing")
    async def transfer(
        _action: Action,
        _observation: Observation,
        source_state: Observation | None,
    ) -> TransferResult:
        assert source_state is None
        return TransferResult(None, reason="omnitransfer_source_state_missing")

    result = asyncio.run(
        recall_functions(
            "Turn bluetooth on",
            observation=Observation(),
            functions=(second, first, function),
            source_states={"settings_page": None, "missing": None},
            limit=2,
            transfer=transfer,
        )
    )

    assert result.functions == ()
    assert {
        item["rejection_reason"] for item in result.audit["decisions"]
    } == {"omnitransfer_source_state_missing"}


class _CheckerRecoveryHost:
    def __init__(self) -> None:
        self.package_name = "com.android.launcher"
        self.actions: list[Action] = []

    def observe(self, **_kwargs: object) -> Observation:
        if self.package_name == "com.android.settings":
            return _page(
                "Bluetooth",
                "bluetooth_switch",
                package=self.package_name,
            )
        return _page(
            "Home",
            "launcher",
            package=self.package_name,
            variant="camera",
        )

    def act(self, action: Action) -> ActionResult:
        self.actions.append(action)
        if action.tool == "open_app":
            self.package_name = str(action.args["package_name"])
        return ActionResult(True)

    def get_state(self, source_state_id: str) -> Observation | None:
        if source_state_id != "settings_page":
            return None
        return _page(
            "Bluetooth",
            "bluetooth_switch",
            package="com.android.settings",
        )


class _SelectThenFinishPlanner:
    def __init__(self, function_id: str) -> None:
        self.function_id = function_id
        self.calls = 0

    def one_step_tool_call(
        self,
        _goal: str,
        _observation: Observation,
        functions: tuple[Function, ...],
        _installed_apps: dict[str, str],
    ) -> ToolCall:
        self.calls += 1
        if self.calls == 1:
            assert functions == ()
            return ToolCall(
                "open_app", {"package_name": "com.android.settings"}
            )
        if self.calls == 2:
            assert self.function_id in {function.id for function in functions}
            return ToolCall(self.function_id, {})
        return ToolCall("finished", {"content": ""})


def test_planner_navigates_to_function_page_before_recall(
    tmp_path,
) -> None:
    function = _function(
        "tap_settings_control",
        "Turn bluetooth on",
        "Use the Settings control to turn bluetooth on.",
        "settings_page",
    )
    store = FunctionStore(tmp_path / "store.json")
    store.put_function(function)
    host = _CheckerRecoveryHost()

    def transfer(
        action: Action,
        observation: Observation,
        _source_state: Observation | None,
    ) -> TransferResult:
        if observation.package_name != "com.android.settings":
            return TransferResult(None, reason="omnitransfer_null_target")
        return TransferResult(
            Action(action.tool, {"x": 800.0, "y": 860.0}),
            reason="test_target_match",
            detail={"absolute_contextual_confidence": 0.95},
        )

    flow = OmniFlow(
        store.path,
        host=host,
        planner=_SelectThenFinishPlanner(function.id),
        installed_apps={"Settings": "com.android.settings"},
        config=OmniFlowConfig(plugins=PluginSet(transfer=transfer)),
    )

    result = flow.run("Turn bluetooth on")

    assert result.success is True
    assert [action.tool for action in host.actions] == ["open_app", "click"]
    assert host.actions[0].args == {"package_name": "com.android.settings"}


class _PageChangingHost:
    def __init__(self) -> None:
        self.page = "first"

    def observe(self, **_kwargs: object) -> Observation:
        if self.page == "first":
            return _page("First page", "first_page")
        return _page("Second page", "second_page", variant="camera")

    def act(self, _action: Action) -> ActionResult:
        self.page = "second"
        return ActionResult(True)

    def get_state(self, source_state_id: str) -> Observation | None:
        if source_state_id == "source_first":
            return _page("First page", "first_page")
        if source_state_id == "source_second":
            return _page("Second page", "second_page", variant="camera")
        return None


class _ChangingPagePlanner:
    def __init__(self) -> None:
        self.visible: list[tuple[str, ...]] = []

    def one_step_tool_call(
        self,
        _goal: str,
        _observation: Observation,
        functions: tuple[Function, ...],
        _installed_apps: dict[str, str],
    ) -> ToolCall:
        self.visible.append(tuple(function.id for function in functions))
        if len(self.visible) == 1:
            return ToolCall("click", {"x": 500.0, "y": 500.0})
        return ToolCall("finished", {"content": ""})


def test_runtime_recalls_again_after_page_changes(tmp_path) -> None:
    store = FunctionStore(tmp_path / "store.json")
    store.put_function(
        _function(
            "first_page_action",
            "Use first page control",
            "Operate the visible first-page control.",
            "source_first",
        )
    )
    store.put_function(
        _function(
            "second_page_action",
            "Use second page control",
            "Operate the visible second-page control.",
            "source_second",
        )
    )
    planner = _ChangingPagePlanner()
    def transfer(
        action: Action,
        observation: Observation,
        source_state: Observation | None,
    ) -> TransferResult:
        current_is_camera = "SurfaceView" in str(observation.xml)
        source_is_camera = "SurfaceView" in str(
            source_state.xml if source_state is not None else ""
        )
        if current_is_camera != source_is_camera:
            return TransferResult(None, reason="omnitransfer_null_target")
        point = (
            {"x": 500.0, "y": 930.0}
            if current_is_camera
            else {"x": 800.0, "y": 860.0}
        )
        return TransferResult(
            Action(action.tool, point),
            reason="omnitransfer_unified_association_v1",
            detail={"absolute_contextual_confidence": 0.95},
        )

    flow = OmniFlow(
        store.path,
        host=_PageChangingHost(),
        planner=planner,
        config=OmniFlowConfig(plugins=PluginSet(transfer=transfer)),
    )

    result = flow.run("Complete the multi-page task")

    assert result.success is True
    assert planner.visible == [
        ("first_page_action",),
        ("second_page_action",),
    ]
    assert [
        event["candidate_function_ids"]
        for event in result.detail["function_resolution"]["recall"]["events"]
    ] == [
        ["first_page_action"],
        ["second_page_action"],
    ]
