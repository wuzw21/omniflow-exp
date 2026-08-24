from __future__ import annotations

import asyncio

from omniflow import Action, ActionResult, Function, Observation, PluginSet
from omniflow.core.model import FunctionStep, StepResult, TransferResult
from omniflow.runtime import core, execution


class RecordingHost:
    def __init__(self, after: Observation) -> None:
        self.after = after
        self.actions: list[Action] = []
        self.observe_requests: list[dict[str, object]] = []

    def act(self, action: Action) -> ActionResult:
        self.actions.append(action)
        return ActionResult(True)

    def observe(self, **kwargs: object) -> Observation:
        self.observe_requests.append(dict(kwargs))
        return self.after


def test_core_executes_one_transferred_action_and_observes_once(monkeypatch) -> None:
    settle_calls: list[float] = []

    async def settle(seconds: float) -> None:
        settle_calls.append(seconds)

    monkeypatch.setattr(core.asyncio, "sleep", settle)
    before = Observation(
        xml=(
            '<hierarchy><node class="android.widget.Button" '
            'bounds="[200,300][400,500]" clickable="true" enabled="true" />'
            "</hierarchy>"
        ),
        package_name="com.example",
        extra={"display": {"width": 1000, "height": 1000}},
    )
    source = Observation(xml="<source/>", package_name="com.example")
    after = Observation(xml="<after/>", package_name="com.example")
    host = RecordingHost(after)
    transfer_calls: list[tuple[Action, Observation, Observation]] = []
    recorded = Action("click", {"x": 100, "y": 200})
    mapped = Action("click", {"x": 300, "y": 400})

    async def transfer(
        action: Action,
        observation: Observation,
        source_state: Observation,
    ) -> TransferResult:
        transfer_calls.append((action, observation, source_state))
        return TransferResult(
            mapped,
            reason="omnitransfer_mapped",
            detail={"score": 1.0},
        )

    result = asyncio.run(
        core.execute_action(
            recorded,
            observation=before,
            host=host,
            plugins=PluginSet(transfer=transfer),
            source_state=source,
        )
    )

    assert result.success is True
    assert result.action == mapped
    assert result.before == before
    assert result.after == after
    assert result.detail == {"score": 1.0}
    assert transfer_calls == [(recorded, before, source)]
    assert host.actions == [mapped]
    assert settle_calls == [1.0]
    assert host.observe_requests == [
        {"xml": True, "screenshot": True, "app_info": True}
    ]


def test_core_reports_transfer_failure_without_dispatch(monkeypatch) -> None:
    monkeypatch.setattr(core, "_ACTION_SETTLE_SECONDS", 0.0)
    before = Observation(xml="<before/>", package_name="com.example")
    source = Observation(xml="<source/>", package_name="com.example")
    host = RecordingHost(Observation(xml="<after/>"))

    async def transfer(
        _action: Action,
        _observation: Observation,
        _source_state: Observation,
    ) -> TransferResult:
        return TransferResult(None, reason="omnitransfer_failed")

    result = asyncio.run(
        core.execute_action(
            Action("click", {"x": 100, "y": 200}),
            observation=before,
            host=host,
            plugins=PluginSet(transfer=transfer),
            source_state=source,
        )
    )

    assert result.success is False
    assert result.error == "omnitransfer_failed"
    assert host.actions == []
    assert host.observe_requests == []


def test_core_blocks_low_confidence_transfer_without_dispatch(monkeypatch) -> None:
    monkeypatch.setattr(core, "_ACTION_SETTLE_SECONDS", 0.0)
    before = Observation(
        xml=(
            '<hierarchy><node class="android.widget.Button" text="Click Me" '
            'bounds="[100,100][300,200]" clickable="true" enabled="true" />'
            "</hierarchy>"
        ),
        package_name="com.example",
        extra={"display": {"width": 400, "height": 800}},
    )
    source = Observation(xml="<hierarchy />", package_name="com.example")
    host = RecordingHost(Observation(xml="<after/>"))

    async def transfer(
        _action: Action,
        _observation: Observation,
        _source_state: Observation,
    ) -> TransferResult:
        return TransferResult(
            Action("click", {"x": 500.0, "y": 187.5}),
            reason="omnitransfer_mapped",
            detail={"score": 0.79},
        )

    result = asyncio.run(
        core.execute_action(
            Action("click", {"x": 200, "y": 150}),
            observation=before,
            host=host,
            plugins=PluginSet(transfer=transfer),
            source_state=source,
        )
    )

    assert result.success is False
    assert result.error == "omnitransfer_low_confidence"
    assert host.actions == []
    assert host.observe_requests == []


def test_core_blocks_transfer_to_non_executable_target(monkeypatch) -> None:
    monkeypatch.setattr(core, "_ACTION_SETTLE_SECONDS", 0.0)
    before = Observation(
        xml=(
            '<hierarchy><node class="android.widget.TextView" text="Counter" '
            'bounds="[100,100][300,200]" clickable="false" enabled="true" />'
            "</hierarchy>"
        ),
        package_name="com.example",
        extra={"display": {"width": 400, "height": 800}},
    )
    host = RecordingHost(Observation(xml="<after/>"))

    async def transfer(*_args: object) -> TransferResult:
        return TransferResult(
            Action("click", {"x": 500.0, "y": 187.5}),
            reason="omnitransfer_mapped",
            detail={"score": 0.99},
        )

    result = asyncio.run(
        core.execute_action(
            Action("click", {"x": 200, "y": 150}),
            observation=before,
            host=host,
            plugins=PluginSet(transfer=transfer),
            source_state=Observation(xml="<hierarchy />"),
        )
    )

    assert result.success is False
    assert result.error == "omnitransfer_target_not_executable"
    assert host.actions == []


def test_core_blocks_clickable_transfer_with_mismatched_target_semantics(
    monkeypatch,
) -> None:
    monkeypatch.setattr(core, "_ACTION_SETTLE_SECONDS", 0.0)
    before = Observation(
        xml=(
            '<hierarchy><node class="android.widget.Button" '
            'content-desc="Shutter" bounds="[100,100][300,200]" '
            'clickable="true" enabled="true" /></hierarchy>'
        ),
        package_name="com.example",
        extra={"display": {"width": 400, "height": 800}},
    )
    host = RecordingHost(Observation(xml="<after/>"))

    async def transfer(*_args: object) -> TransferResult:
        return TransferResult(
            Action("click", {"x": 500.0, "y": 187.5}),
            reason="omnitransfer_mapped",
            detail={
                "absolute_contextual_confidence": 1.0,
                "source": {"content_desc": "Toggle mode list"},
                "candidates": [
                    {
                        "content_desc": "Shutter",
                        "resource_id": "camera:id/shutter",
                    }
                ],
            },
        )

    result = asyncio.run(
        core.execute_action(
            Action("click", {"x": 200, "y": 150}),
            observation=before,
            host=host,
            plugins=PluginSet(transfer=transfer),
            source_state=Observation(xml="<hierarchy />"),
        )
    )

    assert result.success is False
    assert result.error == "omnitransfer_target_semantics_mismatch"
    assert host.actions == []


def test_core_accepts_semantic_child_candidate_with_same_mapped_bounds(
    monkeypatch,
) -> None:
    monkeypatch.setattr(core, "_ACTION_SETTLE_SECONDS", 0.0)
    before = Observation(
        xml=(
            '<hierarchy><node class="android.widget.Button" '
            'content-desc="Search" bounds="[100,100][300,200]" '
            'clickable="true" enabled="true" /></hierarchy>'
        ),
        package_name="com.example",
        extra={"display": {"width": 400, "height": 800}},
    )
    host = RecordingHost(Observation(xml="<after />"))

    async def transfer(*_args: object) -> TransferResult:
        return TransferResult(
            Action("click", {"x": 500.0, "y": 187.5}),
            reason="omnitransfer_mapped",
            detail={
                "absolute_contextual_confidence": 1.0,
                "source": {"class": "android.widget.Button", "content_desc": "Search"},
                "target": {"bounds": [100.0, 100.0, 300.0, 200.0]},
                "candidates": [
                    {
                        "class": "android.view.ViewGroup",
                        "bounds": [100.0, 100.0, 300.0, 200.0],
                    },
                    {
                        "class": "android.widget.Button",
                        "content_desc": "Search",
                        "bounds": [100.0, 100.0, 300.0, 200.0],
                    },
                ],
            },
        )

    result = asyncio.run(
        core.execute_action(
            Action("click", {"x": 200, "y": 150}),
            observation=before,
            host=host,
            plugins=PluginSet(transfer=transfer),
            source_state=Observation(xml="<hierarchy />"),
        )
    )

    assert result.success is True
    assert host.actions


def test_core_has_no_open_app_catalog_gate(monkeypatch) -> None:
    monkeypatch.setattr(core, "_ACTION_SETTLE_SECONDS", 0.0)
    before = Observation(package_name="cn.com.omnimind.bot.debug")
    after = Observation(package_name="com.sankuai.meituan")
    host = RecordingHost(after)
    action = Action("open_app", {"package_name": "com.sankuai.meituan"})

    result = asyncio.run(
        core.execute_action(
            action,
            observation=before,
            host=host,
            plugins=PluginSet(),
        )
    )

    assert result.success is True
    assert host.actions == [action]


def test_function_execution_counts_all_auxiliary_actions(monkeypatch) -> None:
    before = Observation(
        xml=(
            '<hierarchy><node package="com.example" '
            'class="android.widget.FrameLayout" bounds="[0,0][1000,1000]" />'
            "</hierarchy>"
        ),
        package_name="com.example",
        extra={"state_id": "before"},
    )
    host = RecordingHost(before)
    function = Function(
        function_id="two_steps",
        name="Two steps",
        description="Two semantic steps with one auxiliary action each.",
        steps=(
            FunctionStep(0, Action("wait", {"duration_ms": 0}), "source"),
            FunctionStep(1, Action("wait", {"duration_ms": 0}), "source"),
        ),
    )

    async def execute_with_auxiliary(action, **kwargs):
        auxiliary = StepResult(
            True,
            action=Action("press_back", {}),
            before=kwargs["observation"],
            after=kwargs["observation"],
            result=ActionResult(True),
            actions_executed=1,
            origin="page_gate",
        )
        primary = StepResult(
            True,
            action=action,
            before=kwargs["observation"],
            after=kwargs["observation"],
            result=ActionResult(True),
            actions_executed=1,
        )
        return StepResult(
            True,
            action=action,
            before=kwargs["observation"],
            after=kwargs["observation"],
            result=ActionResult(True),
            actions_executed=2,
            executed_steps=(primary, auxiliary),
        )

    monkeypatch.setattr(execution, "execute_robust_action", execute_with_auxiliary)

    result = asyncio.run(
        execution.execute_function(
            function,
            host=host,
            plugins=PluginSet(),
            observation=before,
            state_loader=lambda _state_id: before,
        )
    )

    assert result.success is True
    assert result.actions_executed == 4
    assert [step["metadata"]["function_step_index"] for step in result.detail["trace"]] == [
        0,
        0,
        1,
        1,
    ]
