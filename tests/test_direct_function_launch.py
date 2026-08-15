from __future__ import annotations

from omniflow import RunResult

from src.experiment.direct_function_launch import _install_direct_runs


class _Store:
    def get_function(self, function_id: str) -> object | None:
        return object() if function_id in {"create_note", "save_note"} else None


class _Flow:
    def __init__(self) -> None:
        self.store = _Store()
        self.calls = []

    def call_tool(self, call, *, experiment=None):
        self.calls.append((call, experiment))
        return RunResult(
            True,
            function_id=call.name,
            actions_executed=1,
            detail={"trace": [{"result": {"success": True}}]},
        )


def test_direct_function_sequence_advances_once_and_finishes_after_last_call() -> None:
    flow = _install_direct_runs(
        _Flow(),
        calls=[
            {"function_id": "create_note", "arguments": {"name": "note"}},
            {"function_id": "save_note", "arguments": {"text": "body"}},
        ],
    )

    first = flow.run("Create note", experiment="source")
    second = flow.run("Create note", experiment="source")

    assert first.function_id == "create_note"
    assert first.detail.get("done_reason") is None
    assert second.function_id == "save_note"
    assert second.detail["done_reason"] == "finished"
    assert [call.name for call, _ in flow.calls] == ["create_note", "save_note"]
