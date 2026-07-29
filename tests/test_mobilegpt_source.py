from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.experiment import androidworld as pipeline
from src.experiment import mobilegpt_source
from src.integrations.mobilegpt_runtime import _mobilegpt_chat_model


def _write_source_index(
    root: Path,
    *,
    method: str = "ours",
    official_success: bool = True,
) -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    source_run_log = root / "source.run_log.json"
    source_run_log.write_text(
        json.dumps(
            {
                "schema_version": "omniflow.canonical_run_log.v1",
                "run_id": "source-run",
                "goal": "Turn Bluetooth on.",
                "status": "succeeded",
                "success": True,
                "steps": [
                    {
                        "step_index": 0,
                        "before_state_id": "state-0",
                        "action": {
                            "tool": "click",
                            "args": {"x": 500, "y": 500},
                        },
                        "result": {"success": True},
                        "after_state_id": "state-1",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    state_catalog = root / "transfer_states.json"
    state_catalog.write_text(
        json.dumps(
            {
                "schema_version": "omniflow.transfer-state-catalog.v1",
                "run_id": "source-run",
                "states": {
                    "state-0": {
                        "state_id": "state-0",
                        "package_name": "com.android.settings",
                        "xml": (
                            '<hierarchy><node text="Bluetooth" '
                            'resource-id="android:id/switch_widget" '
                            'clickable="true" bounds="[0,0][100,100]" />'
                            "</hierarchy>"
                        ),
                        "display": {"width": 100, "height": 100},
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    provenance = root / "provenance_manifest.json"
    provenance.write_text(
        json.dumps(
            {
                "schema_version": "omniflow.source-replay-transfer-store.v1",
                "source_target_audit": {
                    "source_target_audit_complete": True,
                    "source_targets": [
                        {
                            "step_index": 0,
                            "source_state_id": "state-0",
                            "target": {
                                "text": "Bluetooth",
                                "resource_id": "android:id/switch_widget",
                            },
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    index = root / "index.json"
    index.write_text(
        json.dumps(
            {
                "SystemBluetoothTurnOn": {
                    "goal": "Turn Bluetooth on.",
                    "params": {},
                    "replay_seed": 111,
                    "step_count": 1,
                    "retained_source_run_log": str(source_run_log),
                    "method": method,
                    "latest_official_success_source": official_success,
                    "source_kind": (
                        "androidworld_validator_success_source_runlog"
                    ),
                    "source_run_log_sha256": hashlib.sha256(
                        source_run_log.read_bytes()
                    ).hexdigest(),
                    "source_state_catalog": str(state_catalog),
                    "source_state_catalog_sha256": hashlib.sha256(
                        state_catalog.read_bytes()
                    ).hexdigest(),
                    "store_provenance": str(provenance),
                    "store_provenance_sha256": hashlib.sha256(
                        provenance.read_bytes()
                    ).hexdigest(),
                }
            }
        ),
        encoding="utf-8",
    )
    return index, source_run_log


def test_mobilegpt_deterministic_preflight_does_not_claim_output(
    tmp_path: Path,
) -> None:
    index, _ = _write_source_index(tmp_path / "source")
    payload = json.loads(index.read_text(encoding="utf-8"))
    row = payload["SystemBluetoothTurnOn"]
    row["source_state_catalog_sha256"] = "0" * 64
    index.write_text(json.dumps(payload), encoding="utf-8")
    output_root = tmp_path / "never-created"

    with pytest.raises(ValueError, match="source_state_catalog_hash_mismatch"):
        mobilegpt_source.prepare_mobilegpt_source_memory(
            index_path=index,
            task_name="SystemBluetoothTurnOn",
            mobilegpt_root=tmp_path / "mobilegpt",
            android_world_root=tmp_path / "android_world",
            output_root=output_root,
            model="qwen3-vl-plus",
        )

    assert not output_root.exists()


def test_mobilegpt_source_accepts_successful_canonical_seed111(
    tmp_path: Path,
) -> None:
    accepted_index, source_run_log = _write_source_index(tmp_path / "accepted")
    item = mobilegpt_source.load_canonical_source_item(
        accepted_index,
        task_name="SystemBluetoothTurnOn",
    )
    assert item.source_run_log == source_run_log
    assert item.replay_seed == 111

    rejected_root = tmp_path / "rejected"
    rejected_root.mkdir()
    rejected_index, _ = _write_source_index(
        rejected_root,
        method="fixed_replay",
    )
    item = mobilegpt_source.load_canonical_source_item(
        rejected_index,
        task_name="SystemBluetoothTurnOn",
    )
    assert mobilegpt_source.source_method_label(item) == "fixed_replay"


def test_mobilegpt_preflight_resolves_target_from_frozen_source_states(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index, _ = _write_source_index(tmp_path / "source-package")
    monkeypatch.setattr(
        pipeline,
        "_infer_mobilegpt_target_from_source_run_log",
        lambda _item: {
            "target_package": "",
            "target_app": "",
            "target_source": "unresolved",
        },
    )

    result = mobilegpt_source.preflight_mobilegpt_source(
        index_path=index,
        task_name="SystemBluetoothTurnOn",
    )

    assert result["target_package"] == "com.android.settings"
    assert result["target_source"] == "frozen_source_states"


def test_mobilegpt_source_reads_explicit_source_seed(tmp_path: Path) -> None:
    index, _ = _write_source_index(tmp_path / "source-seed")
    payload = json.loads(index.read_text(encoding="utf-8"))
    row = payload["SystemBluetoothTurnOn"]
    row["source_seed"] = row.pop("replay_seed")
    index.write_text(json.dumps(payload), encoding="utf-8")

    item = mobilegpt_source.load_canonical_source_item(
        index,
        task_name="SystemBluetoothTurnOn",
    )

    assert item.replay_seed == 111


def test_mobilegpt_configured_model_overrides_upstream_model_argument(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MOBILEGPT_CHAT_MODEL", "qwen3-vl-plus")
    assert _mobilegpt_chat_model("gpt-4") == "qwen3-vl-plus"


def test_mobilegpt_source_generation_has_no_model_or_episode_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    index, _ = _write_source_index(source_root)
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        pipeline,
        "build_mobilegpt_teacher_source",
        lambda *_args, **_kwargs: {"action_count": 2},
    )
    monkeypatch.setattr(
        pipeline,
        "_infer_mobilegpt_target_from_source_run_log",
        lambda _item: {
            "target_package": "com.android.settings",
            "target_app": "Settings",
        },
    )
    monkeypatch.setattr(
        pipeline,
        "_patch_mobilegpt_stats",
        lambda **_kwargs: [],
    )
    monkeypatch.setattr(
        pipeline,
        "_patch_mobilegpt_server_runtime_context",
        lambda **_kwargs: [],
    )

    def build_server(*_args: object, **_kwargs: object) -> pipeline.CommandSpec:
        return pipeline.CommandSpec(
            label="mobilegpt-teacher",
            argv=["python", "teacher.py"],
            env={},
            cwd=tmp_path,
        )

    monkeypatch.setattr(pipeline, "build_mobilegpt_command", build_server)
    monkeypatch.setattr(
        pipeline,
        "_start_mobilegpt_browser_task_server",
        lambda **_kwargs: ({}, None),
    )

    episode_output = tmp_path / "episode"

    def build_episode(
        *_args: object,
        **kwargs: object,
    ) -> pipeline.CommandSpec:
        captured["episode_kwargs"] = kwargs
        return pipeline.CommandSpec(
            label="mobilegpt-source",
            argv=["python", "episode.py"],
            env={},
            cwd=tmp_path,
            output_path=episode_output,
            metadata={},
        )

    monkeypatch.setattr(
        pipeline,
        "build_mobilegpt_androidworld_command",
        build_episode,
    )

    def start_server(
        spec: pipeline.CommandSpec,
        *,
        warmup_sec: float,
    ) -> tuple[object, int]:
        del warmup_sec
        captured["server"] = spec
        return object(), 0

    monkeypatch.setattr(pipeline, "_start_background_command", start_server)
    monkeypatch.setattr(
        pipeline,
        "_stop_background_command",
        lambda _process: None,
    )

    def run_episode(spec: pipeline.CommandSpec) -> int:
        assert spec.output_path is not None
        spec.output_path.mkdir(parents=True)
        (spec.output_path / "task_results.jsonl").write_text(
            json.dumps(
                {
                    "task_name": "SystemBluetoothTurnOn",
                    "official_validator_used": True,
                    "official_validator_success": True,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        stats_path = (
            tmp_path / "bundle" / "source_stats.jsonl"
        )
        stats_path.write_text(
            json.dumps(
                {
                    "event": "chat_call",
                    "model": "qwen3-vl-plus",
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "total_tokens": 12,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return 0

    monkeypatch.setattr(pipeline, "run_command", run_episode)
    monkeypatch.setattr(
        pipeline,
        "seal_mobilegpt_adapted_memory",
        lambda **kwargs: {"source_method": kwargs["source_method"]},
    )

    result = mobilegpt_source.prepare_mobilegpt_source_memory(
        index_path=index,
        task_name="SystemBluetoothTurnOn",
        mobilegpt_root=tmp_path / "mobilegpt",
        android_world_root=tmp_path / "android_world",
        output_root=tmp_path / "bundle",
        model="qwen3-vl-plus",
    )

    server = captured["server"]
    episode_kwargs = captured["episode_kwargs"]
    assert isinstance(server, pipeline.CommandSpec)
    assert isinstance(episode_kwargs, dict)
    assert server.env["MOBILEGPT_CHAT_MODEL"] == "qwen3-vl-plus"
    assert server.env["MOBILEGPT_CHAT_MAX_ATTEMPTS"] == "1"
    assert server.env["MOBILEGPT_OOB_OBSERVE_RETRIES"] == "1"
    assert server.metadata["episode_retries"] == 0
    assert episode_kwargs["rebroadcast_limit"] == 0
    assert result["source_method"] == "ours"
