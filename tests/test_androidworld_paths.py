from src.experiment.androidworld_paths import (
    canonical_device_model,
    canonical_device_seed_name,
    canonical_method_name,
)
from src.experiment.run_task import _experiment_run_dir


def test_archive_names_use_avd_model_not_cli_alias() -> None:
    assert canonical_device_model(
        label="small5554", serial="emulator-5554", console_port=5554
    ) == "WXGA_Tablet_test_00"
    assert canonical_device_seed_name(
        label="small5562", serial="emulator-5562", console_port=5562
    ) == "OmniFlowTargetSmall_seed111_eval113"
    assert canonical_device_seed_name(
        label="fold5564", serial="emulator-5564", console_port=5564
    ) == "OmniFlowTargetFold_seed111_eval113"
    assert canonical_device_model(
        label="source5560", serial="emulator-5560", console_port=5560
    ) == "OmniFlowSourceSmall"
    assert canonical_device_model(
        label="small5554", serial="emulator-5554", console_port=5554
    ) == "WXGA_Tablet_test_00"


def test_retired_pixel_alias_is_read_only_compatibility() -> None:
    assert canonical_device_model(
        label="pixel5576", serial="emulator-5576", console_port=5576
    ) == "AndroidWorldAvd4090"


def test_method_aliases_are_canonicalized() -> None:
    assert canonical_method_name("Ours") == "omniflow"
    assert canonical_method_name("fixed replay") == "fixed_replay"
    assert canonical_method_name("t3a+hint") == "t3a_hint"


def test_live_batch_route_has_runlog_attempt_leaf(monkeypatch) -> None:
    monkeypatch.setenv("OMNIFLOW_BATCH_ATTEMPT_ID", "attempt.omniflow.small5554")
    path = _experiment_run_dir(
        "/tmp/androidworld",
        task="TaskOne",
        method="Ours",
        device="small5554",
        serial="emulator-5554",
        console_port=5554,
    )
    assert str(path).endswith(
        "/TaskOne/omniflow/WXGA_Tablet_test_00_seed111_eval113/"
        "runlog/attempt.omniflow.small5554"
    )


def test_source_collection_route_uses_source_model_and_explicit_attempt() -> None:
    path = _experiment_run_dir(
        "/tmp/androidworld",
        task="TaskOne",
        method="source",
        device="source5560",
        serial="emulator-5560",
        console_port=5560,
        attempt_id="source-attempt",
    )
    assert str(path).endswith(
        "/TaskOne/source/OmniFlowSourceSmall_seed111/runlog/source-attempt"
    )
