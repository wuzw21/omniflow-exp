from pathlib import Path

from src.experiment.memory_ref import MemoryRef


def test_memory_ref_is_optional_provider_data() -> None:
    ref = MemoryRef(
        provider="mobilegpt",
        task_name="DemoTask",
        root=Path("/tmp/memory"),
        schema_version="omniflow.mobilegpt.memory.v2",
        sha256="abc",
    )

    assert ref.to_record() == {
        "provider": "mobilegpt",
        "task_name": "DemoTask",
        "root": "/tmp/memory",
        "schema_version": "omniflow.mobilegpt.memory.v2",
        "sha256": "abc",
        "manifest": "",
        "metadata": {},
    }
