from pathlib import Path

from src.experiment.memory_interface import (
    MemoryAdapter,
    MemoryCheck,
    MemoryPackage,
    MemoryRequest,
    operation_name,
)
from src.experiment.appagent_source import AppAgentMemoryAdapter
from src.experiment.mobilegpt_source import MobileGPTMemoryAdapter


def test_memory_request_is_provider_neutral() -> None:
    request = MemoryRequest(
        task_name="DemoTask",
        source_index=Path("data/current.json"),
        model="model",
    )

    assert request.task_name == "DemoTask"
    assert request.source_index == Path("data/current.json")
    assert request.options == {}


def test_memory_package_and_check_have_stable_records() -> None:
    package = MemoryPackage(
        provider="mobilegpt",
        task_name="DemoTask",
        root=Path("/tmp/memory"),
        bundle_root=Path("/tmp"),
        schema_version="omniflow.mobilegpt.memory.v2",
        sha256="abc",
    )
    check = MemoryCheck(
        provider="mobilegpt",
        task_name="DemoTask",
        valid=True,
        root=package.root,
    )

    assert package.to_record()["provider"] == "mobilegpt"
    assert package.to_record()["schema_version"] == (
        "omniflow.mobilegpt.memory.v2"
    )
    assert check.to_record()["valid"] is True
    assert check.to_record()["errors"] == []


def test_operation_name_is_the_external_interface_label() -> None:
    assert operation_name("prepare") == "prepare"
    assert operation_name("check") == "check"


def test_provider_adapters_implement_the_same_interface() -> None:
    assert isinstance(AppAgentMemoryAdapter(), MemoryAdapter)
    assert isinstance(MobileGPTMemoryAdapter(), MemoryAdapter)
    assert AppAgentMemoryAdapter().name == "appagent"
    assert MobileGPTMemoryAdapter().name == "mobilegpt"
