"""Registry and validation for same-harness AndroidWorld baselines.

The registry is deliberately small. A baseline changes the reuse mechanism
while the AndroidWorld lifecycle, Planner, OOB host, budget, and validator stay
owned by the existing experiment runner.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from omniflow.core.model import Action, Observation, TransferResult


TransferFactory = Callable[[], Callable[[Action, Observation, Observation | None], TransferResult] | None]


@dataclass(frozen=True)
class BaselineSpec:
    """One reproducible same-harness baseline condition."""

    name: str
    label: str
    description: str
    requires_memory: bool
    requires_source_run_log: bool
    requires_element_docs: bool
    function_memory_enabled: bool
    transfer_kind: str | None = None

    def transfer_factory(self) -> TransferFactory:
        """Return a lazy mapper factory so registry import stays lightweight."""

        if self.transfer_kind is None:
            return lambda: None
        if self.transfer_kind == "attribute":
            from omniflow.transfer.attribute import attribute_transfer

            return lambda: attribute_transfer
        if self.transfer_kind == "agentrr":
            from omniflow.transfer.agentrr import DEFAULT_POLICY, agentrr_transfer_with

            return lambda: agentrr_transfer_with(DEFAULT_POLICY)
        raise ValueError(f"unknown_baseline_transfer:{self.transfer_kind}")


BASELINE_SPECS: tuple[BaselineSpec, ...] = (
    BaselineSpec(
        name="none",
        label="OmniFlow",
        description="Function memory with canonical OmniTransfer and Checker.",
        requires_memory=True,
        requires_source_run_log=False,
        requires_element_docs=False,
        function_memory_enabled=True,
    ),
    BaselineSpec(
        name="mobilegpt_style",
        label="MobileGPT-style",
        description="Function memory with attribute-based action relocation.",
        requires_memory=True,
        requires_source_run_log=False,
        requires_element_docs=False,
        function_memory_enabled=True,
        transfer_kind="attribute",
    ),
    BaselineSpec(
        name="appagent_style",
        label="AppAgent-style",
        description="Planner-visible element documentation without Function replay.",
        requires_memory=True,
        requires_source_run_log=False,
        requires_element_docs=True,
        function_memory_enabled=False,
    ),
    BaselineSpec(
        name="demo_hint",
        label="Demo-hint",
        description="Redacted demonstration hint with the shared Planner and Host.",
        requires_memory=False,
        requires_source_run_log=True,
        requires_element_docs=False,
        function_memory_enabled=False,
    ),
    BaselineSpec(
        name="mobimem_style",
        label="MobiMem-style",
        description="AgentRR action replay with package and signature guards.",
        requires_memory=True,
        requires_source_run_log=False,
        requires_element_docs=False,
        function_memory_enabled=True,
        transfer_kind="agentrr",
    ),
)

BASELINE_BY_NAME = {spec.name: spec for spec in BASELINE_SPECS}
BASELINE_SUITE = tuple(spec.name for spec in BASELINE_SPECS)
BASELINE_SUITE_ALIAS = "all"


def baseline_names(*, include_suite_alias: bool = False) -> tuple[str, ...]:
    names = BASELINE_SUITE
    return names + (BASELINE_SUITE_ALIAS,) if include_suite_alias else names


def baseline_spec(name: str) -> BaselineSpec:
    try:
        return BASELINE_BY_NAME[str(name or "").strip()]
    except KeyError as error:
        raise ValueError(f"unknown_same_harness_baseline:{name}") from error


def baseline_result_slot(method: str, baseline: str) -> str:
    """Return the stable output slot for one method and baseline condition."""

    selected = str(baseline or "none").strip() or "none"
    return selected if selected != "none" else str(method).strip()


def validate_baseline_request(
    name: str,
    *,
    action: str,
    method: str,
    harness: str,
    completion_oracle: str,
    memory: str,
    source_run_log: str,
    element_docs: str,
) -> None:
    """Validate one baseline request at the public launcher boundary."""

    if name == BASELINE_SUITE_ALIAS:
        if (
            action != "run"
            or method != "omniflow"
            or harness != "builtin"
            or completion_oracle != "off"
            or not memory
            or not source_run_log
            or not element_docs
        ):
            raise ValueError(
                "baseline_suite_requires_builtin_omniflow_run_with_memory_source_run_log_and_element_docs"
            )
        return

    spec = baseline_spec(name)
    if name == "none":
        return
    if (
        action != "run"
        or method != "omniflow"
        or harness != "builtin"
        or completion_oracle != "off"
        or (spec.requires_memory and not memory)
        or (spec.requires_source_run_log and not source_run_log)
    ):
        raise ValueError(
            "same_harness_baseline_requires_builtin_omniflow_run_with_store_and_no_oracle"
        )
    if spec.requires_element_docs and not element_docs:
        raise ValueError("appagent_style_requires_element_docs")


__all__ = [
    "BASELINE_BY_NAME",
    "BASELINE_SPECS",
    "BASELINE_SUITE",
    "BASELINE_SUITE_ALIAS",
    "BaselineSpec",
    "baseline_names",
    "baseline_result_slot",
    "baseline_spec",
    "validate_baseline_request",
]
