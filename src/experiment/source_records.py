"""Plain data records shared by source preparation and task scheduling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CanonicalRunLog:
    """One source RunLog selected from the canonical data index."""

    task: str
    goal: str
    params: dict
    source_run_log: Path
    replay_seed: int
    step_count: int
    meta: dict


@dataclass(frozen=True)
class SourceRunLogProfile:
    """Read-only facts about the format and replayability of one source log."""

    task: str
    source_run_log: Path
    replay_format: str
    step_count: int
    card_count: int
    latest_official_success_source: bool
    direct_replay_ready: bool
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "task": self.task,
            "source_run_log": str(self.source_run_log),
            "replay_format": self.replay_format,
            "step_count": self.step_count,
            "card_count": self.card_count,
            "latest_official_success_source": self.latest_official_success_source,
            "direct_replay_ready": self.direct_replay_ready,
            "notes": list(self.notes),
        }
