import pytest


@pytest.fixture(autouse=True)
def isolate_failure_evidence(tmp_path, monkeypatch):
    """Unit tests never append synthetic mappings to benchmark evidence."""
    monkeypatch.setenv("OMNIFLOW_TRANSFER_ERROR_POOL", str(tmp_path / "pairs.jsonl"))
