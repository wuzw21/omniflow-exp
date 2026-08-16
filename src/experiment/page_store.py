"""Capture the current Android page and incrementally curate a Page Store."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
from typing import Callable, Protocol, Sequence
import xml.etree.ElementTree as ET

import numpy as np

from omniflow.core.model import Observation
from omniflow.transfer.embedding import (
    PageEncoder,
    SoftPageWordWeights,
    TreeEmbedding,
    pool_dynamic_page_words,
    pool_soft_page_words,
)
from omniflow.transfer.page_store import EmbeddingConfig, PageProposal, PageStore

RunCommand = Callable[..., str | bytes]


@dataclass(frozen=True)
class CapturedPage:
    observation: Observation
    embedding: TreeEmbedding | DescriptorPageEmbedding
    screenshot_path: Path


@dataclass(frozen=True)
class DescriptorPageEmbedding:
    vector: np.ndarray
    element_count: int
    encoder_version: str
    page_word_counts: tuple[int, ...]
    word_presence: np.ndarray | None = None


@dataclass(frozen=True)
class RecordedPageAsset:
    state_id: str
    run_ids: tuple[str, ...]
    metadata: dict[str, object]
    xml: str
    screenshot: bytes
    remote_paths: dict[str, str]


class PageEmbeddingEncoder(Protocol):
    embedding_config: EmbeddingConfig

    def embed(
        self,
        value: Observation | dict[str, object] | str,
    ) -> TreeEmbedding | DescriptorPageEmbedding: ...


class OmniFlowNativePageEncoder:
    """Expose OmniFlow's native 512D page encoder through one store contract."""

    def __init__(self) -> None:
        self.encoder = PageEncoder()
        self.embedding_config = EmbeddingConfig(
            name="omniflow_native_512d_page_embedding",
            dimension=512,
            source_dimension=512,
            pooling="native_tree_spatial_pooling",
            provenance={
                "encoder_version": self.encoder.version,
                "weights_hash": self.encoder.weights.hash,
            },
        )

    def embed(
        self,
        value: Observation | dict[str, object] | str,
    ) -> TreeEmbedding:
        return self.encoder.embed(value)


class _OmniTransferDescriptorBackend:
    def __init__(
        self,
        *,
        omnitransfer_root: Path,
        checkpoint: Path,
        device: str,
    ) -> None:
        root = str(omnitransfer_root)
        if root not in sys.path:
            sys.path.insert(0, root)
        source = str(omnitransfer_root / "src")
        if source not in sys.path:
            sys.path.insert(0, source)
        from dataclasses import replace

        from omnitransfer.learned_matcher import (
            ALL_NODE_CANDIDATE_POLICY,
            RelationAwareMatcher,
            matcher_inputs,
        )
        from omnitransfer.ui_graph import graph_from_record

        self._torch = __import__("torch")
        self._matcher_inputs = matcher_inputs
        self._graph_from_record = graph_from_record
        self.matcher = RelationAwareMatcher.from_checkpoint(
            checkpoint,
            device=device,
        )
        self.config = replace(
            self.matcher.config,
            candidate_policy=ALL_NODE_CANDIDATE_POLICY,
        )
        self.model = self.matcher.model.to(device).eval()
        self.device = device
        self.architecture = self.matcher.config.architecture
        self.text_encoder = self.matcher.config.text_encoder

    def encode_descriptors(
        self,
        xml: str,
        *,
        graph_id: str,
        screenshot_path: str,
    ) -> np.ndarray:
        record: dict[str, object] = {"xml": xml}
        screenshot = Path(screenshot_path) if screenshot_path else None
        if screenshot is not None and screenshot.is_file():
            record["screenshot_path"] = str(screenshot)
        graph = self._graph_from_record(record, graph_id=graph_id)
        inputs = self._matcher_inputs(
            graph,
            graph,
            config=self.config,
            device=self.device,
        )
        with self._torch.inference_mode():
            _, descriptors, _ = self.model._encode_nodes(
                inputs[0],
                inputs[1],
                inputs[7],
                inputs[8],
            )
        bounded = [
            index
            for index, node in enumerate(graph.nodes)
            if node.bbox is not None
            and node.bbox[2] > node.bbox[0]
            and node.bbox[3] > node.bbox[1]
        ]
        return descriptors[bounded].detach().cpu().numpy()


class OmniTransferDescriptorPageEncoder:
    """Build eight dynamic page words from learned v9.2 node descriptors."""

    dimension = 1024
    source_dimension = 128
    version = "omnitransfer-v9.2-dynamic-page-words.v1"

    def __init__(
        self,
        *,
        omnitransfer_root: Path,
        checkpoint: Path,
        device: str = "cpu",
        page_word_checkpoint: Path | None = None,
        backend: object | None = None,
    ) -> None:
        self.omnitransfer_root = omnitransfer_root.expanduser().resolve()
        self.checkpoint = checkpoint.expanduser().resolve()
        if not self.checkpoint.is_file():
            raise ValueError(f"omnitransfer_checkpoint_missing:{self.checkpoint}")
        self.backend = backend or _OmniTransferDescriptorBackend(
            omnitransfer_root=self.omnitransfer_root,
            checkpoint=self.checkpoint,
            device=device,
        )
        self.page_word_checkpoint = (
            page_word_checkpoint.expanduser().resolve()
            if page_word_checkpoint is not None
            else None
        )
        if (
            self.page_word_checkpoint is not None
            and not self.page_word_checkpoint.is_file()
        ):
            raise ValueError(
                f"page_word_checkpoint_missing:{self.page_word_checkpoint}"
            )
        self.page_word_weights = (
            SoftPageWordWeights.from_npz(str(self.page_word_checkpoint))
            if self.page_word_checkpoint is not None
            else None
        )
        if self.page_word_weights is not None:
            self.version = "omnitransfer-v9.2-soft-page-words.v1"
        soft_checkpoint_provenance = (
            {
                "page_word_checkpoint_path": str(self.page_word_checkpoint),
                "page_word_checkpoint_sha256": _file_sha256(
                    self.page_word_checkpoint
                ),
                "page_word_parameter_count": self.page_word_weights.parameter_count,
                "outputs": ["8x128d_page_words", "8d_presence"],
            }
            if self.page_word_weights is not None
            else {}
        )
        self.embedding_config = EmbeddingConfig(
            name=(
                "omnitransfer_v9_2_soft_page_words_1024"
                if self.page_word_weights is not None
                else "omnitransfer_v9_2_dynamic_page_words_1024"
            ),
            dimension=self.dimension,
            source_dimension=self.source_dimension,
            pooling=(
                "prior_guided_soft_page_words_x_128d"
                if self.page_word_weights is not None
                else "eight_dynamic_page_words_x_128d"
            ),
            provenance={
                "omnitransfer_root": str(self.omnitransfer_root),
                "checkpoint_path": str(self.checkpoint),
                "checkpoint_sha256": _file_sha256(self.checkpoint),
                "architecture": str(getattr(self.backend, "architecture", "")),
                "text_encoder": str(getattr(self.backend, "text_encoder", "")),
                "node_descriptor_dimension": self.source_dimension,
                "page_words": [
                    "text",
                    "top_text",
                    "bottom_text",
                    "actionable_text",
                    "stateful_text",
                    "all_nodes",
                    "middle_nodes",
                    "large_surfaces",
                ],
                "padding": "none",
                **soft_checkpoint_provenance,
            },
        )

    def embed(self, value: Observation | dict[str, object] | str) -> DescriptorPageEmbedding:
        observation = (
            Observation(xml=value)
            if isinstance(value, str)
            else Observation.from_value(value)
        )
        xml = str(observation.xml or "")
        if not xml.strip():
            raise ValueError("omnitransfer_page_xml_required")
        screenshot_path = str(observation.extra.get("screenshot_path") or "")
        graph_id = hashlib.sha256(xml.encode("utf-8")).hexdigest()[:20]
        descriptors = np.asarray(
            self.backend.encode_descriptors(
                xml,
                graph_id=graph_id,
                screenshot_path=screenshot_path,
            ),
            dtype=np.float32,
        )
        if descriptors.ndim != 2 or descriptors.shape[1] != self.source_dimension:
            raise ValueError("omnitransfer_node_descriptors_must_be_n_by_128")
        if descriptors.shape[0] == 0 or not np.all(np.isfinite(descriptors)):
            raise ValueError("omnitransfer_node_descriptors_unusable")
        if self.page_word_weights is None:
            page_vector, word_counts = pool_dynamic_page_words(
                observation,
                descriptors,
            )
            word_presence = None
        else:
            soft_output = pool_soft_page_words(
                observation,
                descriptors,
                self.page_word_weights,
            )
            page_vector = soft_output.vector
            word_counts = soft_output.word_counts
            word_presence = soft_output.presence
        if page_vector.shape != (self.dimension,):
            raise ValueError("omnitransfer_dynamic_page_words_must_be_1024d")
        return DescriptorPageEmbedding(
            vector=page_vector,
            element_count=descriptors.shape[0],
            encoder_version=self.version,
            page_word_counts=word_counts,
            word_presence=word_presence,
        )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def capture_current_page(
    *,
    adb: str,
    serial: str,
    work_dir: Path,
    run: RunCommand | None = None,
    encoder: PageEmbeddingEncoder | None = None,
    state_package: str = "cn.com.omnimind.bot.debug",
) -> CapturedPage:
    execute = run or _run
    work_dir.mkdir(parents=True, exist_ok=True)
    result_path = "files/debug-human-recording-result.json"
    receiver = f"{state_package}/.DebugHumanRecordingReceiver"
    shell = (adb, "-s", serial, "shell")
    execute(
        (*shell, "run-as", state_package, "rm", "-f", result_path),
        binary=False,
    )
    execute(
        (
            *shell,
            "am",
            "broadcast",
            "-a",
            f"{state_package}.HUMAN_RECORDING",
            "-n",
            receiver,
            "--es",
            "op",
            "capture_state",
        ),
        binary=False,
    )
    payload: dict[str, object] | None = None
    deadline = time.monotonic() + 30.0
    last_error = ""
    while time.monotonic() < deadline:
        try:
            raw_value = execute(
                (*shell, "run-as", state_package, "cat", result_path),
                binary=False,
            )
            raw = (
                raw_value.decode("utf-8")
                if isinstance(raw_value, bytes)
                else raw_value
            )
            decoded = json.loads(raw)
            if isinstance(decoded, dict):
                payload = decoded
                break
        except (json.JSONDecodeError, OSError, subprocess.SubprocessError) as error:
            last_error = str(error)
        time.sleep(0.25)
    if payload is None:
        raise RuntimeError(f"omniflow_get_state_timeout:{last_error}")
    if payload.get("success") is not True:
        message = payload.get("error_message") or payload.get("error_code") or "failed"
        raise RuntimeError(f"omniflow_get_state_failed:{message}")
    state = payload.get("state")
    if not isinstance(state, dict):
        raise RuntimeError("omniflow_get_state_missing_state")
    xml = str(state.get("xml") or "")
    if not xml.strip().startswith("<"):
        raise RuntimeError("current_page_xml_capture_failed")
    remote_screenshot = str(state.get("screenshot_path") or "")
    if not remote_screenshot:
        raise RuntimeError("omniflow_get_state_screenshot_path_missing")
    screenshot_value = execute(
        (
            adb,
            "-s",
            serial,
            "exec-out",
            "run-as",
            state_package,
            "cat",
            remote_screenshot,
        ),
        binary=True,
    )
    screenshot = (
        screenshot_value
        if isinstance(screenshot_value, bytes)
        else screenshot_value.encode("latin1")
    )
    if not screenshot:
        raise RuntimeError("current_page_screenshot_capture_failed")
    screenshot_path = work_dir / "current.jpg"
    screenshot_path.write_bytes(screenshot)
    package_name = str(state.get("package_name") or "")
    activity_name = str(state.get("activity_name") or "")
    display = state.get("display")
    app_metadata = payload.get("app")
    device_metadata = payload.get("device")
    observation = Observation(
        xml=xml,
        package_name=package_name,
        activity_name=activity_name,
        extra={
            "screenshot_path": str(screenshot_path),
            "state_id": str(state.get("state_id") or ""),
            "display": dict(display) if isinstance(display, dict) else {},
            "app": dict(app_metadata) if isinstance(app_metadata, dict) else {},
            "device": {
                **(
                    dict(device_metadata)
                    if isinstance(device_metadata, dict)
                    else {}
                ),
                "serial": serial,
            },
            "state_backend": str(payload.get("state_backend") or ""),
            "device_serial": serial,
        },
    )
    selected_encoder = encoder or OmniFlowNativePageEncoder()
    embedding = selected_encoder.embed(observation)
    if (
        embedding.vector.shape != (selected_encoder.embedding_config.dimension,)
        or _element_count(embedding) <= 0
    ):
        raise RuntimeError("current_page_embedding_unusable")
    return CapturedPage(observation, embedding, screenshot_path)


def interactive_loop(
    *,
    store: PageStore,
    baseline_store: PageStore,
    adb: str,
    serial: str,
    encoder: PageEmbeddingEncoder,
    baseline_encoder: PageEmbeddingEncoder,
    state_package: str,
    top_k: int = 5,
) -> int:
    if set(store.clusters) != set(baseline_store.clusters):
        raise ValueError("page_store_comparison_indexes_out_of_sync")
    print(
        "我们的 Embedding 配置:\n"
        + json.dumps(encoder.embedding_config.to_dict(), ensure_ascii=False, indent=2)
    )
    print(
        "原生 OmniFlow Embedding 配置:\n"
        + json.dumps(
            baseline_encoder.embedding_config.to_dict(),
            ensure_ascii=False,
            indent=2,
        )
    )
    print("Page Store: 1=捕捉当前页, 0=退出")
    while True:
        command = input("\n选择: ").strip().lower()
        if command in {"0", "q", "quit", "exit"}:
            return 0
        if command != "1":
            print("请输入 1 或 0。")
            continue
        with tempfile.TemporaryDirectory(prefix="omniflow-page-capture-") as temp:
            captured = capture_current_page(
                adb=adb,
                serial=serial,
                work_dir=Path(temp),
                encoder=encoder,
                state_package=state_package,
            )
            proposal = store.propose(
                captured.embedding.vector,
                word_presence=getattr(captured.embedding, "word_presence", None),
                package_name=captured.observation.package_name or "",
                limit=top_k,
            )
            baseline_embedding = baseline_encoder.embed(captured.observation)
            baseline_proposal = baseline_store.propose(
                baseline_embedding.vector,
                package_name=captured.observation.package_name or "",
                limit=top_k,
            )
            _print_capture(captured, proposal, baseline_proposal)
            choice = _read_decision(proposal, baseline_proposal)
            if choice is None:
                print("已跳过；Page Store 未修改。")
                continue
            decision, cluster_id, cluster_name = choice
            result = store.add_page(
                xml=captured.observation.xml or "",
                vector=captured.embedding.vector,
                word_presence=getattr(captured.embedding, "word_presence", None),
                package_name=captured.observation.package_name or "",
                activity_name=captured.observation.activity_name or "",
                screenshot_path=captured.screenshot_path,
                device_serial=serial,
                capture_metadata=_capture_metadata(captured.observation),
                decision=decision,
                cluster_id=cluster_id,
                cluster_name=cluster_name or "",
                proposal=proposal,
            )
            baseline_result = baseline_store.add_page(
                xml=captured.observation.xml or "",
                vector=baseline_embedding.vector,
                package_name=captured.observation.package_name or "",
                activity_name=captured.observation.activity_name or "",
                device_serial=serial,
                capture_metadata=_capture_metadata(captured.observation),
                decision=decision,
                cluster_id=cluster_id,
                cluster_name=cluster_name or "",
                proposal=baseline_proposal,
            )
            if result.cluster_id != baseline_result.cluster_id:
                raise RuntimeError("page_store_comparison_cluster_id_mismatch")
            print(
                json.dumps(
                    {
                        "saved": True,
                        "page_id": result.page_id,
                        "cluster_id": result.cluster_id,
                        "cluster_name": result.cluster_name,
                        "decision": result.decision,
                        "our_matched_score": result.matched_score,
                        "original_matched_score": baseline_result.matched_score,
                        "contribution_weight": result.contribution_weight,
                        "cluster_page_count": store.clusters[
                            result.cluster_id
                        ].page_count,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )


def import_recorded_pages(
    *,
    store: PageStore,
    baseline_store: PageStore,
    adb: str,
    serial: str,
    encoder: PageEmbeddingEncoder,
    baseline_encoder: PageEmbeddingEncoder,
    state_package: str,
    top_k: int = 5,
    merge_threshold: float = 0.95,
    run_limit: int = 0,
    run: RunCommand | None = None,
) -> dict[str, object]:
    if set(store.clusters) != set(baseline_store.clusters):
        raise ValueError("page_store_comparison_indexes_out_of_sync")
    if not 0.0 <= merge_threshold <= 1.0:
        raise ValueError("page_store_merge_threshold_out_of_range")
    assets = load_recorded_page_assets(
        adb=adb,
        serial=serial,
        state_package=state_package,
        run_limit=run_limit,
        run=run,
    )
    known_page_ids = {
        page_id
        for cluster in store.clusters.values()
        for page_id in cluster.page_ids
    }
    rows: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="omniflow-page-import-") as temp:
        temp_root = Path(temp)
        for index, asset in enumerate(assets):
            xml_digest = hashlib.sha256(asset.xml.encode("utf-8")).hexdigest()
            page_id = f"page-{xml_digest[:20]}"
            if page_id in known_page_ids:
                rows.append(
                    {
                        "state_id": asset.state_id,
                        "page_id": page_id,
                        "decision": "exact_duplicate",
                        "run_ids": list(asset.run_ids),
                    }
                )
                continue
            screenshot_path = temp_root / f"{index:05d}-{asset.state_id}.jpg"
            screenshot_path.write_bytes(asset.screenshot)
            observation = _recorded_observation(
                asset,
                screenshot_path=screenshot_path,
                serial=serial,
            )
            embedded = encoder.embed(observation)
            baseline_embedding = baseline_encoder.embed(observation)
            proposal = store.propose(
                embedded.vector,
                word_presence=getattr(embedded, "word_presence", None),
                package_name=observation.package_name or "",
                limit=top_k,
            )
            baseline_proposal = baseline_store.propose(
                baseline_embedding.vector,
                package_name=observation.package_name or "",
                limit=top_k,
            )
            selected = proposal.candidates[0] if proposal.candidates else None
            package_compatible = selected is not None and (
                not selected.package_names
                or not observation.package_name
                or observation.package_name in selected.package_names
            )
            if (
                selected is not None
                and selected.score >= merge_threshold
                and package_compatible
            ):
                decision = "merge"
                cluster_id = selected.cluster_id
                cluster_name = ""
            else:
                decision = "new"
                cluster_id = None
                cluster_name = propose_page_cluster_name(
                    observation,
                    store=store,
                )
            decision_metadata = {
                "state_id": asset.state_id,
                "run_ids": list(asset.run_ids),
                "remote_paths": dict(asset.remote_paths),
                "display": observation.extra.get("display") or {},
                "import_mode": "recorded_page_auto_v1",
                "merge_threshold": merge_threshold,
                "package_compatible": package_compatible,
            }
            result = store.add_page(
                xml=asset.xml,
                vector=embedded.vector,
                word_presence=getattr(embedded, "word_presence", None),
                package_name=observation.package_name or "",
                activity_name=observation.activity_name or "",
                screenshot_path=screenshot_path,
                device_serial=serial,
                capture_metadata=decision_metadata,
                decision=decision,
                cluster_id=cluster_id,
                cluster_name=cluster_name,
                proposal=proposal,
                decision_source="automatic_omnitransfer_threshold_v1",
            )
            baseline_result = baseline_store.add_page(
                xml=asset.xml,
                vector=baseline_embedding.vector,
                package_name=observation.package_name or "",
                activity_name=observation.activity_name or "",
                device_serial=serial,
                capture_metadata=decision_metadata,
                decision=decision,
                cluster_id=cluster_id,
                cluster_name=cluster_name,
                proposal=baseline_proposal,
                decision_source="automatic_omnitransfer_threshold_v1",
            )
            if result.cluster_id != baseline_result.cluster_id:
                raise RuntimeError("page_store_comparison_cluster_id_mismatch")
            known_page_ids.add(result.page_id)
            rows.append(
                {
                    "state_id": asset.state_id,
                    "page_id": result.page_id,
                    "package_name": observation.package_name or "",
                    "cluster_id": result.cluster_id,
                    "cluster_name": result.cluster_name,
                    "decision": result.decision,
                    "our_top_k": _proposal_rows(proposal),
                    "original_top_k": _proposal_rows(baseline_proposal),
                    "run_ids": list(asset.run_ids),
                }
            )
    summary = {
        "mode": "recorded_page_auto_v1",
        "device_serial": serial,
        "merge_threshold": merge_threshold,
        "recorded_states": len(assets),
        "imported": sum(row["decision"] != "exact_duplicate" for row in rows),
        "exact_duplicates": sum(row["decision"] == "exact_duplicate" for row in rows),
        "new_clusters": sum(row["decision"] == "new" for row in rows),
        "merged_pages": sum(row["decision"] == "merge" for row in rows),
        "cluster_count": len(store.clusters),
        "pages": rows,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def load_recorded_page_assets(
    *,
    adb: str,
    serial: str,
    state_package: str,
    run_limit: int = 0,
    run: RunCommand | None = None,
) -> tuple[RecordedPageAsset, ...]:
    execute = run or _run
    prefix = (adb, "-s", serial, "exec-out", "run-as", state_package)
    run_paths = _lines(
        execute(
            (
                *prefix,
                "find",
                "files/run_logs",
                "-maxdepth",
                "1",
                "-type",
                "f",
                "-name",
                "*_gui-*.json",
            ),
            binary=False,
        )
    )
    runs: list[tuple[int, str, dict[str, object]]] = []
    for path in run_paths:
        payload = _json_object(execute((*prefix, "cat", path), binary=False))
        runs.append((int(payload.get("started_at_ms") or 0), path, payload))
    runs.sort(key=lambda item: (item[0], item[1]))
    if run_limit > 0:
        runs = runs[-run_limit:]

    state_paths = _lines(
        execute(
            (
                *prefix,
                "find",
                "files/run_logs/states",
                "-maxdepth",
                "1",
                "-type",
                "f",
            ),
            binary=False,
        )
    )
    asset_paths: dict[str, dict[str, str]] = {}
    for path in state_paths:
        match = re.search(r"_(state_[^.]+)\.(json|xml|jpg)$", path)
        if match:
            asset_paths.setdefault(match.group(1), {})[match.group(2)] = path

    ordered_state_ids: list[str] = []
    state_runs: dict[str, list[str]] = {}
    for _, _, payload in runs:
        run_id = str(payload.get("run_id") or "")
        for state_id in _run_state_ids(payload):
            if state_id not in state_runs:
                ordered_state_ids.append(state_id)
                state_runs[state_id] = []
            if run_id and run_id not in state_runs[state_id]:
                state_runs[state_id].append(run_id)

    assets: list[RecordedPageAsset] = []
    for state_id in ordered_state_ids:
        paths = asset_paths.get(state_id) or {}
        missing = sorted({"json", "xml", "jpg"}.difference(paths))
        if missing:
            raise RuntimeError(
                f"recorded_state_assets_missing:{state_id}:{','.join(missing)}"
            )
        metadata = _json_object(
            execute((*prefix, "cat", paths["json"]), binary=False)
        )
        xml_value = execute((*prefix, "cat", paths["xml"]), binary=False)
        xml = (
            xml_value.decode("utf-8")
            if isinstance(xml_value, bytes)
            else xml_value
        )
        screenshot_value = execute((*prefix, "cat", paths["jpg"]), binary=True)
        screenshot = (
            screenshot_value
            if isinstance(screenshot_value, bytes)
            else screenshot_value.encode("latin1")
        )
        if not xml.strip().startswith("<") or not screenshot:
            raise RuntimeError(f"recorded_state_assets_unusable:{state_id}")
        assets.append(
            RecordedPageAsset(
                state_id=state_id,
                run_ids=tuple(state_runs[state_id]),
                metadata=metadata,
                xml=xml,
                screenshot=screenshot,
                remote_paths=dict(paths),
            )
        )
    return tuple(assets)


def propose_page_cluster_name(observation: Observation, *, store: PageStore) -> str:
    package_name = str(observation.package_name or "")
    app_name = _existing_app_name(package_name, store) or _package_display_name(
        package_name
    )
    page_name = _salient_page_name(
        str(observation.xml or ""),
        package_name=package_name,
    )
    base = f"{app_name}-{page_name}"
    names = {cluster.name for cluster in store.clusters.values()}
    if base not in names:
        return base
    suffix = 2
    while f"{base}-{suffix}" in names:
        suffix += 1
    return f"{base}-{suffix}"


def _recorded_observation(
    asset: RecordedPageAsset,
    *,
    screenshot_path: Path,
    serial: str,
) -> Observation:
    display = asset.metadata.get("display")
    return Observation(
        xml=asset.xml,
        package_name=str(asset.metadata.get("package_name") or ""),
        activity_name=str(asset.metadata.get("activity_name") or ""),
        extra={
            "screenshot_path": str(screenshot_path),
            "state_id": asset.state_id,
            "display": dict(display) if isinstance(display, dict) else {},
            "device_serial": serial,
            "state_backend": "oob_runlog_state_asset",
        },
    )


def _run_state_ids(payload: dict[str, object]) -> tuple[str, ...]:
    values: list[str] = []
    steps = payload.get("steps")
    if isinstance(steps, list):
        for step in steps:
            if not isinstance(step, dict):
                continue
            for key in ("before_state_id", "after_state_id"):
                value = str(step.get(key) or "")
                if value and value not in values:
                    values.append(value)
    final_state = str(payload.get("final_state_id") or "")
    if final_state and final_state not in values:
        values.append(final_state)
    return tuple(values)


def _lines(value: str | bytes) -> tuple[str, ...]:
    text_value = value.decode("utf-8") if isinstance(value, bytes) else value
    return tuple(line.strip() for line in text_value.splitlines() if line.strip())


def _json_object(value: str | bytes) -> dict[str, object]:
    text_value = value.decode("utf-8") if isinstance(value, bytes) else value
    decoded = json.loads(text_value)
    if not isinstance(decoded, dict):
        raise RuntimeError("recorded_json_object_required")
    return decoded


def _proposal_rows(proposal: PageProposal) -> list[dict[str, object]]:
    return [
        {
            "cluster_id": candidate.cluster_id,
            "cluster_name": candidate.cluster_name,
            "score": candidate.score,
            "page_count": candidate.page_count,
        }
        for candidate in proposal.candidates
    ]


def _existing_app_name(package_name: str, store: PageStore) -> str:
    counts: dict[str, int] = {}
    for cluster in store.clusters.values():
        if package_name not in cluster.package_names:
            continue
        prefix = re.split(r"[-—–]", cluster.name, maxsplit=1)[0].strip()
        if prefix and not re.fullmatch(r"[A-Za-z0-9_.]+", prefix):
            counts[prefix] = counts.get(prefix, 0) + cluster.page_count
    return max(counts, key=lambda name: (counts[name], name)) if counts else ""


def _package_display_name(package_name: str) -> str:
    known = {
        "cn.com.omnimind.bot.debug": "OmniFlow",
        "com.android.contacts": "电话",
        "com.android.settings": "设置",
        "com.android.camera": "相机",
        "com.android.chrome": "Chrome",
    }
    if package_name in known:
        return known[package_name]
    pieces = [piece for piece in package_name.split(".") if piece]
    return (pieces[-1] if pieces else "应用").replace("_", " ").title()


def _salient_page_name(xml: str, *, package_name: str = "") -> str:
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return "页面"
    all_text = "\n".join(
        " ".join((node.attrib.get(key) or "").split())
        for node in root.iter()
        for key in ("text", "content-desc")
        if node.attrib.get(key)
    )
    if package_name == "cn.com.omnimind.bot.debug":
        if "GUI 任务未完成" in all_text:
            return "GUI任务失败"
        if "vlm_task" in all_text or "执行中" in all_text:
            return "工具执行"
        return "对话"
    if package_name == "com.android.contacts":
        if "同意并使用" in all_text and "不同意" in all_text:
            return "营业厅授权"
        if "未获取授权" in all_text and "营业厅" in all_text:
            return "营业厅未授权"
        if "点击粘贴" in all_text and all(
            marker in all_text for marker in ("ABC", "DEF", "GHI")
        ):
            return "拨号盘"
        if "话费余额" in all_text and "流量管理" in all_text:
            return "营业厅"
    values: list[tuple[float, str]] = []
    page_terms = (
        "首页",
        "联系人",
        "新建",
        "编辑",
        "详情",
        "设置",
        "订单",
        "购物车",
        "搜索结果",
        "通话",
        "拨号",
        "消息",
        "登录",
        "注册",
        "收藏",
        "个人中心",
        "流量管理",
        "营业厅",
    )
    generic = {"返回", "搜索", "更多", "更多选项", "确定", "取消", "保存"}
    for node in root.iter():
        bounds = _parse_bounds(node.attrib.get("bounds", ""))
        for key in ("text", "content-desc"):
            value = " ".join((node.attrib.get(key) or "").split())
            if (
                not value
                or value in generic
                or len(value) > 18
                or not any(character.isalpha() or "\u4e00" <= character <= "\u9fff" for character in value)
            ):
                continue
            score = 0.0
            if any(term in value for term in page_terms):
                score += 8.0
            if 2 <= len(value) <= 8:
                score += 2.0
            if bounds is not None:
                _, top, _, bottom = bounds
                if bottom <= 600:
                    score += 2.0
                score -= min(top / 2400.0, 1.0)
            if node.attrib.get("clickable") == "true":
                score -= 0.5
            values.append((score, value))
    if not values:
        return "页面"
    values.sort(key=lambda item: (-item[0], len(item[1]), item[1]))
    return values[0][1].replace("/", "-")


def _parse_bounds(value: str) -> tuple[int, int, int, int] | None:
    match = re.fullmatch(r"\[(-?\d+),(-?\d+)\]\[(-?\d+),(-?\d+)\]", value)
    return tuple(int(item) for item in match.groups()) if match else None


def _print_capture(
    captured: CapturedPage,
    proposal: PageProposal,
    baseline_proposal: PageProposal,
) -> None:
    print(
        f"\n当前页: {captured.observation.package_name or '-'} "
        f"{captured.observation.activity_name or '-'} "
        f"nodes={_element_count(captured.embedding)}"
    )
    app = captured.observation.extra.get("app")
    device = captured.observation.extra.get("device")
    display = captured.observation.extra.get("display")
    print(
        "状态来源: "
        f"state={captured.observation.extra.get('state_id') or '-'} "
        f"backend={captured.observation.extra.get('state_backend') or '-'} "
        f"display={display or '-'}"
    )
    if isinstance(app, dict):
        print(
            f"App: label={app.get('label') or '-'} "
            f"version={app.get('version_name') or '-'} "
            f"versionCode={app.get('version_code') or '-'}"
        )
    if isinstance(device, dict):
        print(
            f"设备: {device.get('manufacturer') or '-'} "
            f"{device.get('model') or '-'} serial={device.get('serial') or '-'} "
            f"Android={device.get('android_release') or '-'}"
        )
    page_word_counts = getattr(captured.embedding, "page_word_counts", ())
    if page_word_counts:
        print(
            "动态页面词节点数: "
            + ", ".join(
                f"{name}={count}"
                for name, count in zip(
                    (
                        "text",
                        "top_text",
                        "bottom_text",
                        "actionable_text",
                        "stateful_text",
                        "all_nodes",
                        "middle_nodes",
                        "large_surfaces",
                    ),
                    page_word_counts,
                    strict=True,
                )
            )
        )
    word_presence = getattr(captured.embedding, "word_presence", None)
    if word_presence is not None:
        print(
            "页面词 presence: "
            + ", ".join(
                f"{name}={score:.3f}"
                for name, score in zip(
                    (
                        "text",
                        "top_text",
                        "bottom_text",
                        "actionable_text",
                        "stateful_text",
                        "all_nodes",
                        "middle_nodes",
                        "large_surfaces",
                    ),
                    word_presence,
                    strict=True,
                )
            )
        )
    if not proposal.candidates and not baseline_proposal.candidates:
        print("Page Store 为空，请输入 n 新建第一个簇。")
        return
    _print_proposal("我们的 OmniTransfer Top-K", "o", proposal)
    _print_proposal("原生 OmniFlow Top-K", "b", baseline_proposal)


def _print_proposal(title: str, prefix: str, proposal: PageProposal) -> None:
    print(f"{title}:")
    for index, candidate in enumerate(proposal.candidates, start=1):
        packages = ",".join(candidate.package_names) or "-"
        print(
            f"  {prefix}{index}. name={candidate.cluster_name} "
            f"score={candidate.score:.6f} "
            f"cluster={candidate.cluster_id} pages={candidate.page_count} "
            f"apps={packages}"
        )


def _read_decision(
    proposal: PageProposal,
    baseline_proposal: PageProposal,
) -> tuple[str, str | None, str | None] | None:
    while True:
        value = input(
            "合并到 o编号(我们的) / b编号(原生) / n=新建 / s=跳过: "
        ).strip().lower()
        if value in {"n", "new"}:
            name = " ".join(input("新页面簇名称: ").strip().split())
            if not name:
                print("名称不能为空。")
                continue
            return "new", None, name
        if value in {"s", "skip", "0"}:
            return None
        selected_proposal = None
        index_value = ""
        if value.startswith("o"):
            selected_proposal = proposal
            index_value = value[1:]
        elif value.startswith("b"):
            selected_proposal = baseline_proposal
            index_value = value[1:]
        elif value.isdigit():
            selected_proposal = proposal
            index_value = value
        if selected_proposal is not None and index_value.isdigit():
            index = int(index_value) - 1
            if 0 <= index < len(selected_proposal.candidates):
                return "merge", selected_proposal.candidates[index].cluster_id, None
        print("请输入 o/b 候选编号、n 或 s。")


def _capture_metadata(observation: Observation) -> dict[str, object]:
    return {
        key: value
        for key, value in observation.extra.items()
        if key != "screenshot_path"
    }


def _run(command: tuple[str, ...], *, binary: bool = False) -> str | bytes:
    completed = subprocess.run(
        command,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=not binary,
    )
    return completed.stdout


def _element_count(embedding: TreeEmbedding | DescriptorPageEmbedding) -> int:
    elements = getattr(embedding, "elements", None)
    if elements is not None:
        return len(elements)
    return int(getattr(embedding, "element_count", 0))


def build_encoder(
    name: str,
    *,
    omnitransfer_root: Path,
    checkpoint: Path | None,
    page_word_checkpoint: Path | None,
    device: str,
) -> PageEmbeddingEncoder:
    if name == "omniflow-native-512":
        return OmniFlowNativePageEncoder()
    if name not in {
        "omnitransfer-v9.2-soft-1024",
        "omnitransfer-v9.2-dynamic-1024",
    }:
        raise ValueError(f"unsupported_page_store_encoder:{name}")
    selected_checkpoint = checkpoint or (
        omnitransfer_root
        / "src/omnitransfer/checkpoints"
        / "omnitransfer_spatial_xml_alignment_v9_20260805"
        / "v9_spatial_xml_alignment_seed29.pt"
    )
    return OmniTransferDescriptorPageEncoder(
        omnitransfer_root=omnitransfer_root,
        checkpoint=selected_checkpoint,
        device=device,
        page_word_checkpoint=(
            page_word_checkpoint
            if name == "omnitransfer-v9.2-soft-1024"
            else None
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--serial", required=True)
    parser.add_argument("--adb", default="adb")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--encoder",
        choices=(
            "omnitransfer-v9.2-soft-1024",
            "omnitransfer-v9.2-dynamic-1024",
            "omniflow-native-512",
        ),
        default="omnitransfer-v9.2-soft-1024",
    )
    parser.add_argument(
        "--omnitransfer-root",
        type=Path,
        default=Path("~/Projects/Omni/OmniTransfer").expanduser(),
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--page-word-checkpoint",
        type=Path,
        default=Path(
            "~/OmniFlowPageStore/models/soft_page_words_v1/soft_page_words_seed41.npz"
        ).expanduser(),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--import-recordings", action="store_true")
    parser.add_argument("--recording-run-limit", type=int, default=0)
    parser.add_argument("--auto-merge-threshold", type=float, default=0.95)
    parser.add_argument(
        "--state-package",
        default="cn.com.omnimind.bot.debug",
        help="Android package exposing DebugHumanRecordingReceiver capture_state",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.top_k <= 0:
        raise SystemExit("--top-k must be positive")
    if not args.store.is_absolute():
        raise SystemExit("--store must be an absolute directory")
    encoder = build_encoder(
        args.encoder,
        omnitransfer_root=args.omnitransfer_root,
        checkpoint=args.checkpoint,
        page_word_checkpoint=args.page_word_checkpoint,
        device=args.device,
    )
    baseline_encoder = OmniFlowNativePageEncoder()
    baseline_store_root = args.store / "comparisons" / "omniflow-native-512"
    store = PageStore(args.store, embedding_config=encoder.embedding_config)
    baseline_store = PageStore(
        baseline_store_root,
        embedding_config=baseline_encoder.embedding_config,
    )
    common = dict(
        store=store,
        baseline_store=baseline_store,
        adb=args.adb,
        serial=args.serial,
        encoder=encoder,
        baseline_encoder=baseline_encoder,
        state_package=args.state_package,
        top_k=args.top_k,
    )
    if args.import_recordings:
        import_recorded_pages(
            **common,
            merge_threshold=args.auto_merge_threshold,
            run_limit=args.recording_run_limit,
        )
        return 0
    return interactive_loop(**common)


if __name__ == "__main__":
    sys.exit(main())
