import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from omniflow.transfer.embedding import (
    SoftPageWordWeights,
    pool_dynamic_page_words,
    pool_soft_page_words,
)
from omniflow.transfer.page_store import EmbeddingConfig, PageStore
from src.experiment.page_store import (
    OmniFlowNativePageEncoder,
    OmniTransferDescriptorPageEncoder,
    build_parser,
    capture_current_page,
    import_recorded_pages,
    load_recorded_page_assets,
)


def _vector(x: float, y: float, *, dimension: int = 512) -> np.ndarray:
    vector = np.zeros(dimension, dtype=np.float32)
    vector[0] = x
    vector[1] = y
    return vector / np.linalg.norm(vector)


def test_page_store_ranks_then_records_confirmed_cluster_contribution(
    tmp_path: Path,
) -> None:
    store = PageStore(tmp_path / "page-store")
    first = store.add_page(
        xml="<hierarchy />",
        vector=_vector(1.0, 0.0),
        package_name="com.sankuai.meituan",
        activity_name="MainActivity",
        capture_metadata={"state_id": "state-home"},
        decision="new",
        cluster_name="美团首页",
    )

    proposal = store.propose(
        _vector(0.98, 0.20),
        package_name="com.sankuai.meituan",
        limit=3,
    )

    assert proposal.candidates[0].cluster_id == first.cluster_id
    assert proposal.candidates[0].cluster_name == "美团首页"
    assert proposal.candidates[0].score > 0.97

    merged = store.add_page(
        xml="<hierarchy><node text='外卖' /></hierarchy>",
        vector=_vector(0.98, 0.20),
        package_name="com.sankuai.meituan",
        activity_name="MainActivity",
        decision="merge",
        cluster_id=first.cluster_id,
        proposal=proposal,
    )
    reloaded = PageStore(tmp_path / "page-store")

    assert merged.cluster_id == first.cluster_id
    assert merged.cluster_name == "美团首页"
    assert merged.contribution_weight == 0.5
    assert reloaded.clusters[first.cluster_id].page_count == 2
    assert reloaded.clusters[first.cluster_id].centroid.shape == (512,)
    assert len(reloaded.events) == 2
    assert reloaded.clusters[first.cluster_id].name == "美团首页"
    assert reloaded.events[0]["evidence"]["capture_metadata"]["state_id"] == "state-home"


def test_page_store_locks_omnitransfer_descriptor_configuration(
    tmp_path: Path,
) -> None:
    config = EmbeddingConfig(
        name="omnitransfer_v9_2_dynamic_page_words_1024",
        dimension=1024,
        source_dimension=128,
        pooling="eight_dynamic_page_words_x_128d",
        provenance={"checkpoint_sha256": "a" * 64},
    )
    store = PageStore(tmp_path / "page-store", embedding_config=config)
    store.add_page(
        xml="<hierarchy />",
        vector=_vector(1.0, 0.0, dimension=1024),
        decision="new",
        cluster_name="首页",
    )

    reloaded = PageStore(tmp_path / "page-store", embedding_config=config)

    assert reloaded.embedding_config == config
    assert reloaded.clusters[next(iter(reloaded.clusters))].centroid.shape == (1024,)
    with np.testing.assert_raises_regex(ValueError, "page_store_embedding_config_mismatch"):
        PageStore(tmp_path / "page-store")


def test_new_page_cluster_requires_human_readable_name(tmp_path: Path) -> None:
    store = PageStore(tmp_path / "page-store")

    with np.testing.assert_raises_regex(
        ValueError, "page_store_new_cluster_name_required"
    ):
        store.add_page(
            xml="<hierarchy />",
            vector=_vector(1.0, 0.0),
            decision="new",
        )


def test_page_store_cluster_rename_is_audited(tmp_path: Path) -> None:
    store = PageStore(tmp_path / "page-store")
    created = store.add_page(
        xml="<hierarchy />",
        vector=_vector(1.0, 0.0),
        decision="new",
        cluster_name="临时名称",
        decision_source="automatic_test",
    )

    store.rename_cluster(
        created.cluster_id,
        "电话-拨号盘",
        decision_source="screenshot_review",
    )

    assert store.clusters[created.cluster_id].name == "电话-拨号盘"
    assert store.events[-1]["event_type"] == "cluster_rename"
    assert store.events[-1]["previous_cluster_name"] == "临时名称"


def test_dynamic_page_words_pool_128d_descriptors_into_eight_real_slices() -> None:
    xml = """<hierarchy>
      <node text="Title" bounds="[0,0][100,20]" />
      <node text="Open" clickable="true" bounds="[0,40][100,60]" />
      <node text="Footer" bounds="[0,80][100,100]" />
    </hierarchy>"""
    descriptors = np.zeros((3, 128), dtype=np.float32)
    descriptors[0, 0] = 1.0
    descriptors[1, 1] = 1.0
    descriptors[2, 2] = 1.0

    vector, counts = pool_dynamic_page_words(xml, descriptors)

    assert vector.shape == (1024,)
    assert counts == (3, 1, 1, 1, 0, 3, 1, 3)
    assert np.any(vector[:128])
    assert np.any(vector[128:256])
    assert np.any(vector[256:384])
    assert np.any(vector[384:512])
    assert not np.any(vector[512:640])
    assert np.any(vector[640:])


def test_soft_page_word_layer_returns_embedding_and_presence() -> None:
    xml = """<hierarchy>
      <node text="Title" bounds="[0,0][100,20]" />
      <node text="Open" clickable="true" bounds="[0,40][100,60]" />
      <node text="Footer" bounds="[0,80][100,100]" />
    </hierarchy>"""
    descriptors = np.zeros((3, 128), dtype=np.float32)
    descriptors[0, 0] = 1.0
    descriptors[1, 1] = 1.0
    descriptors[2, 2] = 1.0
    generator = np.random.default_rng(7)
    weights = SoftPageWordWeights(
        input_projection=generator.normal(0.0, 0.02, (144, 32)).astype(np.float32),
        input_bias=np.zeros(32, dtype=np.float32),
        attention_output=generator.normal(0.0, 0.02, (32, 8)).astype(np.float32),
        attention_bias=np.zeros(8, dtype=np.float32),
        prior_strength=np.ones(8, dtype=np.float32),
        presence_output=generator.normal(0.0, 0.02, (32, 8)).astype(np.float32),
        presence_bias=np.zeros(8, dtype=np.float32),
    )

    output = pool_soft_page_words(xml, descriptors, weights)

    assert weights.parameter_count == 5176
    assert output.vector.shape == (1024,)
    assert output.words.shape == (8, 128)
    assert output.presence.shape == (8,)
    assert output.attention.shape == (8, 3)
    np.testing.assert_allclose(output.attention.sum(axis=1), np.ones(8), atol=1e-6)
    assert np.all((output.presence > 0.0) & (output.presence < 1.0))
    np.testing.assert_allclose(np.linalg.norm(output.vector), 1.0, atol=1e-6)


def test_omnitransfer_descriptor_encoder_builds_dynamic_1024d_page_words(
    tmp_path: Path,
) -> None:
    class Backend:
        architecture = "omnitransfer_geometric_alignment_v9"
        text_encoder = "learned_token_lookup"

        def encode_descriptors(self, xml: str, *, graph_id: str, screenshot_path: str):
            del xml, graph_id, screenshot_path
            rows = np.zeros((2, 128), dtype=np.float32)
            rows[0, 0] = 1.0
            rows[1, 1] = 1.0
            return rows

    checkpoint = tmp_path / "v9.pt"
    checkpoint.write_bytes(b"checkpoint")
    encoder = OmniTransferDescriptorPageEncoder(
        omnitransfer_root=tmp_path,
        checkpoint=checkpoint,
        backend=Backend(),
    )

    embedded = encoder.embed(
        {
            "xml": (
                '<hierarchy><node text="Top" bounds="[0,0][100,20]" />'
                '<node text="Go" clickable="true" bounds="[0,80][100,100]" />'
                "</hierarchy>"
            ),
            "extra": {"screenshot_path": "/tmp/page.png"},
        }
    )

    assert embedded.vector.shape == (1024,)
    assert embedded.element_count == 2
    assert embedded.page_word_counts == (2, 1, 1, 1, 0, 2, 0, 2)
    assert np.any(embedded.vector[:128])
    assert np.any(embedded.vector[128:256])
    assert np.any(embedded.vector[256:384])
    assert np.any(embedded.vector[384:512])
    assert np.any(embedded.vector[640:768])
    assert np.any(embedded.vector[896:])
    assert encoder.embedding_config.dimension == 1024
    assert encoder.embedding_config.source_dimension == 128
    assert encoder.embedding_config.provenance["padding"] == "none"


def test_capture_current_page_uses_adb_xml_screenshot_and_foreground_app(
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, ...]] = []

    def run(command: tuple[str, ...], *, binary: bool = False):
        calls.append(command)
        joined = " ".join(command)
        if "rm -f files/debug-human-recording-result.json" in joined:
            return b"" if binary else ""
        if "am broadcast" in joined and "capture_state" in joined:
            return b"Broadcast completed" if binary else "Broadcast completed"
        if "cat files/debug-human-recording-result.json" in joined:
            value = json.dumps(
                {
                    "success": True,
                    "state_backend": "android_gui_environment_observe",
                    "state": {
                        "state_id": "state-meituan-home",
                        "package_name": "com.sankuai.meituan",
                        "activity_name": "com.sankuai.meituan/.MainActivity",
                        "display": {"width": 1080, "height": 2400},
                        "xml": '<hierarchy><node text="美团外卖" bounds="[0,0][100,200]" /></hierarchy>',
                        "screenshot_path": "/data/user/0/test/files/state.jpg",
                    },
                    "app": {
                        "package_name": "com.sankuai.meituan",
                        "label": "美团",
                        "version_name": "12.0",
                    },
                    "device": {"manufacturer": "OPPO", "model": "PJE110"},
                },
                ensure_ascii=False,
            )
            return value.encode() if binary else value
        if "exec-out run-as" in joined and "/data/user/0/test/files/state.jpg" in joined:
            return b"\xff\xd8\xffpage"
        raise AssertionError(command)

    captured = capture_current_page(
        adb="/android/adb",
        serial="phone-a",
        work_dir=tmp_path,
        run=run,
    )

    assert captured.observation.package_name == "com.sankuai.meituan"
    assert captured.observation.activity_name == "com.sankuai.meituan/.MainActivity"
    assert captured.observation.extra["state_id"] == "state-meituan-home"
    assert captured.observation.extra["app"]["version_name"] == "12.0"
    assert captured.observation.extra["device"]["model"] == "PJE110"
    assert captured.observation.extra["display"] == {"width": 1080, "height": 2400}
    assert captured.screenshot_path.read_bytes() == b"\xff\xd8\xffpage"
    assert captured.embedding.vector.shape == (512,)
    assert any("capture_state" in " ".join(call) for call in calls)
    assert not any("uiautomator dump" in " ".join(call) for call in calls)


def test_page_store_cli_defaults_to_omnitransfer_descriptor() -> None:
    args = build_parser().parse_args(
        ["--store", "/tmp/pages", "--serial", "phone-a"]
    )

    assert args.encoder == "omnitransfer-v9.2-soft-1024"
    assert args.device == "cpu"
    assert args.state_package == "cn.com.omnimind.bot.debug"


def test_page_store_uses_shared_word_presence_for_soft_page_similarity(
    tmp_path: Path,
) -> None:
    config = EmbeddingConfig(
        name="soft",
        dimension=1024,
        source_dimension=128,
        pooling="soft",
    )
    store = PageStore(tmp_path / "page-store", embedding_config=config)
    first = np.zeros(1024, dtype=np.float32)
    first[0] = 1.0
    first[128] = 1.0
    second = np.zeros(1024, dtype=np.float32)
    second[0] = 1.0
    second[129] = 1.0
    first_cluster = store.add_page(
        xml='<node bounds="[0,0][1,1]" />',
        vector=first,
        word_presence=np.asarray([1.0, 0.0, 0, 0, 0, 0, 0, 0]),
        decision="new",
        cluster_name="第一页",
    )
    second_cluster = store.add_page(
        xml='<node bounds="[0,0][2,2]" />',
        vector=second,
        word_presence=np.asarray([0.0, 1.0, 0, 0, 0, 0, 0, 0]),
        decision="new",
        cluster_name="第二页",
    )

    proposal = store.propose(
        second,
        word_presence=np.asarray([0.0, 1.0, 0, 0, 0, 0, 0, 0]),
    )

    assert proposal.candidates[0].cluster_id == second_cluster.cluster_id
    assert proposal.candidates[0].score == 1.0
    assert proposal.candidates[1].cluster_id == first_cluster.cluster_id


def test_native_encoder_exposes_locked_512d_configuration() -> None:
    encoder = OmniFlowNativePageEncoder()

    assert encoder.embedding_config.dimension == 512
    assert encoder.embedding_config.source_dimension == 512
    assert encoder.embedding_config.provenance["weights_hash"]


def test_load_recorded_pages_selects_recent_runs_and_complete_state_triplets() -> None:
    older = {
        "run_id": "gui-old",
        "started_at_ms": 10,
        "steps": [
            {"before_state_id": "state_old", "after_state_id": "state_shared"}
        ],
    }
    newer = {
        "run_id": "gui-new",
        "started_at_ms": 20,
        "steps": [
            {"before_state_id": "state_new", "after_state_id": "state_shared"}
        ],
    }
    state_ids = ("state_old", "state_new", "state_shared")

    def run(command: tuple[str, ...], *, binary: bool = False):
        joined = " ".join(command)
        if "-name *_gui-*.json" in joined:
            return "files/run_logs/old_gui-a.json\nfiles/run_logs/new_gui-b.json\n"
        if joined.endswith("cat files/run_logs/old_gui-a.json"):
            return json.dumps(older)
        if joined.endswith("cat files/run_logs/new_gui-b.json"):
            return json.dumps(newer)
        if "find files/run_logs/states" in joined:
            return "".join(
                f"files/run_logs/states/hash_{state_id}.{extension}\n"
                for state_id in state_ids
                for extension in ("json", "xml", "jpg")
            )
        for state_id in state_ids:
            if joined.endswith(f"cat files/run_logs/states/hash_{state_id}.json"):
                return json.dumps(
                    {
                        "state_id": state_id,
                        "package_name": "com.android.contacts",
                    }
                )
            if joined.endswith(f"cat files/run_logs/states/hash_{state_id}.xml"):
                return f'<hierarchy text="{state_id}" />'
            if joined.endswith(f"cat files/run_logs/states/hash_{state_id}.jpg"):
                return b"jpeg" if binary else "jpeg"
        raise AssertionError(command)

    assets = load_recorded_page_assets(
        adb="adb",
        serial="phone",
        state_package="oob.debug",
        run_limit=1,
        run=run,
    )

    assert [asset.state_id for asset in assets] == ["state_new", "state_shared"]
    assert assets[1].run_ids == ("gui-new",)
    assert all(set(asset.remote_paths) == {"json", "xml", "jpg"} for asset in assets)


def test_import_recorded_pages_uses_ours_for_auto_merge_and_skips_exact_duplicates(
    tmp_path: Path,
) -> None:
    config = EmbeddingConfig(
        name="test-page-embedding",
        dimension=2,
        source_dimension=2,
        pooling="test",
    )

    class Encoder:
        embedding_config = config

        def embed(self, value):
            xml = value.xml
            vector = (
                np.asarray([1.0, 0.0], dtype=np.float32)
                if "联系人" in xml
                else np.asarray([0.0, 1.0], dtype=np.float32)
            )
            return SimpleNamespace(vector=vector, element_count=1, word_presence=None)

    run_payload = {
        "run_id": "gui-new",
        "started_at_ms": 20,
        "steps": [
            {"before_state_id": "state_a", "after_state_id": "state_b"},
            {"before_state_id": "state_b", "after_state_id": "state_c"},
        ],
    }
    xml_by_state = {
        "state_a": '<hierarchy><node text="联系人" bounds="[0,0][100,100]" /></hierarchy>',
        "state_b": '<hierarchy><node text="新建联系人" bounds="[0,0][100,100]" /></hierarchy>',
        "state_c": '<hierarchy><node text="新建联系人" bounds="[0,0][100,100]" /></hierarchy>',
    }

    def run(command: tuple[str, ...], *, binary: bool = False):
        joined = " ".join(command)
        if "-name *_gui-*.json" in joined:
            return "files/run_logs/run_gui-a.json\n"
        if joined.endswith("cat files/run_logs/run_gui-a.json"):
            return json.dumps(run_payload)
        if "find files/run_logs/states" in joined:
            return "".join(
                f"files/run_logs/states/hash_{state_id}.{extension}\n"
                for state_id in xml_by_state
                for extension in ("json", "xml", "jpg")
            )
        for state_id, xml in xml_by_state.items():
            if joined.endswith(f"cat files/run_logs/states/hash_{state_id}.json"):
                return json.dumps(
                    {
                        "state_id": state_id,
                        "package_name": "com.android.contacts",
                        "display": {"width": 1080, "height": 2400},
                    }
                )
            if joined.endswith(f"cat files/run_logs/states/hash_{state_id}.xml"):
                return xml
            if joined.endswith(f"cat files/run_logs/states/hash_{state_id}.jpg"):
                return b"jpeg" if binary else "jpeg"
        raise AssertionError(command)

    summary = import_recorded_pages(
        store=PageStore(tmp_path / "ours", embedding_config=config),
        baseline_store=PageStore(tmp_path / "baseline", embedding_config=config),
        adb="adb",
        serial="phone",
        encoder=Encoder(),
        baseline_encoder=Encoder(),
        state_package="oob.debug",
        merge_threshold=0.95,
        run=run,
    )

    assert summary["recorded_states"] == 3
    assert summary["imported"] == 2
    assert summary["exact_duplicates"] == 1
    assert summary["new_clusters"] == 1
    assert summary["merged_pages"] == 1
    assert summary["pages"][0]["cluster_name"] == "电话-联系人"
    assert summary["pages"][1]["decision"] == "merge"
