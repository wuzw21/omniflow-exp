from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from src.experiment.observation_evidence import AndroidWorldEpisodeRecorder
from src.integrations.android_world import host as host_module
from src.integrations.android_world import state as state_module


def _state(color=0):
    return SimpleNamespace(
        pixels=np.full((16, 12, 3), color, dtype=np.uint8),
        forest='<hierarchy><node package="com.example" bounds="[0,0][12,16]" /></hierarchy>',
        ui_elements=[],
        auxiliaries={"package_name": "com.example", "display": {"width": 12, "height": 16}},
    )


def test_identical_frames_share_bytes_but_distinct_pixels_never_merge(tmp_path):
    first = state_module.snapshot_androidworld_state(_state(), evidence_root=tmp_path)
    second = state_module.snapshot_androidworld_state(_state(), evidence_root=tmp_path)
    changed = state_module.snapshot_androidworld_state(_state(255), evidence_root=tmp_path)
    assert first["pixels"] == second["pixels"]
    assert first["pixels"]["path"] != changed["pixels"]["path"]
    assert len(list(tmp_path.rglob('*.png'))) == 2
    with Image.open(first['pixels']['path']) as image:
        assert np.array_equal(np.asarray(image), _state().pixels)


def test_concurrent_identical_snapshots_publish_one_complete_image(tmp_path):
    with ThreadPoolExecutor(max_workers=6) as pool:
        refs = list(pool.map(lambda _: state_module.snapshot_androidworld_state(
            _state(), evidence_root=tmp_path)['pixels']['path'], range(12)))
    assert len(set(refs)) == 1
    with Image.open(refs[0]) as image:
        image.verify()
    assert not list(tmp_path.rglob('.frame-*'))
    Path(refs[0]).write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='hash_collision'):
        state_module.snapshot_androidworld_state(_state(), evidence_root=tmp_path)


def test_host_and_recorder_share_capture_and_post_action_state(tmp_path, monkeypatch):
    state = _state()
    captures = []
    original = state_module._png_bytes
    def encode(pixels):
        captures.append(True)
        return original(pixels)
    monkeypatch.setattr(state_module, '_png_bytes', encode)
    monkeypatch.setattr(host_module, 'OobControlClient', lambda *a, **kw: SimpleNamespace(
        observe=lambda **kw: {}))
    monkeypatch.setattr(host_module, 'oob_state_from_payload', lambda *a, **kw: state)
    recorder = AndroidWorldEpisodeRecorder(lambda: state, lambda action: None, evidence_root=tmp_path)
    recorder.start_episode()
    host = host_module.AndroidWorldHost(SimpleNamespace(_recorder=recorder), evidence_root=tmp_path)
    observation = host.observe(xml=True, screenshot=False, app_info=True)
    assert len(captures) == 1
    assert observation.extra['screenshot_path'] == recorder.latest_observation['screenshot']['path']
    assert observation.image_base64 is None
    host._after_action_state = state
    after = host.take_after_action_observation()
    assert after.xml == observation.xml
    assert after.package_name == 'com.example'
    assert after.extra['screenshot_path'] == observation.extra['screenshot_path']
    assert host.take_after_action_observation() is None
    assert len(captures) == 1
    assert len(recorder.persist_observations()) == 1
    assert 'xml' not in recorder.persist_observations()[0]
