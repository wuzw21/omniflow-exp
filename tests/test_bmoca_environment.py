from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from omniflow import Action
from src.integrations.android_world.launch import build_parser
from src.integrations.bmoca import (
    BMocaHost,
    discover_bmoca_episodes,
)


class _Driver:
    page_source = (
        '<hierarchy><node package="com.android.settings" '
        'bounds="[0,0][1080,1920]" /></hierarchy>'
    )
    current_package = "com.android.settings"
    current_activity = ".Settings"

    def activate_app(self, package_name: str) -> None:
        self.current_package = package_name


class _TimeStep:
    def __init__(self, *, reward: float = 0.0, done: bool = False) -> None:
        self.curr_rew = reward
        self.curr_obs = {"pixel": np.zeros((4, 3, 3), dtype=np.uint8)}
        self._done = done

    def last(self) -> bool:
        return self._done


class _Environment:
    def __init__(self) -> None:
        self.gestures: list[np.ndarray] = []
        self._coordinator = SimpleNamespace(
            _driver=_Driver(),
            _screen_size=(1920, 1080),
            _is_tablet=False,
        )
        self._simulator = SimpleNamespace(_adb_port=5555)

    def reset(self, *, target_env_id: str) -> _TimeStep:
        assert target_env_id == "test_env_100"
        return _TimeStep()

    def step(self, gesture: np.ndarray) -> _TimeStep:
        self.gestures.append(gesture)
        return _TimeStep(reward=1.0, done=True)


def test_bmoca_host_is_only_an_omniflow_host_adapter() -> None:
    environment = _Environment()
    host = BMocaHost(environment, snapshot_id="test_env_100")

    host.reset()
    before = host.observe(xml=True, screenshot=True, app_info=True)
    result = host.act(Action("click", {"x": 500, "y": 250}))

    assert before.package_name == "com.android.settings"
    assert before.extra["benchmark"] == "b-moca"
    assert before.image_base64
    assert result.success is True
    assert host.official_success is True
    assert environment.gestures[0].tolist() == [0.25, 0.5, 0.25, 0.5]


def test_bmoca_episode_discovery_uses_every_requested_environment(
    tmp_path: Path,
) -> None:
    task_dir = tmp_path / "asset/tasks"
    task_dir.mkdir(parents=True)
    (task_dir / "open_settings.textproto").write_text(
        'max_episode_steps: 7\ninstruction: "Open settings"\n',
        encoding="utf-8",
    )
    config_dir = tmp_path / "asset/environments/config"
    config_dir.mkdir(parents=True)
    (config_dir / "environments_test.csv").write_text(
        "idx,device_id\n100,pixel_3\n109,pixel_c\n",
        encoding="utf-8",
    )

    episodes = discover_bmoca_episodes(
        tmp_path,
        task_id="open_settings",
        environment_ids=("100", "109"),
    )

    assert [episode.environment_id for episode in episodes] == ["100", "109"]
    assert [episode.avd_name for episode in episodes] == [
        "pixel_3_test_00",
        "pixel_c_test_00",
    ]
    assert all(episode.goal == "Open settings" for episode in episodes)


def test_unified_e2e_parser_selects_bmoca_as_an_environment() -> None:
    args = build_parser().parse_args(
        [
            "--environment",
            "bmoca",
            "--bmoca-root",
            "/opt/bmoca",
            "--tasks",
            "open_settings",
            "--store-path",
            "/opt/store.json",
        ]
    )

    assert args.environment == "bmoca"
    assert args.bmoca_root == "/opt/bmoca"
