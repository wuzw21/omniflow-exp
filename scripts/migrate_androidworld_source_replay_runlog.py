from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from omniflow.trajectory import canonicalize_run_log
from src.integrations.runlog import import_run_log


EVIDENCE_ONLY_ACTION_ARGS = {
    "clickPrompt",
    "source_context",
    "target_evidence",
}
EXECUTION_ONLY_ACTION_ARGS = {
    "post_action_wait_s",
    "post_wait_s",
    "wait_after_s",
}
COORDINATE_ARGS = {
    "click": {"x": "x", "y": "y"},
    "long_press": {"x": "x", "y": "y"},
    "input_text": {"x": "x", "y": "y"},
    "swipe": {
        "x": "x",
        "y": "y",
        "x1": "x",
        "y1": "y",
        "x2": "x",
        "y2": "y",
    },
}
CANONICAL_STEP_REQUIRED_FIELDS = {
    "step_index",
    "before_state_id",
    "action",
    "result",
    "after_state_id",
}
CANONICAL_STEP_ALLOWED_FIELDS = CANONICAL_STEP_REQUIRED_FIELDS | {"metadata"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"json_object_required:{path}")
    return value


def _raw_actions(step: dict[str, Any]) -> list[dict[str, Any]]:
    value = step.get("executed_actions") or step.get("actions")
    if isinstance(value, list):
        return [action for action in value if isinstance(action, dict)]
    action = step.get("action") or step.get("tool_call")
    return [action] if isinstance(action, dict) else []


def _has_canonical_truth_steps(run_log: dict[str, Any]) -> bool:
    steps = run_log.get("steps")
    if not isinstance(steps, list) or not steps:
        return False
    for step in steps:
        if not isinstance(step, dict):
            return False
        fields = set(step)
        if not CANONICAL_STEP_REQUIRED_FIELDS.issubset(fields):
            return False
        if not fields.issubset(CANONICAL_STEP_ALLOWED_FIELDS):
            return False
        action = step.get("action")
        result = step.get("result")
        if (
            not isinstance(action, dict)
            or set(action) != {"tool", "args"}
            or not isinstance(action.get("args"), dict)
            or not isinstance(result, dict)
            or "success" not in result
        ):
            return False
    return True


def _action_args(action: dict[str, Any]) -> dict[str, Any] | None:
    function = action.get("function")
    candidates = (
        action.get("args"),
        action.get("arguments"),
        action.get("params"),
        function.get("arguments") if isinstance(function, dict) else None,
    )
    return next((value for value in candidates if isinstance(value, dict)), None)


def _action_tool(action: dict[str, Any]) -> str:
    function = action.get("function")
    return str(
        action.get("tool")
        or action.get("type")
        or action.get("name")
        or (function.get("name") if isinstance(function, dict) else "")
        or ""
    ).strip().lower()


def _set_action_tool(action: dict[str, Any], tool: str) -> None:
    function = action.get("function")
    if isinstance(function, dict) and function.get("name") is not None:
        function["name"] = tool
        return
    for key in ("tool", "type", "name", "action_type"):
        if action.get(key) is not None:
            action[key] = tool
            return
    action["type"] = tool


def _migrate_legacy_action_semantics(
    action: dict[str, Any],
    args: dict[str, Any],
) -> dict[str, int]:
    counts = {
        "press_back_to_press_key_count": 0,
        "answer_to_finished_count": 0,
        "start_settings_to_open_app_count": 0,
    }
    tool = _action_tool(action)
    if tool in {"back", "navigate_back", "press_back"}:
        _set_action_tool(action, "press_key")
        if args.get("key") is not None and str(args["key"]).strip().casefold() != "back":
            raise ValueError("source_replay_press_back_key_conflict")
        args["key"] = "back"
        counts["press_back_to_press_key_count"] = 1
    elif tool == "answer":
        _set_action_tool(action, "finished")
        text = args.pop("text", None)
        if text is None:
            raise ValueError("source_replay_answer_text_required")
        if args.get("content") is not None and args["content"] != text:
            raise ValueError("source_replay_answer_content_conflict")
        args["content"] = str(text)
        counts["answer_to_finished_count"] = 1
    elif tool == "start_activity":
        activity_action = str(args.pop("action", "") or "").strip()
        flags = args.pop("flags", None)
        if activity_action != "android.settings.SETTINGS":
            raise ValueError(
                f"source_replay_start_activity_unmapped:{activity_action}"
            )
        if flags not in (None, "", "0x04000000", 0x04000000):
            raise ValueError(f"source_replay_start_activity_flags_unmapped:{flags}")
        package_name = str(args.get("package_name") or "").strip()
        if package_name and package_name != "com.android.settings":
            raise ValueError("source_replay_start_activity_package_conflict")
        _set_action_tool(action, "open_app")
        args["package_name"] = "com.android.settings"
        counts["start_settings_to_open_app_count"] = 1
    return counts


def _canonical_key(value: Any) -> str:
    key = str(value or "").strip().casefold()
    aliases = {
        "back": "back",
        "keycode_back": "back",
        "home": "home",
        "keycode_home": "home",
        "enter": "enter",
        "keyboard_enter": "enter",
        "keycode_enter": "enter",
        "del": "delete",
        "delete": "delete",
        "keycode_del": "delete",
    }
    canonical = aliases.get(key)
    if canonical is None and len(key) == 1 and key.isdigit():
        canonical = key
    if canonical is None and key.startswith("keycode_"):
        suffix = key.removeprefix("keycode_")
        if len(suffix) == 1 and suffix.isdigit():
            canonical = suffix
    if canonical is None:
        raise ValueError(f"source_replay_keycode_unmapped:{key}")
    return canonical


def _positive_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if number > 0 else None


def _display_size(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, dict):
        return None
    screenshot = value.get("screenshot")
    candidates = [value]
    if isinstance(screenshot, dict):
        candidates.insert(0, screenshot)
    for candidate in candidates:
        width = _positive_number(
            candidate.get("original_width")
            or candidate.get("display_width")
            or candidate.get("screen_width")
            or candidate.get("width")
        )
        height = _positive_number(
            candidate.get("original_height")
            or candidate.get("display_height")
            or candidate.get("screen_height")
            or candidate.get("height")
        )
        if width is not None and height is not None:
            return width, height
    xml = str(value.get("xml") or value.get("page") or "")
    bounds = [
        tuple(float(item) for item in match)
        for match in re.findall(
            r'bounds="\[(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?)\]'
            r'\[(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?)\]"',
            xml,
        )
    ]
    if bounds:
        width = max(item[2] for item in bounds)
        height = max(item[3] for item in bounds)
        if width > 0 and height > 0:
            return width, height
    return None


def _step_display_size(
    step: dict[str, Any],
    *,
    fallback: tuple[float, float] | None,
) -> tuple[float, float] | None:
    if fallback is not None:
        return fallback
    for key in (
        "observation_before_act",
        "before_observation",
        "observation",
        "before",
        "state",
    ):
        if size := _display_size(step.get(key)):
            return size
    return fallback


def _normalized_coordinate(value: Any, *, extent: float, name: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"source_replay_pixel_coordinate_type_invalid:{name}")
    number = float(value)
    if number < 0 or number > extent:
        raise ValueError(f"source_replay_pixel_coordinate_out_of_bounds:{name}")
    normalized = round(number / extent * 1000.0, 6)
    return int(normalized) if normalized.is_integer() else normalized


def _normalize_legacy_pixel_coordinates(
    action: dict[str, Any],
    *,
    display_size: tuple[float, float] | None,
) -> int:
    args = _action_args(action)
    axes = COORDINATE_ARGS.get(_action_tool(action))
    if args is None or axes is None or display_size is None:
        return 0
    width, height = display_size
    count = 0
    for name, axis in axes.items():
        if name not in args or args[name] is None:
            continue
        args[name] = _normalized_coordinate(
            args[name],
            extent=width if axis == "x" else height,
            name=name,
        )
        count += 1
    return count


def _migrate_legacy_swipe_args(args: dict[str, Any]) -> tuple[int, int]:
    aliases = {
        "start_x": "x1",
        "start_y": "y1",
        "end_x": "x2",
        "end_y": "y2",
    }
    present = {key for key in aliases if key in args}
    if present and present != set(aliases):
        raise ValueError("source_replay_swipe_coordinate_alias_pair_required")
    alias_count = 0
    if present:
        for legacy_name, canonical_name in aliases.items():
            legacy_value = args.pop(legacy_name)
            existing = args.get(canonical_name)
            if existing is not None and existing != legacy_value:
                raise ValueError(
                    f"source_replay_swipe_coordinate_alias_conflict:{canonical_name}"
                )
            args[canonical_name] = legacy_value
            alias_count += 1

    if args.get("direction") is not None:
        return alias_count, 0
    if not all(args.get(key) is not None for key in ("x1", "y1", "x2", "y2")):
        return alias_count, 0
    dx = float(args["x2"]) - float(args["x1"])
    dy = float(args["y2"]) - float(args["y1"])
    if dx == 0 and dy == 0:
        raise ValueError("source_replay_swipe_direction_ambiguous")
    if abs(dx) >= abs(dy):
        args["direction"] = "right" if dx > 0 else "left"
    else:
        args["direction"] = "down" if dy > 0 else "up"
    return alias_count, 1


def load_androidworld_app_name_to_package(
    android_world_root: str | Path,
) -> tuple[dict[str, str], Path]:
    root = Path(android_world_root).expanduser().resolve()
    source = root / "android_world/env/adb_utils.py"
    if not source.is_file():
        raise FileNotFoundError(source)
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    patterns: dict[str, str] | None = None
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "_PATTERN_TO_ACTIVITY"
            for target in node.targets
        ):
            continue
        value = node.value
        if isinstance(value, ast.Call) and value.args:
            decoded = ast.literal_eval(value.args[0])
            if isinstance(decoded, dict):
                patterns = {
                    str(pattern): str(activity)
                    for pattern, activity in decoded.items()
                }
        break
    if not patterns:
        raise ValueError(f"androidworld_app_mapping_missing:{source}")
    mapping: dict[str, str] = {}
    for pattern, activity in patterns.items():
        package_name = activity.split("/", 1)[0].strip()
        if not package_name:
            raise ValueError(f"androidworld_app_package_missing:{pattern}")
        for alias in pattern.split("|"):
            name = alias.strip().casefold()
            if not name:
                continue
            existing = mapping.get(name)
            if existing and existing != package_name:
                raise ValueError(f"androidworld_app_mapping_conflict:{name}")
            mapping[name] = package_name
    return mapping, source


def _drop_legacy_open_app_args(
    args: dict[str, Any],
    *,
    app_name_to_package: dict[str, str],
) -> tuple[int, int]:
    if "app_name" not in args:
        return 0, 0
    app_name = str(args.pop("app_name") or "").strip().casefold()
    existing = str(args.get("package_name") or "").strip()
    if existing:
        mapped = app_name_to_package.get(app_name)
        if mapped and mapped != existing:
            raise ValueError(f"source_replay_open_app_package_conflict:{app_name}")
        return 1, 0
    package_name = app_name_to_package.get(app_name)
    if not package_name:
        raise ValueError(f"source_replay_open_app_package_unmapped:{app_name}")
    args["package_name"] = package_name
    return 1, 1


def _migrate_legacy_action_args(
    value: dict[str, Any],
    *,
    drop_clear_text: bool,
    app_name_to_package: dict[str, str],
) -> tuple[dict[str, Any], dict[str, int]]:
    migrated = copy.deepcopy(value)
    payload = migrated.get("payload")
    run_log = payload if isinstance(payload, dict) else migrated
    wrapped = run_log.get("run_log")
    run_log = wrapped if isinstance(wrapped, dict) else run_log
    steps = run_log.get("steps") or run_log.get("cards")
    if not isinstance(steps, list):
        raise ValueError("source_replay_steps_required")
    counts = {
        "stripped_evidence_arg_count": 0,
        "stripped_execution_arg_count": 0,
        "keycode_to_key_count": 0,
        "dropped_clear_text_count": 0,
        "input_text_coordinate_split_count": 0,
        "normalized_pixel_action_count": 0,
        "normalized_pixel_coordinate_count": 0,
        "swipe_coordinate_alias_count": 0,
        "swipe_direction_inferred_count": 0,
        "open_app_legacy_arg_count": 0,
        "mapped_open_app_package_count": 0,
        "press_back_to_press_key_count": 0,
        "answer_to_finished_count": 0,
        "start_settings_to_open_app_count": 0,
    }
    canonical_input = (
        str(run_log.get("schema_version") or "").strip()
        == "omniflow.canonical_run_log.v1"
        and _has_canonical_truth_steps(run_log)
    )
    fallback_display_size = _display_size(run_log.get("device"))
    if fallback_display_size is None:
        observed_sizes = [
            size
            for step in steps
            if isinstance(step, dict)
            for key in (
                "observation_before_act",
                "before_observation",
                "observation",
                "before",
                "state",
            )
            if (size := _display_size(step.get(key))) is not None
        ]
        if observed_sizes:
            fallback_display_size = (
                max(size[0] for size in observed_sizes),
                max(size[1] for size in observed_sizes),
            )
    for step in steps:
        if not isinstance(step, dict):
            continue
        display_size = _step_display_size(step, fallback=fallback_display_size)
        for action in _raw_actions(step):
            args = _action_args(action)
            if args is None:
                continue
            semantic_counts = _migrate_legacy_action_semantics(action, args)
            for key, count in semantic_counts.items():
                counts[key] += count
            for key in EVIDENCE_ONLY_ACTION_ARGS:
                if key in args:
                    args.pop(key)
                    counts["stripped_evidence_arg_count"] += 1
            for key in EXECUTION_ONLY_ACTION_ARGS:
                if key in args:
                    args.pop(key)
                    counts["stripped_execution_arg_count"] += 1
            if "keycode" in args:
                raw_keycode = args.pop("keycode")
                if not str(raw_keycode or "").strip():
                    raise ValueError("source_replay_keycode_empty")
                keycode = _canonical_key(raw_keycode)
                existing = (
                    _canonical_key(args["key"])
                    if str(args.get("key") or "").strip()
                    else ""
                )
                if existing and existing != keycode:
                    raise ValueError("source_replay_keycode_conflicts_with_key")
                args["key"] = keycode
                counts["keycode_to_key_count"] += 1
            elif _action_tool(action) == "press_key" and "key" in args:
                args["key"] = _canonical_key(args["key"])
            if _action_tool(action) == "swipe":
                alias_count, direction_count = _migrate_legacy_swipe_args(args)
                counts["swipe_coordinate_alias_count"] += alias_count
                counts["swipe_direction_inferred_count"] += direction_count
            if not canonical_input:
                normalized_count = _normalize_legacy_pixel_coordinates(
                    action,
                    display_size=display_size,
                )
                if normalized_count:
                    counts["normalized_pixel_action_count"] += 1
                    counts["normalized_pixel_coordinate_count"] += normalized_count
            if _action_tool(action) == "open_app":
                removed, mapped = _drop_legacy_open_app_args(
                    args,
                    app_name_to_package=app_name_to_package,
                )
                counts["open_app_legacy_arg_count"] += removed
                counts["mapped_open_app_package_count"] += mapped
            if drop_clear_text and "clear_text" in args:
                args.pop("clear_text")
                counts["dropped_clear_text_count"] += 1
        expanded_actions: list[dict[str, Any]] = []
        for action in _raw_actions(step):
            args = _action_args(action)
            if _action_tool(action) != "input_text" or args is None:
                expanded_actions.append(action)
                continue
            has_x = args.get("x") is not None
            has_y = args.get("y") is not None
            if has_x != has_y:
                raise ValueError("source_replay_input_text_coordinate_pair_required")
            if not has_x:
                expanded_actions.append(action)
                continue
            click_args = {"x": args.pop("x"), "y": args.pop("y")}
            expanded_actions.append({"type": "click", "params": click_args})
            expanded_actions.append(action)
            counts["input_text_coordinate_split_count"] += 1
        step["executed_actions"] = expanded_actions
    return migrated, counts


def migrate_source_replay_runlog(
    *,
    source_path: str | Path,
    output_root: str | Path,
    drop_clear_text: bool = False,
    app_name_to_package: dict[str, str] | None = None,
    app_mapping_source: str | Path | None = None,
) -> dict[str, Any]:
    source = Path(source_path).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if output.exists():
        raise FileExistsError(f"immutable output already exists:{output}")

    sanitized, migration_counts = _migrate_legacy_action_args(
        _load_object(source),
        drop_clear_text=drop_clear_text,
        app_name_to_package={
            str(key).strip().casefold(): str(value).strip()
            for key, value in (app_name_to_package or {}).items()
            if str(key).strip() and str(value).strip()
        },
    )
    canonical = canonicalize_run_log(import_run_log(sanitized))
    if canonical.get("success") is not True or not canonical.get("steps"):
        raise ValueError("successful_source_replay_runlog_required")
    for step in canonical["steps"]:
        unknown = sorted(set(step) - CANONICAL_STEP_ALLOWED_FIELDS)
        if unknown:
            raise ValueError(f"source_replay_step_fields_invalid:{','.join(unknown)}")

    output.mkdir(parents=True)
    output_run_log = output / "source.replay.run_log.json"
    output_run_log.write_text(
        json.dumps(canonical, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "omniflow.androidworld-source-replay-migration.v1",
        "source_run_log": str(source),
        "source_run_log_sha256": _sha256(source),
        "output_run_log": str(output_run_log),
        "output_run_log_sha256": _sha256(output_run_log),
        "step_count": len(canonical["steps"]),
        "evidence_only_action_args": sorted(EVIDENCE_ONLY_ACTION_ARGS),
        "execution_only_action_args": sorted(EXECUTION_ONLY_ACTION_ARGS),
        "drop_clear_text": drop_clear_text,
        "app_mapping_source": str(app_mapping_source or ""),
        "app_mapping_source_sha256": (
            _sha256(Path(app_mapping_source).expanduser().resolve())
            if app_mapping_source
            else ""
        ),
        **migration_counts,
        "target_inputs_read": False,
    }
    manifest_path = output / "provenance_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-log", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--drop-clear-text", action="store_true")
    parser.add_argument("--android-world-root", type=Path)
    args = parser.parse_args()
    app_mapping: dict[str, str] = {}
    app_mapping_source: Path | None = None
    if args.android_world_root is not None:
        app_mapping, app_mapping_source = load_androidworld_app_name_to_package(
            args.android_world_root
        )
    manifest = migrate_source_replay_runlog(
        source_path=args.source_run_log,
        output_root=args.output_root,
        drop_clear_text=args.drop_clear_text,
        app_name_to_package=app_mapping,
        app_mapping_source=app_mapping_source,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
