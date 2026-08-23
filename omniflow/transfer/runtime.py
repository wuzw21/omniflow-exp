from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any
import xml.etree.ElementTree as ET

TRANSFER_STATE_CATALOG_FILENAME = "transfer_states.json"
TRANSFER_STATE_CATALOG_VERSION = "omniflow.transfer-state-catalog.v1"
_TRANSFER_STATE_FIELDS = {
    "state_id",
    "xml",
    "package_name",
    "activity_name",
    "display",
    "screenshot_path",
}


def capture_transfer_state(observation: Any) -> dict[str, Any]:
    extra = observation.extra if isinstance(observation.extra, dict) else {}
    androidworld_state = (
        extra.get("androidworld_state")
        if isinstance(extra.get("androidworld_state"), dict)
        else {}
    )
    pixels = (
        androidworld_state.get("screenshot")
        if isinstance(androidworld_state.get("screenshot"), dict)
        else androidworld_state.get("pixels")
        if isinstance(androidworld_state.get("pixels"), dict)
        else {}
    )
    identity = {
        key: value
        for key, value in observation.to_dict().items()
        if key in {"xml", "package_name", "activity_name"} and value not in {None, ""}
    }
    identity.update(
        {
            key: value
            for key, value in extra.items()
            if key in {"display", "screenshot_path"}
            and value is not None
            and value != ""
        }
    )
    screenshot_path = str(
        extra.get("screenshot_path") or pixels.get("path") or ""
    ).strip()
    if screenshot_path:
        identity["screenshot_path"] = screenshot_path
    explicit_state_id = str(extra.get("state_id") or "").strip()
    encoded = json.dumps(
        identity,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    state = {
        "state_id": explicit_state_id
        or "state_" + hashlib.sha256(encoded.encode()).hexdigest()[:20]
    }
    for key in ("xml", "package_name", "activity_name"):
        value = identity.get(key)
        if isinstance(value, str) and value:
            state[key] = value
    display = identity.get("display")
    if isinstance(display, dict) and set(display) == {"width", "height"}:
        state["display"] = dict(display)
    screenshot_path = identity.get("screenshot_path")
    if isinstance(screenshot_path, str) and screenshot_path:
        state["screenshot_path"] = screenshot_path
    return state


def load_transfer_state_catalog(path: str | Path) -> dict[str, dict[str, Any]]:
    catalog_path = Path(path).expanduser().resolve()
    if not catalog_path.is_file():
        return {}
    payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version",
        "run_id",
        "states",
    }:
        raise ValueError("transfer_state_catalog_contract_invalid")
    if payload.get("schema_version") != TRANSFER_STATE_CATALOG_VERSION:
        raise ValueError("unsupported_transfer_state_catalog_version")
    if not isinstance(payload.get("run_id"), str) or not payload["run_id"].strip():
        raise ValueError("transfer_state_catalog_run_id_required")
    raw_states = payload.get("states")
    if not isinstance(raw_states, dict):
        raise ValueError("transfer_state_catalog_states_invalid")
    states: dict[str, dict[str, Any]] = {}
    for state_id, value in raw_states.items():
        state = _canonicalize_transfer_state(value)
        if str(state_id) != state["state_id"]:
            raise ValueError("transfer_state_catalog_key_mismatch")
        states[state["state_id"]] = state
    return states


def _canonicalize_transfer_state(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("transfer_state_must_be_object")
    unknown = sorted(set(value) - _TRANSFER_STATE_FIELDS)
    if unknown:
        raise ValueError(f"transfer_state_unknown_fields:{','.join(unknown)}")
    state_id = str(value.get("state_id") or "").strip()
    if not state_id:
        raise ValueError("transfer_state_id_required")
    state: dict[str, Any] = {"state_id": state_id}
    for key in ("xml", "package_name", "activity_name"):
        item = value.get(key)
        if item is None:
            continue
        if not isinstance(item, str):
            raise ValueError(f"transfer_state_{key}_must_be_string")
        state[key] = item
    display = value.get("display")
    if display is not None:
        if not isinstance(display, dict) or set(display) != {"width", "height"}:
            raise ValueError("transfer_state_display_invalid")
        width = display.get("width")
        height = display.get("height")
    else:
        width = height = None
    if width is not None or height is not None:
        if not all(
            isinstance(item, int) and not isinstance(item, bool) and item > 0
            for item in (width, height)
        ):
            raise ValueError("transfer_state_display_invalid")
        state["display"] = {"width": width, "height": height}
    screenshot_path = value.get("screenshot_path")
    if screenshot_path is not None:
        if not isinstance(screenshot_path, str) or not screenshot_path.strip():
            raise ValueError("transfer_state_screenshot_path_invalid")
        state["screenshot_path"] = screenshot_path.strip()
    return state


def load_omnitransfer() -> Any:
    configured_root = str(os.environ.get("OMNITRANSFER_ROOT") or "").strip()
    packaged_root = (
        Path(__file__).resolve().parents[3] / ".runtime" / "omnitransfer"
    ).resolve()
    root = (
        Path(configured_root)
        if configured_root
        else packaged_root
        if (packaged_root / "src" / "omnitransfer").is_dir()
        else Path.home() / "Projects" / "Omni" / "OmniTransfer"
    )
    return _load_omnitransfer_from_root(root)


def _load_omnitransfer_from_root(root: Path) -> Any:
    root = root.expanduser().resolve()
    source_root = (root / "src").resolve()
    package_root = source_root / "omnitransfer"
    if not package_root.is_dir():
        raise RuntimeError(f"omnitransfer_root_missing:{root}")
    loaded = sys.modules.get("omnitransfer")
    if loaded is not None and _module_is_from(loaded, package_root):
        return loaded
    for name in tuple(sys.modules):
        if name == "omnitransfer" or name.startswith("omnitransfer."):
            del sys.modules[name]
    source_path = str(source_root)
    sys.path[:] = [
        item for item in sys.path if str(Path(item or ".").resolve()) != source_path
    ]
    sys.path.insert(0, source_path)
    importlib.invalidate_caches()
    module = importlib.import_module("omnitransfer")
    if not _module_is_from(module, package_root):
        raise RuntimeError(f"omnitransfer_import_outside_root:{module.__file__}")
    return module


def _module_is_from(module: Any, package_root: Path) -> bool:
    module_path = Path(str(getattr(module, "__file__", "") or "")).resolve()
    try:
        module_path.relative_to(package_root)
    except ValueError:
        return False
    return True


def transfer_action(
    *,
    source_xml: str,
    target_xml: str,
    source_point: tuple[float, float] | None = None,
    source_element_id: str | None = None,
    source_offset: tuple[float, float] | None = None,
    source_screenshot_path: str | None = None,
    target_screenshot_path: str | None = None,
    source_visual_rgb: dict[str, Any] | None = None,
    target_visual_rgb: dict[str, Any] | None = None,
    action_type: str = "click",
    top_k: int = 1,
) -> dict[str, Any]:
    request = {
        "source_xml": source_xml,
        "target_xml": target_xml,
        "source_point": source_point,
        "source_element_id": source_element_id,
        "source_offset": source_offset,
        "source_screenshot_path": source_screenshot_path,
        "target_screenshot_path": target_screenshot_path,
        "source_visual_rgb": source_visual_rgb,
        "target_visual_rgb": target_visual_rgb,
        "action_type": action_type,
        "top_k": top_k,
    }
    module = load_omnitransfer()
    rank_candidates = getattr(module, "rank_action_candidates", None)
    if not callable(rank_candidates):
        raise RuntimeError("omnitransfer_candidate_ranking_unavailable")
    ranking = rank_candidates(**request)
    if not isinstance(ranking, dict):
        raise RuntimeError("omnitransfer_result_invalid")
    return _select_transfer_candidate(ranking, request)


def preflight_omnitransfer() -> dict[str, Any]:
    """Exercise the candidate API while keeping readiness policy in OmniFlow."""

    module = load_omnitransfer()
    rank_candidates = getattr(module, "rank_action_candidates", None)
    if not callable(rank_candidates):
        raise RuntimeError("omnitransfer_candidate_ranking_unavailable")
    source_xml = (
        '<hierarchy bounds="[0,0][100,100]">'
        '<node resource-id="preflight:id/search" text="Search" '
        'class="android.widget.Button" clickable="true" enabled="true" '
        'bounds="[10,20][50,60]" />'
        "</hierarchy>"
    )
    target_xml = (
        '<hierarchy bounds="[0,0][200,400]">'
        '<node resource-id="preflight:id/search" text="Search" '
        'class="android.widget.Button" clickable="true" enabled="true" '
        'bounds="[40,100][120,260]" />'
        "</hierarchy>"
    )
    result = rank_candidates(
        source_xml=source_xml,
        target_xml=target_xml,
        source_point=(30.0, 40.0),
    )
    if not isinstance(result, dict):
        raise RuntimeError("omnitransfer_result_invalid")
    candidates = result.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        reason = str(result.get("reason") or "candidates_missing")
        error = str(result.get("error") or "")
        raise RuntimeError(
            f"omnitransfer_preflight_failed:{reason}:{error}".rstrip(":")
        )
    return {
        "ready": True,
        "backend": str(result.get("matcher_backend") or "unknown"),
        "mapping_mode": str(result.get("mapping_mode") or ""),
        "candidate_ranking_schema": str(result.get("schema_version") or ""),
        "matcher_release": str(result.get("matcher_release") or ""),
        "matcher_checkpoint_sha256": str(
            result.get("matcher_checkpoint_sha256") or ""
        ),
        "matcher_feature_schema": str(result.get("matcher_feature_schema") or ""),
        "matcher_feature_schema_sha256": str(
            result.get("matcher_feature_schema_sha256") or ""
        ),
    }


def _select_transfer_candidate(
    ranking: dict[str, Any],
    _request: dict[str, Any],
) -> dict[str, Any]:
    """Apply OmniFlow policy to an OmniTransfer candidate ranking."""

    candidates = ranking.get("candidates")
    if not isinstance(candidates, list):
        raise RuntimeError("omnitransfer_candidates_invalid")
    if not candidates:
        return {
            **ranking,
            "mapped": False,
            "reason": str(
                ranking.get("reason")
                or ranking.get("status")
                or "target_candidates_missing"
            ),
        }
    selected = candidates[0]
    if not isinstance(selected, dict):
        raise RuntimeError("omnitransfer_candidate_invalid")
    try:
        new_x = float(selected["new_x"])
        new_y = float(selected["new_y"])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("omnitransfer_candidate_coordinates_invalid") from error
    bounds = list(selected.get("bbox") or ())
    return {
        **ranking,
        "mapped": True,
        "selection_policy": "omniflow_top_candidate",
        "new_x": new_x,
        "new_y": new_y,
        "target_candidate_id": str(selected.get("candidate_id") or ""),
        "target_bbox": bounds,
        "target_center": (
            [(float(bounds[0]) + float(bounds[2])) / 2.0,
             (float(bounds[1]) + float(bounds[3])) / 2.0]
            if len(bounds) == 4
            else []
        ),
}
