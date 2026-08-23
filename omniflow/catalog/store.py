from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
import json
import lzma
from pathlib import Path
from typing import Any

from omniflow.core.model import Function, Observation
from omniflow.functions.artifact import parse_function_artifact

CATALOG_POINTER_SCHEMA = "omniflow.catalog-pointer.v1"
CATALOG_MANIFEST_SCHEMA = "omniflow.catalog-manifest.v1"
FUNCTION_STORE_SCHEMA = "omniflow.store.v2"


@dataclass(frozen=True)
class CatalogSnapshot:
    release_id: str
    root: Path
    functions: dict[str, Function]
    states: dict[str, Observation]
    manifest: dict[str, Any]

    def function_store_payload(self) -> dict[str, Any]:
        return {
            "schema_version": FUNCTION_STORE_SCHEMA,
            "functions": {
                key: value.to_dict() for key, value in sorted(self.functions.items())
            },
        }

    def get_state(self, state_id: str) -> Observation | None:
        return self.states.get(str(state_id or "").strip())


def default_catalog_root() -> Path:
    pointer_path = Path(__file__).with_name("default.json")
    pointer = _read_object(pointer_path)
    if pointer.get("schema_version") != CATALOG_POINTER_SCHEMA:
        raise ValueError("unsupported_catalog_pointer_version")
    release_id = str(pointer.get("release_id") or "").strip()
    if not release_id or Path(release_id).name != release_id:
        raise ValueError("catalog_release_id_invalid")
    return Path(__file__).with_name("releases") / release_id


def load_default_catalog() -> CatalogSnapshot:
    return load_catalog(default_catalog_root())


def load_catalog(root: str | Path) -> CatalogSnapshot:
    catalog_root = Path(root)
    manifest = _read_object(catalog_root / "manifest.json")
    if manifest.get("schema_version") != CATALOG_MANIFEST_SCHEMA:
        raise ValueError("unsupported_catalog_manifest_version")
    release_id = str(manifest.get("release_id") or "").strip()
    if release_id != catalog_root.name:
        raise ValueError("catalog_release_id_mismatch")
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != {
        "function_store.json",
        "states.json.xz.b64",
    }:
        raise ValueError("catalog_manifest_files_invalid")
    for name, expected_sha256 in files.items():
        path = catalog_root / name
        actual_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_sha256 != str(expected_sha256 or "").strip().lower():
            raise ValueError(f"catalog_file_checksum_mismatch:{name}")

    store = _read_object(catalog_root / "function_store.json")
    if store.get("schema_version") != FUNCTION_STORE_SCHEMA:
        raise ValueError("unsupported_catalog_function_store_version")
    raw_functions = store.get("functions")
    if not isinstance(raw_functions, dict):
        raise ValueError("catalog_functions_invalid")
    functions: dict[str, Function] = {}
    for key, raw_function in raw_functions.items():
        function = parse_function_artifact(raw_function)
        if function.id != str(key):
            raise ValueError("catalog_function_key_mismatch")
        functions[function.id] = function

    encoded_states = (catalog_root / "states.json.xz.b64").read_bytes()
    try:
        raw_states = json.loads(lzma.decompress(base64.b64decode(encoded_states)))
    except (ValueError, lzma.LZMAError, json.JSONDecodeError) as error:
        raise ValueError("catalog_states_invalid") from error
    if not isinstance(raw_states, dict):
        raise ValueError("catalog_states_invalid")
    states = {
        str(key): Observation.from_value(value)
        for key, value in raw_states.items()
        if isinstance(value, dict)
    }
    referenced_state_ids = {
        step.source_state_id
        for function in functions.values()
        for step in function.steps
        if step.source_state_id
    }
    missing_state_ids = sorted(referenced_state_ids - states.keys())
    if missing_state_ids:
        raise ValueError(f"catalog_source_states_missing:{','.join(missing_state_ids)}")
    return CatalogSnapshot(
        release_id=release_id,
        root=catalog_root,
        functions=functions,
        states=states,
        manifest=manifest,
    )


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"catalog_json_object_required:{path.name}")
    return payload
