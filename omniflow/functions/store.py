from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Iterable

from omniflow.core.model import Function
from omniflow.functions.artifact import (
    parse_function_artifact,
    validate_function_artifact,
)

STORE_VERSION = "omniflow.store.v2"


class FunctionStore:
    def __init__(
        self,
        path: str | Path,
        *,
        seed_functions: Iterable[Function] = (),
        replace_seeded: bool = False,
    ):
        self.path = Path(path)
        self.functions: dict[str, Function] = {}
        self.load_errors: dict[str, str] = {}
        self._load()
        self._seed(seed_functions, replace=replace_seeded)

    def list_functions(
        self,
        *,
        offset: int = 0,
        limit: int = 100,
        include_hidden: bool = True,
    ) -> list[Function]:
        start = max(0, int(offset))
        end = start + max(1, min(int(limit), 500))
        functions = (
            self.functions.values()
            if include_hidden
            else (item for item in self.functions.values() if item.agent_visible)
        )
        return sorted(functions, key=lambda item: item.id)[start:end]

    def get_function(self, function_id: str) -> Function | None:
        return self.functions.get(str(function_id or "").strip())

    def put_function(self, value: Function | dict) -> Function:
        function = value if isinstance(value, Function) else parse_function_artifact(value)
        validate_function_artifact(function)
        self.functions[function.id] = function
        self.load_errors.clear()
        self._normalize_visibility()
        self.save()
        return function

    def import_transfer_states(self, catalog_path: str | Path) -> None:
        """Freeze compiler evidence before publishing Functions that reference it."""
        from omniflow.transfer.runtime import (
            TRANSFER_STATE_CATALOG_FILENAME,
            TRANSFER_STATE_CATALOG_VERSION,
            load_transfer_state_catalog,
        )

        destination = self.path.parent / TRANSFER_STATE_CATALOG_FILENAME
        states = load_transfer_state_catalog(destination)
        incoming = load_transfer_state_catalog(catalog_path)
        for state_id, state in incoming.items():
            screenshot = state.get("screenshot_path")
            if screenshot:
                source = Path(screenshot)
                content = source.read_bytes()
                digest = hashlib.sha256(content).hexdigest()
                frozen = self.path.parent / "transfer_screenshots" / (digest + source.suffix)
                frozen.parent.mkdir(parents=True, exist_ok=True)
                if not frozen.exists():
                    temporary = frozen.with_suffix(frozen.suffix + ".tmp")
                    temporary.write_bytes(content)
                    temporary.replace(frozen)
                elif frozen.read_bytes() != content:
                    raise ValueError("function_source_screenshot_conflict")
                state["screenshot_path"] = str(frozen.resolve())
            if state_id in states and states[state_id] != state:
                raise ValueError(f"function_source_state_conflict:{state_id}")
            states[state_id] = state
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".json.tmp")
        temporary.write_text(json.dumps({
            "schema_version": TRANSFER_STATE_CATALOG_VERSION,
            # This is the aggregate Store catalog, not one source trajectory.
            "run_id": self.path.stem,
            "states": states,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(destination)

    def delete_function(self, function_id: str) -> bool:
        normalized = str(function_id or "").strip()
        if normalized not in self.functions:
            return False
        del self.functions[normalized]
        self.load_errors.clear()
        self.save()
        return True

    def clear_functions(self) -> int:
        deleted = len(self.functions)
        self.functions.clear()
        self.load_errors.clear()
        self.save()
        return deleted

    def reload(self) -> None:
        self.functions = {}
        self.load_errors = {}
        self._load()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": STORE_VERSION,
            "functions": {
                key: value.to_dict() for key, value in sorted(self.functions.items())
            },
        }
        temporary = self.path.with_suffix(f"{self.path.suffix}.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(self.path)

    def _load(self) -> None:
        if not self.path.exists():
            return
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("schema_version") != STORE_VERSION:
            raise ValueError("unsupported_store_version")
        raw_functions = payload.get("functions")
        if not isinstance(raw_functions, dict):
            raise ValueError("function_store_functions_must_be_object")
        loaded: dict[str, Function] = {}
        load_errors: dict[str, str] = {}
        for key, value in raw_functions.items():
            try:
                function = parse_function_artifact(value)
                if str(key) != function.id:
                    raise ValueError("function_store_key_mismatch")
            except (TypeError, ValueError) as error:
                load_errors[str(key)] = str(error) or type(error).__name__
                continue
            loaded[function.id] = function
        self.functions = loaded
        self.load_errors = load_errors
        self._normalize_visibility()

    def _seed(
        self,
        seed_functions: Iterable[Function],
        *,
        replace: bool,
    ) -> None:
        changed = False
        for function in seed_functions:
            validate_function_artifact(function)
            if function.id in self.functions and not replace:
                continue
            if self.functions.get(function.id) == function:
                continue
            self.functions[function.id] = function
            changed = True
        if changed:
            self._normalize_visibility()
            self.save()

    def _normalize_visibility(self) -> None:
        """Keep compiler-authored local and complete Functions agent-visible.

        A local fragment may begin at an intermediate GUI state while the
        complete Function begins at the source entry state.  Hiding fragments
        merely because their steps are contained by the complete envelope
        destroys the state-level reuse contract and forces Planner fallback.
        Recall applies page/state matching, so visibility must not act as an
        implicit ranking or admission policy.
        """
        return
