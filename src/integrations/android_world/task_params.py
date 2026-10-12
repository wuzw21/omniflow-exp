"""Rehydration of archived AndroidWorld task parameters (split out of run_episode).

Moved verbatim from ``src.integrations.android_world.run_episode``; that module re-exports these names.
"""

from __future__ import annotations

import ast
import random
import re


def _rehydrate_task_params(
    *,
    params: dict[str, object],
    task_type: type[object] | None = None,
) -> dict[str, object]:
    """Restore AndroidWorld task params that were serialized through JSON."""

    hydrated = dict(params)
    # JSON task-parameter transport cannot carry the PIL image used by
    # ``SaveCopyOfReceiptTaskEval``; ``run_task`` serializes it as a stable
    # descriptive string.  The official initializer only needs an image
    # object to materialize the fixture (the validator checks the filename),
    # so regenerate an equivalent receipt image at the lifecycle boundary.
    # This keeps the formal task semantics intact while avoiding an
    # environment-startup ``'str' object has no attribute save`` failure.
    if isinstance(hydrated.get("receipt_image"), str):
        try:
            from android_world.task_evals.utils import receipt_generator

            receipt_image, _ = receipt_generator.create_receipt()
            hydrated["receipt_image"] = receipt_image
        except Exception:
            # Leave non-placeholder strings untouched; the task initializer
            # will report the original parameter error with full evidence.
            pass
    is_information_retrieval = task_type is not None and any(
        base.__name__ == "InformationRetrieval"
        for base in getattr(task_type, "__mro__", ())
    )
    if is_information_retrieval:
        if "seed" not in hydrated:
            raise ValueError("InformationRetrieval task params require seed")
        from android_world.task_evals.information_retrieval import datetime_utils
        task_template = task_type(hydrated).task_template
        canonical_params = dict(hydrated)
        for task_param in task_template.task_params:
            key = str(task_param.name)
            if key not in hydrated:
                raise ValueError(
                    "InformationRetrieval task params missing canonical field: "
                    f"{key}"
                )
            serialized_value = hydrated[key]
            possible_values = list(task_param.possible_values)
            matches: list[str] = []
            serialized_date = None
            weekday_name = None
            if isinstance(serialized_value, str):
                try:
                    serialized_date = datetime_utils.get_date(serialized_value)
                except (TypeError, ValueError):
                    serialized_date = None
                weekday_match = re.fullmatch(
                    r"(?:this |the )?(monday|tuesday|wednesday|thursday|friday|saturday|sunday)(?: after next)?",
                    serialized_value.strip().lower(),
                )
                if weekday_match and "after next" not in serialized_value.strip().lower():
                    weekday_name = weekday_match.group(1)
            for possible_value in possible_values:
                equivalent = serialized_value == possible_value
                if not equivalent and isinstance(serialized_value, str):
                    try:
                        equivalent = (
                            possible_value.format(**hydrated) == serialized_value
                        )
                    except (KeyError, ValueError):
                        equivalent = False
                    if not equivalent:
                        # Prefer an exact interpretation of relative date
                        # phrases (for example ``the Sunday after next``)
                        # before falling back to weekday-only compatibility.
                        # Comparing only weekdays would make every Sunday in
                        # the candidate window match and incorrectly reject
                        # an otherwise valid serialized task parameter.
                        equivalent = (
                            serialized_date is not None
                            and datetime_utils.get_date(possible_value)
                            == serialized_date
                        )
                    if not equivalent and serialized_date is None:
                        # Older successful RunLogs sometimes store a bare
                        # weekday (for example ``Monday``), while the
                        # current task registry stores the corresponding
                        # concrete date.  Resolve against the task's own
                        # canonical candidates, rather than assuming the
                        # next occurrence from the device clock.  This is
                        # important for windows that contain the previous
                        # week (Monday 9 Oct) but not the next day (Monday
                        # 16 Oct).
                        weekday_match = re.fullmatch(
                            r"(?:this |the )?(monday|tuesday|wednesday|thursday|friday|saturday|sunday)(?: after next)?",
                            serialized_value.strip().lower(),
                        )
                        if weekday_name:
                            try:
                                candidate_date = datetime_utils.get_date(
                                    possible_value
                                )
                            except (TypeError, ValueError):
                                candidate_date = None
                            if candidate_date is not None:
                                equivalent = (
                                    candidate_date.strftime("%A").lower()
                                    == weekday_name
                                )
                    if not equivalent:
                        # The exact relative-date comparison above is the
                        # primary path.  At this point only the legacy
                        # weekday-compatible fallback remains applicable.
                        equivalent = False
                if equivalent and possible_value not in matches:
                    matches.append(possible_value)
            # If a bare weekday could not be parsed to an exact candidate,
            # retain the legacy compatibility rule as a second pass.  This
            # supports old traces from a previous calendar window without
            # turning an exact relative phrase into an ambiguous match.
            if not matches and weekday_name:
                for possible_value in possible_values:
                    try:
                        candidate_date = datetime_utils.get_date(possible_value)
                    except (TypeError, ValueError):
                        candidate_date = None
                    if (
                        candidate_date is not None
                        and candidate_date.strftime("%A").lower() == weekday_name
                    ):
                        matches.append(possible_value)
            if len(matches) != 1:
                raise ValueError(
                    "InformationRetrieval task params do not match one canonical "
                    f"value: {key}"
                )
            canonical_params[key] = matches[0]
        hydrated = canonical_params
    if task_type is not None and task_type.__name__ == "MarkorTranscribeReceipt":
        if "seed" not in hydrated:
            raise ValueError("MarkorTranscribeReceipt task params require seed")
        random_state = random.getstate()
        try:
            random.seed(int(hydrated["seed"]))
            generated = task_type.generate_random_params()
        finally:
            random.setstate(random_state)
        for key, value in generated.items():
            if key == "img":
                continue
            if key not in hydrated or hydrated[key] != value:
                raise ValueError(
                    "MarkorTranscribeReceipt generated params do not match "
                    f"canonical source: {key}"
                )
        image = generated.get("img")
        if image is None:
            raise ValueError("MarkorTranscribeReceipt generated img is missing")
        hydrated["img"] = image
    serialized_rows = [
        row
        for key in ("row_objects", "noise_row_objects")
        for row in (hydrated.get(key) if isinstance(hydrated.get(key), list) else [])
        if isinstance(row, (dict, str))
    ]
    if not serialized_rows:
        return hydrated
    from android_world.task_evals.utils import sqlite_schema_utils

    row_types = {
        "CalendarEvent": sqlite_schema_utils.CalendarEvent,
        "Expense": sqlite_schema_utils.Expense,
        "Recipe": sqlite_schema_utils.Recipe,
    }

    def deserialize_row(value: object) -> object:
        """Restore legacy JSON strings emitted by AndroidWorld row reprs."""

        if not isinstance(value, str):
            return value
        text = value.strip()
        if not text or "(" not in text or not text.endswith(")"):
            return value
        try:
            expression = ast.parse(text, mode="eval").body
        except (SyntaxError, ValueError):
            return value
        if not isinstance(expression, ast.Call) or not isinstance(
            expression.func, ast.Name
        ):
            return value
        row_type = row_types.get(expression.func.id)
        if row_type is None or expression.args or any(
            keyword.arg is None for keyword in expression.keywords
        ):
            return value
        try:
            fields = {
                str(keyword.arg): ast.literal_eval(keyword.value)
                for keyword in expression.keywords
            }
            return row_type(**fields)
        except (TypeError, ValueError, SyntaxError):
            return value

    def expense_row(row: object) -> object:
        if not isinstance(row, dict):
            return row
        allowed = {
            "name",
            "amount",
            "category",
            "note",
            "created_date",
            "modified_date",
            "expense_id",
        }
        payload = {key: row[key] for key in allowed if key in row}
        return sqlite_schema_utils.Expense(**payload)

    def calendar_event(row: object) -> object:
        if not isinstance(row, dict):
            return row
        allowed = {
            "start_ts",
            "end_ts",
            "title",
            "location",
            "description",
            "repeat_interval",
            "repeat_rule",
            "reminder_1_minutes",
            "reminder_2_minutes",
            "reminder_3_minutes",
            "reminder_1_type",
            "reminder_2_type",
            "reminder_3_type",
            "repeat_limit",
            "repetition_exceptions",
            "attendees",
            "import_id",
            "time_zone",
            "flags",
            "event_type",
            "parent_id",
            "last_updated",
            "source",
            "availability",
            "color",
            "type",
            "id",
        }
        payload = {key: row[key] for key in allowed if key in row}
        return sqlite_schema_utils.CalendarEvent(**payload)

    def recipe_row(row: object) -> object:
        if not isinstance(row, dict):
            return row
        allowed = {
            "title",
            "description",
            "servings",
            "preparationTime",
            "source",
            "ingredients",
            "directions",
            "favorite",
            "imageName",
            "recipeId",
        }
        payload = {key: row[key] for key in allowed if key in row}
        return sqlite_schema_utils.Recipe(**payload)

    def hydrate_row(row: object) -> object:
        row = deserialize_row(row)
        if not isinstance(row, dict):
            return row
        keys = set(row)
        if keys & {"start_ts", "end_ts", "repeat_rule", "event_type"}:
            return calendar_event(row)
        if keys & {"ingredients", "directions", "preparationTime", "recipeId"}:
            return recipe_row(row)
        if keys & {"amount", "expense_id", "created_date", "modified_date"}:
            return expense_row(row)
        return row

    for key in ("row_objects", "noise_row_objects"):
        rows = hydrated.get(key)
        if isinstance(rows, list):
            hydrated[key] = [hydrate_row(row) for row in rows]
    return hydrated
