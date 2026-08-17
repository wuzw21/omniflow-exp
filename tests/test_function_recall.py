from __future__ import annotations

from omniflow import Action, ActionResult, Function, Observation, OmniFlow, ToolCall
from omniflow.core.model import FunctionStep
from omniflow.functions.assets import FUNCTION_ARTIFACT_VERSION, FunctionStore
from omniflow.functions.recall import (
    GOAL_LEXICAL_WEIGHT,
    PAGE_SIMILARITY_WEIGHT,
    recall_functions,
)


def _page(
    text: str,
    resource_id: str,
    *,
    package: str = "com.example",
    variant: str = "form",
) -> Observation:
    if variant == "camera":
        xml = (
            '<hierarchy><node package="%s" class="android.widget.FrameLayout" '
            'bounds="[0,0][1000,1000]"><node package="%s" '
            'class="android.view.SurfaceView" bounds="[0,0][1000,850]" />'
            '<node package="%s" class="android.widget.ImageButton" '
            'content-desc="%s" resource-id="%s" bounds="[420,860][580,1000]" '
            'clickable="true" /></node></hierarchy>'
        ) % (package, package, package, text, resource_id)
    else:
        xml = (
            '<hierarchy><node package="%s" class="android.widget.LinearLayout" '
            'bounds="[0,0][1000,1000]"><node package="%s" '
            'class="android.widget.TextView" text="%s" '
            'bounds="[50,50][950,180]" /><node package="%s" '
            'class="android.widget.EditText" resource-id="%s" '
            'bounds="[50,240][950,360]" editable="true" />'
            '<node package="%s" class="android.widget.Button" text="Continue" '
            'bounds="[650,800][950,920]" clickable="true" /></node></hierarchy>'
        ) % (package, package, text, package, resource_id, package)
    return Observation(
        xml=xml,
        package_name=package,
        activity_name="MainActivity",
        extra={"display": {"width": 1000, "height": 1000}},
    )


def _function(
    function_id: str,
    name: str,
    description: str,
    source_state_id: str,
    *,
    action: Action | None = None,
) -> Function:
    return Function(
        function_id=function_id,
        name=name,
        description=description,
        steps=(
            FunctionStep(
                step_index=0,
                source_state_id=source_state_id,
                action=action or Action("click", {"x": 500.0, "y": 500.0}),
            ),
        ),
        schema_version=FUNCTION_ARTIFACT_VERSION,
        input_schema={
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        agent_visible=True,
    )


def test_function_recall_uses_one_lexical_and_page_score() -> None:
    current = _page("Account details", "account_form")
    matching = _function(
        "submit_current_form",
        "Tap continue",
        "Advance from the current form.",
        "account_page",
    )
    lexical_but_wrong_page = _function(
        "finish_account_setup",
        "Finish account setup",
        "Finish account setup from this screen.",
        "camera_page",
    )

    result = recall_functions(
        "Finish account setup",
        observation=current,
        functions=(matching, lexical_but_wrong_page),
        source_states={
            "account_page": current,
            "camera_page": _page(
                "Camera shutter",
                "camera_shutter",
                variant="camera",
            ),
        },
    )

    assert [function.id for function in result.functions] == [
        lexical_but_wrong_page.id,
        matching.id,
    ]
    decisions = {
        item["function_id"]: item for item in result.audit["decisions"]
    }
    assert decisions[matching.id]["goal_lexical_score"] == 0.0
    assert decisions[matching.id]["page_similarity"] == 1.0
    assert decisions[matching.id]["score"] == PAGE_SIMILARITY_WEIGHT
    assert decisions[lexical_but_wrong_page.id]["score"] > decisions[matching.id][
        "score"
    ]
    assert result.audit["ranking_weights"] == {
        "page_similarity": PAGE_SIMILARITY_WEIGHT,
        "goal_lexical": GOAL_LEXICAL_WEIGHT,
    }
    assert GOAL_LEXICAL_WEIGHT > PAGE_SIMILARITY_WEIGHT


def test_open_app_function_uses_the_same_page_weighted_score() -> None:
    current = _page("Bluetooth settings", "bluetooth_switch")
    open_app_function = _function(
        "z_open_settings",
        "Continue",
        "Continue from this page.",
        "matching_page",
        action=Action("open_app", {"package_name": "com.android.settings"}),
    )
    click_function = _function(
        "a_click_continue",
        "Continue",
        "Continue from this page.",
        "other_page",
    )

    result = recall_functions(
        "Continue",
        observation=current,
        functions=(click_function, open_app_function),
        source_states={
            "matching_page": current,
            "other_page": _page("Camera", "shutter", variant="camera"),
        },
        limit=1,
    )

    assert result.functions == (open_app_function,)
    decisions = {item["function_id"]: item for item in result.audit["decisions"]}
    assert decisions[open_app_function.id]["page_similarity"] == 1.0
    assert decisions[open_app_function.id]["score"] > decisions[click_function.id][
        "score"
    ]


def test_missing_page_evidence_contributes_zero() -> None:
    function = _function(
        "tap_continue",
        "Tap continue",
        "Advance from the current form.",
        "source_page",
    )

    result = recall_functions(
        "Continue",
        observation=Observation(package_name="com.example"),
        functions=(function,),
        source_states={"source_page": _page("Form", "form")},
    )

    assert result.functions == (function,)
    decision = result.audit["decisions"][0]
    assert decision["page_similarity"] == 0.0
    assert decision["score"] == GOAL_LEXICAL_WEIGHT * decision["goal_lexical_score"]


def test_top_k_includes_zero_score_functions_with_stable_id_tiebreak() -> None:
    function = _function(
        "tap_settings_control",
        "Turn bluetooth on",
        "Use the Settings control to turn bluetooth on.",
        "settings_page",
    )

    second = _function("z_second", "Unrelated", "No shared terms.", "missing")
    first = _function("a_first", "Different", "Still unrelated.", "missing")
    result = recall_functions(
        "Turn bluetooth on",
        observation=Observation(),
        functions=(second, first, function),
        source_states={"settings_page": None, "missing": None},
        limit=2,
    )

    assert [item.id for item in result.functions] == [function.id, first.id]




class _PageChangingHost:
    def __init__(self) -> None:
        self.page = "first"

    def observe(self, **_kwargs: object) -> Observation:
        if self.page == "first":
            return _page("First page", "first_page")
        return _page("Second page", "second_page", variant="camera")

    def act(self, _action: Action) -> ActionResult:
        self.page = "second"
        return ActionResult(True)

    def get_state(self, source_state_id: str) -> Observation | None:
        if source_state_id == "source_first":
            return _page("First page", "first_page")
        if source_state_id == "source_second":
            return _page("Second page", "second_page", variant="camera")
        return None


class _ChangingPagePlanner:
    def __init__(self) -> None:
        self.visible: list[tuple[str, ...]] = []

    def one_step_tool_call(
        self,
        _goal: str,
        _observation: Observation,
        functions: tuple[Function, ...],
        _installed_apps: dict[str, str],
    ) -> ToolCall:
        self.visible.append(tuple(function.id for function in functions))
        if len(self.visible) == 1:
            return ToolCall("click", {"x": 500.0, "y": 500.0})
        return ToolCall("finished", {"content": ""})


def test_runtime_recalls_again_after_page_changes(tmp_path) -> None:
    store = FunctionStore(
        tmp_path / "store.json",
        seed_functions=(
            _function(
            "first_page_action",
            "Use first page control",
            "Operate the visible first-page control.",
            "source_first",
            ),
            _function(
            "second_page_action",
            "Use second page control",
            "Operate the visible second-page control.",
            "source_second",
            ),
        ),
    )
    planner = _ChangingPagePlanner()
    flow = OmniFlow(store.path, host=_PageChangingHost(), planner=planner)

    result = flow.run("Complete the multi-page task")

    assert result.success is True
    assert planner.visible == [
        ("first_page_action", "second_page_action"),
        ("second_page_action", "first_page_action"),
    ]
    assert [
        event["candidate_function_ids"]
        for event in result.detail["function_resolution"]["recall"]["events"]
    ] == [
        ["first_page_action", "second_page_action"],
        ["second_page_action", "first_page_action"],
    ]
