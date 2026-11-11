from pathlib import Path

path = Path(
    "/home/wuzewen/Projects/Omni/releases/OmniFlow-exp-mobilegpt-5250817c/"
    "src/integrations/mobilegpt.py"
)
source = path.read_text()

old = '    open_app_evidence: list[dict[str, Any]] = []\n    raw_steps = list(payload.get("steps") or [])\n'
new = '    open_app_evidence: list[dict[str, Any]] = []\n    final_observation = raw_payload.get("final_observation")\n    if not isinstance(final_observation, dict):\n        final_observation = None\n    raw_steps = list(payload.get("steps") or [])\n'
assert source.count(old) == 1
source = source.replace(old, new)

old = """        next_observation = raw_step.get("next_observation")
        if not isinstance(next_observation, dict) and ordinal + 1 < len(raw_steps):
            next_observation = _observation_for_step(raw_steps[ordinal + 1])
        next_forest = (
"""
new = """        next_observation = raw_step.get("next_observation")
        if not isinstance(next_observation, dict) and ordinal + 1 < len(raw_steps):
            next_observation = _observation_for_step(raw_steps[ordinal + 1])
        if not isinstance(next_observation, dict) and ordinal + 1 == len(raw_steps):
            next_observation = final_observation
        next_forest = (
"""
assert source.count(old) == 1
source = source.replace(old, new)

old = """        next_forest = (
            androidworld_observation_xml(next_observation)
            if isinstance(next_observation, dict)
            else ""
        )
        observation_repaired_from_next = False
"""
new = """        next_forest = (
            androidworld_observation_xml(next_observation)
            if isinstance(next_observation, dict)
            else ""
        )
        action = _recover_indexed_action_from_transition(
            action,
            forest,
            next_forest,
        )
        observation_repaired_from_next = False
"""
assert source.count(old) == 1
source = source.replace(old, new)

helper = r'''
def _actionable_xml_elements(
    xml_text: str,
    *,
    action_type: str,
) -> list[ET.Element]:
    """Return native nodes that can represent one recorded action."""

    try:
        root = ET.fromstring(str(xml_text or ""))
    except ET.ParseError:
        return []
    candidates: list[ET.Element] = []
    for element in root.iter():
        if _bounds(element.get("bounds")) is None:
            continue
        if action_type == "input_text":
            is_editable = str(element.get("editable") or "").lower() == "true"
            is_input = "edittext" in str(element.get("class") or "").casefold()
            if is_editable or is_input:
                candidates.append(element)
            continue
        if action_type in {"click", "double_tap", "long_press"}:
            is_clickable = str(element.get("clickable") or "").lower() == "true"
            is_checkable = str(element.get("checkable") or "").lower() == "true"
            is_long_clickable = (
                str(element.get("long-clickable") or "").lower() == "true"
            )
            if is_clickable or is_checkable or is_long_clickable:
                candidates.append(element)
    return candidates


def _transition_match_score(
    current: ET.Element,
    next_elements: Sequence[ET.Element],
) -> int:
    """Score whether a current node is the node whose state changed next."""

    current_bounds = _bounds(current.get("bounds"))
    current_resource = str(current.get("resource-id") or "").strip()
    current_class = str(current.get("class") or "").strip()
    current_semantics = tuple(
        str(current.get(name) or "")
        for name in ("text", "content-desc", "checked", "selected", "clickable")
    )
    best = 0
    for candidate in next_elements:
        candidate_bounds = _bounds(candidate.get("bounds"))
        candidate_resource = str(candidate.get("resource-id") or "").strip()
        same_resource = bool(current_resource and current_resource == candidate_resource)
        same_geometry = current_bounds is not None and current_bounds == candidate_bounds
        same_class = bool(current_class and current_class == candidate.get("class"))
        if not same_resource and not (same_geometry and same_class):
            continue
        candidate_semantics = tuple(
            str(candidate.get(name) or "")
            for name in ("text", "content-desc", "checked", "selected", "clickable")
        )
        if current_semantics == candidate_semantics:
            continue
        score = 5 if same_resource else 3
        if current.get("selected") != candidate.get("selected"):
            score += 2
        if current.get("checked") != candidate.get("checked"):
            score += 2
        best = max(best, score)
    return best


def _recover_indexed_action_from_transition(
    action: dict[str, Any],
    source_xml: str,
    next_xml: str,
) -> dict[str, Any]:
    """Recover a missing action point from settled XML evidence."""

    if _point(action) is not None or not next_xml:
        return action
    action_type = _action_type(action)
    if action_type not in {"click", "double_tap", "input_text", "long_press"}:
        return action
    candidates = _actionable_xml_elements(source_xml, action_type=action_type)
    try:
        next_root = ET.fromstring(str(next_xml or ""))
    except ET.ParseError:
        return action
    next_elements = list(next_root.iter())
    scored = [
        (_transition_match_score(element, next_elements), element)
        for element in candidates
    ]
    scored = [(score, element) for score, element in scored if score > 0]
    if not scored:
        return action
    best_score = max(score for score, _ in scored)
    best = [element for score, element in scored if score == best_score]
    if len(best) != 1:
        return action
    bounds = _bounds(best[0].get("bounds"))
    if bounds is None:
        return action
    left, top, right, bottom = bounds
    recovered = dict(action)
    recovered["x"] = (left + right) / 2.0
    recovered["y"] = (top + bottom) / 2.0
    return recovered
'''
anchor = '''    grounded["y"] = (top + bottom) / 2.0
    return grounded


def _element_contains_action'''
assert source.count(anchor) == 1
source = source.replace(anchor, anchor.replace(
    "\n\ndef _element_contains_action",
    "\n\n" + helper + "\n\ndef _element_contains_action",
), 1)
path.write_text(source)

