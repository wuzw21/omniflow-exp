"""Select natural same-evidence/different-question controls before model inference."""
from collections import defaultdict


def select_pairs(examples, limit=4):
    if limit < 1:
        raise ValueError("Positive pair limit required")
    groups = defaultdict(list)
    for example in examples:
        groups[(example.sharer_context, example.receiver_context)].append(example)
    selected = []
    # Preserve public validation row order. No model predictions participate.
    for group in groups.values():
        first = group[0]
        old_answers = {s.strip().lower() for s in first.answers}
        for current in group[1:]:
            new_answers = {s.strip().lower() for s in current.answers}
            if first.question != current.question and old_answers.isdisjoint(new_answers):
                selected.append((first, current))
                break
        if len(selected) == limit:
            break
    if len(selected) != limit:
        raise ValueError("Not enough same-side evidence pairs with different answers")
    return selected
