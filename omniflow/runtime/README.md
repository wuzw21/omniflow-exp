# Runtime edit guide

`engine.py` owns one persistent Planner/Function lifecycle. `execution.py`
owns transfer, semantic grounding, action dispatch, and normal VLM fallback.
`core.py` is the low-level action primitive; `checker.py` evaluates only rules
registered on the active Function; `semantic_grounding.py` resolves visible
targets.

Do not create a method-specific executor or a second resume state. Missing
OmniTransfer evidence must return to the normal fallback path and must never
execute source coordinates.

