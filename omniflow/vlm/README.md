# VLM edit guide

`planner.py` exposes the model-facing tool space and validates calls;
`context.py` builds evidence; `model_config.py` resolves the canonical model
endpoint; `usage.py` accounts for tokens. Prompts and external baseline
contracts are frozen unless the protocol explicitly changes them.

Keep transport and policy separate: a method adapter may supply a Planner, but
it must not duplicate Planner history, retries, or completion logic.

