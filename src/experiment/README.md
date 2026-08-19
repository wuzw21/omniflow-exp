# Experiment edit guide

`e2e_task_pipeline.py` is the only scheduler. `androidworld.py` handles one
task/method/device result. `artifact_index.py` owns the sole `data/current.json`
index. `result_registry.py` writes the immutable ledger. `preflight.py` checks
inputs; `source_assets.py` validates source evidence; the MobileGPT and
AppAgent source files prepare their native evidence without owning lifecycle.

Do not add another batch runner, matrix, index, result table, source pool, or
conversion command. Add a result field only in `result_schema.py`, its writer,
and its tests.
