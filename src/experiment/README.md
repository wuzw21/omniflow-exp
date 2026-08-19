# Experiment edit guide

`run_tasks.py` is the only scheduler. `run_task.py` handles one
task/method/device result. `data_index.py` owns the sole `data/current.json`
index and exposes `load_data_index`/`refresh_data_index` operations.
`result_registry.py` writes the immutable ledger. `checks.py` checks inputs;
`source_evidence.py` validates source evidence only; the MobileGPT and AppAgent
source files call their own provider conversion functions and prepare native
evidence without owning lifecycle.
There is no shared `method` string dispatcher in the source module.

Path rules live in `paths.py`: repository-relative inputs use `resolve_path`,
index evidence uses `resolve_reference`, and task/method/device names use
`safe_component`. External AndroidWorld, OmniTransfer, and B-MoCA roots remain
explicit separate inputs; normalization is unified, physical ownership is not.

The valuable single-Function replay is an E2E request mode built by
`run_task.build_task_command(function_id=..., function_arguments=...)`. It
uses the same native launcher, Host, OmniTransfer, checker session, evidence
sealing, and result contract as normal `omniflow`; it is not a second runner.

File-level instructions live in the single `docs/FILE_EDIT_GUIDE.md`. Do not
add one repetitive README beside every Python file; update that guide and the
nearest directory README when an owner or interface changes.

Do not add another batch runner, matrix, index, result table, source pool, or
conversion command. Add a result field only in `result_schema.py`, its writer,
and its tests.
