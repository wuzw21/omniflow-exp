# Experiment edit guide

`e2e_task_pipeline.py` is the only scheduler. `androidworld.py` handles one
task/method/device result. `artifact_index.py` owns the sole `data/current.json`
index and exposes the clearly named `load_artifact_index`/
`refresh_artifact_index` operations. `result_registry.py` writes the immutable
ledger. `preflight.py` checks inputs; `source_assets.py` validates source
evidence; the MobileGPT and AppAgent source files prepare their native evidence
without owning lifecycle.

Path rules live in `paths.py`: repository-relative inputs use `resolve_path`,
index evidence uses `resolve_reference`, and task/method/device names use
`safe_component`. External AndroidWorld, OmniTransfer, and B-MoCA roots remain
explicit separate inputs; normalization is unified, physical ownership is not.

The valuable single-Function replay is an E2E request mode built by
`androidworld.build_e2e_command(function_id=..., function_arguments=...)`. It
uses the same native launcher, Host, OmniTransfer, checker session, evidence
sealing, and result contract as normal `omniflow`; it is not a second runner.

File-level instructions live in the single `docs/FILE_EDIT_GUIDE.md`. Do not
add one repetitive README beside every Python file; update that guide and the
nearest directory README when an owner or interface changes.

Do not add another batch runner, matrix, index, result table, source pool, or
conversion command. Add a result field only in `result_schema.py`, its writer,
and its tests.
