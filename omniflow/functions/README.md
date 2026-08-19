# Function lifecycle

`assets.py` is the sole Function compiler, validator, and Store writer.
`recall.py` only selects from a loaded Store. The lifecycle is:

```text
RunLog -> save_function -> validate -> one Function Store -> recall
```

Do not create a catalog writer, generated sidecar, second compiler, or direct
Store mutation. Generated Functions are never hand-edited; fix the compiler or
policy and regenerate from the same RunLog.

## Migrating old JSON

The current Store is `omniflow.store.v2` and contains one complete Function.
Historical `omniflow.function-bundle.v2` files are not Stores: they contain
actions without the source state IDs required by the current runtime. Convert
them with the successful source RunLog so the converter can realign and validate
every action:

```bash
.venv/bin/python -m omniflow.functions.migrate_store \
  --input /absolute/old/codex_function_bundle.json \
  --source-run-log /absolute/source/run_log.json \
  --output /absolute/new/function_store.json
```

An old `omniflow.store.v2` that contains several Functions is split instead of
silently dropping all but one. Pass a directory as the output:

```bash
.venv/bin/python -m omniflow.functions.migrate_store \
  --input /absolute/old/function_store.json \
  --output /absolute/new-stores/
```

The converter never edits its input and refuses an existing output unless
`--force` is explicit. It also requires the old `transfer_states.json` beside
the input Store, or via `--transfer-states`; this prevents producing a Store
that cannot execute. `function-asset-catalog.v1` is an index, not a Function
Store, and must not be renamed into one. Rebuild the canonical `data/current.json`
index after migration.
