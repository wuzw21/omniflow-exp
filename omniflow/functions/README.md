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
that cannot execute. Every migrated Store also receives its source
`run_log.json`; a Store without source-call evidence is reported as blocked.

For a whole historical directory, use one dry run first:

```bash
.venv/bin/python -m omniflow.functions.migrate_store \
  --input-root /absolute/old-data \
  --output /absolute/new-data \
  --dry-run --report /absolute/migration-report.json
```

The scanner recognizes current Stores, old bundles, and
`function-asset-catalog.v1`. A catalog is only an old index: its referenced
Store, RunLog, and transfer states are migrated; the catalog itself is never
copied into runtime data. Duplicate catalog objects are scanned once. Missing
evidence, missing source arguments, ambiguous legacy action alignment, and
existing outputs are listed as `blocked` with a reason instead of being
silently skipped. Multi-Function historical Stores are split into separate
canonical attempts so the data index can register them.

After reviewing the report, rerun without `--dry-run`. Then rebuild the one
runtime index with the existing `src.experiment.data_index refresh` command;
the migration tool deliberately does not become a second index writer.
