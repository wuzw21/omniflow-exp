# Function lifecycle

`assets.py` is the sole Function schema normalizer and Store writer.
`recall.py` only selects from a loaded Store. The lifecycle is:

```text
Function v3 -> save_function -> Function Store -> recall
```

Do not create a catalog writer, generated sidecar, second compiler, or direct
Store mutation. A Store may contain multiple independent Functions. Each
Function embeds its RunLog-shaped observations in `transfer_states`, and steps
reference them through `transfer_state_ids`.
