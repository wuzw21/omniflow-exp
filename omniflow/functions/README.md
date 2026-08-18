# Function lifecycle

`assets.py` is the sole Function compiler, validator, and Store writer.
`recall.py` only selects from a loaded Store. The lifecycle is:

```text
RunLog -> save_function -> validate -> one Function Store -> recall
```

Do not create a catalog writer, generated sidecar, second compiler, or direct
Store mutation. Generated Functions are never hand-edited; fix the compiler or
policy and regenerate from the same RunLog.

