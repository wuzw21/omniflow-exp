# Transfer edit guide

`page_embedding.py` is the only active page encoder and `runtime.py` loads the
immutable transfer-state catalog stored with evidence. OmniTransfer owns
candidate mapping; OmniFlow owns page checks, selection, execution, and
fallback.

Do not add resource-id lookup, coordinate passthrough, local pooling, or a
second encoder. Transfer failures are explicit runtime failures.

