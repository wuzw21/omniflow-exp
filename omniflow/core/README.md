# Core edit guide

`core` owns shared contracts. `model.py` is the domain interface; `schemas.py`
normalizes external action payloads; `trajectory.py` validates RunLogs;
`config.py` resolves runtime settings; `androidworld_accessibility.py` only
projects native accessibility data.

Add a field here only when every caller shares the invariant. Update the
matching JSON schema and focused tests in the same change. Do not add method,
device, or artifact-specific fields to the core models.

