# Schema edit guide

The JSON files under `schemas/oob` are external contracts. A schema change is
always a separate commit with its README and focused tests. Do not introduce a
compatibility alias for a retired interface; convert legacy evidence in memory
and write only the current contract.

