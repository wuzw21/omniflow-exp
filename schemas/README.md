# Schema edit guide

The JSON files under `schemas/oob` are external contracts. A schema change is
always a separate commit with its README and focused tests. Do not introduce a
compatibility alias for a retired interface; convert legacy evidence in memory
and write only the current contract.

Experiment-owned JSON is separate from the shared OOB wire contracts. Its
simple lifecycle is:

| Schema | Meaning | Owner |
| --- | --- | --- |
| `experiment/appagent_memory.v3.json` | AppAgent prepared memory | `src/integrations/appagent.py` |
| `experiment/mobilegpt_memory.v2.json` | MobileGPT prepared memory | `src/integrations/mobilegpt.py` |

Provider schemas describe only provider-owned prepared memory. Runtime inputs
are passed directly to the public launcher; there is no experiment index.
