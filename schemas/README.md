# Schema edit guide

The JSON files under `schemas/oob` are external contracts. A schema change is
always a separate commit with its README and focused tests. Do not introduce a
compatibility alias for a retired interface; convert legacy evidence in memory
and write only the current contract.

Experiment-owned JSON is separate from the shared OOB wire contracts. Its
simple lifecycle is:

| Schema | Meaning | Owner |
| --- | --- | --- |
| `experiment/source_evidence.v2.json` | source RunLog, grounding evidence, and safety facts | `src/experiment/source_evidence.py` |
| `experiment/appagent_memory.v3.json` | AppAgent prepared memory | `src/integrations/appagent.py` |
| `experiment/mobilegpt_memory.v2.json` | MobileGPT prepared memory | `src/integrations/mobilegpt.py` |
| `experiment/run_check.v2.json` | one read-only run readiness report | `src/experiment/checks.py` |
| `experiment/data_index.v2.json` | the single local data index | `src/experiment/data_index.py` |

Provider schemas describe only provider-owned prepared memory. The source
evidence schema never names AppAgent or MobileGPT, and the data index stores
prepared memory records by provider instead of making one provider part of the
index's core meaning.
