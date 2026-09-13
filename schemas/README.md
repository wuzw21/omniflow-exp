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

AndroidWorld result rows keep validator outcomes separate from method/process
outcomes. `official_validator_success` and
`androidworld_validator_result.success` report the official reward threshold
(`reward > 0.5`), even if the method subsequently fails. MobileGPT's
`mobilegpt_protocol.task_finished`, `process_returncode`, `classification`,
and `failure_reason` retain its execution outcome independently. A positive
official reward must not turn a failed process exit into zero; a nonzero exit
must not rewrite a positive official reward as validator failure. This corrects
the MobileGPT implementation to the existing result contract without adding
fields or changing historical evidence.
# Function recovery evidence

`diagnostics.function_resume` may contain `omniflow.function-resume.v1`.
Its `events` enumerate actual kernel invocations (fresh and resumed), with
Function id, start step, trace interval, dispatch outcome and execution result.
`after_failure` means a prior Function invocation failed in the same task; it
does not assert that the later action is semantically correct. `attempt_count`
and `success_count` count resumed invocations, not task completion. Legacy
unversioned diagnostics remain readable; absent counts are unknown, not zero.
An interrupted invocation may lack a sealed report and must remain unknown.
