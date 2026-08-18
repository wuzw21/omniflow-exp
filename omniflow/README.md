# `omniflow`

This package is the reusable runtime. Experiment scheduling belongs in
`src/experiment`; do not put AndroidWorld or B-MoCA orchestration here.

| File/directory | Edit for |
| --- | --- |
| `bridge.py` | external JSON-line tools and `run_gui` |
| `runlog.py` | canonical RunLog evidence loading |
| `core/` | shared models, schemas, and config |
| `functions/` | Function compilation, validation, Store writing, recall |
| `runtime/` | Planner lifecycle and action execution |
| `transfer/` | canonical OmniTransfer integration |
| `vlm/` | Planner context, model configuration, and usage |
| `vlm_coordinates.py` | coordinate conversion only |

Keep the package independent of experiment result paths and method names.
Public changes require bridge/schema tests; runtime changes require runtime
tests through the public interface.

