# Script entry points

`scripts/exp/run_androidworld.sh` is the only public experiment entry. It
selects the repository-local Python runtime, validates roots, and dispatches
the scheduler. Keep protocol values in `config/paper_androidworld.json` and
keep shell logic limited to process setup and dispatch.

