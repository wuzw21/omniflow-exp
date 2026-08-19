# Script entry points

`scripts/exp/run_androidworld.sh` is the only public experiment entry. It
selects the repository-local Python runtime, validates roots, and dispatches
the scheduler. Keep protocol values in `config/paper_androidworld.json` and
keep shell logic limited to process setup and dispatch. The optional
`--control-backend oob` flag is the sole way to select the OOB transport and is
restricted to development/source/E2E runs.
