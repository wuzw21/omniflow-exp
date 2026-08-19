# Test edit guide

Tests cross the same interface as callers. Keep tests beside the owning seam:
runtime behavior in `test_runtime_*`, Function lifecycle in `test_function_*`,
experiment scheduling in `test_run_tasks.py` and `test_exp_script.py`,
and external adapters in their matching test file.

When deleting a module, delete only tests that assert the deleted contract;
move coverage to the surviving owner when behavior remains. Do not preserve a
test solely to keep a retired alias alive.

