# AndroidWorld integration

`launch.py` owns the native lifecycle and official runner. `host.py` exposes
native observe/act/reset. `environment.py` adapts official task validation.
`methods.py` constructs method adapters. `agent.py` is the OmniFlow adapter;
`mobilegpt_agent.py` is the MobileGPT adapter; `apps.py` and `state.py` hold
shared AndroidWorld setup/state helpers. `oob_control.py` is an explicit
development/source transport adapter selected by
`run_androidworld.sh --control-backend oob`.

Never add a method-specific reset, step loop, coordinate executor, or launcher.
All formal methods re-enter the shared launcher.
