# AndroidWorld integration

`run_episode.py` owns the native lifecycle and official runner. `host.py` exposes
native observe/act/reset. `environment.py` adapts official task validation.
`methods.py` constructs the small set of local method adapters. `agent.py` is
the OmniFlow adapter; `apps.py` and `state.py` hold
shared AndroidWorld setup/state helpers. `oob_control.py` is an explicit
development/source transport adapter selected by
`run_androidworld.sh --control-backend oob`.

MobileGPT and AppAgent are not AndroidWorld method adapters here. Start their
official repositories directly and provide only their native warm-start memory.
Never add a method-specific reset, step loop, coordinate executor, or launcher.
All local methods re-enter the shared launcher.
