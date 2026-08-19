# External baselines

MobileGPT and AppAgent are not OmniFlow agents. Their planner, memory reader,
action parser, controller, and task loop belong to their pinned upstream
repositories. OmniFlow only forwards the inputs they need.

## One path

```text
scripts/exp/run_androidworld.sh
  -> src/experiment/run_tasks.py
  -> src/experiment/run_task.py
  -> official external process
```

The external process does not call `src/integrations/android_world/run_episode.py`
and does not use the OmniFlow action schema.

| Baseline | Official entry | OmniFlow provides | OmniFlow does not implement |
| --- | --- | --- | --- |
| AppAgent | `AppAgent/run.py` | temporary `config.yaml`, `apps/<app>/` link to sealed `demo_docs`, one-device ADB proxy, task text | AppAgent parsing, UI tags, model loop, ADB actions |
| MobileGPT | staged upstream `Server/main.py` plus upstream Android app | temporary Server workspace with official Python package and copied native memory, client host patch in a temporary checkout, APK build/install, official broadcast | Explore/Select/Derive/Subtask logic, socket protocol, XML parser, action execution |

## AppAgent edit point

Change only the sealed AppAgent memory input or the provider configuration in
`src/experiment/run_task.py`. The forwarder creates:

```text
<result>/official_workspace/
  config.yaml
  apps/<app>/ -> <sealed_memory>/apps/<app>
  scripts/  -> <official_AppAgent>/scripts
  bin/adb
  tasks/
```

The task is sent through stdin to the official `run.py` wrapper, which starts
the upstream `scripts/task_executor.py` itself. Do not add an AppAgent parser
or controller under `src/`.

## MobileGPT edit point

The only conversion-specific input is the existing native memory converter.
The forwarder overlays its `frozen_memory` data into a copied official
`Server/memory` directory while preserving the official Python package files.
The official server therefore still imports and runs its own `memory` package.
The Android client is copied only to a temporary build directory so its official
`HOST_IP` constant can point to the server; the checked-out MobileGPT repository
is never edited.

MobileGPT requires an Android SDK, Gradle/Android dependencies, an enabled
Accessibility Service, and a reachable server host. For Android emulators the
default host is `10.0.2.2`; set `MOBILEGPT_CLIENT_HOST` for another device.

## Evidence rule

External execution evidence is saved under the attempt output and includes the
official revision, original entry, staged workspace, command log, and return
code. It is not an AndroidWorld validator result unless a separate official
validator conclusion is present. The runner therefore does not register an
external-only row into the formal validator ledger or invent a success value.
