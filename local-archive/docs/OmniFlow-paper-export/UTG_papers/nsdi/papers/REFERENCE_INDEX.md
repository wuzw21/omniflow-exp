# OmniFlow NSDI Local Reference Index

Status: audit artifact
Last updated: 2026-05-21
Scope: citation keys currently used by `UTG_papers/nsdi/overleaf/*.tex`.

This file maps each current citation key to a local reference file or an
external documentation source. It does not change the paper text. Entries marked
`needs_verification` should not be used for precise claims until the exact
publication is confirmed.

## Citation Key Coverage

| Key | Local reference | Verified title / source | Status | Notes |
|---|---|---|---|---|
| `actionengine` | `actionengine_2602.20502.pdf` | ActionEngine | verified_pdf | Key comparison paper for programmatic mobile GUI agents. |
| `aitw` | `aitw_2307.10088.pdf` | Android in the Wild | verified_pdf | Dataset / benchmark reference. |
| `android-monkey` | external docs | Android UI/Application Exerciser Monkey, Android Developers | docs_only | Tool documentation, not a paper PDF. |
| `androidworld` | `androidworld_2405.14573.pdf` | AndroidWorld: A Dynamic Benchmarking Environment for Autonomous Agents | verified_pdf | PDF says ICLR 2025 and 116 programmatic tasks. Paper text currently says "50-goal", which needs correction before submission. |
| `appagent` | `appagent_2312.13771.pdf` | AppAgent | verified_pdf | Memory/exploration-document baseline. |
| `appium` | external docs | Appium official documentation | docs_only | Tool documentation, not a paper PDF. |
| `autodroid` | `autodroid_2308.15272.pdf` | AutoDroid | verified_pdf | Correct AutoDroid PDF. |
| `autodroid2` | `autodroid_v2_2412.18116.pdf` | AutoDroid-V2 | verified_pdf | MobiSys 2025 paper. |
| `backstage` | `backstage_mobilesoft2018.pdf` | Analyzing the User Interface of Android Apps | verified_pdf | Static UI analysis / BACKSTAGE reference. |
| `cradle` | `cradle_2403.03186.pdf` | Cradle: Empowering Foundation Agents Towards General Computer Control | verified_pdf | General computer-control agent reference. |
| `droidbot` | `droidbot_gpt_2304.07061.pdf` | DroidBot-GPT: GPT-powered UI Automation for Android | verified_pdf | LLM-driven mobile GUI automation baseline. |
| `droidbot-tool` | `droidbot_tool_icse2017.pdf` | DroidBot: A Lightweight UI-Guided Test Input Generator for Android | verified_pdf | UTG exploration tool reference. |
| `droidspeak` | `droidspeak_2411.02820.pdf` | DroidSpeak: KV Cache Sharing for Cross-LLM Communication and Multi-LLM Serving | verified_pdf | LLM serving cache comparison. |
| `fastserve` | `fastserve_2305.05920.pdf` | Fast Distributed Inference Serving for Large Language Models | verified_pdf | LLM serving / scheduling comparison. |
| `federated-gui` | `fedgui_2604.14956.pdf`; related: `fedmabench_2025_emnlp.pdf`, `mobilea3gent_2025_hcinlp.pdf` | FedGUI / FedMABench / MobileA3gent | verified_related_but_claim_needs_check | The current text says differential privacy over learned GUI representations. The downloaded FedGUI/FedMABench/MobileA3gent references are about federated/decentralized GUI agent data and benchmarking; the DP-specific wording needs verification. |
| `gator` | `gator_static_window_transition_graphs_ase2015.pdf` | Static Window Transition Graphs for Android | verified_pdf | The paper builds static WTG from app binaries using Android static analysis. |
| `hermes` | none | not verified | needs_verification | Current text attributes compressed coordinate/action sharing and self-evolution claims to Hermes. I did not find a matching verified paper during this audit. Treat as unsafe until exact source is supplied. |
| `infinigen` | `infinigen_2406.19707.pdf` | InfiniGen | verified_pdf | LLM serving / KV cache context. |
| `lowe2004` | `lowe2004_sift_ijcv.pdf` | Distinctive Image Features from Scale-Invariant Keypoints | verified_pdf | Classical local-feature matching reference. |
| `mobile-agent` | `mobile_agent_2401.16158.pdf` | Mobile-Agent: Autonomous Multi-Modal Mobile Device Agent with Visual Perception | verified_pdf | VLM mobile-agent baseline. |
| `mobilegpt` | `mobilegpt_2312.03003.pdf`; duplicate `mobilegpt_real.pdf` | MobileGPT | verified_pdf | Key memory/replay/state-machine comparison paper. |
| `mobileviews` | `mobileviews_2409.14337.pdf` | MobileViews | verified_pdf | Corpus reference for large-scale mobile UI analysis. |
| `seeclick` | `seeclick_2401.10935.pdf` | SeeClick | verified_pdf | GUI grounding model reference. |
| `sglang` | `sglang_2312.07104.pdf` | SGLang: Efficient Execution of Structured Language Model Programs | verified_pdf | LLM program execution / KV cache comparison. |
| `ufo` | `ufo_2402.07939.pdf` | UFO: A UI-Focused Agent for Windows OS Interaction | verified_pdf | Desktop/computer-use agent reference. |
| `uground` | `uground_2410.05243.pdf` | UGround | verified_pdf | GUI grounding model reference. |
| `uiautomator2` | external docs / GitHub | openatx/uiautomator2 | docs_only | Tool documentation, not a paper PDF. |
| `vdroid` | `vdroid_2503.15937.pdf` | V-Droid | verified_pdf | State-of-the-art verification / GUI-agent reference cited by current draft. |
| `verisafe` | `verisafe_2503.18492.pdf` | VeriSafe | verified_pdf | Runtime action-safety comparison. |
| `vhm` | none | not verified | needs_verification | Current draft cites VHM as a research system building approximate UTGs. I could not verify the exact publication. |

## Non-Key Local Files

- `autodroid_2312.02644.pdf` is not AutoDroid. `pdftotext` identifies it as
  "Algorithmic collusion under competitive design". Do not cite it for
  AutoDroid.

## External Documentation Sources

- Appium: https://appium.io/docs/
- UiAutomator2: https://github.com/openatx/uiautomator2
- Android Monkey: https://developer.android.com/studio/test/other-testing-tools/monkey
