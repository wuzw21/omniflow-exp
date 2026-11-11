# Storage-only revision, fixed before timing

Reuse the exact previous video, 41 immutable two-frame blocks, BF16 visual features, short=8 and mid=64 original FluxMem frames, and original queries at blocks 2/26/41. Do not change selection, thresholds, model weights, sampling or vision chunk size.

C2 replaces C1's historical concatenation with a fixed short-window ring and per-frame selected tensors. Drop raw/normalized features only after their original policy dependencies end; original helpers make every pruning/merge decision. Absolute retained indices preserve original ordering. Snapshot packs output at query time; this cost is included, not described as zero-copy.

Validate all 41 prefixes against original full recomputation and C1 on real MPS features. Check repeated reads, first pruning/long merge, ring wrap, reset and replay. Run old and new two-block append regression: old must fail, new must pass. Count all append cat/clone logical bytes separately from timing, including original helper internals. Verify three-query snapshots through the original full-model QA harness.

Timing: memory-only replay has 12 paired repetitions; full visual+memory frontend has 6 pairs. Each condition starts empty and handles the same three queries. One unrecorded full workload per condition warms the executable paths. Alternate BC/CB pair order. No profiler or model-output export within timing. Synchronize at workload/query boundaries, include cache insertion and snapshot assembly, exclude loading/validation/state reporting. No sleep to simulate arrival. B invokes original full_memory at each query. C2 maintains state for each new block. Actual vision calls are included independently for both conditions in frontend tests. No concurrent heavy tests. Report paired raw deltas, median and range, bootstrap interval for mean paired difference; same-video repetitions are timing repetitions, not independent videos.

Measure actual deduplicated tensor storage after each query; input bank and shared model weights separate. Neither this 82-second trace nor the ring implies bounded long-term retained output. No full LM serving, energy or multi-video claim from the frontend benchmark.
