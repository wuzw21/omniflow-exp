# 未发送的作者资源请求

收件人：Dohwan Ko；地址见[作者主页](https://ikodoh.github.io/)。未使用工具发送。

Subject: SVMemAgent inference code and checkpoint for on-device reproduction

Dear Dohwan,

We would like to reproduce SVMemAgent on an edge device and separately measure memory-maintenance and downstream QA latency and memory, without changing the original policy.

We found the public Hugging Face repository `temptemp1234/SVMemAgent`, revision `7b7ba15a2d35064287f449259607759e36259440`. It contains checkpoints and configurations for `VideoStreamingMemoryAgent`, but no model implementation or inference scripts. We could not locate an official code link on your publications page. Native Transformers 4.57.6 and 5.16.1 do not recognize its custom model type without additional code.

Could you point us to the official inference implementation and confirm which checkpoint/configuration reproduces the paper’s main results? If that Hugging Face repository is yours, an example command and sample for the Qwen3-VL-8B evaluation path would help us preserve the original preprocessing and memory-update semantics.

Thank you for your help.
