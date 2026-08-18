# B-MoCA integration

This directory contains the B-MoCA environment seam. Preserve official device,
reward, and replay contracts here; campaign scheduling stays in
`src/experiment/e2e_task_pipeline.py`. New reuse behavior must use the shared
Function/OmniTransfer runtime or the official baseline adapter rather than
adding another scheduler.

