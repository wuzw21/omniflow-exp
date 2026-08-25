# AndroidWorld 4090 runbook

部署完成后，在 canonical checkout 直接运行：

```bash
cd /home/zewen/Projects/OmniFlow-exp
bash scripts/exp/run_androidworld.sh \
  --task CameraTakePhoto \
  --method omniflow \
  --device standard45562
```

常用可选参数：

```bash
--source-seed 111
--evaluation-seed 113
--max-steps 20
--max-fallback-steps 5
--deadline 600
--model GLM-4.6V
```

批量方法或设备使用逗号或 `all`。入口不执行部署检查、seed/path 校验、结果扫描或
smoke test；AndroidWorld episode 自己负责设备 lifecycle、任务 setup 和官方 validator。
