# MobileGPT 4090 固定实验环境

这套环境只准备一次。之后正式实验统一从权威 checkout 启动：

```bash
cd /home/zewen/Projects/OmniFlow-exp
bash scripts/exp/prepare_mobilegpt_environment.sh
```

准备脚本固定以下边界：

- 官方 MobileGPT checkout：`/home/zewen/Projects/Omni/mobilegpt-official-clean`
- 实验运行时：`data/runtime/mobilegpt/`
- 持久化 server 模板：`data/runtime/mobilegpt/server/`
- 持久化官方 client APK：`data/runtime/mobilegpt/client_<device>.apk`
- chat model：`Qwen3.6-Plus`
- embedding model：`GLM-Embedding-2`
- source client server：`10.0.2.2:12345`
- target client server：按 `12000 + console_port % 40000` 固定派生；三台设备各自独立
- Python：`/home/zewen/Projects/OmniFlow-exp/.venv/bin/python`

`server/` 和四个按设备端口固定的 `client_<device>.apk` 是一次性准备的实验版运行资产。每个 episode 只建立独立
的 Memory overlay 和日志目录，不重新 Gradle、安装 MobileGPT client、重写官方
Explore/Select/Derive/Memory，也不使用 OmniFlow Checker 改变 MobileGPT 行为。

一键运行仍然只有统一入口：

```bash
bash scripts/exp/run_androidworld.sh run \
  --task ContactsAddContact \
  --method mobilegpt \
  --device source5560 \
  --source-run-log data/androidworld/ContactsAddContact/source/OmniFlowSourceSmall_seed111/runlog/current/run_log.json
```

迁移时复制 `data/runtime/mobilegpt/`、对应的 AndroidWorld 数据目录，以及
`config/runtime.env`；API key 只放在目标机被 Git 忽略的
`config/runtime.secrets.env`。运行前只需把 `runtime.env` 中的机器根路径替换成
目标机路径，实验参数、模型、端口和相对 Memory 地址保持不变。

检查固定环境：

```bash
test -x .venv/bin/python
test -f data/runtime/mobilegpt/client.apk
test -f data/runtime/mobilegpt/server/main.py
cat data/runtime/mobilegpt/environment.json
```
