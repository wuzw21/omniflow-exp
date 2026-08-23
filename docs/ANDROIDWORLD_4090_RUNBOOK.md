# AndroidWorld 4090 Runbook

本手册固定 AndroidWorld 4090 的运行环境和启动方式。已完成结果不需要重跑；后续任务只复用同一套环境配置。

## 固定环境

远端环境配置文件：

```text
/home/zewen/Desktop/omniflow-exp/androidworld-4090-v1.env
```

它固定以下内容：

- Python：`/home/zewen/Projects/Omni/OmniFlow-exp/.venv/bin/python`
- AndroidWorld：`/home/zewen/Projects/Omni/releases/android-world-632ac95959ace58c8e2ed2db8e4209cc3d9c26ef`
- memory/index：`/home/zewen/Projects/Omni/OmniFlow-exp-authoritative/data`
- 结果根目录：`/home/zewen/Desktop/omniflow-exp`
- 控制后端：`oob`
- source：`emulator-5560`
- Standard：`emulator-45562`
- Fold：`emulator-45564`
- Tablet：`emulator-45554`

不要在命令行重新拼接这些路径，也不要为同一任务创建第二套启动配置。

## 唯一启动命令

```bash
ssh 4090 'source /home/zewen/Desktop/omniflow-exp/androidworld-4090-v1.env && cd /home/zewen/Projects/Omni/OmniFlow-exp && bash scripts/exp/run_androidworld.sh --e2e-task TASK_NAME --e2e-method omniflow --e2e-device standard45562:emulator-45562:45562 --e2e-source-seed 111 --e2e-evaluation-seed 113 --control-backend oob --task-deadline-sec 1800'
```

将 `TASK_NAME` 和 `--e2e-device` 替换为当前待测任务和一个正式 target。每次只启动一个任务；同一任务的其他 target 仍使用同一条命令逐个执行。

## 输出和记录

- 原始结果：`/home/zewen/Desktop/omniflow-exp/<TASK_NAME>/`
- 通过组汇总：`/home/zewen/Desktop/test-omniflow.txt`
- 失败任务保留其独立 attempt 目录，不覆盖既有结果。
- 已完成的 task/device 组跳过，不重复运行。

## 当前环境注意事项

变量和设备拓扑已经固定。启动前不需要刷新旧索引或修正历史记录。

当前仍需单独处理 OmniTransfer checkpoint 的格式匹配；在该资产解决前，启动可能在任务开始前停止。这是环境资产问题，不应通过修改任务、Function 或执行链规避。
