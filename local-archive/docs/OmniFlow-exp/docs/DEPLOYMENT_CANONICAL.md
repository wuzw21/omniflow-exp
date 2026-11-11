# OmniFlow-exp 唯一部署版本

这是 OmniFlow AndroidWorld 实验的长期部署合同。

## 唯一路径

| 位置 | 权威 checkout |
|---|---|
| 本地 | `/Users/wuzewen/Projects/Omni/OmniFlow-exp` |
| 9207 | `/home/wuzewen/Projects/Omni/OmniFlow-exp` |

所有 OmniFlow 自有代码、启动脚本、配置、虚拟环境和实验数据都从上述同名
checkout 读取。唯一公开启动入口是：

```bash
cd /home/wuzewen/Projects/Omni/OmniFlow-exp
bash scripts/exp/run_androidworld.sh ...
```

本地运行时把路径中的 `/home/wuzewen` 换成本地用户目录即可；命令和入口不变。

## 禁止使用的目录

以下目录不是运行时版本：

- `OmniFlow-exp-unified`
- `OmniFlow-exp-official`
- `OmniFlow-exp-mobilegpt-*`
- `OmniFlow-exp-semantic-*`
- 任何带日期、时间戳或临时修复后缀的 OmniFlow checkout

这些目录不删除，以保留历史 provenance；但不得作为 `PYTHON_BIN`、
`OMNIFLOW_EXP_*_ROOT`、`OMNIFLOW_MOBILEGPT_ROOT`、启动 cwd 或部署目标。

## 部署检查

部署前必须确认：

```bash
git -C /home/wuzewen/Projects/Omni/OmniFlow-exp rev-parse --show-toplevel
git -C /home/wuzewen/Projects/Omni/OmniFlow-exp rev-parse HEAD
test -x /home/wuzewen/Projects/Omni/OmniFlow-exp/.venv/bin/python
```

若启动脚本不是从这个 checkout 解析出来，视为部署错误，先停止，不启动设备实验。
