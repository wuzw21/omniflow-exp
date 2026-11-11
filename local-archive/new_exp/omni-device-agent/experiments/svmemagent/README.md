# SVMemAgent 原版复现：实际入口检查

2026-10-31。**未跑通；当前阻塞是缺少可核验的作者模型实现和推理入口。** 本轮实际执行了公开资源检查与两套原生 Transformers 配置加载，没有执行 writer/QA forward，没有时间或内存实测。

## 已找到的资源

- [第一作者论文列表](https://ikodoh.github.io/publications/) 的 SVMemAgent 条目仅链接 arXiv，没有 code/project 链接。
- [实验室论文列表](https://mlv.korea.ac.kr/publications) 的该条目显示 `[Code]` 等占位文字，但实际 HTML 段内没有链接。
- GitHub 按仓库名 `SVMemAgent`、Python 代码 `SVMemAgent`、类名 `VideoStreamingMemoryAgent` 检索，保存的三次成功请求均为空。另查作者与实验室公开仓库列表未找到实现。搜索结果为空不证明代码绝对未发布。
- Hugging Face API 找到 [temptemp1234/SVMemAgent](https://huggingface.co/temptemp1234/SVMemAgent/tree/7b7ba15a2d35064287f449259607759e36259440)，固定 revision `7b7ba15a2d35064287f449259607759e36259440`。该仓库没有 README/model card，未查到作者页面对它的引用，**官方来源与发布许可未确认**。

同名 HF 仓库有 1,060 个文件：20 个 safetensors、配置、训练状态、数据标注与历史评测输出，但没有 `.py/.sh/.md/.yaml/.yml/.toml` 文件或代码压缩包。已获取完整目录元数据和两个小配置；未下载大权重或反序列化 pickle。HF 中的历史评测结果不是我们的复现结果。

候选配置指向自定义 `model_type=videostreaming_memory_agent`、`architectures=[VideoStreamingMemoryAgent]`，没有 `auto_map`。DINOv3 encoder、8 层语义分支、2 层时间分支和 budget 16 等字段与论文部分描述相似，但这种相似不足以确认作者身份、最终 checkpoint 或完整推理语义。配置中的阈值也不能在缺源码时直接映射为论文决策阈值。

## 实际加载尝试

在物理 Apple M4/macOS 26.5.1 上，用本地保存的候选配置执行原生 `AutoConfig.from_pretrained(..., local_files_only=True, trust_remote_code=False)`：

| 运行时 | 真实结果 | writer / QA 计时、内存 |
|---|---|---|
| Python 3.12.14 / Transformers 4.57.6 | ValueError：不识别 `videostreaming_memory_agent` | 未运行 / 未测 |
| Python 3.14.3 / Transformers 5.16.1 | 同样在配置加载失败 | 未运行 / 未测 |

候选配置标注 Transformers 4.57.0。以上检查说明现有两个原生安装都不能直接构建这个自定义模型，**不证明拿到作者注册代码后仍不能运行**。`trust_remote_code=True` 也不能凭空补出仓库缺失的代码。没有因此安装任意 fork、用随机权重或自行实现论文架构。

证据：目录中的 `release_probe_20261031/` 保存候选元数据、配置、GitHub 查询输出、作者页面链接摘录和两次真实异常。复跑必须选择新输出文件：

```sh
# 仓库根；依赖原机器既有环境
../edgecloud-research/.venv-latent/bin/python scripts/probe_svmemagent_load.py \
  --config-dir experiments/svmemagent/release_probe_20261031 \
  --output runs/svmemagent-new-probe/load.json
```

退出码 2 表示配置加载阻塞，不能记作推理成功。该命令只验证前置入口，不会下载/加载权重，也不产生性能数据。仓库回归入口为 `python3 -m unittest discover -s tests -v`，覆盖本次失败证据、缺代码/缺来源与测量值必须为空、已有输出不覆盖。

## 继续原版复现所缺的具体材料

作者的 `VideoStreamingMemoryAgent` 实现和配置注册；原始逐帧预处理/记忆更新/评测入口；正式 checkpoint 与上述候选文件的对应关系；Qwen3-VL-8B 分支的原始运行参数及输入样本。拿到这些才能固定源码并加载实际模型，然后测 writer、QA 与整体路径。

已准备[资源请求邮件草稿](AUTHOR_REQUEST.md)，**未发送**。用户尚未授权外部发信。没有擅自更换论文、模型或改写方法。本轮也没有占用 9207 GPU，因为缺代码的前置条件不会因更换设备而解决。

工具边界：按 agent-reach 技能使用 GitHub CLI 与 Exa；其更新检查因 `ModuleNotFoundError: agent_reach` 失败，未更改全局环境。这不影响已成功完成的资源查询。
