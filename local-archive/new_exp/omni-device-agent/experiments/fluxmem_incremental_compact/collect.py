"""Archive completed real runs and render tables directly from their measurements."""
import argparse,hashlib,json,shutil
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--validate',type=Path,required=True);p.add_argument('--memory',type=Path,required=True);p.add_argument('--frontend',type=Path,required=True);p.add_argument('--qa',type=Path,required=True);a=p.parse_args()
base=Path(__file__).resolve().parent;e=base/'evidence';e.mkdir(exist_ok=False)
records={}
for name in ['validate','memory','frontend','qa']:
 src=getattr(a,name);r=json.loads((src/'results.json').read_text());assert r['status']=='completed',name
 records[name]=r;shutil.copy2(src/'results.json',e/f'{name}_results.json')
 folder=e/f'{name}_code';folder.mkdir()
 for filename,sha in r['source_hashes'].items():
  assert hashlib.sha256((src/filename).read_bytes()).hexdigest()==sha
  shutil.copy2(src/filename,folder/filename)
 log=Path('/tmp')/f'{src.name}.log'
 if log.exists():shutil.copy2(log,e/f'{name}_stdout.log')
 if (src/'phases.jsonl').exists():shutil.copy2(src/'phases.jsonl',e/f'{name}_phases.jsonl')
outputs=[]
for f in sorted(a.qa.glob('*_outputs.pt')):
 outputs.append({'path':str(f.resolve()),'sha256':hashlib.sha256(f.read_bytes()).hexdigest()})
(e/'qa_outputs_manifest.json').write_text(json.dumps(outputs,indent=2)+'\n')
r=records['validate'];last=r['records'][-1];bbytes=41*120*2048*2;cbytes=last['C2_state']['tensor_storage_bytes'];oldbytes=last['C1_state']['tensor_storage_bytes']
mem=records['memory']['summary']['memory_seconds'];front=records['frontend']['summary']['total_seconds'];fm=records['frontend']['summary']['memory_seconds'];fv=records['frontend']['summary']['visual_seconds']
def row(label,summary):
 b=summary['conditions']['B'];c=summary['conditions']['C2']
 return f"| {label} | {b['median']:.3f} [{b['min']:.3f}, {b['max']:.3f}] | {c['median']:.3f} [{c['min']:.3f}, {c['max']:.3f}] | {summary['mean_paired_saving']:.3f} [{summary['mean_paired_saving_bootstrap_95'][0]:.3f}, {summary['mean_paired_saving_bootstrap_95'][1]:.3f}] | {summary['C2_faster_pairs']}/{len(summary['paired_B_minus_C2_seconds'])} |"
text=f'''# 去除历史拼接后的实际结果

采样与 FluxMem 策略均不变。C2 用固定短期环形缓冲和按帧保存的保留特征替代 C1 的稠密历史追加；原匹配、剪枝、Otsu 和聚类 helper 直接复用。物理设备仍为 Apple M4 / 16 GiB，Torch 2.13、Qwen2.5-VL-3B BF16。原问题为 4、53、82 秒，固定两帧块对应 2、26、41 块；53 秒只读取 52 个已配对帧。

41 个前缀与作者完整重算、旧 C1 均逐位一致。重复读取不改变状态，读出结果被外部修改不会污染状态，reset 后重放也精确一致。完整模型三题×三条件共 9 次生成，记忆输出的全部张量、首 token logits、生成 token 全部精确相同。验证仍使用既有视频，不据此推断多视频质量。

## 少存了什么

| 82 秒末的跨问题持久 tensor storage | MiB |
|---|---:|
| B：全部原始视觉缓存，记忆重算 | {bbytes/2**20:.2f} |
| C1：原始、归一化、合并特征完整保留 | {oldbytes/2**20:.2f} |
| C2：短期工作区＋各帧已保留特征 | {cbytes/2**20:.2f} |

C2 比 C1 少 {(oldbytes-cbytes)/2**20:.2f} MiB（{(1-cbytes/oldbytes)*100:.1f}%），比 B 少 {(bbytes-cbytes)/2**20:.2f} MiB（{(1-cbytes/bbytes)*100:.1f}%）。五槽归一化 ring 占 2.34 MiB，包含尚未使用的容量；输出特征与绝对索引占 {last['C2_state']['components']['blocks']/2**20:.2f} MiB。它删除的是不再参与原策略后续执行的 raw/norm/已丢弃 token，不是改变选择策略。首次预分配会使短前缀的 C2 状态多于 B；并非所有时间点都省内存。

这些是实际 storage 去重后的持久状态，不含共同输入、验证参考、权重、allocator 缓存和临时输出；不是部署峰值。环形缓冲有固定容量，长期保留输出仍增长，没有声称整个系统有界。完整 QA 从文件读取核验状态，并未验证只持有上述状态的常驻完整模型服务。

## 不再复制整段历史，但仍需写入新状态和组装答案输入

同一批 41 块在 MPS 重放：C1 的 append 有 280 次 cat，累计逻辑输出 808.91 MiB；C2 的 append 为 **0 次 cat**。旧/新两块最小回归分别失败/通过（7→0 次）。原策略内部的算子也纳入 append 跟踪，未只检查源码。

C2 仍有 clone {r['append_copies']['C2']['logical_bytes']['aten.clone.default']/2**20:.2f} MiB、copy_ {r['append_copies']['C2']['logical_bytes']['aten.copy_.default']/2**20:.2f} MiB 的累计逻辑输出，此外还有索引选择与算子临时量；不能称为零复制或用这些项相加当总 DRAM 流量。查询时拼接保留特征及索引，这项输出组装计入维护耗时。C2 没有保存一份供审计使用的完整原始特征。

## 与 B 的重复计时

每次都是从空状态处理原三题的累计秒数；先各一次完整预热，再按 BC/CB 交替配对。记忆重放 12 对；独立实际视觉编码＋记忆前端 6 对。GPU 同步边界；计时内包括新增特征入库、状态维护和查询输出组装，不包括验等、storage 报告、加载或算子跟踪。两种实验顺序运行，没有与 QA 同时争用 GPU。

| 累计耗时（秒） | B 中位数 [最小, 最大] | C2 中位数 [最小, 最大] | 配对平均节省 [bootstrap 95% 区间] | C2 更快的对数 |
|---|---:|---:|---:|---:|
{row('仅记忆重放，12 对',mem)}
{row('含视觉的前端总计，6 对',front)}
{row('前端内的记忆阶段，6 对',fm)}
{row('前端内的视觉阶段，6 对',fv)}

区间对完整 B/C2 配对差值做 10,000 次 bootstrap，seed=1234；反映本机同视频重复的时间波动，不是跨视频/设备置信区间。视觉块数两者都为 41；匹配调用 B/C2=125/77，剪枝=59/37，长期合并均为 5。计数与上轮相同。原始每次耗时、同步边界及状态量保留于 JSON，不能用各阶段中位数相加替代总计中位数。

'''
if front['mean_paired_saving_bootstrap_95'][0]>0:
 text+='本轮前端平均配对差值为正，bootstrap 区间下界接近零；6 对中有一对 C2 更慢。因此证据支持本例的维护时间下降，前端整体只有初步的节省迹象，尚不足以称为稳定的部署加速。'
else:
 text+='记忆维护是否节省时间与整个前端是否稳定加速必须分开：本轮前端配对区间包含零，尚不能确认稳定的前端总加速。'
text+='没有重复测量 C1 的时间，因此不能把 C2 相对 B 的全部差值归因于消除 cat，也不能与上轮单次数字相减估计本轮修复收益。尚未测完整语言模型常驻时的问答延迟、功耗、多视频或长时间在线负载。\n\n证据：[逐前缀与复制](evidence/validate_results.json)、[12 对维护计时](evidence/memory_results.json)、[6 对前端计时](evidence/frontend_results.json)、[完整 QA](evidence/qa_results.json)。复跑见 [README](README.md)，事先固定的设置见 [PROTOCOL](PROTOCOL.md)。\n'
(base/'RESULTS.md').write_text(text)
print('Archived completed evidence and rendered RESULTS.md')
