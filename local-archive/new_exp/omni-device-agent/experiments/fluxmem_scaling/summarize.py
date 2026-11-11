import argparse, json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument("--inputs",type=Path,required=True);p.add_argument("--features",type=Path,required=True);p.add_argument("--audit",type=Path,required=True);p.add_argument("--memory",type=Path,nargs=3,required=True);p.add_argument("--frontend",type=Path,nargs=3,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args()
out=a.output;out.mkdir(parents=True,exist_ok=False);ev=out/"evidence";ev.mkdir()
def read(pth):
    name="results.json" if (pth/"results.json").exists() else "manifest.json"
    return json.loads((pth/name).read_text())
srcs={"inputs":a.inputs,"features":a.features,"audit":a.audit,"memory_40":a.memory[0],"memory_10":a.memory[1],"memory_4":a.memory[2],"frontend_40":a.frontend[0],"frontend_10":a.frontend[1],"frontend_4":a.frontend[2]}
rs={k:read(v) for k,v in srcs.items()}
for k,v in srcs.items(): (ev/f"{k}_results.json").write_text((v/("results.json" if (v/"results.json").exists() else "manifest.json")).read_text())
audit=rs["audit"];assert audit["status"]=="completed" and len(audit["records"])==94;assert all(x["equivalence"]["features_bitwise_equal"] and x["equivalence"]["indices_equal"] for x in audit["records"])
mem=[{"cadence":c,"result":rs[f"memory_{c}"]} for c in [40,10,4]];front=[{"cadence":c,"result":rs[f"frontend_{c}"]} for c in [40,10,4]]
def line(xs,ys,color,ox,oy,w,h,xmax,ymax):
 pts=" ".join(f"{ox+x/xmax*w:.1f},{oy+h-y/ymax*h:.1f}" for x,y in zip(xs,ys));dots="".join(f'<circle cx="{ox+x/xmax*w:.1f}" cy="{oy+h-y/ymax*h:.1f}" r="3" fill="{color}"/>' for x,y in zip(xs,ys));return f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2"/>{dots}'
def panel(title,xlabel,ylabel,series,xmax,ymax,ox,oy,w=480,h=260):
 s=f'<g><text x="{ox}" y="{oy-28}" class="title">{title}</text><line x1="{ox}" y1="{oy+h}" x2="{ox+w}" y2="{oy+h}" class="axis"/><line x1="{ox}" y1="{oy}" x2="{ox}" y2="{oy+h}" class="axis"/><text x="{ox+w/2}" y="{oy+h+34}" class="label">{xlabel}</text><text transform="translate({ox-42},{oy+h/2}) rotate(-90)" class="label">{ylabel}</text>'
 for t in [0,.25,.5,.75,1]:
  xx=ox+w*t;yy=oy+h-h*t;s+=f'<line x1="{xx}" y1="{oy}" x2="{xx}" y2="{oy+h}" class="grid"/><line x1="{ox}" y1="{yy}" x2="{ox+w}" y2="{yy}" class="grid"/><text x="{xx}" y="{oy+h+17}" class="tick">{xmax*t:g}</text><text x="{ox-9}" y="{yy+4}" text-anchor="end" class="tick">{ymax*t:g}</text>'
 for i,(label,xs,ys,color) in enumerate(series):s+=line(xs,ys,color,ox,oy,w,h,xmax,ymax)+f'<text x="{ox+w-120}" y="{oy+18+i*18}" class="legend" fill="{color}">● {label}</text>'
 return s+"</g>"
series=[];maxy=0
for item,color in zip(mem,["#1f77b4","#d62728","#2ca02c"]):
 run=next(x for x in item["result"]["runs"] if not x["warmup"]);xs=[e["seconds"] for e in run["events"]];ys=[];total=0
 for e in run["events"]:total+=e["memory_seconds"];ys.append(total)
 maxy=max(maxy,max(ys));series.append((f'B, q={item["cadence"]}s',xs,ys,color))
for item,color in zip(mem,["#72a7d8","#ed8f8f","#7cc47c"]):
 run=next(x for x in item["result"]["runs"] if not x["warmup"]);xs=[e["seconds"] for e in run["events"]];ys=[];total=0
 for e in run["events"]:total+=e["memory_seconds"];ys.append(total)
 series.append((f'C, q={item["cadence"]}s',xs,ys,color))
svg='<svg xmlns="http://www.w3.org/2000/svg" width="1060" height="760"><style>.title{font:16px sans-serif;font-weight:bold}.label{font:12px sans-serif}.tick,.legend{font:11px sans-serif}.axis{stroke:#333}.grid{stroke:#ddd}.note{font:12px sans-serif}</style>'
svg+=panel("Cumulative memory-maintenance time","video time (s)","seconds",series,188,maxy*1.08,70,70)
xs=[x["seconds"] for x in audit["records"]];bstate=[x["blocks"]*120*2048*2/2**20 for x in audit["records"]];cstate=[x["state"]["tensor_storage_bytes"]/2**20 for x in audit["records"]]
svg+=panel("Persistent tensor state vs video length","video time (s)","MiB",[("B cache",xs,bstate,"#1f77b4"),("C2 state",xs,cstate,"#d62728")],188,max(bstate)*1.08,600,70)
series=[]
for item,color in zip(mem,["#1f77b4","#d62728","#2ca02c"]):
 run=next(x for x in item["result"]["runs"] if not x["warmup"]);series.append((f'q={item["cadence"]}s',[e["seconds"] for e in run["events"]],[e["memory_seconds"] for e in run["events"]],color))
svg+=panel("Per-readout memory cost","video time (s)","seconds",series,188,max(max(z[2]) for z in series)*1.15,70,430)
svg+='<text x="70" y="735" class="note">B recomputes original memory at each readout; C maintains once per new block and packs retained output.</text></svg>'
(out/"scaling_curves.svg").write_text(svg)
text="# 长视频与查询密度的实际 scaling\n\n同一条真实 StreamingBench sample_1.mp4 从前 82 秒扩展到 188 秒；1 fps、两帧一块、120 visual tokens/块，采样和 FluxMem short=8、mid=64 策略全部不变。前 41 个视觉块哈希与上一轮一致。94 个前缀（2、4、…、188 秒）中，C2 与原版全重算的保留位置和合并特征全部逐位一致；最终 188 秒保留 4258 tokens。\n\n查询密度是受控的状态读出到达，不是新增语义问题：每 40、10、4 秒读出一次，并强制在 82、120、188 秒读出。188 秒分别有 6、20、48 次读出。B 在每个读出重新执行原完整记忆；C2 按新增视觉块维护状态，再组装读出。\n\n## 维护时间如何随查询变密增长\n\n| 查询间隔 | 读出数（188 秒） | B 维护中位数 | C2 维护中位数 | B−C2 平均 |\n|---|---:|---:|---:|---:|\n"
for item in mem:
 s=item["result"]["summary"]["total_memory_seconds"];run=next(x for x in item["result"]["runs"] if not x["warmup"]);text+=f'| {item["cadence"]} 秒 | {len(run["events"])} | {s["B_median"]:.2f}s | {s["C_median"]:.2f}s | {s["mean_saving"]:.2f}s |\n'
text+="""\nB 的 188 秒维护时间从约 16–18 秒（40 秒间隔）增加到 48–65 秒（10 秒间隔），再到 90–118 秒（4 秒间隔）；C2 始终约 4.8–5.8 秒。记忆阶段使用同一批真实视觉特征，因此 B 的增长来自读出次数，而不是视觉编码。C2 的策略调用只随新增块发生；读出仍有输出组装成本。\n\n## 视频变长时状态与读出成本\n\n| 视频长度 | B 持久状态 | C2 持久状态 | C2 读出末状态 token |\n|---:|---:|---:|---:|\n"""
for x in [41,60,94]:
 q=next(z for z in audit["records"] if z["blocks"]==x);b=x*120*2048*2/2**20;c=q["state"]["tensor_storage_bytes"]/2**20;text+=f'| {2*x}s | {b:.2f} MiB | {c:.2f} MiB | {q["retained_tokens"]} |\n'
text+="""\nB 存储所有视觉特征，按块线性增长；C2 在本视频上从 82 秒的 14.45 MiB 增长到约 19.03 MiB。C2 的短期 ring 是固定项，长期保留特征仍增长；它不是有界长期记忆。每次读出的组装成本仍随当前保留 token 数增长。\n\n## 含真实视觉编码的前端总时间\n\n| 查询间隔 | B 总前端 | C2 总前端 | C2 更快 |\n|---|---:|---:|---:|\n"""
for item in front:
 s=item["result"]["summary"]["total_seconds"];text+=f'| {item["cadence"]} 秒 | {s["B_median"]:.2f}s | {s["C_median"]:.2f}s | {s["mean_saving"]:.2f}s |\n'
text+="""\n前端包含每个新块的真实视觉编码、状态维护和查询读出组装；每个密度只有一对 BC/CB，因此这些是测量值，不是稳定性区间。B/C2 的视觉块数相同；查询变密时，B 的总时间继续受到重复记忆执行影响，而 C2 增长较慢。前端峰值包含视觉权重与公共运行时状态，不是手机部署峰值。\n\n图：[实际 scaling 曲线](scaling_curves.svg)。原始记录：[94 前缀验等](evidence/audit_results.json)、[维护计时](evidence/memory_40_results.json)、[前端计时](evidence/frontend_40_results.json)。\n\n这组实验回答的是执行 scaling，不是最终 agent 任务质量：没有新增语义问题，也没有修改记忆策略、采样或阈值。最窄结论是：在该真实视频上，B 的重复记忆计算对查询密度敏感；C2 的维护成本主要随视频新增内容增长，但其长期状态和每次读出组装仍随保留历史增长。\n"""
(out/"RESULTS.md").write_text(text)
print("wrote",out/"RESULTS.md",out/"scaling_curves.svg")
