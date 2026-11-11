import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parent

def main():
    a=json.loads((ROOT/'analysis.json').read_text())
    audit=json.loads((ROOT/'measurement_audit.json').read_text())
    t=json.loads((ROOT/'timing.json').read_text())
    if t['status']!='completed': raise RuntimeError('Native runs incomplete')
    qa=json.loads((ROOT/'case_qa.json').read_text()) if (ROOT/'case_qa.json').exists() else None
    names={'all':'全编码','uniform':'每两步采样','last':'比较最近保留画面','memory':'比较四张保留画面'}
    baseline=next(r['median_seconds'] for r in t['summaries'] if r['kind']=='all')
    lines=['# 端侧视觉编码门控：首轮实际测试','',
      '问题：利用已有画面记忆，能否减少端侧视觉编码，同时保住短暂文字证据？本轮没有新建 Agent 框架，也没有复现 ActiveSight/SVMemAgent。','',
      '## 设置','',
      '- Apple M4 / 16 GiB / macOS '+t['hardware']['macos']+'；原生 Qwen3-VL-2B 视觉编码器，BF16、MPS、SDPA。',
      '- 12 条公开 AndroidWorld 完整轨迹、139 张截图。开发集 6 条/50 张，保留集 6 条/89 张；固定数据版本和选取顺序见 PROTOCOL.md。',
      '- 所有策略最多保留 4 张 FIFO 关键帧及其视觉输出。阈值只根据开发集选择：最近画面和记忆比较均为 0.01。固定间隔策略未通过开发集 95% 召回门槛，只作失败诊断。',
      '- 在本机实际执行截图读取、廉价筛选、官方预处理、视觉编码和 MPS 同步；3 次交错重复。没有计入摄像头、解码器或语音；没有能耗测量。','',
      '## 保留集结果','',
      '| 策略 | 编码次数 / 89 | 原始 OCR 证据召回 | 去叠层敏感性结果 | 回放中位时间（秒） | 相对全编码减少 |',
      '|---|---:|---:|---:|---:|---:|']
    for r in t['summaries']:
        clean=next(x for x in audit['aggregate'] if x['split']=='test' and x['kind']==r['kind'] and x['param']==r['param'])
        lines.append(f"| {names[r['kind']]} | {r['calls']} | {r['evidence_recall']:.2%} | {clean['recall']:.2%} | {r['median_seconds']:.3f} | {1-r['median_seconds']/baseline:.2%} |")
    lines+=['','原始 OCR 参考包含 335 个短暂文本事件。人工查看发现 Android pointer-location 调试叠层混入参考，另做顶部 10% 排除的敏感性审计（296 个事件），保留原始结果、所有阈值、分割和决策，不重新调参。两列均是参考证据召回，不是人工真值、问答正确率或任务成功率。','',
      '三次实际回放时间（秒）：','']
    for r in t['summaries']:
        lines.append(f"- {names[r['kind']]}："+', '.join(f'{v:.3f}' for v in r['repeat_seconds'])+f"；编码中位耗时 {r['median_encode_seconds']:.3f}，筛选中位耗时 {r['median_gate_seconds']:.4f}。")
    lines+=['','## 失败核验','',
      '人工查看原始截图确认：计时器第 6 步显示 00h 00m 16s，但后续保留记忆未包含该帧；费用录入第 9–10 步横向类别栏显示 Transportation、Clothes、Health Care，但记忆筛选未保留。步号从 0 开始。','']
    if qa and qa['status']=='completed':
        lines+=['进一步用完整 Qwen3-VL-2B 做真实生成，问：“Clothes 右边的类别标签是什么？没有显示则答 UNKNOWN。”三组采用相同问题、生成参数和模型，只使用各自第 11 步的保留截图。该案例是在看到失败后选出的诊断，不能估计总体问答成功率。','',
                '| 输入策略 | 实际保留步骤 | 模型实际输出 |','|---|---|---|']
        for r in qa['records']:lines.append(f"| {names[r['kind']]} | {r['retained_steps']} | {r['answer'].replace(chr(10),' ')} |")
    else:lines+=['完整模型问答核验：未完成；不得将 OCR 证据损失当作已测得的 QA 损失。']
    lines+=['','## 判断与边界','',
      '预先设定的门槛是相对全编码证据召回下降不超过 2 个百分点且实际变快。当前记忆比较在原始参考和去叠层敏感性分析下均未通过质量门槛。测试支持“能够少算且更快”，不支持“保质加速已实现”，更不支持联合框架必要性。',
      '',
      '这是 Mac 真机上的原生算子与截图回放测试；源 Android 轨迹并非本轮重新执行，Android/穿戴设备的在线感知、响应和续航均待真机验证。静态截图不覆盖帧间运动、采集功耗、传感器休眠、时间截止或模型唤醒。没有真实 ActiveSight/SVMemAgent 强基线，低分辨率像素差只是最小入口对照。',
      '',
      '原计划的峰值内存没有测得：timing.json 只记录每条轨迹结束时的 MPS allocated/driver memory，不能将其称为峰值。',
      '',
      '## 重跑入口','',
      '从 edgecloud-research 目录运行（prepare.py 使用已固定的公开轨迹索引；已存在 timing.json/case_qa.json 时拒绝覆盖，重跑应归档为新 run）：','',
      '```sh',
      '.venv-latent/bin/python stream_gate/prepare.py',
      '.venv-latent/bin/python stream_gate/analyze.py',
      '.venv-latent/bin/python stream_gate/audit.py',
      '.venv-latent/bin/python stream_gate/benchmark.py',
      '.venv-latent/bin/python stream_gate/check_case_qa.py',
      '.venv-latent/bin/python stream_gate/write_report.py',
      '.venv-latent/bin/python -m unittest -v test_stream_gate.py',
      '```','',
      '原始记录：manifest.json（版本、路径、图像哈希），ocr_reference.jsonl（独立 OCR），analysis.json（全部阈值、逐步选择与丢失），measurement_audit.json（清理敏感性与人工核验），timing.json（真实运行），case_qa.json（完整模型输出）。','']
    (ROOT/'RESULTS.md').write_text('\n'.join(lines))

if __name__=='__main__': main()
