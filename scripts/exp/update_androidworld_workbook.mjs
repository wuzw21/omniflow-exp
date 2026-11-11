import crypto from "node:crypto";
import fs from "node:fs/promises";
import path from "node:path";
const { FileBlob, SpreadsheetFile } = await import(
  process.env.ARTIFACT_TOOL_MODULE || "@oai/artifact-tool"
);

const METHOD_TARGETS = {
  omniflow: {
    standard: { column: "C", setting: "OmniFlowTargetPixel6Pro_seed111_eval113", label: "Standard / Pixel 6 Pro" },
    fold: { column: "D", setting: "OmniFlowTargetFold_seed111_eval113", label: "Fold" },
    tablet: { column: "E", setting: "WXGA_Tablet_test_00_seed111_eval113", label: "Tablet" },
  },
  mobilegpt: {
    standard: { column: "F", setting: "OmniFlowTargetPixel6Pro_seed111_eval113", label: "Standard / Pixel 6 Pro" },
    fold: { column: "G", setting: "OmniFlowTargetFold_seed111_eval113", label: "Fold" },
    tablet: { column: "H", setting: "WXGA_Tablet_test_00_seed111_eval113", label: "Tablet" },
  },
  appagent: {
    standard: { column: "I", setting: "OmniFlowTargetPixel6Pro_seed111_eval113", label: "Standard / Pixel 6 Pro" },
    fold: { column: "J", setting: "OmniFlowTargetFold_seed111_eval113", label: "Fold" },
    tablet: { column: "K", setting: "WXGA_Tablet_test_00_seed111_eval113", label: "Tablet" },
  },
  autodroid: {
    standard: { column: "L", setting: "OmniFlowTargetPixel6Pro_seed111_eval113", label: "Standard / Pixel 6 Pro" },
    fold: { column: "M", setting: "OmniFlowTargetFold_seed111_eval113", label: "Fold" },
    tablet: { column: "N", setting: "WXGA_Tablet_test_00_seed111_eval113", label: "Tablet" },
  },
};

const args = process.argv.slice(2);
const valueFor = (name, fallback = "") => {
  const index = args.indexOf(name);
  return index >= 0 ? args[index + 1] || fallback : fallback;
};
const valuesFor = (name) => {
  const values = [];
  for (let index = 0; index < args.length; index += 1) {
    if (args[index] === name && args[index + 1]) values.push(args[index + 1]);
  }
  return values;
};
const required = (name) => {
  const value = valueFor(name);
  if (!value) throw new Error(`missing_${name.replaceAll("-", "_")}`);
  return value;
};
const resolve = (value) => path.resolve(value);
const sha256 = async (filePath) => {
  const bytes = await fs.readFile(filePath);
  return crypto.createHash("sha256").update(bytes).digest("hex");
};
const readJson = async (filePath) => JSON.parse(await fs.readFile(filePath, "utf8"));
const lastJsonl = async (filePath) => {
  const lines = (await fs.readFile(filePath, "utf8")).trim().split(/\n+/).filter(Boolean);
  return JSON.parse(lines.at(-1));
};
const object = (value) => (value && typeof value === "object" && !Array.isArray(value) ? value : {});
const list = (value) => (Array.isArray(value) ? value : []);
const numberOrBlank = (value) => (typeof value === "number" ? value : "");
const finiteNumber = (value) => {
  const numeric = typeof value === "number" ? value : Number(value);
  return Number.isFinite(numeric) ? numeric : "";
};
const seconds = (value) => (typeof value === "number" ? (value / 1000).toFixed(3) : "");
const count = (items, predicate) => list(items).filter(predicate).length;
const formatPercent = (value) => (typeof value === "number" ? `${(value * 100).toFixed(1)}%` : "");
const evidencePath = (value) => resolve(value);
const repoRelative = (filePath, repoRoot) => {
  const relative = path.relative(repoRoot, resolve(filePath));
  return relative && !relative.startsWith("..") ? relative : resolve(filePath);
};
const markdownCell = (value) => String(value || "").replaceAll("|", "\\|").replaceAll("\n", " ");

const runPathFor = async (resultPath) => {
  const resolved = evidencePath(resultPath);
  if (path.basename(resolved) === "run_log.json") return resolved;
  const candidate = path.join(path.dirname(resolved), "run_log.json");
  await fs.access(candidate);
  return candidate;
};

const readEvidence = async (
  resultPath,
  displayResultPath = resultPath,
  pathMetadata = {},
) => {
  const resolvedResult = evidencePath(resultPath);
  const runPath = await runPathFor(resolvedResult);
  const displayPath = String(displayResultPath);
  const displayRunPath = path.basename(displayPath) === "run_log.json"
    ? displayPath
    : path.join(path.dirname(displayPath), "run_log.json");
  const run = await readJson(runPath);
  const result = path.basename(resolvedResult) === "run_log.json" ? object(run) : await lastJsonl(resolvedResult);
  const diagnostics = object(run.diagnostics);
  const execution = object(diagnostics.execution_summary);
  const functionExecution = object(diagnostics.function_execution);
  const functionResolution = object(diagnostics.function_resolution);
  const metadata = object(pathMetadata);
  const functionSteps = list(functionExecution.steps);
  const reuse = object(result.reuse_metrics);
  const validator = object(result.androidworld_validator_result || run.validator);
  const official = result.official_validator_used === true || validator.official === true || validator.uses_androidworld_official_validator === true;
  const validatorSuccess = result.official_validator_success === true
    || validator.success === true
    || run.success === true;
  const success = (result.success === true || result.classification === "success" || run.success === true)
    && validatorSuccess;
  const transferFailures = count(functionSteps, (step) => String(step.error || "").toLowerCase().includes("transfer"));
  const functionSuccesses = count(functionSteps, (step) => step.success === true);
  const functionFailures = count(functionSteps, (step) => step.success === false);
  const functionId = String(result.function_id || functionExecution.function_id || diagnostics.function_id || "");
  const actions = numberOrBlank(result.actions_executed ?? result.step_count ?? list(run.steps).length);
  const physical = numberOrBlank(
    reuse.physical_action_count
      ?? (result.method === "appagent" ? actions : undefined),
  );
  const stateChanging = numberOrBlank(reuse.state_changing_physical_action_count);
  const modelCalls = numberOrBlank(result.model_calls ?? execution.model_calls);
  const fallbackSteps = numberOrBlank(result.fallback_steps ?? execution.fallback_steps);
  const totalTokens = numberOrBlank(result.total_tokens ?? execution.total_tokens);
  const executionSeconds = seconds(
    result.execution_duration_ms
      ?? execution.execution_duration_ms
      ?? execution.duration_ms,
  );
  const durationSeconds = seconds(result.duration_ms);
  const storePath = String(metadata.store || functionResolution.store_path || "");
  const memoryPath = storePath ? path.dirname(storePath) : "";
  return {
    resultPath: resolvedResult,
    displayResultPath: displayPath,
    runPath,
    displayRunPath,
    method: String(result.method || run.method || ""),
    resultSha: await sha256(resolvedResult),
    runSha: await sha256(runPath),
    official,
    success,
    device: result.device || "",
    evaluationSeed: result.task_random_seed ?? "",
    actions,
    physical,
    stateChanging,
    modelCalls,
    promptTokens: numberOrBlank(result.prompt_tokens ?? execution.prompt_tokens),
    completionTokens: numberOrBlank(result.completion_tokens ?? execution.completion_tokens),
    fallbackSteps,
    totalTokens,
    durationSeconds,
    executionSeconds,
    functionId,
    replayStatus: String(functionExecution.replay_status || functionResolution.replay_status || "not_started"),
    functionSuccesses,
    functionFailures,
    functionTotal: functionSuccesses + functionFailures,
    transferFailures,
    functionResolutionStatus: String(functionResolution.status || ""),
    runtimeWarning: String(result.failure_reason || execution.failure_reason || ""),
    checkerTriggers: object(diagnostics.checker_trigger_counts),
    reuseRate: reuse.reuse_rate,
    sourceRunLogPath: String(metadata.source_run_log || ""),
    memoryPath,
    storePath,
    compileReportPath: memoryPath ? path.join(memoryPath, "compile_report.json") : "",
    transferStatesPath: memoryPath ? path.join(memoryPath, "transfer_states.json") : "",
  };
};

const readSummaryEvidence = (summary, displayResultPath = "") => {
  const item = object(summary);
  const diagnostics = object(item.diagnostics);
  const execution = object(diagnostics.execution_summary);
  const functionExecution = object(diagnostics.function_execution);
  const functionResolution = object(diagnostics.function_resolution);
  const reuse = object(item.reuse_metrics);
  const resultPath = String(item.result_path || item.result || "");
  const runPath = String(item.run_log_path || item.run_log || "");
  const displayResult = String(displayResultPath || item.result_display || resultPath);
  const displayRun = String(item.run_log_display || runPath);
  const storePath = String(item.store || functionResolution.store_path || "");
  const memoryPath = storePath ? path.dirname(storePath) : "";
  const official = item.official === true || item.official_validator_used === true;
  const success = item.success === true || item.official_validator_success === true;
  return {
    resultPath,
    displayResultPath: displayResult,
    runPath,
    displayRunPath: displayRun,
    method: String(item.method || ""),
    resultSha: String(item.result_sha256 || item.result_sha || ""),
    runSha: String(item.run_log_sha256 || item.run_log_sha || ""),
    official,
    success: official && success,
    device: String(item.device || ""),
    evaluationSeed: item.evaluation_seed ?? item.task_random_seed ?? "",
    actions: finiteNumber(item.actions_executed ?? item.actions),
    physical: finiteNumber(item.physical_action_count ?? item.physical),
    stateChanging: finiteNumber(item.state_changing_physical_action_count ?? item.state_changing),
    modelCalls: finiteNumber(item.model_calls ?? execution.model_calls),
    promptTokens: finiteNumber(item.prompt_tokens ?? execution.prompt_tokens),
    completionTokens: finiteNumber(item.completion_tokens ?? execution.completion_tokens),
    fallbackSteps: finiteNumber(item.fallback_steps ?? execution.fallback_steps),
    totalTokens: finiteNumber(item.total_tokens ?? execution.total_tokens),
    durationSeconds: String(item.duration_seconds ?? ""),
    executionSeconds: String(item.execution_seconds ?? ""),
    functionId: String(item.function_id || functionExecution.function_id || diagnostics.function_id || ""),
    replayStatus: String(item.replay_status || functionExecution.replay_status || functionResolution.replay_status || "not_started"),
    functionSuccesses: finiteNumber(item.function_successes),
    functionFailures: finiteNumber(item.function_failures),
    functionTotal: finiteNumber(item.function_total),
    transferFailures: finiteNumber(item.transfer_failures),
    functionResolutionStatus: String(item.function_resolution_status || functionResolution.status || ""),
    runtimeWarning: String(item.failure_reason || execution.failure_reason || ""),
    checkerTriggers: object(item.checker_trigger_counts || diagnostics.checker_trigger_counts),
    reuseRate: finiteNumber(item.reuse_rate ?? reuse.reuse_rate),
    sourceRunLogPath: String(item.source_run_log || ""),
    memoryPath,
    storePath,
    compileReportPath: String(
      item.compile_report || (memoryPath ? path.join(memoryPath, "compile_report.json") : ""),
    ),
    transferStatesPath: String(
      item.transfer_states || (memoryPath ? path.join(memoryPath, "transfer_states.json") : ""),
    ),
  };
};

const updateDecision = (candidate) => ({
  update: candidate.official && candidate.success,
  reason: candidate.official && candidate.success
    ? "official_pass"
    : "candidate_not_official_pass",
});

const sourceCell = (source, repoRoot) => {
  const steps = list(source.steps);
  const validator = object(source.validator);
  const success = source.success === true
    && validator.official === true
    && validator.success === true;
  const provenance = object(source.provenance);
  return [
    `状态：${success ? "可用（official success）" : "不可用（source evidence failed）"}`,
    `官方验证：${success ? "PASS" : "FAIL"}`,
    `选择：${source.selection || "explicit source evidence"}`,
    `采集时间：${source.collectedAt || "未记录"}；provenance=${provenance.kind || "unknown"}`,
    `source_seed=${source.seed ?? 111}；source_actions=${steps.length}；physical=${count(steps, (step) => object(step.action).action_type !== "answer")}`,
    `RunLog：${resolve(source.path || "")}`,
    `SHA-256：${source.sha}`,
  ].join("\n");
};

const targetCell = (evidence, target, repoRoot) => {
  const status = evidence.official && evidence.success ? "已完成（official success）" : "已记录（official validator failed）";
  const transfer = evidence.method === "appagent"
    ? "not_applicable（AppAgent official executor）"
    : evidence.transferFailures > 0
    ? `failure=${evidence.transferFailures}；回退到 Planner/VLM`
    : "failure=0";
  const functionStatus = evidence.functionTotal
    ? `${evidence.replayStatus}（${evidence.functionSuccesses}/${evidence.functionTotal} steps）`
    : evidence.method === "appagent" ? "not_applicable" : evidence.replayStatus;
  const checker = Object.keys(evidence.checkerTriggers).length
    ? JSON.stringify(evidence.checkerTriggers)
    : "0";
  const runtimeWarning = evidence.runtimeWarning
    ? `内部诊断：${evidence.runtimeWarning}（官方 validator：${evidence.official && evidence.success ? "PASS" : "FAIL/未到达"}）`
    : "内部诊断：none";
  return [
    `状态：${status}`,
    `官方验证：${evidence.official && evidence.success ? "PASS" : "FAIL"}`,
    `设备：${target.label}；evaluation_seed=${evidence.evaluationSeed || 113}`,
    `动作：${evidence.actions}；physical=${evidence.physical}；state-changing=${evidence.stateChanging}`,
    `模型调用：${evidence.modelCalls}；fallback：${evidence.fallbackSteps}；tokens=${evidence.totalTokens}`,
    `Function：${evidence.functionId || "none"}`,
    `Function replay：${functionStatus}`,
    `Transfer：${transfer}`,
    `Function reuse：${formatPercent(evidence.reuseRate) || "未记录"}；Checker：${checker}`,
    runtimeWarning,
    `耗时：${evidence.durationSeconds}s；执行耗时：${evidence.executionSeconds}s`,
    `Source RunLog：${evidence.sourceRunLogPath || "未记录"}`,
    `Memory：${evidence.memoryPath || "未记录"}`,
    `Function Store：${evidence.storePath || "未记录"}`,
    `Compile report：${evidence.compileReportPath || "未记录"}`,
    `Transfer states：${evidence.transferStatesPath || "未记录"}`,
    `RunLog：${evidence.displayRunPath}`,
    `SHA-256：${evidence.runSha}`,
    `结果证据：${evidence.displayResultPath}`,
    `结果 SHA-256：${evidence.resultSha}`,
  ].join("\n");
};

const receiptRow = (task, method, target, evidence) => [
  task,
  method,
  target.setting,
  path.basename(path.dirname(evidence.resultPath)),
  evidence.official && evidence.success ? "success" : "fail",
  evidence.durationSeconds ? Number(evidence.durationSeconds) : null,
  evidence.executionSeconds ? Number(evidence.executionSeconds) : null,
  evidence.modelCalls,
  evidence.promptTokens,
  evidence.completionTokens,
  evidence.totalTokens,
  evidence.functionId || "",
  evidence.displayResultPath,
  [
    "automatic evidence update",
    `official_validator=${evidence.official && evidence.success ? "PASS" : "FAIL"}`,
    `function_replay=${evidence.method === "appagent" ? "not_applicable" : evidence.replayStatus}`,
    `transfer_failures=${evidence.transferFailures}`,
    `fallback_steps=${evidence.fallbackSteps}`,
    `source_run_log=${evidence.sourceRunLogPath || ""}`,
    `memory=${evidence.memoryPath || ""}`,
    `store=${evidence.storePath || ""}`,
    `compile_report=${evidence.compileReportPath || ""}`,
    `transfer_states=${evidence.transferStatesPath || ""}`,
    `run_log_sha256=${evidence.runSha}`,
    `task_results_sha256=${evidence.resultSha}`,
  ].join("; "),
];

const copySuccessStyle = (sheet, destination) => {
  const range = sheet.getRange(destination);
  range.format.fill = "#ECFDF5";
  range.format.font = { color: "#166534" };
  range.format.borders = { preset: "all", style: "thin", color: "#A7F3D0" };
  range.format.wrapText = true;
  range.format.horizontalAlignment = "left";
};

const workbookPath = required("--workbook");
const batchPath = valueFor("--batch-json");
const task = batchPath ? "" : required("--task");
const method = valueFor("--method", "omniflow").toLowerCase();
const TARGETS = METHOD_TARGETS[method];
if (!TARGETS) throw new Error(`unknown_method:${method}`);
const sourcePath = batchPath ? "" : required("--source-run-log");
const resultArgs = valuesFor("--result");
const displayResultArgs = valuesFor("--result-display");
const repoRoot = resolve(valueFor("--repo-root", process.cwd()));
const parseDeviceMap = (items, optionName) => {
  const parsed = new Map();
  for (const item of items) {
    const separator = item.indexOf("=");
    if (separator <= 0) throw new Error(`invalid_${optionName}:${item}`);
    const device = item.slice(0, separator).toLowerCase();
    if (!TARGETS[device]) throw new Error(`unknown_target:${device}`);
    parsed.set(device, item.slice(separator + 1));
  }
  return parsed;
};
let updates;
if (batchPath) {
  const batch = await readJson(resolve(batchPath));
  if (!Array.isArray(batch) || !batch.length) throw new Error("empty_batch");
  updates = batch.map((item) => {
    const sourceRunLog = String(item.source_run_log || "");
    const results = object(item.results);
    const summaries = object(item.summaries);
    const displayResults = object(item.result_display);
    const metadata = object(item.metadata);
    const resultByDevice = new Map();
    const summaryByDevice = new Map();
    for (const [device, resultPath] of Object.entries(results)) {
      const normalized = String(device).toLowerCase();
      if (!TARGETS[normalized]) throw new Error(`unknown_target:${device}`);
      if (typeof resultPath === "string" && resultPath) {
        resultByDevice.set(normalized, resultPath);
      } else if (object(resultPath) && Object.keys(object(resultPath)).length) {
        summaryByDevice.set(normalized, object(resultPath));
      }
    }
    for (const [device, summary] of Object.entries(summaries)) {
      const normalized = String(device).toLowerCase();
      if (!TARGETS[normalized]) throw new Error(`unknown_target:${device}`);
      if (Object.keys(object(summary)).length) summaryByDevice.set(normalized, object(summary));
    }
    if (
      !String(item.task || "").trim()
      || (!sourceRunLog && !resultByDevice.size && !summaryByDevice.size)
    ) {
      throw new Error("invalid_batch_item");
    }
    return {
      task: String(item.task).trim(),
      sourcePath: sourceRunLog,
      sourceCollectedAt: String(item.source_collected_at || ""),
      sourceSelection: String(item.source_selection || ""),
      resultByDevice,
      summaryByDevice,
      displayResultByDevice: new Map(
        Object.entries(displayResults).map(([device, value]) => [String(device).toLowerCase(), String(value)]),
      ),
      metadataByDevice: new Map(
        Object.entries(metadata).map(([device, value]) => [String(device).toLowerCase(), object(value)]),
      ),
    };
  });
} else {
  const resultByDevice = parseDeviceMap(resultArgs, "result");
  const displayResultByDevice = parseDeviceMap(displayResultArgs, "result_display");
  for (const device of Object.keys(TARGETS)) {
    if (!resultByDevice.has(device)) throw new Error(`missing_result:${device}`);
  }
  updates = [{
    task,
    sourcePath,
    sourceCollectedAt: valueFor("--source-collected-at"),
    sourceSelection: valueFor("--source-selection"),
    resultByDevice,
    summaryByDevice: new Map(),
    displayResultByDevice,
    metadataByDevice: new Map(),
  }];
}

const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(resolve(workbookPath)));
const taskSheet = workbook.worksheets.getItem("116 Tasks");
const used = taskSheet.getUsedRange(true).values;
const receiptsSheet = workbook.worksheets.getItem("Run Receipts");
const receiptUsed = receiptsSheet.getUsedRange(true);
const receiptValues = receiptUsed.values;
const receiptTable = receiptsSheet.tables.items[0];
const runOrderSheet = workbook.worksheets.getItem("Run Order");
const runOrderValues = runOrderSheet.getUsedRange(true).values;
const sourceTaskSet = new Set(updates.filter((item) => item.sourcePath).map((item) => item.task));
const seenRunOrderTasks = new Set();
const surplusRunOrderIndexes = [];
if (sourceTaskSet.size) {
  for (let index = 1; index < runOrderValues.length; index += 1) {
    const runOrderTask = String(runOrderValues[index][3] || "").trim();
    if (!sourceTaskSet.has(runOrderTask) || seenRunOrderTasks.has(runOrderTask)) {
      surplusRunOrderIndexes.push(index);
    } else {
      seenRunOrderTasks.add(runOrderTask);
    }
  }
}
const missingRunOrderTasks = [...sourceTaskSet].filter((item) => !seenRunOrderTasks.has(item));
if (missingRunOrderTasks.length !== surplusRunOrderIndexes.length) {
  throw new Error(`run_order_identity_mismatch:missing=${missingRunOrderTasks.length}:surplus=${surplusRunOrderIndexes.length}`);
}
const repairedRunOrderTasks = [];
for (let index = 0; index < missingRunOrderTasks.length; index += 1) {
  const rowIndex = surplusRunOrderIndexes[index];
  const recoveredTask = missingRunOrderTasks[index];
  runOrderValues[rowIndex][3] = recoveredTask;
  runOrderSheet.getRange(`D${rowIndex + 1}`).values = [[recoveredTask]];
  repairedRunOrderTasks.push({ row: rowIndex + 1, task: recoveredTask });
}
const updatedCells = [];
const updatedRunOrderRows = [];
const updatedEvidence = {};
const decisions = [];
for (const update of updates) {
  const rowIndex = used.findIndex((row) => String(row[0] || "").trim() === update.task);
  if (rowIndex < 0) throw new Error(`task_not_found:${update.task}`);
  const rowNumber = rowIndex + 1;
  if (update.sourcePath) {
    const sourceAbsolute = resolve(update.sourcePath);
    const source = await readJson(sourceAbsolute);
    source.path = sourceAbsolute;
    source.sha = await sha256(sourceAbsolute);
    source.collectedAt = update.sourceCollectedAt;
    source.selection = update.sourceSelection;
    taskSheet.getRange(`B${rowNumber}`).values = [[sourceCell(source, repoRoot)]];
    updatedCells.push(`B${rowNumber}`);
    const runOrderIndex = runOrderValues.findIndex((row) => String(row[3] || "").trim() === update.task);
    if (runOrderIndex < 0) throw new Error(`run_order_task_not_found:${update.task}`);
    const runOrderRow = runOrderIndex + 1;
    const provenance = object(source.provenance).kind || "unknown";
    runOrderSheet.getRange(`C${runOrderRow}`).values = [[list(source.steps).length]];
    runOrderSheet.getRange(`E${runOrderRow}:H${runOrderRow}`).values = [[
      source.seed ?? 111,
      sourceAbsolute,
      source.sha,
      `official success; latest complete across local + 4090; collected=${update.sourceCollectedAt || "unknown"}; provenance=${provenance}`,
    ]];
    updatedRunOrderRows.push(runOrderRow);
  }
  const evidence = new Map();
  for (const [device, resultPath] of update.resultByDevice) {
    evidence.set(device, await readEvidence(
      resultPath,
      update.displayResultByDevice.get(device) || resultPath,
      update.metadataByDevice.get(device) || {},
    ));
  }
  for (const [device, summary] of update.summaryByDevice) {
    evidence.set(device, readSummaryEvidence(
      summary,
      update.displayResultByDevice.get(device) || "",
    ));
  }
  for (const [device, item] of evidence) {
    const target = TARGETS[device];
    const destination = `${target.column}${rowNumber}`;
    const comparison = updateDecision(item);
    decisions.push({
      task: update.task,
      device,
      updated: comparison.update,
      reason: comparison.reason,
      candidate_official_pass: item.official && item.success,
    });
    if (!comparison.update) continue;
    if (item.official && item.success) copySuccessStyle(taskSheet, `${target.column}${rowNumber}`);
    taskSheet.getRange(destination).values = [[targetCell(item, target, repoRoot)]];
    updatedCells.push(destination);
    const row = receiptRow(update.task, method, target, item);
    const existingIndex = receiptValues.findIndex((candidate) => (
      String(candidate[0] || "") === update.task
      && String(candidate[1] || "") === method
      && String(candidate[2] || "") === target.setting
      && String(candidate[12] || "") === item.displayResultPath
    ));
    if (existingIndex >= 0) {
      receiptsSheet.getRange(`A${existingIndex + 1}:N${existingIndex + 1}`).values = [row];
    } else if (receiptTable && typeof receiptTable.rows.add === "function") {
      receiptTable.rows.add(null, [row]);
    } else {
      const nextRow = receiptsSheet.getUsedRange(true).values.length + 1;
      receiptsSheet.getRange(`A${nextRow}:N${nextRow}`).values = [row];
    }
    updatedEvidence[`${update.task}:${device}`] = {
      official: item.official,
      success: item.success,
      run_log: item.displayRunPath,
      result: item.displayResultPath,
    };
  }
  if (updatedCells.some((cell) => cell.endsWith(String(rowNumber)))) {
    taskSheet.getRange(`A${rowNumber}:N${rowNumber}`).format.autofitRows();
  }
}

const outputPath = resolve(valueFor("--output", workbookPath));
const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);
const sourceStatusOutput = valueFor("--source-status-output");
if (sourceStatusOutput) {
  const sourceRows = [];
  for (const update of updates) {
    if (!update.sourcePath) continue;
    const sourceAbsolute = resolve(update.sourcePath);
    const source = await readJson(sourceAbsolute);
    const validator = object(source.validator);
    if (!(source.success === true && validator.official === true && validator.success === true)) {
      throw new Error(`source_not_official_success:${update.task}`);
    }
    sourceRows.push({
      task: update.task,
      collectedAt: update.sourceCollectedAt || "未记录",
      steps: list(source.steps).length,
      provenance: object(source.provenance).kind || "unknown",
      sha: await sha256(sourceAbsolute),
      runLog: repoRelative(sourceAbsolute, repoRoot),
    });
  }
  sourceRows.sort((left, right) => left.task.localeCompare(right.task));
  const lines = [
    "# AndroidWorld Current Source RunLogs",
    "",
    "> 由 `scripts/exp/update_androidworld_workbook.sh` 与主工作簿同步生成。每个任务选择本地权威仓库和 4090 权威仓库中采集时间最新、AndroidWorld 官方 validator 成功且截图/XML/动作链完整的 Source RunLog。",
    "",
    `- 已整理任务：${sourceRows.length}/116`,
    `- 选择规则：latest complete official success across local and 4090`,
    `- 当前路径：\`data/androidworld/<task>/source/OmniFlowSourceSmall_seed111/runlog/current/run_log.json\``,
    "",
    "| Task | Collected at | Steps | Provenance | SHA-256 | Current RunLog |",
    "|---|---:|---:|---|---|---|",
    ...sourceRows.map((item) => (
      `| ${markdownCell(item.task)} | ${markdownCell(item.collectedAt)} | ${item.steps} | ${markdownCell(item.provenance)} | \`${item.sha}\` | \`${markdownCell(item.runLog)}\` |`
    )),
    "",
  ];
  await fs.writeFile(resolve(sourceStatusOutput), lines.join("\n"), "utf8");
}
console.log(JSON.stringify({
  workbook: outputPath,
  source_status: sourceStatusOutput ? resolve(sourceStatusOutput) : "",
  method,
  tasks: updates.map((item) => item.task),
  updated_cells: updatedCells,
  updated_run_order_rows: updatedRunOrderRows,
  repaired_run_order_tasks: repairedRunOrderTasks,
  receipt_rows: Object.keys(updatedEvidence).length,
  evidence: updatedEvidence,
  decisions,
}));
