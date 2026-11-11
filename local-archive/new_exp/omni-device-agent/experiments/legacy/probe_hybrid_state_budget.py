"""Run pinned upstream state-pool allocation and byte accounting on CPU.

This extracts unchanged definitions, not a new cache/compaction implementation.
Only the declared pool API is executed. No model weights, kernels, agent loop,
network measurement, or full FreeToken installation are involved.
"""
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import torch

from feasibility import kv_bytes, serialization_seconds

ROOT = Path(__file__).parent
UPSTREAM = Path.home() / ".agent-reach/research/freetoken-32056cf"
CONFIGS = Path.home() / ".agent-reach/research/hybrid-state-configs"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def definitions(path, names, namespace):
    tree = ast.parse(path.read_text())
    nodes = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
    selected = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            nodes.append(node)
            selected.add(node.name)
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in node.targets):
            nodes.append(node)
            selected.update(t.id for t in node.targets if isinstance(t, ast.Name))
    if selected != set(names):
        raise ValueError(f"upstream definitions missing: {set(names)-selected}")
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(path), "exec"), namespace)


def run():
    manifest = json.loads((ROOT / "results/freetoken_source_manifest.json").read_text())
    for record in manifest["files"]:
        if digest(UPSTREAM / record["path"]) != record["sha256"]:
            raise ValueError("pinned upstream source changed: " + record["path"])
    configs = json.loads((CONFIGS / "manifest.json").read_text())
    ns = {"torch": torch, "ENV": SimpleNamespace(MAMBA_SSM_DTYPE="float32")}
    definitions(UPSTREAM / "python/freetoken/utils/misc.py", {"div_even"}, ns)
    definitions(UPSTREAM / "python/freetoken/kvcache/linear_state_pool.py",
                {"_SSM_DTYPES", "ssm_state_dtype", "_linear_local_dims", "LinearStatePool",
                 "linear_state_bytes_per_req", "_linear_pool_num_slots", "_linear_pool_min_slots"}, ns)
    rows = []
    for record in configs:
        path = Path(record["path"])
        if digest(path) != record["sha256"]:
            raise ValueError("model config changed")
        cfg = json.loads(path.read_text())["text_config"]
        layer_types = cfg["layer_types"]
        if set(layer_types) != {"linear_attention", "full_attention"} or len(layer_types) != cfg["num_hidden_layers"]:
            raise ValueError("unsupported state geometry")
        group = SimpleNamespace(layer_ids=[i for i, t in enumerate(layer_types) if t == "linear_attention"],
            num_key_heads=cfg["linear_num_key_heads"], num_value_heads=cfg["linear_num_value_heads"],
            key_head_dim=cfg["linear_key_head_dim"], value_head_dim=cfg["linear_value_head_dim"],
            conv_kernel_dim=cfg["linear_conv_kernel_dim"])
        pool = ns["LinearStatePool"](group, num_slots=3, dtype=torch.bfloat16, device=torch.device("cpu"), tp_size=1)
        measured = pool.bytes_per_slot()
        estimated = ns["linear_state_bytes_per_req"](group, 1, torch.bfloat16)
        if measured != estimated:
            raise AssertionError("upstream allocation differs from upstream byte estimate")
        src, dst = pool.alloc(2)
        pool.conv_states[:, src].fill_(1)
        pool.recurrent_states[:, src].fill_(2)
        pool.copy_from(src, dst)
        copied = torch.equal(pool.conv_states[:, src], pool.conv_states[:, dst]) and torch.equal(pool.recurrent_states[:, src], pool.recurrent_states[:, dst])
        pool.reset(dst)
        reset_isolated = bool((pool.conv_states[:, dst] == 0).all() and (pool.recurrent_states[:, dst] == 0).all()
                              and (pool.conv_states[:, src] == 1).all() and (pool.recurrent_states[:, src] == 2).all())
        if not copied or not reset_isolated:
            raise AssertionError("upstream state copy/reset lifecycle failed")
        policy_cfg = SimpleNamespace(max_running_req=1, cache_type="hybrid_radix", linear_state_cache_ratio=1)
        rows.append({"model": record["repo"], "config_revision": record["revision"], "config_sha256": record["sha256"],
            "linear_layers": len(group.layer_ids), "full_attention_layers": layer_types.count("full_attention"),
            "state_bytes_per_snapshot": measured,
            "conv_shape_per_slot": list(pool.conv_states[:, 0].shape),
            "recurrent_shape_per_slot": list(pool.recurrent_states[:, 0].shape),
            "conv_dtype": str(pool.conv_states.dtype), "recurrent_dtype": str(pool.recurrent_states.dtype),
            "copy_equal": copied, "reset_isolated": reset_isolated,
            "upstream_minimum_slots_at_one_running_request": ns["_linear_pool_min_slots"](policy_cfg),
            "upstream_slots_with_ratio_one_at_one_running_request": ns["_linear_pool_num_slots"](policy_cfg),
            "raw_snapshot_serialization_seconds": {str(bw): serialization_seconds(measured, bw) for bw in (20, 100, 1000)},
            "attention_kv_bytes_by_tokens": {str(n): kv_bytes(n, layer_types.count("full_attention"), cfg["num_key_value_heads"], cfg["head_dim"]) for n in (1024, 8192)}})
        del pool
    return {"status": "completed_cpu_state_pool_probe", "source_sha256": digest(Path(__file__)),
        "upstream_commit": manifest["commit"], "torch": torch.__version__, "records": rows,
        "limits": ["Allocated state tensors contain test constants, not model-computed state",
                   "CPU allocation/copy/reset and byte accounting only; no full FreeToken runtime or model inference",
                   "Transfer rates are analytical inputs, not measured network; no compression or cross-model state conversion",
                   "TP=1, BF16 conv state and FP32 recurrent state; omitted optional sibling states and allocator overhead",
                   "Configured pool slot counts are policy accounting, not measured peak serving memory",
                   "This does not prove local-model necessity or an end-to-end opportunity"]}


if __name__ == "__main__":
    torch.set_num_threads(4)
    result = run()
    (ROOT / "results/hybrid_state_budget.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))
