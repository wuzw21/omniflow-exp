"""Static audit of pinned upstream mode/model bindings; never starts the demo.

No model invocation, vehicle control, Docker, private environment loading, or
network call. This cannot establish that the application works under an outage.
"""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parent
SOURCE = Path.home() / ".agent-reach/research/voice-car-assistant-d90e4f9"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assignments(tree, name):
    return [n.value for n in tree.body if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)]


def run():
    manifest = json.loads((ROOT / "results/vehicle_source_manifest.json").read_text())
    for file in manifest["files"]:
        if sha(SOURCE / file["path"]) != file["sha256"]:
            raise ValueError("upstream file changed")
    agent = ast.parse((SOURCE / "langchain-agent-service/agent_service.py").read_text())
    provider = ast.parse((SOURCE / "langchain-agent-service/llm_provider.py").read_text())
    graph_bindings = {}
    for name in ("_OFFLINE_AGENT", "_ONLINE_AGENT"):
        values = assignments(agent, name)
        assert len(values) == 1
        call = values[0]
        assert isinstance(call, ast.Call) and ast.unparse(call.func) == "create_react_agent"
        graph_bindings[name] = {k.arg: ast.unparse(k.value) for k in call.keywords}
    assert {g["model"] for g in graph_bindings.values()} == {"_llm"}
    assert {g["checkpointer"] for g in graph_bindings.values()} == {"_checkpointer"}
    assert [ast.unparse(v) for v in assignments(agent, "_llm")] == ["make_chat_llm()"]
    assert [ast.unparse(v) for v in assignments(agent, "_checkpointer")] == ["MemorySaver()"]
    factory = next(n for n in provider.body if isinstance(n, ast.FunctionDef) and n.name == "make_chat_llm")
    assert not factory.args.args and not factory.args.kwonlyargs
    ollama = next(n for n in ast.walk(factory) if isinstance(n, ast.Call) and ast.unparse(n.func) == "ChatOllama")
    kwargs = {k.arg: ast.unparse(k.value) for k in ollama.keywords}
    assert kwargs["num_ctx"] == "1536"
    budget = assignments(agent, "_HISTORY_TOKEN_BUDGET")
    assert len(budget) == 1 and ast.literal_eval(budget[0]) == 600
    entrypoints = {}
    for name in ("run_agent", "stream_agent"):
        fn = next(n for n in agent.body if isinstance(n, ast.FunctionDef) and n.name == name)
        choices = [ast.unparse(n) for n in ast.walk(fn) if isinstance(n, ast.IfExp)
                   and isinstance(n.body, ast.Name) and n.body.id == "_ONLINE_AGENT"]
        assert choices == ["_ONLINE_AGENT if network_mode == 'online' else _OFFLINE_AGENT"]
        entrypoints[name] = choices[0]
    return {"status": "completed_static_source_audit", "source_sha256": sha(Path(__file__)),
        "repository": manifest["repository"], "commit": manifest["commit"],
        "graphs": graph_bindings, "provider_binding": ast.unparse(assignments(provider, "LLM_PROVIDER")[0]),
        "default_ollama_context": 1536, "history_budget_approx_tokens": 600, "entrypoints": entrypoints,
        "checked_invariants": ["both graphs bind the same model object", "both graphs share the message checkpointer",
            "model factory has no per-request network-mode argument", "both request entrypoints select graphs by network mode",
            "default local context and approximate history budgets retained"],
        "limits": ["Static bindings only, no application execution or network-isolation test",
            "History checkpoint is not a numerical KV checkpoint",
            "Environment chooses local/cloud provider at import; does not prove either endpoint is available",
            "No inference latency, cache-hit rate, task quality or physical vehicle validation"]}


if __name__ == "__main__":
    result = run()
    (ROOT / "results/vehicle_mode_audit.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))
