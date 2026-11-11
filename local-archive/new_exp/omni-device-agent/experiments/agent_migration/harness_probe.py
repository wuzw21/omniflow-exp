from __future__ import annotations
import asyncio, hashlib, json, shutil
from pathlib import Path
from typing import Any, List, Dict, Sequence

from openjarvis.agents.native_openhands import NativeOpenHandsAgent
from openjarvis.agents.native_react import NativeReActAgent
from openjarvis.engine._stubs import InferenceEngine
from openjarvis.core.types import Message, ToolResult
from openjarvis.tools._stubs import BaseTool, ToolSpec

ROOT = Path('runs/agent_migration_harness_probe_001/fixture')
SRC = ROOT / 'incoming' / 'invoice.txt'
DST = ROOT / 'bills' / 'invoice.txt'

class MoveTool(BaseTool):
    tool_id = 'move_file'
    @property
    def spec(self):
        return ToolSpec(
            name='move_file',
            description='Move one file from source to destination.',
            parameters={
                'type':'object',
                'properties': {
                    'source': {'type':'string'},
                    'destination': {'type':'string'},
                },
                'required':['source','destination'],
                'additionalProperties': False,
            },
        )
    def execute(self, **params: Any) -> ToolResult:
        source = Path(params['source'])
        destination = Path(params['destination'])
        if not source.exists():
            return ToolResult('move_file', f'missing source: {source}', success=False)
        destination.parent.mkdir(parents=True, exist_ok=True)
        source.rename(destination)
        return ToolResult('move_file', f'moved {source} -> {destination}', success=True)

class ScriptedEngine(InferenceEngine):
    engine_id = 'scripted-probe'
    def __init__(self, harness: str):
        self.harness = harness
        self.calls = []
    def generate(self, messages: Sequence[Message], *, model: str, temperature: float=0.7, max_tokens: int=1024, **kwargs: Any) -> Dict[str, Any]:
        self.calls.append({
            'message_roles': [m.role.value for m in messages],
            'message_text': [m.text for m in messages],
            'kwargs_keys': sorted(kwargs.keys()),
        })
        if self.harness == 'react':
            if len(self.calls) == 1:
                return {'content': 'Thought: use the file tool.\nAction: move_file\nAction Input: {"source": "runs/agent_migration_harness_probe_001/fixture/incoming/invoice.txt", "destination": "runs/agent_migration_harness_probe_001/fixture/bills/invoice.txt"}', 'usage': {'total_tokens': 1}}
            return {'content': 'Thought: verify the result.\nFinal Answer: The invoice was moved.', 'usage': {'total_tokens': 1}}
        if self.harness == 'openhands':
            if len(self.calls) == 1:
                return {'content': '', 'tool_calls': [{'id': 'call-1', 'name': 'move_file', 'arguments': json.dumps({'source': 'runs/agent_migration_harness_probe_001/fixture/incoming/invoice.txt', 'destination': 'runs/agent_migration_harness_probe_001/fixture/bills/invoice.txt'})}], 'usage': {'total_tokens': 1}}
            return {'content': 'The invoice was moved.', 'usage': {'total_tokens': 1}}
        raise ValueError(self.harness)
    async def stream(self, messages: Sequence[Message], *, model: str, temperature: float=0.7, max_tokens: int=1024, **kwargs: Any):
        if False:
            yield ''
    def list_models(self) -> List[str]:
        return ['probe-model']
    def health(self) -> bool:
        return True

def reset_fixture():
    if ROOT.exists():
        shutil.rmtree(ROOT)
    (ROOT/'incoming').mkdir(parents=True)
    (ROOT/'bills').mkdir(parents=True)
    SRC.write_text('invoice-123\n', encoding='utf-8')

def digest(path: Path):
    if not path.exists(): return None
    return hashlib.sha256(path.read_bytes()).hexdigest()

def run_one(harness):
    reset_fixture()
    engine = ScriptedEngine(harness)
    cls = NativeReActAgent if harness == 'react' else NativeOpenHandsAgent
    agent = cls(engine, model='probe-model', tools=[MoveTool()], max_turns=3, temperature=0.0, max_tokens=256)
    result = agent.run('Move the invoice into the bills directory and verify completion.')
    return {
        'harness': harness,
        'result': {'content': result.content, 'turns': result.turns, 'tool_results': [{'name': t.tool_name, 'success': t.success, 'content': t.content} for t in result.tool_results]},
        'source_exists': SRC.exists(),
        'destination_exists': DST.exists(),
        'destination_sha256': digest(DST),
        'engine_calls': engine.calls,
    }

results = [run_one('react'), run_one('openhands')]
Path('runs/agent_migration_harness_probe_001/results.json').write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding='utf-8')
print(json.dumps(results, indent=2, ensure_ascii=False))
