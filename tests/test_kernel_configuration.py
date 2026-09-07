import json
import os
from pathlib import Path
import subprocess
import sys


def test_standalone_kernel_and_mcp_do_not_load_benchmark_configuration(tmp_path):
    env = {**os.environ, 'OMNIFLOW_ANDROIDWORLD_CONFIG': str(tmp_path/'absent.json')}
    result = subprocess.run([sys.executable, '-c',
        'from omniflow import OmniFlow; from src.integrations.gui_agent_mcp import create_server; '
        'from omniflow.core.config import RuntimeSettings; assert RuntimeSettings().max_steps == 30'],
        env=env, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr


def test_androidworld_harness_still_reads_explicit_protocol(tmp_path):
    payload = json.loads((Path(__file__).parents[1]/'config/paper_androidworld.json').read_text())
    payload['protocol']['max_steps'] = 17
    config = tmp_path/'experiment.json'
    config.write_text(json.dumps(payload))
    result = subprocess.run([sys.executable, '-c',
        'from src.experiment.protocol import MAX_STEPS; assert MAX_STEPS == 17'],
        env={**os.environ, 'OMNIFLOW_ANDROIDWORLD_CONFIG': str(config)},
        capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
