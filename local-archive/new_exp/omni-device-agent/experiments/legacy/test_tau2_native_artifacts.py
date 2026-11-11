"""Regress real device evidence and original evaluator boundaries, never fake success."""
import hashlib
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).parent


def read(name):
    return json.loads((ROOT / 'results' / name).read_text())


class NativeTau2Evidence(unittest.TestCase):
    def test_reference_replay_is_repeatable_and_not_agent_success(self):
        a = read('tau2_native_environment_001.json')
        b = read('tau2_native_environment_002.json')
        self.assertEqual(a, b)
        self.assertIn('no model or agent run', a['scope'])
        self.assertEqual(a['status'], 'completed')
        self.assertEqual({x['task_id'] for x in a['cases']}, {'66', '84', '6', '41'})
        for case in a['cases']:
            self.assertEqual(case['noop_reward'], 0)
            self.assertFalse(case['noop_db_match'])
            self.assertEqual(case['reference_reward'], 1)
            self.assertTrue(case['reference_db_match'])
            self.assertFalse(any(case['reference_tool_errors']))

    def test_actual_native_entry_and_source_guard_survive_imports(self):
        from check_tau2_native_environment import verify_source
        first = verify_source()
        from tau2.run import run_task
        self.assertTrue(callable(run_task))
        self.assertEqual(first, verify_source())

    def test_actual_m4_tool_result_and_restart(self):
        r = read('tau2_transport_001/summary.json')
        self.assertEqual(r['status'], 'completed')
        self.assertEqual(r['hardware']['chip'], 'Apple M4')
        self.assertEqual(r['runtime']['mlx-lm'], '0.32.2')
        self.assertTrue(r['weights_verified'])
        self.assertEqual(len(r['runs']), 2)
        self.assertEqual(len({run['server']['pid'] for run in r['runs']}), 2)
        for run in r['runs']:
            self.assertTrue(run['server']['ready'])
            self.assertEqual(run['server']['exit_code'], -15)  # intentional shutdown
            self.assertTrue(run['one_correct_tool_call'])
            self.assertTrue(run['used_tool_result'])
            self.assertEqual([c['finish_reason'] for c in run['calls']], ['tool_calls', 'stop'])
            cache = [c['usage']['prompt_tokens_details']['cached_tokens'] for c in run['calls']]
            self.assertEqual(cache, [0, 224])
            raw = Path(r['trace']) / f"cycle-{run['cycle']}" / 'calls.jsonl'
            self.assertEqual(hashlib.sha256(raw.read_bytes()).hexdigest(), run['raw_calls_sha256'])
        self.assertFalse(r['task_success_claim_permitted'])
        self.assertFalse(r['speedup_claim_permitted'])

    def test_frozen_transport_sources(self):
        r = read('tau2_transport_001/summary.json')
        for name, sha in r['scripts_sha256'].items():
            self.assertEqual(hashlib.sha256((ROOT / 'results/tau2_transport_001' / name).read_bytes()).hexdigest(), sha)

    def test_setup_failures_did_not_produce_agent_results(self):
        for attempt in ['002', '003']:
            r = read(f'tau2_local_{attempt}/summary.json')
            self.assertEqual(r['status'], 'execution_failed')
            self.assertEqual(r['calls'], [])
            self.assertNotIn('reward_info', r)
        first = (ROOT / 'results/tau2_local_001.log').read_text()
        self.assertIn('AssertionError', first)

    def test_real_remote_failure_gates_model_startup(self):
        r = read('tau2_preflight_005/summary.json')
        self.assertEqual(r['status'], 'environment_unavailable')
        self.assertEqual(r['calls'], [])
        self.assertEqual(r['endpoints'], {})
        pre = r['remote_preflight']
        self.assertEqual(pre['status'], 'unavailable')
        self.assertFalse(pre['checks']['write_fsync_rename_delete'])
        self.assertFalse(pre['checks']['model_files_exist'])
        self.assertFalse(pre['checks']['runtime_ready'])
        self.assertIn({'check': 'storage', 'errno': 28, 'type': 'OSError'}, pre['errors'])
        self.assertFalse(r['quality_generalization_permitted'])
        self.assertFalse(r['speedup_claim_permitted'])


if __name__ == '__main__':
    unittest.main()
