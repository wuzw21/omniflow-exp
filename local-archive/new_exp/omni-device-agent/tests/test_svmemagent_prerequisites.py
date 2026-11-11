"""Regression evidence: missing source must not become a successful reproduction."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / 'experiments/svmemagent/release_probe_20261031'

class SVMemAgentPrerequisiteTests(unittest.TestCase):
    def test_recorded_native_attempts_failed_before_inference(self):
        sha = hashlib.sha256((PROBE / 'config.json').read_bytes()).hexdigest()
        for name, version in [('load_transformers_4.json', '4.57.6'),
                              ('load_transformers_5.json', '5.16.1')]:
            row = json.loads((PROBE / name).read_text())
            self.assertEqual(row['status'], 'blocked_at_config_loading')
            self.assertEqual(row['transformers'], version)
            self.assertEqual(row['config_sha256'], sha)
            self.assertIn('videostreaming_memory_agent', row['error'])
            self.assertIsNone(row['writer_measurements'])
            self.assertIsNone(row['qa_measurements'])
            self.assertEqual(row['candidate_official_provenance'], 'unverified')

    def test_checkpoint_listing_is_not_model_implementation(self):
        meta = json.loads((PROBE / 'candidate_metadata.json').read_text())
        self.assertEqual(meta['sha'], '7b7ba15a2d35064287f449259607759e36259440')
        paths = [r['rfilename'] for r in meta['siblings']]
        self.assertEqual(len(paths), 1060)
        self.assertTrue(any(p.endswith('.safetensors') for p in paths))
        self.assertFalse(any(p.endswith(('.py', '.sh', '.zip', '.tar.gz')) for p in paths))
        self.assertNotIn('auto_map', json.loads((PROBE / 'config.json').read_text()))

    def test_search_errors_are_not_reported_as_empty_successful_searches(self):
        for name in ['github_repo_search', 'github_python_search', 'github_class_search']:
            row = json.loads((PROBE / (name + '.json')).read_text())
            self.assertEqual(row['exit_code'], 0)
            self.assertEqual(json.loads(row['stdout']), [])

    def test_repeat_probe_does_not_overwrite_previous_run(self):
        with tempfile.TemporaryDirectory() as d:
            output = Path(d) / 'record.json'
            original = b'{"preserve": true}\n'
            output.write_bytes(original)
            run = subprocess.run([sys.executable, str(ROOT / 'scripts/probe_svmemagent_load.py'),
                                  '--config-dir', str(PROBE), '--output', str(output)],
                                 capture_output=True, text=True)
            self.assertNotEqual(run.returncode, 0)
            self.assertIn('FileExistsError', run.stderr)
            self.assertEqual(output.read_bytes(), original)

if __name__ == '__main__':
    unittest.main()
