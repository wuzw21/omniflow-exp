"""Verify frozen upstream inputs; these checks do not run or simulate an LLM."""
import hashlib
import json
from pathlib import Path
import subprocess
import unittest

ROOT = Path(__file__).parent


class ReproductionContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads((ROOT / 'results/kvcmas_reproduction_manifest.json').read_text())
        cls.source = Path(cls.manifest['source']['path'])

    def test_original_revision_and_files(self):
        actual = subprocess.check_output(['git', '-C', str(self.source), 'rev-parse', 'HEAD'], text=True).strip()
        self.assertEqual(actual, self.manifest['source']['revision'])
        subprocess.run(['git', '-C', str(self.source), 'diff', '--quiet', 'HEAD', '--'], check=True)
        for f in self.manifest['files']:
            self.assertEqual(hashlib.sha256((self.source / f['path']).read_bytes()).hexdigest(), f['sha256'])

    def test_full_original_dataset_and_order(self):
        d = self.manifest['dataset']
        rows = [json.loads(line) for line in (self.source / d['path']).read_text().splitlines() if line.strip()]
        self.assertEqual(len(rows), 161)
        self.assertEqual([r['name'] for r in rows], d['names'])
        self.assertEqual(len(set(d['names'])), 161)
        self.assertTrue(all(r['prompt'] and r['test'] for r in rows))

    def test_no_model_or_reproduction_result_claimed(self):
        m = self.manifest
        self.assertEqual(m['model']['repo'], 'Qwen/Qwen2.5-Coder-7B-Instruct')
        self.assertEqual(len(m['model']['revision']), 40)
        self.assertEqual(m['status'], 'source_and_protocol_pinned_not_executed')
        self.assertEqual(m['model_calls'], 0)
        self.assertIsNone(m['task_results'])
        self.assertIsNone(m['speedup'])


if __name__ == '__main__':
    unittest.main()
