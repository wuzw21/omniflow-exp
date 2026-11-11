import ast
import inspect
import json
from pathlib import Path
import tempfile
import unittest
from probe_minions_collaboration import load_upstream, lcp, SOURCE


class SourceAndAccountingTests(unittest.TestCase):
    def test_original_loop_and_fixtures_are_pinned(self):
        Minion, Usage, manifest = load_upstream()
        self.assertEqual(len(manifest['verified_sha256']), 8)
        self.assertEqual(Path(Minion.__call__.__code__.co_filename), SOURCE / 'minions/minion.py')
        self.assertEqual(inspect.signature(Minion).parameters['max_rounds'].default, 3)
        self.assertEqual((Usage(prompt_tokens=4) + Usage(prompt_tokens=7)).prompt_tokens, 11)
        with tempfile.TemporaryDirectory() as d:
            runner = Minion(log_dir=d, max_rounds=2)
            question = json.loads((SOURCE / 'minions/examples/code/task.json').read_text())['question']
            prompt = runner.supervisor_initial_prompt.format(task=question, mcp_tools_info=None)
            context = (SOURCE / 'minions/examples/code/sample.txt').read_text()
            self.assertNotIn(context, prompt)
            self.assertNotIn('filter_data', question)
            self.assertIn(context, runner.worker_system_prompt.format(context=context, task=question))

    def test_prefix_requires_exact_position_and_tokens(self):
        self.assertEqual(lcp([1, 2, 3], [1, 2, 4]), 2)
        self.assertEqual(lcp([1, 2], [1, 2, 3]), 2)
        self.assertEqual(lcp([], [1]), 0)
        self.assertEqual(lcp([1, 2, 3], [2, 3]), 0)

    def test_final_generated_token_is_not_counted_as_cached(self):
        previous_input, generated = [1, 2], [3, 4, 5]
        materialized = previous_input + generated[:-1]
        self.assertEqual(lcp(materialized, [1, 2, 3, 4, 5, 6]), 4)


if __name__ == '__main__':
    unittest.main()
