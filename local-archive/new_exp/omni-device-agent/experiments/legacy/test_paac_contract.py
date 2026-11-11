"""Deterministic upstream primitives and source contracts, never agent success."""
import json
from pathlib import Path
import unittest

from audit_paac_contract import audit
from prepare_tau2_pilot import build

ROOT = Path(__file__).parent
RESEARCH = Path.home() / '.agent-reach/research'


class PaacContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.audit = audit(RESEARCH / 'paac-f929831')

    def test_missing_old_proposal_is_not_protected_by_alignment(self):
        for row in self.audit['operator_cases'][:2]:
            self.assertTrue(row['alignment_passed'])
            self.assertTrue(row['old_value_remains_in_proposal_path'])
            self.assertFalse(row['old_value_remains_in_explicit_registry_path'])
            self.assertEqual(row['registry_before']['NAME_FIRST_1'], row['registry_after']['NAME_FIRST_1'])

    def test_repeated_known_proposal_reuses_canonical_token(self):
        row = self.audit['operator_cases'][2]
        self.assertFalse(row['old_value_remains_in_proposal_path'])
        self.assertEqual(row['registry_before'], row['registry_after'])
        self.assertEqual(row['proposal_path_output'], row['explicit_registry_sanitize_output'])

    def test_source_branches_are_kept_distinct(self):
        self.assertEqual(len(self.audit['sanitizer_dispatch']), 2)
        self.assertEqual(len(self.audit['tau2_evaluator_override_calls']), 2)
        self.assertEqual(len([c for c in self.audit['run_call_sites'] if 'self.device_sanitize(' in c['expression']]), 2)
        limits = self.audit['scientific_limits']
        self.assertIsNone(limits['observed_real_llm_omission_rate'])
        self.assertFalse(limits['task_quality_measured'])
        self.assertFalse(limits['bug_fix_acceptance'])

    def test_frozen_pilot_is_development_only_and_has_not_run(self):
        result = build(RESEARCH / 'tau2-verified-864350a')
        saved = json.loads((ROOT / 'tau2_capability_protocol.json').read_text())
        self.assertEqual(result, saved)
        self.assertEqual(result['counts'], {'total': 114, 'train': 74, 'test': 40, 'base': 114})
        self.assertEqual(len({r['id'] for r in result['selected']}), 4)
        self.assertTrue(result['local_necessity_unproven'])
        self.assertEqual(result['status'], 'protocol_frozen_not_executed')


if __name__ == '__main__':
    unittest.main()
