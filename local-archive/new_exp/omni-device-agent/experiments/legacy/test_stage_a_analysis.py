"""Statistical/selection regressions use synthetic scores, never model evidence."""
import copy
import unittest
from analyze_stage_a_development import paired_delta,recovery_fraction,summarize


def fixture():
    manifest={'questions':[{'id':q,'split':s} for q,s in [('a','dev'),('b','dev'),('c','test')]]}
    score=lambda f:{'f1':f,'em':float(f==1),'kl_reference_to_variant':0.1}
    report={'status':'completed','settings':{'conditions':['P0','P1','P2'],'starts':list(range(0,28,4)),
            'width':4,'limit':2},'cases':[]}
    for q in ('a','b'):
        for condition in report['settings']['conditions']:
            report['cases'].append({'id':q,'split':'dev','condition':condition,'native':score(.5),'direct':score(0),
                'history_controls':{'native_source_history':score(0),'source_history_shared_with_target_prefix':score(.5)},
                'calibration':{'split_dense_relative_logit_error':.001,'kl_reference_to_variant':.0001,'argmax_agreement':1},
                'intervals':[{'start':s,'width':4,'scores':{'source':score(.75 if s==8 else .5),'oracle':score(1 if s==16 else .75)}}
                             for s in report['settings']['starts']]})
    return report,manifest


class AnalysisTests(unittest.TestCase):
    def test_pairing_preserves_constant_within_question_effect(self):
        result=paired_delta([0,.5,.75],[.25,.75,1],draws=100)
        self.assertEqual(result['paired_percentile_95ci'],[.25,.25])

    def test_nonpositive_reference_gap_never_produces_recovery_fraction(self):
        for source in (.5,.75):
            self.assertIsNone(recovery_fraction(.5,source,1)['fraction'])
        self.assertEqual(recovery_fraction(1,0,.5)['fraction'],.5)

    def test_selection_uses_source_not_oracle_and_requires_complete_design(self):
        report,manifest=fixture()
        full=summarize(report,manifest,draws=20)
        self.assertEqual(full['interval_candidate']['start'],8)
        self.assertIsNone(full['conditions']['P2']['intervals'][2]['recovery']['fraction'])
        report['cases']=[r for r in report['cases'] if r['id']=='a']
        report['settings']['limit']=1
        self.assertIsNone(summarize(report,manifest,draws=20)['interval_candidate'])

    def test_duplicate_heldout_and_missing_condition_cannot_change_sample(self):
        report,manifest=fixture()
        variants=[]
        dup=copy.deepcopy(report);dup['cases'].append(dup['cases'][0]);variants.append(dup)
        held=copy.deepcopy(report);held['cases'][0]['id']='c';variants.append(held)
        missing=copy.deepcopy(report);missing['cases'].pop();variants.append(missing)
        for variant in variants:
            with self.assertRaises(ValueError):summarize(variant,manifest,draws=20)

    def test_incomplete_run_and_nan_metrics_rejected(self):
        report,manifest=fixture()
        report['status']='running'
        with self.assertRaises(ValueError):summarize(report,manifest,draws=20)
        report['status']='completed';report['cases'][0]['native']['f1']=float('nan')
        with self.assertRaises(ValueError):summarize(report,manifest,draws=20)


if __name__=='__main__':unittest.main()
