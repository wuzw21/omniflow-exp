import json
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
RESULT=ROOT/"experiments/fluxmem_scaling/result_002"
def read(name):
    return json.loads((RESULT/"evidence"/name).read_text())

class ScalingEvidenceTests(unittest.TestCase):
    def setUp(self):
        if not RESULT.exists():
            self.skipTest("Scaling evidence has not been generated on the physical device")

    def test_94_prefixes_are_bitwise_equivalent(self):
        r=read("audit_results.json")
        self.assertEqual(r["status"],"completed")
        self.assertEqual(len(r["records"]),94)
        for x in r["records"]:
            self.assertTrue(x["equivalence"]["indices_equal"])
            self.assertTrue(x["equivalence"]["features_bitwise_equal"])
            self.assertEqual(x["equivalence"]["max_abs_error"],0)

    def test_three_query_densities_use_same_video_and_have_expected_endpoints(self):
        expected={40:6,10:20,4:48}
        for cadence,count in expected.items():
            r=read(f"memory_{cadence}_results.json")
            self.assertEqual(r["status"],"completed")
            run=next(x for x in r["runs"] if not x["warmup"])
            self.assertEqual(len(run["events"]),count)
            self.assertEqual(run["events"][-1]["blocks"],94)
            if run["condition"] == "C":
                self.assertEqual(run["counts"]["merge_frames"],58)
            else:
                self.assertGreater(run["counts"]["merge_frames"],58)

    def test_b_grows_with_query_density_c_is_flat_in_memory_replay(self):
        vals={}
        for cadence in [40,10,4]:
            r=read(f"memory_{cadence}_results.json")["summary"]["total_memory_seconds"]
            vals[cadence]=(r["B_median"],r["C_median"])
        self.assertGreater(vals[10][0],vals[40][0]*2)
        self.assertGreater(vals[4][0],vals[10][0]*1.5)
        self.assertLess(max(v[1] for v in vals.values())-min(v[1] for v in vals.values()),1.0)

    def test_state_grows_more_slowly_for_c(self):
        rows=read("audit_results.json")["records"]
        by={x["blocks"]:x for x in rows}
        self.assertEqual(by[41]["retained_tokens"],3087)
        self.assertEqual(by[94]["retained_tokens"],4258)
        b41=41*120*2048*2; b94=94*120*2048*2
        c41=by[41]["state"]["tensor_storage_bytes"]; c94=by[94]["state"]["tensor_storage_bytes"]
        self.assertGreater(b94/b41, c94/c41)

    def test_frontend_runs_complete_and_c_is_faster_in_all_three_single_pairs(self):
        for cadence in [40,10,4]:
            r=read(f"frontend_{cadence}_results.json")
            self.assertEqual(r["status"],"completed")
            s=r["summary"]["total_seconds"]
            self.assertLess(s["C_median"],s["B_median"])
            self.assertEqual(len([x for x in r["runs"] if not x["warmup"]]),2)

if __name__=="__main__":
    unittest.main()
