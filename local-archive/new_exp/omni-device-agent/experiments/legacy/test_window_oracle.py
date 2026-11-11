import unittest
import numpy as np
from window_oracle import contiguous_segments, slots_in_window, analyze


class WindowOracleTests(unittest.TestCase):
    def test_request_session_and_position_gaps_are_all_boundaries(self):
        segments = contiguous_segments(np.array([0,0,0,1,1,1]),
                                       np.array([0,1,3,4,5,6]),
                                       np.array([0,0,0,0,1,1]))
        self.assertEqual(segments, [(0,2),(2,3),(3,4),(4,6)])

    def test_same_expert_id_in_different_layers_costs_two_slots(self):
        routes = np.array([[[0],[0]],[[0],[0]]])
        self.assertEqual(slots_in_window(routes,4).tolist(),[0,4])

    def test_no_cross_request_windows_or_partial_tail(self):
        routes = np.zeros((6,2,1),dtype=int)
        r = analyze(routes,[(0,3),(3,6)],(2,4),4,100,np.zeros(6),['test'])
        self.assertEqual(r['2']['oracle_resident_slots']['count'],2)
        self.assertEqual(r['2']['covered_rows'],4)
        self.assertEqual(r['4']['oracle_resident_slots']['count'],0)
        self.assertEqual(r['2']['slots_new_to_previous_disjoint_window']['count'],0)

    def test_known_oracle_and_new_slots(self):
        routes=np.array([[[0],[0]],[[1],[0]],[[1],[1]],[[2],[1]]])
        r=analyze(routes,[(0,4)],(2,),4,100,np.zeros(4),['test'])['2']
        self.assertEqual(r['oracle_resident_slots']['mean'],3)
        self.assertEqual(r['slots_new_to_previous_disjoint_window']['mean'],2)

    def test_invalid_shape_and_ids_fail(self):
        for routes in (np.empty((0,2,1),dtype=int),np.array([[[4]]]),np.array([[0]])):
            with self.assertRaises(ValueError): slots_in_window(routes,4)

    def test_boundary_length_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            contiguous_segments(np.array([0]),np.array([0,1]),np.array([0]))

    def test_unknown_session_is_not_mislabeled_as_last_named_session(self):
        routes=np.array([[[0]],[[1]]])
        result=analyze(routes,[(0,2)],(2,),4,100,np.array([-1,-1]),['named'])['2']
        self.assertEqual(list(result['per_session_slots']),['unmapped:-1'])
        self.assertEqual(result['request_macro_median_slots']['count'],1)


if __name__=='__main__': unittest.main()
