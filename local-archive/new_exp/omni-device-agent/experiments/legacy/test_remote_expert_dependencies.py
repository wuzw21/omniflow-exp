import itertools
import unittest
import numpy as np
from feasibility import evaluate_fixed_residency

from remote_expert_dependencies import (allocate_coverage, coverage_curves, evaluate,
                                        mask_from_counts, placements, ranked_experts)


class RemoteDependencyTests(unittest.TestCase):
    def test_scalar_reference_agrees_on_integer_misses_and_dependencies(self):
        rng=np.random.default_rng(13)
        routes=np.stack([rng.choice(6,size=(3,2),replace=False) for _ in range(12)])
        for _ in range(20):
            mask=rng.random((3,6)) < .5
            resident=[np.flatnonzero(x).tolist() for x in mask]
            scalar=evaluate_fixed_residency(routes.tolist(),resident,30,expert_count=6)
            vector=evaluate(routes,mask)
            hits=int(mask[np.arange(3)[None,:,None],routes].sum())
            self.assertEqual(hits+scalar['remote_expert_selections'],routes.size)
            self.assertAlmostEqual(scalar['expert_hit_rate'],vector['local_expert_hit_rate'],places=15)
            self.assertEqual(sum(scalar['blocking_rpc_per_token'])/len(routes),vector['remote_layers_per_token']['mean'])

    def test_equal_hit_rate_can_have_different_rpc_count(self):
        routes=np.array([[[0,1],[0,1]]])
        concentrated=np.array([[False,False],[True,True]])
        scattered=np.array([[True,False],[True,False]])
        a,b=evaluate(routes,concentrated),evaluate(routes,scattered)
        self.assertEqual(a['local_expert_hit_rate'],b['local_expert_hit_rate'])
        self.assertEqual(a['remote_layers_per_token']['mean'],1)
        self.assertEqual(b['remote_layers_per_token']['mean'],2)

    def test_knapsack_matches_exhaustive_allocation(self):
        rng=np.random.default_rng(7)
        for _ in range(20):
            curves=np.concatenate([np.zeros((3,1),dtype=int), rng.integers(0,9,(3,3)).cumsum(axis=1)],axis=1)
            for budget in range(10):
                feasible=[a for a in itertools.product(range(4),repeat=3) if sum(a)==budget]
                allocation,value=allocate_coverage(curves,budget)
                self.assertEqual(sum(allocation),budget)
                self.assertEqual(value,max(sum(curves[i,k] for i,k in enumerate(a)) for a in feasible))

    def test_coverage_requires_all_experts_not_mean_hits(self):
        routes=np.array([[[0,1]],[[0,2]],[[1,2]]])
        curves=coverage_curves(routes,np.array([[0,1,2,3]]),1)
        self.assertEqual(curves.tolist(),[[0,0,1,3,3]])

    def test_empty_and_full_residency(self):
        routes=np.array([[[0,1],[1,2]],[[1,2],[0,2]]])
        self.assertEqual(evaluate(routes,np.zeros((2,3),bool))['remote_layers_per_token']['mean'],2)
        self.assertEqual(evaluate(routes,np.ones((2,3),bool))['fully_local_token_fraction'],1)

    def test_tie_break_is_stable_and_layer_ids_separate(self):
        ranking=ranked_experts(np.ones((2,4),dtype=int))
        self.assertEqual(ranking.tolist(),[[0,1,2,3],[0,1,2,3]])
        mask=mask_from_counts(ranking,[1,0])
        self.assertEqual(evaluate(np.array([[[0],[0]]]),mask)['remote_layers_per_token']['mean'],1)

    def test_all_policies_get_exact_same_budget(self):
        routes=np.array([[[0,1],[1,2]],[[0,2],[1,3]],[[1,3],[0,2]]])
        for slots in range(9):
            masks,_=placements(routes,4,slots,step=1)
            self.assertEqual({int(m.sum()) for m in masks.values()},{slots})

    def test_evaluation_never_modifies_calibration_placement(self):
        train=np.array([[[0,1],[1,2]],[[0,2],[1,3]]])
        masks,_=placements(train,4,4,step=1)
        before={k:v.copy() for k,v in masks.items()}
        for mask in masks.values():
            evaluate(np.array([[[2,3],[0,3]]]),mask)
        for k in masks:
            np.testing.assert_array_equal(masks[k],before[k])


if __name__=='__main__':
    unittest.main()
