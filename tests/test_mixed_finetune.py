import unittest
from collections import Counter
from types import SimpleNamespace
import torch


class MixedFineTuneTests(unittest.TestCase):
    def test_curriculum_skips_first_reset_and_fixed_height_columns(self):
        from go2_stairs.mixed_finetune import curriculum_eligible_ids
        ids = torch.arange(3)
        lengths = torch.tensor([0, 20, 20])
        types = torch.tensor([0, 1, 2])
        fixed_columns = torch.tensor([False, True, False])
        self.assertEqual(curriculum_eligible_ids(ids, lengths, types, fixed_columns).tolist(), [2])

    def test_guard_does_not_trade_basic_walking_for_stair_score(self):
        from go2_stairs.mixed_finetune_job import development_decision
        keys = ['flat', 'up10', 'down10', 'up20', 'down20']
        base = {k: {'n': 8, 'completed': 8, 'clean': 8, 'invalid': 0} for k in keys}
        candidate = {k: dict(v) for k, v in base.items()}
        candidate['flat'].update(completed=2, clean=2)
        exposure = {'groups': {k: {'actual_tile_samples': 20, 'target_riser_zone_samples': 10}
                    for k in ['pyramid_stairs_h20', 'pyramid_stairs_inv_h20']}}
        self.assertEqual(development_decision(base, candidate, exposure), 'STOP_BASIC_REGRESSION')
        candidate['flat'] = dict(base['flat'])
        self.assertEqual(development_decision(base, candidate, exposure), 'CONTINUE')
        candidate['up20']['invalid'] = 1
        self.assertEqual(development_decision(base, candidate, exposure), 'STOP_INVALID_EVALUATION')

    def test_exposure_tracks_current_position_not_spawn_assignment(self):
        from go2_stairs.mixed_finetune import mixture_specs, current_tile_labels
        points = torch.tensor([[-36., -196.], [-34.8, -148.], [100., 0.]])
        labels, riser_zone = current_tile_labels(points, mixture_specs(), 10, 50, 8.)
        self.assertEqual(labels.tolist(), [0, 2, 12])
        self.assertEqual(riser_zone.tolist(), [False, True, False])

    def test_public_warmstart_keeps_weights_but_not_old_adam_momentum_or_counter(self):
        from go2_stairs.mixed_finetune import warm_start_public
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=.1)
        model(torch.ones(1, 2)).sum().backward(); optimizer.step()
        state = {'model_state_dict': {k: v.detach().clone()+.2 for k, v in model.state_dict().items()}, 'iter': 1499,
                 'optimizer_state_dict': optimizer.state_dict()}
        runner = SimpleNamespace(alg=SimpleNamespace(policy=model, optimizer=optimizer, learning_rate=.1),
                                 current_learning_iteration=1499)
        warm_start_public(runner, state, 1e-4)
        self.assertEqual(runner.current_learning_iteration, 0)
        self.assertEqual(len(runner.alg.optimizer.state), 0)
        self.assertEqual(runner.alg.optimizer.param_groups[0]['lr'], 1e-4)
        for k, v in model.state_dict().items():
            self.assertTrue(torch.equal(v, state['model_state_dict'][k]))

    def test_mixture_preserves_all_native_families_and_allocates_target_heights(self):
        from go2_stairs.mixed_finetune import mixture_specs, column_labels
        specs = mixture_specs()
        self.assertAlmostEqual(sum(x['proportion'] for x in specs), 1.)
        labels = column_labels(specs, 50)
        counts = Counter(labels)
        by_name = {x['name']: x for x in specs}
        families = Counter()
        for name, count in counts.items():
            families[by_name[name]['family']] += count
        self.assertEqual(sorted(families.values()), [5, 5, 10, 10, 10, 10])
        for family in ['pyramid_stairs', 'pyramid_stairs_inv']:
            self.assertEqual(counts[family + '_h18'], 2)
            self.assertEqual(counts[family + '_h20'], 3)
            self.assertEqual(counts[family + '_h22'], 1)
            self.assertEqual(counts[family + '_broad'], 4)


if __name__ == '__main__':
    unittest.main()
