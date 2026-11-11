import unittest
import torch
from flat_fixture import flat_report


class TrainingHealthTests(unittest.TestCase):
    def test_movement_reward_rejects_standing_shortcut_and_rewards_stop(self):
        from go2_stairs.training_health import relative_tracking_reward
        cmd = torch.tensor([[.1, 0.], [.2, 0.], [0., 0.]])
        stopped = relative_tracking_reward(cmd, torch.zeros_like(cmd), .2)
        self.assertTrue(torch.allclose(stopped, torch.tensor([0.,0.,1.]), atol=1e-6))
        self.assertTrue(torch.allclose(relative_tracking_reward(cmd, cmd, .2), torch.ones(3)))
        self.assertTrue(bool((relative_tracking_reward(cmd[:2], -cmd[:2], .2)<0).all()))

    def test_monitor_latches_saturation_and_serializes_its_state(self):
        from go2_stairs.training_health import TrainingHealth
        monitor = TrainingHealth(patience=3)
        healthy = dict(near_bound_fraction=.01, hard_clip_fraction=0., commanded_speed=.2,
                       actual_speed=.1, latent_std_mean=.4)
        self.assertIsNone(monitor.update(healthy))
        bad = dict(healthy, near_bound_fraction=.99)
        self.assertIsNone(monitor.update(bad))
        restored = TrainingHealth.from_dict(monitor.to_dict())
        self.assertIsNone(restored.update(bad))
        self.assertEqual(restored.update(bad), 'ACTION_SATURATION')
        self.assertEqual(restored.update(healthy), 'ACTION_SATURATION')

    def test_nonfinite_and_out_of_range_actions_stop_immediately(self):
        from go2_stairs.training_health import TrainingHealth
        metrics = dict(near_bound_fraction=0., hard_clip_fraction=0., commanded_speed=.2,
                       actual_speed=.1, latent_std_mean=.4)
        self.assertEqual(TrainingHealth().update(dict(metrics, actual_speed=float('nan'))), 'NONFINITE_METRIC')
        self.assertEqual(TrainingHealth().update(dict(metrics, hard_clip_fraction=.001)), 'OUT_OF_RANGE_ACTION')

    def test_no_progress_gate_does_not_confuse_no_fall_with_walking(self):
        from go2_stairs.training_health import development_decision
        stand=flat_report()
        self.assertEqual(development_decision(stand,300), 'STOP_NO_PROGRESS')
        self.assertEqual(development_decision(stand,100), 'CONTINUE_WARMUP')
        walking=flat_report(True)
        self.assertEqual(development_decision(walking,300), 'FLAT_GATE_PASSED')

    def test_even_claimed_pass_must_have_finite_physical_measurements(self):
        from go2_stairs.training_health import development_decision
        invalid={'passed_count':20,'cases':[{'passed':True,'forward_distance_m':float('nan'),
                                           'complete':True,'failure':None} for _ in range(20)]}
        self.assertEqual(development_decision(invalid,300),'STOP_INVALID_EVALUATION')

    def test_declared_success_is_recomputed_and_missing_or_nan_data_fail_closed(self):
        from go2_stairs.training_health import development_decision
        import copy
        original=flat_report(True)
        edits=[lambda r:r.pop('action_metrics'),
               lambda r:r['action_metrics'].update(hard_clip_fraction=float('nan')),
               lambda r:r['cases'][1].update(case_id='flat-dev-00'),
               lambda r:r['cases'][0].update(failure='terminated'),
               lambda r:r['cases'][0].update(stable_stop_s=0.),
               lambda r:r['cases'][0].update(move_rmse_mps=float('nan')),
               lambda r:r['cases'][0].update(turn_rmse_radps=99.),
               lambda r:r['cases'][0].update(forward_distance_m=0.),
               lambda r:r['cases'][0].update(complete=1),
               lambda r:r.update(passed_count=19)]
        for index,edit in enumerate(edits):
            report=copy.deepcopy(original);edit(report)
            with self.subTest(index=index):
                self.assertEqual(development_decision(report,300),'STOP_INVALID_EVALUATION')
