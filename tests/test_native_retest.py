import unittest
import torch
from types import SimpleNamespace


class NativeRetestTests(unittest.TestCase):
    def test_release_stabilization_is_an_explicit_opt_in(self):
        from go2_stairs.native_factory import restore_release_stabilization
        sim = SimpleNamespace(physx=SimpleNamespace(enable_stabilization=False))
        restore_release_stabilization(sim)
        self.assertTrue(sim.physx.enable_stabilization)

    def test_release_velocity_limit_is_written_to_solver_not_only_motor(self):
        from go2_stairs.native_factory import restore_release_velocity_limits
        actuator = SimpleNamespace(velocity_limit=30., velocity_limit_sim=None)
        restore_release_velocity_limits({'legs': actuator})
        self.assertEqual(actuator.velocity_limit_sim, 30.)
        self.assertEqual(actuator.velocity_limit, 30.)

    def test_frozen_case_matrix(self):
        from go2_stairs.native_retest import native_cases
        rows = native_cases(.2, 'up', 8)
        self.assertEqual(len(rows), 8)
        self.assertEqual([x['speed'] for x in rows], [.8] * 4 + [1.] * 4)
        self.assertTrue(all(x['height'] == .2 and x['direction'] == 'up' for x in rows))
        self.assertEqual(rows, native_cases(.2, 'up', 8))

    def test_reject_mixed_terrain_batch(self):
        from go2_stairs.native_retest import native_cases, validate_native_batch
        rows = native_cases(.2, 'up', 4)
        rows[1]['direction'] = 'down'
        with self.assertRaises(ValueError):
            validate_native_batch(rows)

    def test_actor_uses_exact_published_feedforward_weights(self):
        from go2_stairs.native_retest import FrozenActor
        torch.manual_seed(9)
        network = torch.nn.Sequential(torch.nn.Linear(235, 512), torch.nn.ELU(),
            torch.nn.Linear(512, 256), torch.nn.ELU(), torch.nn.Linear(256, 128),
            torch.nn.ELU(), torch.nn.Linear(128, 12))
        state = {'actor.' + k: v for k, v in network.state_dict().items()}
        obs = torch.randn(7, 235)
        actor = FrozenActor(state, 'cpu')
        self.assertTrue(torch.equal(actor.act_inference({'policy': obs}), network(obs)))
        self.assertFalse(any(x.requires_grad for x in actor.parameters()))

    def test_native_geometry_outside_original_tile(self):
        from go2_stairs.native_retest import native_cases, validate_native_batch
        self.assertEqual(validate_native_batch(native_cases(.2, 'down', 8)), (.2, 'down'))
        with self.assertRaises(ValueError):
            native_cases(.0, 'up', 8)


if __name__ == '__main__':
    unittest.main()
