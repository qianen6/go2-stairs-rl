import importlib
import math
import unittest

import torch


class ContractsTests(unittest.TestCase):
    def setUp(self):
        try:
            self.m = importlib.import_module('go2_stairs.contracts')
        except ModuleNotFoundError:
            self.fail('Observation/action contract implementation is missing')

    def parts(self, n=2):
        return [torch.full((n, width), float(i))
                for i, width in enumerate((3, 3, 3, 12, 12, 12))]

    def manifest(self, **changes):
        names = tuple(f'joint_{i}' for i in range(12))
        data = dict(joint_names=names, runtime_joint_names=names[::-1],
                    q_nominal=(0.0,) * 12, action_scale=(0.25,) * 12,
                    lower=(-1.0,) * 12, upper=(1.0,) * 12,
                    kp=(20.0,) * 12, kd=(0.5,) * 12,
                    torque_limits=(20.0,) * 12)
        data.update(changes)
        return self.m.ControlManifest(**data)

    def test_actor_layout_and_no_privileged_argument(self):
        result = self.m.actor_frame(*self.parts())
        self.assertEqual(result.shape, (2, 45))
        self.assertEqual(result[0].tolist(), [0.] * 3 + [1.] * 3 + [2.] * 3
                         + [3.] * 12 + [4.] * 12 + [5.] * 12)
        with self.assertRaises(TypeError):
            self.m.actor_frame(*self.parts(), heights=torch.zeros(2, 121))

    def test_actor_rejects_mismatched_batch_and_nonfinite(self):
        parts = self.parts()
        parts[-1] = torch.zeros(1, 12)
        with self.assertRaises(ValueError):
            self.m.actor_frame(*parts)
        parts = self.parts()
        parts[0][0, 0] = float('nan')
        with self.assertRaises(ValueError):
            self.m.actor_frame(*parts)

    def test_critic_preserves_actor_and_appends_truth(self):
        h = torch.zeros(2, 225)
        result = self.m.critic_frame(h, torch.ones(2, 3), torch.full((2, 121), 2.0))
        self.assertEqual(result.shape, (2, 349))
        self.assertEqual(result[0, 225:].tolist(), [1.] * 3 + [2.] * 121)
        self.assertEqual(h.sum().item(), 0.)

    def test_history_reset_order_and_partial_reset(self):
        h = self.m.ObservationHistory(2, 'cpu')
        frames = torch.stack([torch.ones(45), torch.full((45,), 2.)])
        h.reset(torch.tensor([0, 1]), frames)
        h.push(frames * 3)
        self.assertEqual(h.tensor()[0].tolist(), [1.] * 180 + [3.] * 45)
        h.reset(torch.tensor([1]), torch.full((1, 45), 7.))
        self.assertEqual(h.tensor()[1].tolist(), [7.] * 225)
        self.assertEqual(h.tensor()[0, -1].item(), 3.)

    def test_history_requires_reset_and_does_not_expose_storage(self):
        h = self.m.ObservationHistory(1, 'cpu')
        with self.assertRaises(RuntimeError):
            h.push(torch.zeros(1, 45))
        h.reset(torch.tensor([0]), torch.ones(1, 45))
        view = h.tensor()
        view.fill_(42.)
        self.assertEqual(h.tensor()[0, 0].item(), 1.)
        with self.assertRaises(ValueError):
            h.reset(torch.tensor([-1]), torch.zeros(1, 45))

    def test_joint_targets_clip_and_reorder_by_name(self):
        action = torch.linspace(-1, 1, 12).reshape(1, 12)
        target = self.m.joint_target(action, self.manifest())
        self.assertAlmostEqual(target[0, 0].item(), 0.25)
        self.assertAlmostEqual(target[0, -1].item(), -0.25)
        clipped = self.m.joint_target(torch.full((1, 12), 4.), self.manifest())
        self.assertTrue(torch.allclose(clipped, torch.full((1, 12), .25)))

    def test_joint_limits_are_applied_after_offset(self):
        manifest = self.manifest(q_nominal=(.9,) * 12)
        result = self.m.joint_target(torch.ones(1, 12), manifest)
        self.assertEqual(result.tolist(), [[1.] * 12])

    def test_bad_manifest_and_action_rejected(self):
        with self.assertRaises(ValueError):
            self.manifest(runtime_joint_names=('duplicate',) * 12)
        with self.assertRaises(ValueError):
            self.manifest(kp=(-1.,) * 12)
        with self.assertRaises(ValueError):
            self.manifest(action_scale=(float('nan'),) * 12)
        with self.assertRaises(ValueError):
            self.manifest(policy_dt=.021)
        with self.assertRaises(ValueError):
            self.m.joint_target(torch.full((1, 12), float('inf')), self.manifest())

    def test_manifest_roundtrip_and_hash_change(self):
        manifest = self.manifest()
        restored = self.m.ControlManifest.from_dict(manifest.to_dict())
        self.assertEqual(manifest.digest(), restored.digest())
        other = self.manifest(action_scale=(.2,) * 12)
        self.assertNotEqual(manifest.digest(), other.digest())

    def test_projected_gravity_convention(self):
        q = torch.tensor([[1., 0, 0, 0], [math.sqrt(.5), math.sqrt(.5), 0, 0]])
        actual = self.m.projected_gravity(q)
        torch.testing.assert_close(actual, torch.tensor([[0., 0, -1], [0., -1, 0]]),
                                   atol=1e-6, rtol=0)
        with self.assertRaises(ValueError):
            self.m.projected_gravity(torch.zeros(1, 4))

    def test_manifest_rejects_values_not_representable_by_float32(self):
        with self.assertRaises(ValueError):
            self.manifest(q_nominal=(1e39,) * 12, lower=(-1e40,) * 12, upper=(1e40,) * 12)

    def test_scaled_target_overflow_is_rejected(self):
        manifest = self.manifest(q_nominal=(1e38,) * 12, action_scale=(3e38,) * 12,
                                 lower=(-3e38,) * 12, upper=(3e38,) * 12)
        with self.assertRaises(ValueError):
            self.m.joint_target(torch.ones(1, 12), manifest)

    def test_large_finite_quaternion_normalizes_stably(self):
        actual = self.m.projected_gravity(torch.tensor([[1e30, 1e30, 0., 0.]]))
        torch.testing.assert_close(actual, torch.tensor([[0., -1., 0.]]), atol=1e-6, rtol=0)

    def test_observation_cache_does_not_append_twice_per_step(self):
        self.assertTrue(hasattr(self.m, 'ObservationCache'), 'Step-keyed observation cache is missing')
        cache = self.m.ObservationCache(2, 'cpu')
        first = cache.update(0, torch.ones(2, 45))
        second = cache.update(0, torch.full((2, 45), 99.))
        torch.testing.assert_close(first, second)
        newer = cache.update(1, torch.full((2, 45), 2.))
        self.assertEqual(newer[0].tolist(), [1.] * 180 + [2.] * 45)

    def test_observation_cache_partial_reset_does_not_advance_other_envs(self):
        self.assertTrue(hasattr(self.m, 'ObservationCache'), 'Step-keyed observation cache is missing')
        cache = self.m.ObservationCache(2, 'cpu')
        cache.update(0, torch.ones(2, 45))
        before = cache.update(1, torch.full((2, 45), 2.))
        cache.mark_reset(torch.tensor([1]))
        after = cache.update(1, torch.full((2, 45), 7.))
        torch.testing.assert_close(after[0], before[0])
        self.assertEqual(after[1].tolist(), [7.] * 225)
        with self.assertRaises(ValueError):
            cache.update(0, torch.zeros(2, 45))


if __name__ == '__main__':
    unittest.main()
