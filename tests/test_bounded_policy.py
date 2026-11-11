import contextlib
import io
import math
import unittest

import torch
from tensordict import TensorDict


class BoundedPolicyTests(unittest.TestCase):
    def build(self, n=256):
        from go2_stairs.policy import BoundedBetaActorCritic
        torch.manual_seed(10)
        obs = TensorDict({'policy': torch.randn(n, 225), 'critic': torch.randn(n, 349)}, batch_size=[n])
        with contextlib.redirect_stdout(io.StringIO()):
            policy = BoundedBetaActorCritic(obs, {'policy': ['policy'], 'critic': ['critic']}, 12,
                                           actor_hidden_dims=[32, 16], critic_hidden_dims=[32, 16],
                                           activation='elu', actor_obs_normalization=False,
                                           critic_obs_normalization=False)
        return policy, obs

    def test_sampling_inference_and_density_use_same_bounded_distribution(self):
        policy, obs = self.build()
        action = policy.act(obs)
        self.assertTrue(bool((action.abs() < 1).all()))
        expected = policy.distribution.log_prob((action+1)/2).sum(-1) - 12*math.log(2)
        self.assertTrue(torch.allclose(policy.get_actions_log_prob(action), expected, atol=1e-5))
        self.assertTrue(torch.allclose(policy.act_inference(obs), 2*policy.distribution.mean-1))
        self.assertTrue(torch.allclose(policy.entropy, policy.distribution.entropy().sum(-1)+12*math.log(2)))
        self.assertLessEqual(float(policy.entropy.max()), 12*math.log(2)+1e-5)

    def test_unchanged_policy_ppo_ratio_is_one_and_gradients_are_finite(self):
        policy, obs = self.build()
        action = policy.act(obs)
        old = policy.get_actions_log_prob(action).detach()
        policy.act(obs)
        new = policy.get_actions_log_prob(action)
        self.assertTrue(torch.allclose((new-old).exp(), torch.ones_like(old), atol=1e-5))
        loss = -new.mean() - .005*policy.entropy.mean() + policy.evaluate(obs).square().mean()
        loss.backward()
        self.assertTrue(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in policy.parameters()))

    def test_invalid_actions_and_nonfinite_network_output_fail(self):
        policy, obs = self.build()
        policy.act(obs)
        for value in (1., -1., float('nan'), float('inf')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                policy.get_actions_log_prob(torch.full((256, 12), value))
        with torch.no_grad():
            policy.actor[-1].bias.fill_(float('nan'))
        with self.assertRaises(ValueError):
            policy.act(obs)

    def test_actual_rsl_ppo_update_changes_parameters_without_clipping(self):
        from rsl_rl.algorithms import PPO
        policy, obs = self.build(64)
        algorithm = PPO(policy, num_learning_epochs=2, num_mini_batches=2, learning_rate=3e-4,
                        schedule='fixed', desired_kl=None, entropy_coef=.005, device='cpu')
        algorithm.init_storage('rl', 64, 8, obs, [12])
        before = {k:v.clone() for k,v in policy.state_dict().items()}
        with torch.inference_mode():
            for _ in range(8):
                action = algorithm.act(obs)
                self.assertTrue(bool((action.abs()<1).all()))
                reward = 1-(action-.25).square().mean(-1)
                algorithm.process_env_step(obs, reward, torch.zeros(64, dtype=torch.long), {})
            algorithm.compute_returns(obs)
        losses = algorithm.update()
        self.assertTrue(all(math.isfinite(v) for v in losses.values()))
        self.assertTrue(any(not torch.equal(before[k],v) for k,v in policy.state_dict().items()))
