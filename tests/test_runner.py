import copy
import tempfile
from pathlib import Path
import unittest

import torch
from tensordict import TensorDict


class RunnerTest(unittest.TestCase):
    def test_real_ppo_model_optimizer_rng_and_counter_roundtrip(self):
        from go2_stairs.runner import CheckpointRunner, runner_config
        class ObservationOnlyEnv:
            num_envs = 2
            num_actions = 12
            device = 'cpu'
            def get_observations(self):
                return TensorDict({'policy': torch.zeros(2, 225),
                                   'critic': torch.zeros(2, 349)}, batch_size=[2])
        with tempfile.TemporaryDirectory() as folder:
            config = runner_config()
            untouched_config = copy.deepcopy(config)
            meta = {'stage': 'flat', 'seed': 0, 'ppo': config, 'asset_bundle_hash': 'b'*64,
                    'task_parameters': {'tracking_std': .5}}
            one = CheckpointRunner(ObservationOnlyEnv(), copy.deepcopy(config), folder,
                                   'cpu', 'a'*64, meta)
            direct = copy.deepcopy(config)
            CheckpointRunner(ObservationOnlyEnv(), direct, folder, 'cpu', 'a'*64, meta)
            self.assertEqual(direct, untouched_config)
            # Create real Adam optimizer state and a nontrivial model update.
            one.alg.optimizer.zero_grad()
            sum(p.square().sum() for p in one.alg.policy.parameters()).backward()
            one.alg.optimizer.step()
            one.alg.learning_rate = .00073
            one.current_learning_iteration = 1
            one.tot_timesteps = 96
            one.save(str(Path(folder)/'first.pt'))
            expected = torch.rand(5)
            two = CheckpointRunner(ObservationOnlyEnv(), copy.deepcopy(config), folder,
                                   'cpu', 'a'*64, meta)
            two.restore(Path(folder)/'first.pt')
            self.assertTrue(torch.equal(torch.rand(5), expected))
            self.assertEqual(two.current_learning_iteration, 2)
            self.assertEqual(two.tot_timesteps, 96)
            self.assertEqual(two.alg.learning_rate, .00073)
            for key, tensor in one.alg.policy.state_dict().items():
                self.assertTrue(torch.equal(two.alg.policy.state_dict()[key], tensor))
            self.assertEqual(len(two.alg.optimizer.state_dict()['state']),
                             len(one.alg.optimizer.state_dict()['state']))
            inference = CheckpointRunner(ObservationOnlyEnv(), config, folder, 'cpu', 'a'*64, meta)
            rng_before = torch.get_rng_state().clone()
            inference.restore(Path(folder)/'first.pt', training=False)
            self.assertTrue(torch.equal(rng_before, torch.get_rng_state()))
            self.assertEqual(len(inference.alg.optimizer.state_dict()['state']), 0)
            changed = copy.deepcopy(config)
            changed['algorithm']['gamma'] = .5
            mismatch = CheckpointRunner(ObservationOnlyEnv(), changed, folder, 'cpu', 'a'*64,
                                        dict(meta, ppo=changed))
            with self.subTest('gamma'), self.assertRaisesRegex(ValueError, 'PPO'):
                mismatch.restore(Path(folder)/'first.pt')
            wrong_asset = CheckpointRunner(ObservationOnlyEnv(), config, folder, 'cpu', 'a'*64,
                                          dict(meta, asset_bundle_hash='c'*64))
            with self.subTest('asset'), self.assertRaisesRegex(ValueError, 'asset'):
                wrong_asset.restore(Path(folder)/'first.pt')
            different_task = CheckpointRunner(ObservationOnlyEnv(), config, folder, 'cpu', 'a'*64,
                                             dict(meta, task_parameters={'tracking_std': .2}))
            with self.subTest('task'), self.assertRaisesRegex(ValueError, 'task'):
                different_task.restore(Path(folder)/'first.pt')
            fresh = CheckpointRunner(ObservationOnlyEnv(), config, folder, 'cpu', 'a'*64,
                                     dict(meta, asset_bundle_hash='c'*64))
            fresh.initialize_policy(Path(folder)/'first.pt')
            self.assertEqual(fresh.current_learning_iteration, 0)
            self.assertEqual(len(fresh.alg.optimizer.state_dict()['state']), 0)
            self.assertTrue(fresh.run_config['initialization']['weights_only'])
            for key, tensor in one.alg.policy.state_dict().items():
                self.assertTrue(torch.equal(fresh.alg.policy.state_dict()[key], tensor))
            with self.assertRaises(FileExistsError):
                two.save(str(Path(folder)/'first.pt'))

    def test_config_is_explicit_and_fresh(self):
        from go2_stairs.runner import runner_config
        one = runner_config()
        self.assertEqual(one['policy']['actor_hidden_dims'], [256, 256, 128])
        self.assertEqual(one['obs_groups'], {'policy': ['policy'], 'critic': ['critic']})
        self.assertFalse(one['policy']['actor_obs_normalization'])
        one['policy']['actor_hidden_dims'][0] = 1
        self.assertEqual(runner_config()['policy']['actor_hidden_dims'][0], 256)

    def test_bounded_profile_disables_gaussian_kl_and_initializes_no_gaussian_std(self):
        from go2_stairs.runner import runner_config
        cfg = runner_config()
        self.assertEqual(cfg['policy']['class_name'], 'BoundedBetaActorCritic')
        self.assertEqual(cfg['algorithm']['schedule'], 'fixed')
        self.assertIsNone(cfg['algorithm']['desired_kl'])
        self.assertNotIn('init_noise_std', cfg['policy'])


if __name__ == '__main__':
    unittest.main()
