import importlib
import random
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        try:
            self.m = importlib.import_module('go2_stairs.checkpoints')
        except ModuleNotFoundError:
            self.fail('Checkpoint persistence implementation is missing')

    def state(self):
        return dict(model_state_dict={'actor': torch.tensor([1., 2.]), 'critic': torch.tensor([3.])},
                    optimizer_state_dict={'state': {}, 'param_groups': [{'lr': .001}]},
                    normalization={'mean': torch.zeros(225)}, curriculum={'level': 2},
                    rng=self.m.capture_rng_state(), iter=7, contract_hash='a' * 64,
                    config={'stage': 'flat', 'policy_hz': 50})

    def test_roundtrip_preserves_model_optimizer_normalization_curriculum(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / '中文模型.pt'
            digest = self.m.save_checkpoint(path, self.state())
            loaded = self.m.load_checkpoint(path, 'a' * 64)
            self.assertEqual(len(digest), 64)
            torch.testing.assert_close(loaded['model_state_dict']['critic'], torch.tensor([3.]))
            self.assertEqual(loaded['optimizer_state_dict']['param_groups'][0]['lr'], .001)
            self.assertEqual(loaded['normalization']['mean'].shape, (225,))
            self.assertEqual(loaded['curriculum']['level'], 2)
            self.assertEqual(loaded['iter'], 7)

    def test_rng_resume_reproduces_next_draws(self):
        saved = self.m.capture_rng_state()
        expected = (random.random(), np.random.rand(), torch.rand(3))
        self.m.restore_rng_state(saved)
        self.assertEqual(random.random(), expected[0])
        self.assertEqual(np.random.rand(), expected[1])
        torch.testing.assert_close(torch.rand(3), expected[2])

    def test_existing_checkpoint_and_run_directory_are_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp) / 'run'
            self.m.create_run_dir(run)
            path = run / 'model.pt'
            self.m.save_checkpoint(path, self.state())
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                self.m.save_checkpoint(path, self.state())
            with self.assertRaises(FileExistsError):
                self.m.create_run_dir(run)
            self.assertEqual(path.read_bytes(), before)

    def test_wrong_contract_or_missing_resume_state_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'model.pt'
            self.m.save_checkpoint(path, self.state())
            with self.assertRaises(ValueError):
                self.m.load_checkpoint(path, 'b' * 64)
            state = self.state()
            del state['rng']
            with self.assertRaises(ValueError):
                self.m.save_checkpoint(Path(temp) / 'partial.pt', state)

    def test_failed_save_never_publishes_partial_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'model.pt'
            with mock.patch.object(self.m.torch, 'save', side_effect=OSError('simulated disk failure')):
                with self.assertRaises(OSError):
                    self.m.save_checkpoint(path, self.state())
            self.assertFalse(path.exists())
            self.assertEqual(list(Path(temp).iterdir()), [])


if __name__ == '__main__':
    unittest.main()
