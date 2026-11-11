import unittest
import torch
from types import SimpleNamespace
from tempfile import TemporaryDirectory
from pathlib import Path
import json


class GuardedFineTuneTests(unittest.TestCase):
    def test_continuation_requires_target_gain_and_retained_basics(self):
        from go2_stairs.finetune_diagnostics import continuation_gate
        base = {'flat':{'completed':8,'clean':8},'up10':{'completed':4,'clean':4},'down10':{'completed':4,'clean':4},
                'up20':{'completed':3,'clean':2},'down20':{'completed':7,'clean':4}}
        candidate = {k:dict(v) for k,v in base.items()}
        self.assertEqual(continuation_gate(base,candidate),'NO_TARGET_GAIN')
        candidate['up20']={'completed':4,'clean':3}
        self.assertEqual(continuation_gate(base,candidate),'CANDIDATE_FOR_INDEPENDENT_CONFIRMATION')
        candidate['up20']['n']=99
        self.assertEqual(continuation_gate(base,candidate),'INVALID_EVALUATION')
        candidate['up20'].pop('n')
        candidate['flat']['completed']=7
        self.assertEqual(continuation_gate(base,candidate),'BASIC_REGRESSION')

    def test_restore_adam_preserves_moments_but_uses_declared_finetune_lr(self):
        from go2_stairs.finetune_diagnostics import restore_public_adam
        model = torch.nn.Linear(2,1)
        old = torch.optim.Adam(model.parameters(), lr=.01)
        model(torch.ones(1,2)).sum().backward();old.step()
        state = {'model_state_dict':model.state_dict(),'optimizer_state_dict':old.state_dict()}
        new_model = torch.nn.Linear(2,1)
        runner = SimpleNamespace(alg=SimpleNamespace(policy=new_model,optimizer=torch.optim.Adam(new_model.parameters())), current_learning_iteration=1499)
        restore_public_adam(runner,state,1e-4)
        self.assertEqual(runner.current_learning_iteration,0)
        self.assertEqual(runner.alg.optimizer.param_groups[0]['lr'],1e-4)
        for current, previous in zip(runner.alg.optimizer.state.values(), old.state.values()):
            self.assertTrue(torch.equal(current['exp_avg'],previous['exp_avg']))
            self.assertTrue(torch.equal(current['exp_avg_sq'],previous['exp_avg_sq']))

    def test_completed_record_ignores_stale_status_and_rejects_overwrite(self):
        from go2_stairs.finetune_diagnostics import write_final_record, read_final_record
        with TemporaryDirectory() as folder:
            root = Path(folder); checkpoint = root/'model.pt'
            torch.save({'iter': 25}, checkpoint)
            write_final_record(root, {'checkpoint': str(checkpoint), 'completed_finetune_updates': 26, 'train_exit_code': 0})
            (root/'status.json').write_text(json.dumps({'status':'TRAINING','completed_finetune_updates':4}))
            self.assertEqual(read_final_record(root)['completed_finetune_updates'], 26)
            with self.assertRaises(FileExistsError):
                write_final_record(root, {'checkpoint': str(checkpoint), 'completed_finetune_updates': 4, 'train_exit_code': 0})
            checkpoint.write_bytes(b'changed')
            with self.assertRaises(ValueError):
                read_final_record(root)

    def test_gaussian_kl_measures_action_shift_and_is_zero_for_equal_policies(self):
        from go2_stairs.finetune_diagnostics import gaussian_kl
        mu = torch.zeros(4, 12); sigma = torch.ones(12)
        self.assertTrue(torch.equal(gaussian_kl(mu, sigma, mu, sigma), torch.zeros(4)))
        self.assertTrue(torch.allclose(gaussian_kl(mu, sigma, mu+1, sigma), torch.full((4,), 6.)))


if __name__ == '__main__':
    unittest.main()
