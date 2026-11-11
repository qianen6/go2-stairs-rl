import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
import torch


class TrainingRepairTests(unittest.TestCase):
    def test_initial_reset_does_not_promote_unrun_terrain_curriculum(self):
        from go2_stairs.repair_train import eligible_curriculum_ids
        ids=torch.arange(4)
        self.assertEqual(eligible_curriculum_ids(torch.zeros(4,dtype=torch.long),ids).numel(),0)
        torch.testing.assert_close(eligible_curriculum_ids(torch.tensor([0,10,0,5]),ids),torch.tensor([1,3]))

    def test_commands_cover_low_speed_stop_and_general_motion(self):
        from go2_stairs.repair_train import sample_repair_commands
        torch.manual_seed(52)
        commands,regime=sample_repair_commands(10000,'cpu')
        self.assertEqual(commands.shape,(10000,3))
        low=regime==0;general=regime==1;stop=regime==2
        self.assertTrue(5700<int(low.sum())<6300)
        self.assertTrue(2200<int(general.sum())<2800)
        self.assertTrue(1200<int(stop.sum())<1800)
        self.assertTrue(bool(((commands[low,0]>=.08)&(commands[low,0]<=.25)).all()))
        self.assertTrue(bool((commands[low,1:]==0).all()))
        self.assertTrue(bool((commands[stop]==0).all()))
        self.assertTrue(bool((abs(commands[general])<=1).all()))

    def test_moving_command_does_not_reward_standing_as_tracking(self):
        from go2_stairs.repair_train import repair_tracking
        cmd=torch.tensor([[.1,0],[.2,0],[0,0]])
        reward=repair_tracking(cmd,torch.zeros_like(cmd))
        torch.testing.assert_close(reward,torch.tensor([0.,0.,1.]))
        torch.testing.assert_close(repair_tracking(cmd,cmd),torch.ones(3))
        self.assertTrue(bool((repair_tracking(cmd,-cmd)>=-1).all()))

    def test_resume_restores_optimizer_and_advances_next_update(self):
        from go2_stairs.repair_train import restore_runner
        p=torch.nn.Linear(3,2);o=torch.optim.Adam(p.parameters(),lr=.0002)
        p(torch.ones(2,3)).sum().backward();o.step()
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'model_1499.pt'
            torch.save(dict(model_state_dict=p.state_dict(),optimizer_state_dict=o.state_dict(),iter=1499,infos=None),path)
            q=torch.nn.Linear(3,2);optim=torch.optim.Adam(q.parameters(),lr=.001)
            runner=SimpleNamespace(alg=SimpleNamespace(policy=q,optimizer=optim,learning_rate=.001),current_learning_iteration=0)
            result=restore_runner(runner,path)
            self.assertEqual(runner.current_learning_iteration,1500)
            self.assertEqual(result['base_completed_updates'],1500)
            self.assertAlmostEqual(runner.alg.learning_rate,.0002)
            self.assertTrue(optim.state)
            for a,b in zip(p.parameters(),q.parameters()):torch.testing.assert_close(a,b)

    def test_low_speed_telemetry_distinguishes_stall_from_tracking(self):
        from go2_stairs.repair_train import command_metrics
        c=torch.tensor([[.1,0,0],[.2,0,0],[.8,0,0],[0,0,0]])
        v=torch.tensor([[0.,0],[.2,0],[.8,0],[0,0]])
        m=command_metrics(c,v)
        self.assertEqual(float(m['low_count']),2)
        self.assertAlmostEqual(float(m['low_stall_fraction']),.5)
        self.assertAlmostEqual(float(m['low_tracking_error']),.05,places=6)


if __name__=='__main__':unittest.main()
