"""Orchestration seam only; physical PPO is verified by the native smoke runs."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from flat_fixture import flat_report


class GuardedTrainTests(unittest.TestCase):
    def run_case(self, *, passes=False, child_failure=False, corrupt_report=False):
        from go2_stairs.guarded_train import main
        calls = []
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)/'job'
            def child(command, **kwargs):
                calls.append(command)
                destination = Path(command[command.index('--output')+1])
                destination.mkdir(parents=True)
                if command[2]=='go2_stairs.train':
                    if child_failure:
                        return SimpleNamespace(returncode=1)
                    completed = sum(int(c[c.index('--iterations')+1]) for c in calls if c[2]=='go2_stairs.train')
                    checkpoint=destination/'model.pt';checkpoint.write_bytes(b'fixture checkpoint')
                    result={'status':'PPO_JOB_COMPLETED','checkpoint':str(checkpoint),
                            'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                            'completed_updates':completed}
                    (destination/'status.json').write_text(json.dumps(result))
                else:
                    checkpoint=Path(command[command.index('--checkpoint')+1])
                    result=flat_report(passes)
                    result['checkpoint_sha256']=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                    if corrupt_report:
                        result['cases'][0]['failure']='terminated'
                        result['cases'][1]['case_id']='flat-dev-00'
                        result['action_metrics']['near_bound_fraction']=float('nan')
                    (destination/'summary.json').write_text(json.dumps(result))
                return SimpleNamespace(returncode=0)
            with patch.object(sys,'argv',['guard','--output',str(output),'--iterations','1000']), \
                 patch('go2_stairs.guarded_train.subprocess.run',side_effect=child):
                code=main()
            return code,json.loads((output/'status.json').read_text()),calls

    def test_zero_motion_stops_at_300_not_1000_and_resumes_each_verified_checkpoint(self):
        code,status,calls=self.run_case()
        self.assertEqual((code,status['status'],len(calls)),(2,'STOP_NO_PROGRESS',6))
        self.assertNotIn('--resume',calls[0])
        self.assertIn('--resume',calls[2])
        self.assertEqual(status['completed_updates'],300)

    def test_passed_gate_stops_instead_of_launching_unneeded_blocks(self):
        code,status,calls=self.run_case(passes=True)
        self.assertEqual((code,status['status'],len(calls)),(0,'FLAT_GATE_PASSED',2))

    def test_failed_training_never_launches_replay_or_more_training(self):
        code,status,calls=self.run_case(child_failure=True)
        self.assertEqual((code,status['status'],len(calls)),(2,'STOP_TRAINING_FAILURE',1))

    def test_corrupt_success_report_cannot_pass_the_actual_guarded_driver(self):
        code,status,calls=self.run_case(passes=True,corrupt_report=True)
        self.assertEqual((code,status['status'],len(calls)),(2,'STOP_INVALID_EVALUATION',2))
        self.assertFalse(status['flat_gate_passed'])
