import unittest
import tempfile
from pathlib import Path
import numpy as np
from go2_stairs.evaluate import build_core_suite
from go2_stairs.terrain import StairSpec,height_at


class StairEvalTests(unittest.TestCase):
    def test_active_and_terminal_numeric_faults_keep_all120_cases(self):
        from go2_stairs.stair_eval import ensure_finite,persist_invalid_run
        import torch,json
        cases=build_core_suite()
        for fault_index in (0,1):
            partial=[None]*120
            partial[0]=dict(done=True,success=True,reason='success',elapsed_s=1.)
            values=torch.zeros(120,3);values[fault_index,0]=float('nan')
            with tempfile.TemporaryDirectory() as folder:
                try:
                    ensure_finite(values,'injected observation')
                except ValueError as exc:
                    persist_invalid_run(Path(folder),cases,partial,2.,exc,[np.zeros((120,23))],
                                        [np.zeros((120,37))],.005,.02,{'training_seed':7})
                else:self.fail('nonfinite fault was swallowed')
                report=json.loads((Path(folder)/'summary.json').read_text())
                self.assertFalse(report['evaluation_valid'])
                self.assertFalse(report['summary']['passed'])
                self.assertEqual(report['summary']['episodes'],120)
                self.assertEqual(report['summary']['directions']['up']['episodes'],60)
                self.assertEqual(report['summary']['directions']['down']['episodes'],60)
                self.assertEqual(report['summary']['failure_reasons']['runtime_error'],119)
                self.assertTrue(report['results'][0]['success'])
                self.assertTrue((Path(folder)/'physics_trace.partial.npz').is_file())
                self.assertTrue((Path(folder)/'state_trace.partial.npz').is_file())

    def test_verified_training_seed_is_taken_from_record(self):
        from go2_stairs.stair_eval import verify_training
        from unittest import mock
        import json,hashlib
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);p=root/'native'/'run'/'model_1499.pt';p.parent.mkdir(parents=True)
            p.write_bytes(b'fixture');params=p.parent/'params';params.mkdir()
            for n in ('env.yaml','agent.yaml'):(params/n).write_text('{}')
            record=dict(status='TRAINING_COMPLETED_EVALUATION_PENDING',checkpoint=str(p.resolve()),
                budget_updates=1500,seed=7,num_envs=4096,source_identity={'x':1},
                checkpoint_metadata={'sha256':'a'},effective_config_sha256={n:hashlib.sha256((params/n).read_bytes()).hexdigest() for n in ('env.yaml','agent.yaml')})
            (root/'status.json').write_text(json.dumps(record))
            with mock.patch('go2_stairs.official_rough.validate_checkpoint',return_value={'sha256':'a'}), \
                 mock.patch('go2_stairs.official_rough.source_identity',return_value={'x':1}), \
                 mock.patch('go2_stairs.official_rough.validate_effective_config'):
                self.assertEqual(verify_training(p)['training_seed'],7)

    def test_frozen_starts_are_repeatable_bounded_and_on_correct_platform(self):
        from go2_stairs.stair_eval import case_setup
        cases=build_core_suite()
        for i,c in enumerate(cases):
            a=case_setup(c,i);b=case_setup(c,i)
            self.assertEqual(a,b)
            self.assertLessEqual(abs(a['lateral']),.05)
            self.assertLessEqual(abs(a['yaw']),np.pi/36)
            self.assertEqual(a['start_x'],.5)
            self.assertAlmostEqual(a['start_z'],.4+(c['height']*c['steps'] if c['direction']=='down' else 0))

    def test_mesh_matches_analytic_stairs_and_nominal_floor(self):
        from go2_stairs.stair_eval import frozen_mesh,mesh_height
        cases=build_core_suite()[::5]
        mesh,offsets,origins=frozen_mesh(cases)
        self.assertEqual(origins.shape,(1,24,3))
        self.assertEqual(len(set(map(tuple,offsets))),24)
        for i,c in enumerate(cases):
            spec=StairSpec(c['height'],c['depth'],c['steps'],c['direction'])
            for x in [.3,spec.landing+.1,spec.finish_x+.3]:
                h=mesh_height(mesh,x+offsets[i,0],offsets[i,1])
                self.assertAlmostEqual(h,height_at(x,0,spec),places=6)

    def test_stop_requires_entire_body_and_all_feet_beyond_finish(self):
        from go2_stairs.stair_eval import on_finish
        s=StairSpec(.06,.3,3,'up');x=s.finish_x+.16
        self.assertTrue(on_finish((x,0,.4),[(x,0,.2)]*4,s))
        self.assertFalse(on_finish((x,0,.4),[(x-.02,0,.2)]*4,s))
        self.assertFalse(on_finish((x,.7,.4),[(x,0,.2)]*4,s))

    def test_physics_trace_recomputes_success_contact_and_timeout(self):
        from go2_stairs.stair_eval import judge_trace
        c=build_core_suite()[0];s=StairSpec(c['height'],c['depth'],c['steps'],c['direction'])
        row=np.array([s.finish_x+.2,0,.4]+[s.finish_x+.2,0,.12]*4+[0,0,0]+[0,0,0]+[0,0],dtype=float)
        self.assertEqual(row.shape,(23,))
        self.assertTrue(judge_trace(c,np.tile(row,(200,1)),.005)['success'])
        trace=np.tile(row,(200,1));trace[:8,21]=21
        self.assertEqual(judge_trace(c,trace,.005)['reason'],'body_contact')
        trace=np.tile(row,(200,1));trace[:,18]=.1
        self.assertEqual(judge_trace(c,trace,.005)['reason'],'incomplete_trace')

    def test_replay_selection_is_deterministic_and_keeps_failure_examples(self):
        from go2_stairs.stair_eval import select_replays
        rows=[dict(case_id='a',direction='up',success=False),dict(case_id='b',direction='up',success=True),
              dict(case_id='c',direction='down',success=False),dict(case_id='d',direction='down',success=False)]
        self.assertEqual(select_replays(rows),['b','a','c'])

    def test_replay_uses_saved_state_coordinates_and_binds_trace_hash(self):
        from go2_stairs.stair_replay import relocate_state,verify_trace_files
        row=np.arange(37,dtype=float)
        shifted=relocate_state(row,[16,8,0],[0,0,0])
        np.testing.assert_array_equal(shifted[:3],row[:3]-[16,8,0])
        np.testing.assert_array_equal(shifted[3:],row[3:])
        import hashlib
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);p=root/'state_trace.npz';p.write_bytes(b'original')
            q=root/'physics_trace.npz';q.write_bytes(b'original-physics')
            record={'trace_sha256':{n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in ('state_trace.npz','physics_trace.npz')}}
            verify_trace_files(root,record)
            p.write_bytes(b'changed')
            with self.assertRaises(ValueError):verify_trace_files(root,record)

    def test_all_replay_robots_initialized_to_own_recorded_lane(self):
        from go2_stairs.stair_replay import relocate_batch
        states=np.zeros((120,37));states[:,3]=1
        old=np.array([[float(i%12*8),float(i//12*8),0.] for i in range(120)])
        states[:,:3]=old+[.5,0,.4]
        indices=[0,5,28];new=np.array([[0.,0,0],[8.,0,0],[16.,0,0]])
        actual=relocate_batch(states,indices,old,new)
        np.testing.assert_array_equal(actual[:,:3],new+[.5,0,.4])
        self.assertEqual(len(set(actual[:,0])),3)


if __name__=='__main__':unittest.main()
