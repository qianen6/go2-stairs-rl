import unittest
from pathlib import Path
import tempfile
import torch
import json
import hashlib


class ReferenceTests(unittest.TestCase):
    def test_forward_cue_ablation_changes_only_one_training_config_field(self):
        from go2_stairs.reference import build_command
        base=build_command(Path('reference fixture'),2048,300,0)
        cue=build_command(Path('reference fixture'),2048,300,0,forward_cue=True)
        self.assertEqual(cue[:-1],base)
        self.assertEqual(cue[-1],'env.commands.base_velocity.ranges.lin_vel_x=[0.6,0.8]')

    def test_official_command_does_not_override_dynamics_rewards_or_policy(self):
        from go2_stairs.reference import build_command
        cmd=build_command(Path('reference fixture'),128,2,0)
        self.assertIn('Unitree-Go2-Velocity',cmd)
        overrides=[x for x in cmd if x.startswith('env.')]
        self.assertEqual({x.split('=')[0] for x in overrides},
                         {'env.scene.robot.spawn.usd_path','env.scene.terrain.visual_material','env.scene.sky_light.spawn.texture_file'})
        self.assertFalse(any('clip_actions' in x or 'rewards' in x or 'observations' in x or 'sim.dt' in x for x in cmd))
        with self.assertRaises(ValueError):
            build_command(Path('reference fixture'),128,1000,0)

    def checkpoint(self):
        state={'std':torch.ones(12)}
        for net,dims in [('actor',[45,512,256,128,12]),('critic',[60,512,256,128,1])]:
            for index,(a,b) in enumerate(zip(dims,dims[1:])):
                state[f'{net}.{index*2}.weight']=torch.zeros(b,a)
                state[f'{net}.{index*2}.bias']=torch.zeros(b)
        return {'model_state_dict':state,'optimizer_state_dict':{'state':{},'param_groups':[]},'iter':1,'infos':None}

    def test_native_checkpoint_dimensions_counter_and_finiteness(self):
        from go2_stairs.reference import validate_native_checkpoint
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'model_1.pt';data=self.checkpoint();torch.save(data,path)
            result=validate_native_checkpoint(path,2)
            self.assertEqual((result['actor_obs'],result['critic_obs'],result['completed_updates']),(45,60,2))
            with self.assertRaises(ValueError):
                validate_native_checkpoint(path,3)
            data['model_state_dict']['std'][0]=float('nan');torch.save(data,path)
            with self.assertRaises(ValueError):
                validate_native_checkpoint(path,2)

    def test_reference_report_requires_twenty_unique_completed_physical_cases(self):
        from go2_stairs.reference import reference_pass_count
        from flat_fixture import flat_report
        data=flat_report(True)
        self.assertEqual(reference_pass_count(data),20)
        data['cases'][1]['case_id']=data['cases'][0]['case_id']
        with self.assertRaises(ValueError):
            reference_pass_count(data)

    def test_replay_origin_binds_asset_source_checkpoint_and_effective_config(self):
        from go2_stairs.reference import validate_reference_origin
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);native=root/'native'/'run';(native/'params').mkdir(parents=True)
            checkpoint=native/'model_1.pt';checkpoint.write_bytes(b'fixture')
            files={}
            for name in ('env.yaml','agent.yaml'):
                p=native/'params'/name;p.write_text('fixture: 1')
                files[name]=hashlib.sha256(p.read_bytes()).hexdigest()
            meta={'sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
            record={'profile':'UPSTREAM_REFERENCE_ONLY','checkpoint_metadata':meta,'asset_bundle':{'sha256':'a'*64},
                    'source_identity':{'commit':'b'*40},'effective_config_sha256':files}
            (root/'status.json').write_text(json.dumps(record))
            validate_reference_origin(checkpoint,meta,{'commit':'b'*40},{'sha256':'a'*64})
            with self.assertRaises(ValueError):
                validate_reference_origin(checkpoint,meta,{'commit':'b'*40},{'sha256':'c'*64})
            (native/'params/env.yaml').write_text('fixture: 2')
            with self.assertRaises(ValueError):
                validate_reference_origin(checkpoint,meta,{'commit':'b'*40},{'sha256':'a'*64})
