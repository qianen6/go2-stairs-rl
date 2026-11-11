import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


def fixture():
    return {
        'scene': {'num_envs': 2048, 'terrain': {'terrain_type': 'generator',
            'terrain_generator': {'curriculum': True, 'sub_terrains': {
                name: {'proportion': .2, 'step_height_range': [.05, .23], 'step_width': .3}
                for name in ('pyramid_stairs', 'pyramid_stairs_inv')}}}},
        'observations': {'policy': {'base_lin_vel': {'func': 'base_lin_vel'},
                                    'height_scan': {'func': 'height_scan'}}},
        'curriculum': {'terrain_levels': {'func': 'terrain_levels_vel'}},
    }, {'max_iterations': 1500, 'policy': {'actor_hidden_dims': [512, 256, 128],
        'critic_hidden_dims': [512, 256, 128]}, 'algorithm': {'class_name': 'PPO'}}


class OfficialRoughTests(unittest.TestCase):
    def test_validation_remains_active_under_optimized_python(self):
        code = '''
from go2_stairs.official_rough import validate_effective_config
try:
    validate_effective_config({'scene': {'terrain': {'terrain_generator': {'sub_terrains': {
        'pyramid_stairs': {}, 'pyramid_stairs_inv': {}}}}}}, {}, 2048, 1500)
except ValueError:
    print('INVALID_CONFIG_REJECTED')
else:
    raise SystemExit(7)
'''
        result=subprocess.run([sys.executable,'-O','-c',code],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertIn('INVALID_CONFIG_REJECTED',result.stdout)

    def test_main_records_failures_and_never_treats_zero_exit_as_checkpoint(self):
        from go2_stairs import official_rough as m
        for result,error,status in [(1,RuntimeError,'FAILED'),(0,ValueError,'FAILED'),
                                    (KeyboardInterrupt(),KeyboardInterrupt,'CANCELLED')]:
            with self.subTest(result=result),tempfile.TemporaryDirectory() as folder:
                root=Path(folder)/'run'
                child=mock.Mock(side_effect=result) if isinstance(result,BaseException) else mock.Mock(return_value=result)
                with mock.patch.object(m,'source_identity',return_value={'fixture':1}), \
                     mock.patch.object(m,'adapter_identity',return_value={'fixture':1}), \
                     mock.patch.object(m,'run_reference_child',child), self.assertRaises(error):
                    m.main(['--output',str(root),'--updates','2','--num-envs','128'])
                record=json.loads((root/'status.json').read_text(encoding='utf-8'))
                self.assertEqual(record['status'],status)
                self.assertFalse(record['stair_skill_verified'])

    def test_main_rejects_existing_output_without_launching_or_overwriting(self):
        from go2_stairs import official_rough as m
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);sentinel=root/'status.json';sentinel.write_text('untouched')
            with mock.patch.object(m,'source_identity',return_value={}), \
                 mock.patch.object(m,'run_reference_child') as child, self.assertRaises(FileExistsError):
                m.main(['--output',str(root),'--updates','2','--num-envs','128'])
            child.assert_not_called()
            self.assertEqual(sentinel.read_text(),'untouched')

    def test_command_uses_isaac_rough_without_changing_learning_or_asset(self):
        from go2_stairs.official_rough import build_command
        cmd = build_command(Path('rough fixture'), 2048, 1500, 0)
        self.assertIn('Isaac-Velocity-Rough-Unitree-Go2-v0', cmd)
        self.assertIn('1500', cmd)
        self.assertIn('IsaacLab/scripts/reinforcement_learning/rsl_rl/train.py', cmd[1].replace('\\', '/'))
        self.assertEqual({v.split('=')[0] for v in cmd if v.startswith('env.')},
                         {'env.scene.terrain.visual_material', 'env.scene.sky_light.spawn.texture_file'})
        self.assertEqual({v.split('=')[0] for v in cmd if v.startswith('agent.')},
                         {'agent.experiment_name'})

    def test_bad_budgets_rejected(self):
        from go2_stairs.official_rough import build_command
        for n, u, seed in [(0,1500,0), (4097,1500,0), (128,0,0), (128,1501,0),
                            (True,2,0), (128,2,-1)]:
            with self.subTest(n=n,u=u,seed=seed), self.assertRaises(ValueError):
                build_command(Path('fixture'), n, u, seed)

    def test_effective_config_requires_both_stair_types_and_original_range(self):
        from go2_stairs.official_rough import validate_effective_config
        env, agent = fixture()
        result = validate_effective_config(env, agent, 2048, 1500)
        self.assertTrue(result['stairs_enabled'])
        self.assertFalse(result['stair_skill_verified'])
        self.assertTrue(result['requires_height_scan'])
        for mutate in [
            lambda e,a: e['scene']['terrain'].update(terrain_type='plane'),
            lambda e,a: e['scene']['terrain']['terrain_generator']['sub_terrains'].pop('pyramid_stairs_inv'),
            lambda e,a: e['scene']['terrain']['terrain_generator']['sub_terrains']['pyramid_stairs'].update(proportion=0),
            lambda e,a: e['scene']['terrain']['terrain_generator']['sub_terrains']['pyramid_stairs'].update(step_height_range=[.04,.08]),
            lambda e,a: e['observations']['policy'].update(height_scan=None),
            lambda e,a: e['curriculum'].update(terrain_levels=None),
            lambda e,a: a.update(max_iterations=300),
        ]:
            e,a=copy.deepcopy(env),copy.deepcopy(agent);mutate(e,a)
            with self.assertRaises(ValueError):
                validate_effective_config(e,a,2048,1500)

    def test_yaml_is_data_only_even_for_python_tags(self):
        from go2_stairs.official_rough import load_yaml_data
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'test.yaml'
            path.write_text('dims: !!python/tuple [512, 256, 128]\nfunc: !!python/name:os.system \'\'\n')
            data=load_yaml_data(path)
            self.assertEqual(data['dims'],[512,256,128])
            self.assertFalse(callable(data['func']))

    def test_checkpoint_native_rough_shape_and_finite(self):
        import torch
        from go2_stairs.official_rough import validate_checkpoint
        state={'std':torch.ones(12)}
        for net,dims in [('actor',[235,512,256,128,12]),('critic',[235,512,256,128,1])]:
            for i,(a,b) in enumerate(zip(dims,dims[1:])):
                state[f'{net}.{i*2}.weight']=torch.zeros(b,a)
                state[f'{net}.{i*2}.bias']=torch.zeros(b)
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'model_1.pt'
            record={'model_state_dict':state,'optimizer_state_dict':{'state':{},'param_groups':[]},'iter':1}
            torch.save(record,p)
            self.assertEqual(validate_checkpoint(p,2)['actor_obs'],235)
            with self.assertRaises(ValueError):validate_checkpoint(p,3)
            state['actor.0.weight'][0,0]=float('nan');torch.save(record,p)
            with self.assertRaises(ValueError):validate_checkpoint(p,2)


if __name__ == '__main__':
    unittest.main()
