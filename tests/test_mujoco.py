import importlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock
from contextlib import redirect_stdout
import io

import numpy as np


class MujocoSceneTests(unittest.TestCase):
    def setUp(self):
        try:
            self.m = importlib.import_module('go2_stairs.mujoco_runner')
        except ModuleNotFoundError:
            self.fail('MuJoCo scene implementation is missing')
        self.assets = Path(os.environ.get('GO2_MODEL_DIR', Path(__file__).parents[1] / 'assets' / 'go2'))
        self.assertTrue((self.assets / 'go2.xml').is_file(), 'Go2 assets must be provisioned for integration tests')

    def test_real_asset_loads_and_starts_on_correct_platform(self):
        from go2_stairs.terrain import StairSpec
        for direction, expected_z in [('up', .27), ('down', .45)]:
            spec = StairSpec(.06, .3, 3, direction)
            model = self.m.load_scene(self.assets, spec)
            data = self.m.initialize(model, spec)
            self.assertEqual(model.nu, 12)
            self.assertAlmostEqual(model.opt.timestep, .002)
            self.assertAlmostEqual(data.qpos[2], expected_z)
            self.assertAlmostEqual(data.qpos[0], .5)
            manifest = self.m.manifest_from_model(model)
            self.assertEqual(len(set(manifest.runtime_joint_names)), 12)

    def test_smoke_is_finite_and_labels_nonlearned_controller(self):
        from go2_stairs.terrain import StairSpec
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / '中文 scene'
            report = self.m.run_smoke(self.assets, StairSpec(.04, .35, 3, 'down'), .20, output)
            self.assertTrue(report['finite_state'])
            self.assertTrue(report['standing_end'])
            self.assertAlmostEqual(report['sim_time_s'], .20)
            self.assertEqual(report['controller'], 'PD_STAND_SMOKE_NOT_LEARNED')
            self.assertFalse(report['trained_policy_evaluated'])
            self.assertTrue((output / 'scene.xml').exists())
            saved = json.loads((output / 'summary.json').read_text(encoding='utf-8'))
            self.assertEqual(saved['physics_steps'], 100)
            with self.assertRaises(FileExistsError):
                self.m.run_smoke(self.assets, StairSpec(.04, .35, 3, 'down'), .20, output)

    def test_invalid_duration_rejected_without_output(self):
        from go2_stairs.terrain import StairSpec
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / 'never-created'
            with self.assertRaises(ValueError):
                self.m.run_smoke(self.assets, StairSpec(.04, .35, 3, 'up'), float('nan'), target)
            self.assertFalse(target.exists())

    def test_real_mujoco_auto_reset_is_latched_as_failure(self):
        from go2_stairs.terrain import StairSpec
        actual_step = self.m.mujoco.mj_step
        def unstable_step(model, data):
            data.qvel[0] = 1e20
            actual_step(model, data)
        cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as temp:
            try:
                os.chdir(temp)  # MuJoCo's deliberate instability log stays in the fixture.
                with mock.patch.object(self.m.mujoco, 'mj_step', side_effect=unstable_step):
                    report = self.m.run_smoke(self.assets, StairSpec(.04, .3, 3, 'up'), .004, Path(temp) / 'run')
                self.assertFalse(report['smoke_passed'])
                self.assertEqual(report['failure_reason'], 'simulation_instability')
            finally:
                os.chdir(cwd)

    def test_nonfinite_state_and_time_preserve_failure_json(self):
        from go2_stairs.terrain import StairSpec
        actual_step = self.m.mujoco.mj_step
        for field in ('qpos', 'time'):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temp:
                def broken_step(model, data):
                    actual_step(model, data)
                    if field == 'qpos':
                        data.qpos[0] = float('nan')
                    else:
                        data.time = float('nan')
                try:
                    with mock.patch.object(self.m.mujoco, 'mj_step', side_effect=broken_step):
                        report = self.m.run_smoke(self.assets, StairSpec(.04, .3, 3, 'up'), .004, Path(temp) / 'run')
                except ValueError as exc:
                    self.fail(f'Failure evidence lost during JSON serialization: {exc}')
                self.assertFalse(report['smoke_passed'])
                saved = json.loads((Path(temp) / 'run' / 'summary.json').read_text(encoding='utf-8'))
                self.assertFalse(saved['smoke_passed'])
                self.assertIsNone(saved['final_position'][0] if field == 'qpos' else saved['sim_time_s'])

    def test_render_failure_preserves_simulation_summary_and_cli_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / 'run'
            argv = ['--model-dir', str(self.assets), '--direction', 'up', '--seconds', '.004',
                    '--output', str(target), '--render']
            with mock.patch.object(self.m.mujoco, 'Renderer', side_effect=RuntimeError('render unavailable')), \
                    mock.patch('sys.argv', ['go2-stairs-scene', *argv]):
                try:
                    with redirect_stdout(io.StringIO()):
                        code = self.m.main()
                except (RuntimeError, TypeError) as exc:
                    self.fail(f'Expected durable failure report and exit code, got {exc}')
            self.assertEqual(code, 1)
            saved = json.loads((target / 'summary.json').read_text(encoding='utf-8'))
            self.assertTrue(saved['finite_state'])
            self.assertEqual(saved['render_status'], 'failed')
            self.assertIn('render unavailable', saved['render_error'])


if __name__ == '__main__':
    unittest.main()
