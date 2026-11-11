"""Execute the installed wrapper's real step body, without loading Kit/GPU."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
from tensordict import TensorDict


class WrapperOrderTests(unittest.TestCase):
    def test_nonfinite_action_is_rejected_before_upstream_clipping(self):
        from go2_stairs.contracts import FiniteActionMixin
        package = Path(importlib.util.find_spec('isaaclab_rl').origin).parent
        source = package/'rsl_rl/vecenv_wrapper.py'
        tree = ast.parse(source.read_text(encoding='utf-8'))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'RslRlVecEnvWrapper')
        step = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'step')
        module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0),
                                 ast.ClassDef(name='NativeWrapperStep', bases=[], keywords=[], body=[step], decorator_list=[])],
                            type_ignores=[])
        space = {'torch': torch, 'TensorDict': TensorDict}
        exec(compile(ast.fix_missing_locations(module), str(source), 'exec'), space)
        class Checked(FiniteActionMixin, space['NativeWrapperStep']):
            pass
        wrapper = Checked()
        wrapper.num_envs = 1
        wrapper.clip_actions = 1.
        forwarded = []
        def physics_step(action):
            forwarded.append(action.clone())
            return {'policy': torch.zeros(1, 225)}, torch.zeros(1), torch.zeros(1, dtype=torch.bool), torch.zeros(1, dtype=torch.bool), {}
        wrapper.env = SimpleNamespace(step=physics_step)
        wrapper.unwrapped = SimpleNamespace(cfg=SimpleNamespace(is_finite_horizon=False))
        for value in (float('inf'), -float('inf'), float('nan')):
            with self.assertRaises(ValueError):
                wrapper.step(torch.full((1, 12), value))
        self.assertEqual(forwarded, [])
        wrapper.step(torch.full((1, 12), 3.))
        self.assertTrue(torch.equal(forwarded[0], torch.ones(1, 12)))
