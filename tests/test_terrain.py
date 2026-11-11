import importlib
import unittest


class TerrainTests(unittest.TestCase):
    def setUp(self):
        try:
            self.m = importlib.import_module('go2_stairs.terrain')
        except ModuleNotFoundError:
            self.fail('Stair geometry implementation is missing')

    def spec(self, direction='up'):
        return self.m.StairSpec(.06, .30, 3, direction)

    def test_up_heights_are_discrete_not_a_ramp(self):
        for x, expected in [(.3, 0), (1.21, .06), (1.49, .06),
                            (1.51, .12), (1.81, .18), (2.5, .18)]:
            self.assertAlmostEqual(self.m.height_at(x, 0, self.spec()), expected)

    def test_down_starts_on_upper_platform(self):
        for x, expected in [(.3, .18), (1.21, .12), (1.51, .06), (1.81, 0), (2.5, 0)]:
            self.assertAlmostEqual(self.m.height_at(x, 0, self.spec('down')), expected)

    def test_outside_corridor_has_ground_height(self):
        self.assertEqual(self.m.height_at(.3, .7, self.spec('down')), 0)
        self.assertEqual(self.m.height_at(-.1, 0, self.spec('down')), 0)

    def test_boxes_match_analytic_height(self):
        for direction in ('up', 'down'):
            spec = self.spec(direction)
            boxes = self.m.build_boxes(spec)
            for x in (.3, 1.35, 1.65, 1.95, 2.7):
                tops = [b.center[2] + b.half_size[2] for b in boxes
                        if abs(x - b.center[0]) < b.half_size[0]]
                self.assertAlmostEqual(max([0] + tops), self.m.height_at(x, 0, spec))
            for b in boxes:
                self.assertTrue(all(v > 0 for v in b.half_size))

    def test_landing_and_bad_dimensions(self):
        self.assertAlmostEqual(self.spec().finish_x, 2.1)
        self.assertAlmostEqual(self.spec().length, 3.3)
        for kwargs in [dict(height=-.1), dict(depth=0), dict(steps=0),
                       dict(steps=2.5), dict(direction='sideways'), dict(width=float('nan'))]:
            values = dict(height=.06, depth=.3, steps=3, direction='up')
            values.update(kwargs)
            with self.assertRaises(ValueError):
                self.m.StairSpec(**values)

    def test_height_rejects_nonfinite_coordinates(self):
        with self.assertRaises(ValueError):
            self.m.height_at(float('nan'), 0, self.spec())


if __name__ == '__main__':
    unittest.main()
