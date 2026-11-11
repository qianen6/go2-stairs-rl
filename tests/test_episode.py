import importlib
import unittest

from go2_stairs.terrain import StairSpec


class EpisodeJudgeTests(unittest.TestCase):
    def setUp(self):
        self.m = importlib.import_module('go2_stairs.evaluate')
        self.assertTrue(hasattr(self.m, 'EpisodeJudge'), 'Physical episode judge is missing')
        self.spec = StairSpec(.06, .3, 3, 'up')

    def sample(self, t, **changes):
        x = 2.4
        data = dict(time_s=t, body_position=(x, 0., .45),
                    foot_positions=((x, .1, .18),) * 4, planar_speed=0., yaw_rate=0.,
                    tilt_rad=0., command=(0., 0., 0.), body_contact_force=0., leg_contact_force=0.)
        data.update(changes)
        return self.m.EpisodeSample(**data)

    def test_goal_requires_all_feet_zero_command_and_full_second(self):
        judge = self.m.EpisodeJudge(self.spec)
        for i in range(500):
            status = judge.step(self.sample(i*.002))
            self.assertFalse(status['done'])
        status = judge.step(self.sample(1.0))
        self.assertTrue(status['success'])
        self.assertEqual(status['reason'], 'success')

    def test_forward_command_does_not_count_as_stopped_goal(self):
        judge = self.m.EpisodeJudge(self.spec, timeout_s=.01)
        for i in range(6):
            status = judge.step(self.sample(i*.002, command=(.1, 0, 0)))
        self.assertEqual(status['reason'], 'timeout')
        self.assertFalse(status['success'])

    def test_one_foot_behind_platform_prevents_success(self):
        judge = self.m.EpisodeJudge(self.spec, timeout_s=.01)
        for i in range(6):
            status = judge.step(self.sample(i*.002, foot_positions=((1.5, 0, .1),) * 4))
        self.assertEqual(status['reason'], 'timeout')

    def test_body_contact_uses_duration_and_latches_failure(self):
        judge = self.m.EpisodeJudge(self.spec)
        judge.step(self.sample(0.))
        for i in range(1, 20):
            status = judge.step(self.sample(i*.002, body_contact_force=21.))
            self.assertFalse(status['done'])
        failed = judge.step(self.sample(.04, body_contact_force=21.))
        self.assertEqual(failed['reason'], 'body_contact')
        self.assertEqual(judge.step(self.sample(.042)), failed)

    def test_contact_gap_resets_duration(self):
        judge = self.m.EpisodeJudge(self.spec)
        for i in range(25):
            force = 0. if i == 10 else 21.
            status = judge.step(self.sample(i*.002, body_contact_force=force))
        self.assertFalse(status['done'])

    def test_nonfinite_and_time_gap_become_explicit_failures(self):
        judge = self.m.EpisodeJudge(self.spec)
        status = judge.step(self.sample(0, planar_speed=float('nan')))
        self.assertEqual(status['reason'], 'runtime_error')
        judge = self.m.EpisodeJudge(self.spec)
        judge.step(self.sample(0))
        status = judge.step(self.sample(.2))
        self.assertEqual(status['reason'], 'runtime_error')

    def test_lateral_exit_and_tilt(self):
        judge = self.m.EpisodeJudge(self.spec)
        self.assertEqual(judge.step(self.sample(0, body_position=(.5, .7, .3)))['reason'], 'lateral_exit')
        judge = self.m.EpisodeJudge(self.spec)
        for i in range(51):
            result = judge.step(self.sample(i*.002, tilt_rad=1.2))
        self.assertEqual(result['reason'], 'tilt')


if __name__ == '__main__':
    unittest.main()
