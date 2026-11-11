import unittest


class FlatDevelopmentTest(unittest.TestCase):
    def test_numpy_samples_produce_builtin_measurements_for_direct_validation(self):
        import numpy as np
        from go2_stairs.flat_development import FlatEpisodeMetrics
        state=FlatEpisodeMetrics()
        state.update('move',np.array([.2,0,0],dtype=np.float32),np.array([.1,0,0],dtype=np.float32),False)
        result=state.result()
        self.assertIs(type(result['forward_distance_m']),float)
        self.assertIs(type(result['commanded_distance_m']),float)

    def rollout(self, velocity=.2, fall=False, stopped=True):
        from go2_stairs.flat_development import FlatEpisodeMetrics
        state = FlatEpisodeMetrics()
        for i in range(900):
            t = i * .02
            if 1 <= t < 11:
                state.update('move', (.2, 0, 0), (velocity, 0, 0), fall and i == 100)
            elif 11 <= t < 14:
                state.update('turn', (0, 0, .2), (0, 0, .2), False)
            else:
                state.update('stop' if t >= 14 else 'stand', (0, 0, 0),
                             (0 if stopped else .1, 0, 0), False)
        return state.result()

    def test_tracking_turn_and_stop_pass(self):
        self.assertTrue(self.rollout()['passed'])

    def test_standing_still_is_not_locomotion(self):
        self.assertFalse(self.rollout(velocity=0)['passed'])

    def test_fall_remains_failed_after_automatic_reset(self):
        self.assertFalse(self.rollout(fall=True)['passed'])

    def test_stop_must_be_sustained(self):
        self.assertFalse(self.rollout(stopped=False)['passed'])

    def test_restarting_after_stop_is_not_a_completed_stop(self):
        from go2_stairs.flat_development import FlatEpisodeMetrics
        state = FlatEpisodeMetrics()
        for i in range(900):
            if 50 <= i < 550:
                state.update('move', (.2, 0, 0), (.2, 0, 0), False)
            elif 550 <= i < 700:
                state.update('turn', (0, 0, .2), (0, 0, .2), False)
            else:
                state.update('stop' if i >= 700 else 'stand', (0, 0, 0),
                             (0.2 if i >= 750 else 0, 0, 0), False)
        self.assertFalse(state.result()['passed'])

    def test_partial_or_nonfinite_rollout_does_not_pass(self):
        from go2_stairs.flat_development import FlatEpisodeMetrics
        state = FlatEpisodeMetrics()
        self.assertFalse(state.result()['passed'])
        state.update('move', (.2, 0, 0), (float('nan'), 0, 0), False)
        self.assertEqual(state.result()['failure'], 'nonfinite_state')


if __name__ == '__main__':
    unittest.main()
