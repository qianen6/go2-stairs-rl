import unittest
import numpy as np
from go2_stairs.stack_retest import RetestJudge,build_retest_cases,foot_riser_event


class StackRetestTests(unittest.TestCase):
    def test_declared_approach_angle_preserves_seeded_lateral_start(self):
        from go2_stairs.reference_reproduction import sample_initial_pose
        base=sample_initial_pose({'seed':53000});angled=sample_initial_pose({'seed':53000,'heading_offset':np.pi/6})
        self.assertEqual(base[0],angled[0]);self.assertAlmostEqual(angled[1]-base[1],np.pi/6)

    def test_physics_refinement_preserves_fifty_hz_policy_period(self):
        from go2_stairs.reference_reproduction import physics_decimation
        self.assertEqual(physics_decimation(.005),4)
        self.assertEqual(physics_decimation(.0025),8)
        with self.assertRaises(ValueError):physics_decimation(.003)

    def row(self):
        r=np.zeros(55);r[2]=.4;r[3:15]=np.array([[0,.1,0],[0,-.1,0],[-.2,.1,0],[-.2,-.1,0]]).ravel();r[51:55]=[-.4,.4,-.2,.2]
        return r

    def test_quality_contact_is_recorded_without_stopping_native_traversal(self):
        j=RetestJudge(3.,6.,0.);r=self.row();r[22]=5.
        for k in range(40):j.step((k+1)*.005,r)
        self.assertIsNone(j.result);self.assertTrue(j.quality['leg_contact'])

    def test_completion_requires_body_and_feet_crossed_then_stopped(self):
        j=RetestJudge(3.,6.,0.);r=self.row();r[0]=4.;r[51:55]=[3.6,4.4,-.2,.2]
        r[3:15]=np.array([[4.2,.1,0],[4.2,-.1,0],[3.8,.1,0],[3.8,-.1,0]]).ravel()
        for k in range(199):self.assertIsNone(j.step((k+1)*.005,r))
        self.assertTrue(j.step(1.,r)['completed'])
        self.assertTrue(j.result['clean'])

    def test_native_failure_overrides_candidate_success_in_same_policy_step(self):
        j=RetestJudge(3.,6.,0.);j.finish(1.,'completed');j.native_termination(1.)
        self.assertFalse(j.result['completed'])

    def test_empty_height_coverage_marks_invalid_not_model_failure(self):
        j=RetestJudge(3.,6.,0.);j.invalidate(.5,'height_coverage')
        self.assertFalse(j.result['valid']);self.assertFalse(j.result['completed'])

    def test_frozen_evaluation_has_balanced_20cm_directions_and_speeds(self):
        cases=build_retest_cases('evaluation')
        self.assertEqual(len(cases),40)
        self.assertEqual(len({c['seed'] for c in cases}),40)
        for d in ['up','down']:
            for v in [.8,1.]:self.assertEqual(sum(c['direction']==d and c['speed']==v and c['height']==.2 for c in cases),10)

    def test_foot_clearance_is_relative_to_start_floor_and_riser_top(self):
        e=foot_riser_event(np.array([1.19,0.,-1.25]),np.array([1.21,0.,-1.25]),1.2,-1.4,.2)
        self.assertAlmostEqual(e['height_above_start'],.15)
        self.assertAlmostEqual(e['clearance_above_step'], -.05)
        self.assertTrue(e['crossed_plane'])


if __name__=='__main__':unittest.main()
