import unittest
import numpy as np

from go2_stairs.reference_reproduction import (
    build_cases, walk_command, ReferenceJudge, flat_metrics, aggregate,
    verify_public_checkpoint, profile_geometry, ordered_body_indices, choose_native_stair_actors, classify_stair_profile,
)


class ReferenceReproductionTests(unittest.TestCase):
    def test_missing_terrain_ray_must_fail_even_if_clipping_makes_observation_finite(self):
        import torch
        from go2_stairs.reference_reproduction import require_height_hit_coverage
        hits=torch.tensor([[0.,float('inf')]])
        self.assertTrue(bool(torch.isfinite((.4-hits-.5).clamp(-1,1)).all()))
        with self.assertRaises(ValueError):require_height_hit_coverage(hits)
        require_height_hit_coverage(torch.tensor([[0.,.2,-.4]]))

    def test_initial_scan_pose_guard_rejects_stale_or_reordered_environment_pose(self):
        import torch
        from go2_stairs.reference_reproduction import require_scan_pose_alignment
        expected=torch.tensor([[0.,0.,.4],[70.,0.,1.8]])
        self.assertEqual(require_scan_pose_alignment(expected,expected.clone()),0.)
        with self.assertRaises(ValueError):require_scan_pose_alignment(expected,expected.flip(0))
        stale=expected.clone();stale[1,0]=2.5
        with self.assertRaises(ValueError):require_scan_pose_alignment(expected,stale)

    def test_initial_scan_pose_guard_rejects_nonfinite_values(self):
        import torch
        from go2_stairs.reference_reproduction import require_scan_pose_alignment
        expected=torch.zeros(2,3);invalid=expected.clone();invalid[0,0]=float('nan')
        with self.assertRaises(ValueError):require_scan_pose_alignment(expected,invalid)

    def test_native_camera_selection_uses_current_origins_and_avoids_overlapping_actors(self):
        origins=np.array([[0,0,-.7],[0,0,0],[20,0,-.7],[30,0,.7],[30,0,.7],[40,0,.7]])
        self.assertEqual(choose_native_stair_actors(origins,['up',None,'up','down','down','down']),[2,5])
        with self.assertRaises(ValueError):choose_native_stair_actors(np.zeros((4,3)),[None]*4)

    def test_native_stair_classification_rejects_slopes_despite_negative_origin(self):
        self.assertIsNone(classify_stair_profile(np.linspace(-.7,0,160))['direction'])
        up=classify_stair_profile(np.repeat(np.arange(-7,1)*.1,20))
        self.assertEqual(up['direction'],'up');self.assertEqual(up['risers'],7)
        self.assertAlmostEqual(up['height'],.1)
        self.assertEqual(classify_stair_profile(np.repeat(np.arange(7,-1,-1)*.1,20))['direction'],'down')

    def row(self):
        r=np.zeros(55);r[2]=.4;r[3:15]=np.array([[0,.1,0],[0,-.1,0],[-.2,.1,0],[-.2,-.1,0]]).ravel();r[51:55]=[-.4,.4,-.16,.16]
        return r

    def test_native_reset_failure_latches_even_after_good_motion(self):
        j=ReferenceJudge('flat');j.native_termination(.1)
        self.assertEqual(j.step(.105,self.row())['reason'],'native_base_contact')

    def test_native_failure_in_same_policy_step_overrides_provisional_success(self):
        j=ReferenceJudge('stairs');j.finish(1.,'success');j.native_termination(1.015)
        self.assertFalse(j.result['success'])
        k=ReferenceJudge('stairs');k.finish(1.,'success');k.native_termination(1.04)
        self.assertTrue(k.result['success'])

    def test_foot_force_mapping_does_not_assume_sensor_robot_body_order(self):
        self.assertEqual(ordered_body_indices(['RF_foot','LF_foot'],['base','LF_foot','RF_foot']),[2,1])
        with self.assertRaises(ValueError):ordered_body_indices(['RF_foot'],['base','LF_foot'])

    def test_stairs_require_all_feet_then_one_second_stop(self):
        j=ReferenceJudge('stairs',finish_x=3.,end_x=4.,ground_min=-1.4)
        r=self.row();r[0]=3.6;r[51:55]=[3.2,4.,-.16,.16];r[3:15]=np.array([[3.6,.1,0],[3.6,-.1,0],[3.0,.1,0],[3.0,-.1,0]]).ravel()
        for k in range(200):j.step((k+1)*.005,r)
        self.assertFalse(j.crossed)
        r[[9,12]]=3.2
        for k in range(199):self.assertIsNone(j.step(1+(k+1)*.005,r))
        self.assertTrue(j.step(2.,r)['success'])

    def test_whole_body_rear_must_cross_even_when_root_and_feet_have_crossed(self):
        j=ReferenceJudge('stairs');r=self.row();r[0]=3.5;r[3:15]=np.array([[3.6,.1,0],[3.6,-.1,0],[3.2,.1,0],[3.2,-.1,0]]).ravel()
        r[51:55]=[3.1,3.9,-.16,.16]
        for k in range(201):j.step((k+1)*.005,r)
        self.assertFalse(j.crossed);self.assertIsNone(j.result)

    def test_collision_duration_and_terminal_failure(self):
        j=ReferenceJudge('flat');r=self.row();r[22]=2.
        for k in range(39):self.assertIsNone(j.step((k+1)*.005,r))
        self.assertEqual(j.step(.2,r)['reason'],'leg_contact')
        r[22]=0;self.assertFalse(j.step(.205,r)['success'])

    def test_nonfinite_or_missing_trace_never_passes(self):
        j=ReferenceJudge('flat');r=self.row();r[15]=float('nan')
        self.assertEqual(j.step(.005,r)['reason'],'runtime_error')
        self.assertFalse(flat_metrics(np.zeros((0,55)),.005)['passed'])

    def test_inverted_stairs_negative_world_height_is_valid(self):
        j=ReferenceJudge('stairs',ground_min=-1.4);r=self.row();r[2]=-1.
        self.assertIsNone(j.step(.005,r))

    def test_walk_commands_cover_turn_stop_and_segments(self):
        self.assertEqual(walk_command(0.),(0.,0.,True))
        self.assertEqual(walk_command(3.),(.5,0.,False))
        self.assertEqual(walk_command(13.),(.8,0.,False))
        self.assertAlmostEqual(walk_command(23.)[1],np.pi/2)
        self.assertEqual(walk_command(38.)[0],0.)

    def test_fixed_suite_has_all_heights_directions_and_complete_denominator(self):
        c=build_cases('stairs');self.assertEqual(len(c),120)
        self.assertEqual(len({x['seed'] for x in c}),120)
        a=aggregate(c,[]);self.assertEqual(a['episodes'],120);self.assertEqual(a['successes'],0)
        self.assertEqual(a['failure_reasons'],{'missing_result':120})

    def test_profile_geometry_measures_risers_not_total_height(self):
        x=np.array([0.,1.,1.5,2.,2.5,3.,3.5]);z=np.array([-.6,-.6,-.4,-.2,0.,0.,0.])
        g=profile_geometry(x,z,'up',.2)
        self.assertEqual(g['riser_count'],3);self.assertAlmostEqual(g['total_height'],.6)
        self.assertAlmostEqual(g['last_edge_x'],2.25)
        with self.assertRaises(ValueError):profile_geometry(x,z,'down',.2)

    def test_flat_metrics_reject_missing_final_stop_and_sliding(self):
        dt=.005;t=np.arange(1,8401)*dt;a=np.tile(self.row(),(8400,1))
        for k,tm in enumerate(t):
            vx,target,stop=walk_command(max(0.,tm-dt));a[k,18]=vx;a[k,23]=vx;a[k,25]=target;a[k,26]=target
        a[:,35:39]=10
        self.assertTrue(flat_metrics(a,dt)['passed'])
        a[-200:,15]=.2
        self.assertFalse(flat_metrics(a,dt)['passed'])
        a[-200:,15]=0;a[:,27:35]=.3
        self.assertFalse(flat_metrics(a,dt)['passed'])

    def test_checkpoint_validation_rejects_substituted_weights(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'fake.pt';p.write_bytes(b'not the published model')
            with self.assertRaises(ValueError):verify_public_checkpoint(p)


if __name__=='__main__':unittest.main()
