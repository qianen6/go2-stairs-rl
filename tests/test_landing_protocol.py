import unittest
import numpy as np


class InteriorLandingTests(unittest.TestCase):
    def test_stop_trigger_has_interior_margin_but_completion_threshold_stays_original(self):
        from go2_stairs.landing_protocol import InteriorLandingJudge
        judge=InteriorLandingJudge(3.,6.,0.,30.)
        row=np.zeros(55);row[2]=.4;row[3:15]=[3.4,0,0]*4
        row[51:55]=[3.20,4.,-.2,.2];row[18]=.8
        judge.step(.005,row)
        self.assertTrue(judge.crossed)
        self.assertFalse(judge.stop_requested)
        row[51]=3.36
        judge.step(.010,row)
        self.assertTrue(judge.stop_requested)
        row[18]=0.;row[51]=3.20
        for i in range(3,204):
            judge.step(i*.005,row)
        self.assertTrue(judge.result['completed'])
        self.assertTrue(judge.result['clean'])

    def test_robot_that_slides_back_onto_stairs_does_not_pass(self):
        from go2_stairs.landing_protocol import InteriorLandingJudge
        judge=InteriorLandingJudge(3.,6.,0.,30.)
        row=np.zeros(55);row[2]=.4;row[3:15]=[3.5,0,0]*4;row[51:55]=[3.4,4.,-.2,.2];row[18]=.8
        judge.step(.005,row);row[18]=0.;row[51]=3.1
        for i in range(2,404):judge.step(i*.005,row)
        self.assertIsNone(judge.result)


if __name__=='__main__':unittest.main()
