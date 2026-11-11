"""Move the stop trigger inside the landing without relaxing acceptance geometry."""
import numpy as np
from .native_retest import RetestJudge


class InteriorLandingJudge(RetestJudge):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.stop_requested=False
        self.stop_time=None

    def step(self,t,row):
        if self.result is None and np.isfinite(row).all():
            feet=row[3:15].reshape(4,3)
            interior=(row[51]>=self.edge+.35 and row[52]<=self.end and row[53]>=-4 and row[54]<=4
                      and np.all(feet[:,0]>=self.edge+.35) and np.all(feet[:,0]<=self.end))
            if interior and not self.stop_requested:
                self.stop_requested=True
                self.stop_time=float(t)
        # Original +.15m full-body/four-feet acceptance and1s stop are unchanged.
        return super().step(t,row)
