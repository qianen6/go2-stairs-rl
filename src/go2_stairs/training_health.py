"""Reward counterexamples and explicit bounded-training stop decisions."""
import math
import torch


def relative_tracking_reward(command, velocity, std):
    if not .05 <= float(std) <= 1.:
        raise ValueError('invalid tracking width')
    if command.shape != velocity.shape or command.ndim != 2:
        raise ValueError('tracking inputs must have identical [N,D] shape')
    error = (command-velocity).square().sum(-1)
    magnitude = command.square().sum(-1)
    actual = torch.exp(-error/std**2)
    standing = torch.exp(-magnitude/std**2)
    gain = ((actual-standing)/(1-standing).clamp_min(.01)).clamp(-1.,1.)
    return torch.where(magnitude > .05**2, gain, actual)


class TrainingHalt(RuntimeError):
    def __init__(self, reason):
        self.guard_reason = reason
        super().__init__(reason)


class TrainingHealth:
    def __init__(self, patience=5):
        self.patience = patience
        self.saturation_streak = 0
        self.reason = None

    def update(self, metrics):
        if self.reason is not None:
            return self.reason
        if not all(math.isfinite(float(x)) for x in metrics.values()):
            self.reason = 'NONFINITE_METRIC'
        elif metrics['hard_clip_fraction'] > 0:
            self.reason = 'OUT_OF_RANGE_ACTION'
        else:
            self.saturation_streak = self.saturation_streak+1 if metrics['near_bound_fraction']>.6 else 0
            if self.saturation_streak >= self.patience:
                self.reason = 'ACTION_SATURATION'
        return self.reason

    def to_dict(self):
        return dict(patience=self.patience, saturation_streak=self.saturation_streak, reason=self.reason)

    @classmethod
    def from_dict(cls, value):
        obj = cls(value['patience'])
        obj.saturation_streak = value['saturation_streak']
        obj.reason = value['reason']
        return obj


def _validated_physical_development(report):
    """Recompute the fixed gate from measurements, never from success labels."""
    if not isinstance(report, dict) or report['status']!='FLAT_DEVELOPMENT_EVALUATED':
        raise ValueError('wrong report type/status')
    if report['formal_stair_acceptance'] is not False or report['noise'] is not False or report['scene_seed']!=2000:
        raise ValueError('wrong evaluation conditions')
    protocol='18s: stand1, forward10(vx=.1/.2), turn3(yaw=.2), stop4; 20 parallel trials'
    thresholds=dict(no_termination=True,stable_stop_s=1.,move_rmse_mps=.1,
                    turn_rmse_radps=.15,minimum_commanded_distance_fraction=.5)
    if report['protocol']!=protocol or report['thresholds']!=thresholds:
        raise ValueError('evaluation protocol/thresholds changed')
    cases=report['cases']
    if not isinstance(cases,list) or len(cases)!=20 or not all(isinstance(c,dict) for c in cases):
        raise ValueError('expected twenty cases')
    if {c['case_id'] for c in cases}!={f'flat-dev-{i:02}' for i in range(20)}:
        raise ValueError('missing, repeated or unknown case identity')
    def number(value):
        if type(value) not in (int,float) or not math.isfinite(value):
            raise ValueError('nonfinite or nonnumeric physical measurement')
        return value
    passed=0
    for c in cases:
        if c['complete'] is not True or type(c['steps']) is not int or c['steps']!=900 or type(c['passed']) is not bool:
            raise ValueError('incomplete case or nonboolean result')
        if c['failure'] not in (None,'terminated','nonfinite_state'):
            raise ValueError('unknown failure status')
        distance=number(c['forward_distance_m'])
        requested=number(c['commanded_distance_m'])
        move=number(c['move_rmse_mps'])
        turn=number(c['turn_rmse_radps'])
        stop=number(c['stable_stop_s'])
        longest=number(c['longest_stop_s'])
        expected=1. if int(c['case_id'][-2:])%2==0 else 2.
        if not math.isclose(requested,expected,abs_tol=1e-6) or min(move,turn)<0 or not 0<=stop<=longest<=4.000001:
            raise ValueError('measurement range or command mismatch')
        # Cauchy-Schwarz consistency: ten-second integrated error <= ten * RMS error.
        if abs(distance-requested)>10*move+1e-4:
            raise ValueError('distance and velocity error contradict each other')
        success=bool(c['failure'] is None and stop>=1. and distance>=.5*requested and move<=.1 and turn<=.15)
        if c['passed']!=success:
            raise ValueError('case label contradicts physical measurements')
        passed+=success
    if type(report['passed_count']) is not int or report['passed_count']!=passed:
        raise ValueError('success count contradicts cases')
    if type(report['passed']) is not bool or report['passed']!=(passed==20):
        raise ValueError('summary label contradicts cases')
    return cases,passed


def _validated_development(report):
    cases,passed=_validated_physical_development(report)
    def number(value):
        if type(value) not in (int,float) or not math.isfinite(value):
            raise ValueError('nonfinite or nonnumeric action measurement')
        return value
    action=report['action_metrics']
    hard=number(action['hard_clip_fraction']);near=number(action['near_bound_fraction']);maximum=number(action['max_abs_action'])
    if not 0<=hard<=near<=1 or maximum<0 or (maximum<1 and hard>0) or (maximum>=1 and hard==0):
        raise ValueError('invalid action measurements')
    if maximum<=.95 and near>0 or maximum>.95 and near==0:
        raise ValueError('action magnitude and tail fraction contradict each other')
    return cases,action,passed


def development_decision(report, completed_updates):
    try:
        cases,action_metrics,passed=_validated_development(report)
    except (KeyError,TypeError,ValueError,OverflowError):
        return 'STOP_INVALID_EVALUATION'
    if action_metrics['hard_clip_fraction'] > 0:
        return 'STOP_ACTION_CLIPPING'
    if action_metrics['near_bound_fraction'] > .6:
        return 'STOP_ACTION_SATURATION'
    if passed==20:
        return 'FLAT_GATE_PASSED'
    if completed_updates < 300:
        return 'CONTINUE_WARMUP'
    if sum(c.get('failure') is not None for c in cases) > 4:
        return 'STOP_UNSTABLE_ROLLOUTS'
    distance = sum(c['forward_distance_m'] for c in cases)/20
    if not math.isfinite(distance) or distance < .1:
        return 'STOP_NO_PROGRESS'
    return 'CONTINUE_LEARNING'
