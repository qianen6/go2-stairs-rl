"""Frozen evaluation cases and honest result aggregation (no robot commands)."""

from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import product
import math


@dataclass(frozen=True)
class EpisodeSample:
    """One physics-step sample; lengths/metres, times/seconds, angles/radians."""
    time_s: float
    body_position: tuple
    foot_positions: tuple
    planar_speed: float
    yaw_rate: float
    tilt_rad: float
    command: tuple
    body_contact_force: float = 0.
    leg_contact_force: float = 0.


class EpisodeJudge:
    """Offline scalar acceptance judge. Terminal failures are latched."""

    def __init__(self, spec, physics_dt=.002, timeout_s=60.):
        if not all(math.isfinite(v) and v > 0 for v in (physics_dt, timeout_s)):
            raise ValueError('invalid evaluation timing')
        self.spec = spec
        self.dt = physics_dt
        self.timeout = timeout_s
        self._last_time = 0.
        self._seen = False
        self._body_duration = self._leg_duration = self._tilt_duration = self._goal_duration = 0.
        self._terminal = None

    def _finish(self, reason):
        self._terminal = dict(done=True, success=reason == 'success', reason=reason,
                              elapsed_s=self._last_time)
        return self._terminal.copy()

    def step(self, sample):
        if self._terminal is not None:
            return self._terminal.copy()
        try:
            if len(sample.body_position) != 3 or len(sample.command) != 3 or len(sample.foot_positions) != 4:
                return self._finish('runtime_error')
            if any(len(p) != 3 for p in sample.foot_positions):
                return self._finish('runtime_error')
            values = (sample.time_s, *sample.body_position, *sample.command,
                      *(v for foot in sample.foot_positions for v in foot), sample.planar_speed,
                      sample.yaw_rate, sample.tilt_rad, sample.body_contact_force, sample.leg_contact_force)
            if not all(math.isfinite(v) for v in values):
                return self._finish('runtime_error')
            if min(sample.planar_speed, sample.body_contact_force, sample.leg_contact_force) < 0 or not 0 <= sample.tilt_rad <= math.pi:
                return self._finish('runtime_error')
        except (TypeError, AttributeError):
            return self._finish('runtime_error')
        delta = sample.time_s - self._last_time
        if delta < 0 or (self._seen and delta <= 0) or delta > self.dt * 1.5:
            return self._finish('runtime_error')
        self._last_time = float(sample.time_s)
        self._seen = True
        self._body_duration = self._body_duration + delta if sample.body_contact_force > 20 else 0.
        self._leg_duration = self._leg_duration + delta if sample.leg_contact_force > 0 else 0.
        self._tilt_duration = self._tilt_duration + delta if sample.tilt_rad > math.pi/3 else 0.
        if abs(sample.body_position[1]) > self.spec.width/2:
            return self._finish('lateral_exit')
        if sample.body_position[2] < 0:
            return self._finish('fell_below_ground')
        for duration, threshold, reason in ((self._body_duration, .04, 'body_contact'),
                                             (self._leg_duration, .20, 'leg_contact'),
                                             (self._tilt_duration, .10, 'tilt')):
            if duration >= threshold - 1e-10:
                return self._finish(reason)
        points = (sample.body_position, *sample.foot_positions)
        on_platform = all(self.spec.finish_x+.15 <= p[0] <= self.spec.length and abs(p[1]) <= self.spec.width/2 for p in points)
        stopped = all(abs(v) <= 1e-6 for v in sample.command) and sample.planar_speed < .05 and abs(sample.yaw_rate) < .10
        self._goal_duration = self._goal_duration + delta if on_platform and stopped else 0.
        if self._goal_duration >= 1. - 1e-10:
            return self._finish('success')
        if sample.time_s >= self.timeout - 1e-10:
            return self._finish('timeout')
        return dict(done=False, success=False, reason='running', elapsed_s=self._last_time)


def build_core_suite():
    cases = []
    for height, depth, steps, direction in product((.04, .06, .08), (.30, .35), (3, 5), ('up', 'down')):
        for repeat in range(5):
            index = len(cases)
            cases.append(dict(case_id=f'core-{index:03d}', height=height, depth=depth,
                              steps=steps, direction=direction, seed=10000+index,
                              command_vx=.10 if repeat % 2 == 0 else .20))
    return cases


def next_level(level, up_rate, down_rate):
    if type(level) is not int or not 0 <= level <= 3:
        raise ValueError('invalid curriculum level')
    if not all(math.isfinite(v) and 0 <= v <= 1 for v in (up_rate, down_rate)):
        raise ValueError('rates must be finite and between zero and one')
    weaker = min(up_rate, down_rate)
    if weaker >= .8:
        return min(level+1, 3)
    if weaker < .6:
        return max(level-1, 0)
    return level


def _stats(successes, count):
    p = successes / count if count else 0.
    if not count:
        return dict(episodes=0, successes=0, success_rate=0., wilson_95=[0., 1.])
    z = 1.959963984540054
    denominator = 1 + z*z/count
    center = (p + z*z/(2*count)) / denominator
    radius = z * math.sqrt(p*(1-p)/count + z*z/(4*count*count)) / denominator
    return dict(episodes=count, successes=successes, success_rate=p,
                wilson_95=[max(0., center-radius), min(1., center+radius)])


def summarize(cases, results):
    case_by_id = {c['case_id']: c for c in cases}
    if not cases or len(case_by_id) != len(cases):
        raise ValueError('cases must be nonempty and unique')
    by_id = {}
    for row in results:
        case_id = row['case_id']
        if case_id not in case_by_id or case_id in by_id:
            raise ValueError('unknown or duplicate result')
        if type(row['success']) is not bool:
            raise ValueError('success must be a JSON boolean')
        if not isinstance(row.get('reason'), str) or not row['reason']:
            raise ValueError('result needs a nonempty reason')
        if row['success'] != (row['reason'] == 'success'):
            raise ValueError('success and reason disagree')
        elapsed = row.get('elapsed_s')
        if not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError('invalid elapsed_s')
        by_id[case_id] = row
    directions = defaultdict(lambda: [0, 0])
    groups = defaultdict(lambda: [0, 0])
    failures = Counter()
    for c in cases:
        direction = c['direction']
        if direction not in ('up', 'down'):
            raise ValueError('invalid case direction')
        group = (c['height'], c['depth'], c['steps'], direction)
        row = by_id.get(c['case_id'], {'success': False, 'reason': 'missing_result'})
        directions[direction][1] += 1
        groups[group][1] += 1
        directions[direction][0] += int(row['success'])
        groups[group][0] += int(row['success'])
        if not row['success']:
            failures[row['reason']] += 1
    direction_stats = {name: _stats(*directions[name]) for name in ('up', 'down')}
    group_rows = [dict(height=k[0], depth=k[1], steps=k[2], direction=k[3], **_stats(*v))
                  for k, v in groups.items()]
    frozen_core = {c['case_id']: c for c in build_core_suite()}
    core_suite_valid = case_by_id == frozen_core
    passed = (core_suite_valid
              and all(v['episodes'] > 0 and v['success_rate'] >= .90 for v in direction_stats.values())
              and all(g['success_rate'] >= .8 for g in group_rows))
    return dict(episodes=len(cases), successes=sum(v[0] for v in directions.values()),
                passed=passed, core_suite_valid=core_suite_valid, directions=direction_stats, groups=group_rows,
                failure_reasons=dict(failures))
