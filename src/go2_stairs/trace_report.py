"""Explain how each evaluation case ended, from a landing_eval cell directory (trace.npz + summary.json).

    python -m go2_stairs.trace_report reports/stairs25-baseline-public-eval/up20
    python -m go2_stairs.trace_report <cell dir> --json report.json

Per case: outcome, risers passed, stall time, head-contact time, which non-foot bodies touched the stairs
just before the end, and how often thigh/calf actuators hit their torque limit. Each case gets a mechanism:
  completed | stall_head_on_riser | stall | fall_after_head_contact | fall_after_leg_catch | fall | <other judge reason>
This is the analysis that found the original failure (head pressed on the first riser for 20+ s).
"""
import argparse
import json
import re
from pathlib import Path
import numpy as np

# landing_eval measurement columns (physics rate)
X, PLANAR_SPEED, CMD_X = 0, 15, 18
NATIVE_GROUPS = {'base_legs': dict(joint_names_expr=['.*_hip_joint', '.*_thigh_joint', '.*_calf_joint'],
                                   effort_limit=23.5, velocity_limit=30., stiffness=25., damping=.5)}


def joint_limits(meta, joint_names):
    """Per-joint (effort, velocity, Kp, Kd) used by the evaluated policy's actuators."""
    groups = meta.get('actuators')
    if groups is None and meta.get('actuator_profile') == 'STAIRS25_V1':
        from .stairs25_profile import actuator_groups, load_profile
        groups = actuator_groups(load_profile())
    groups = groups or NATIVE_GROUPS
    out = np.zeros((len(joint_names), 4))
    for i, joint in enumerate(joint_names):
        owner = [g for g in groups.values() if any(re.fullmatch(p, joint) for p in g['joint_names_expr'])]
        if len(owner) != 1:
            raise ValueError(f'joint {joint} matched {len(owner)} actuator groups')
        g = owner[0]; out[i] = (g['effort_limit'], g['velocity_limit'], g['stiffness'], g['damping'])
    return out


def saturation(policy_states, limits):
    """Fraction of policy steps where the DC-motor clip binds, per joint (T, N, 12) -> (N, 12)."""
    q, qd, target = policy_states[..., 13:25], policy_states[..., 25:37], policy_states[..., 37:49]
    effort, vlim, kp, kd = (limits[:, k] for k in range(4))
    tau = kp*(target-q)-kd*qd
    hi = np.clip(effort*(1-qd/vlim), 0, effort); lo = np.clip(effort*(-1-qd/vlim), -effort, 0)
    return (tau >= hi-1e-3) | (tau <= lo+1e-3)


def classify(reason, stalled_s, head_s, leg_contact_at_end, head_contact_at_end=False):
    if reason == 'completed':
        return 'completed'
    if reason == 'timeout':
        return 'stall_head_on_riser' if head_s >= 1. else 'stall'
    if reason in ('native_base_contact', 'tilt', 'fell_below_ground'):
        if head_contact_at_end or head_s >= .5:
            return 'fall_after_head_contact'
        return 'fall_after_leg_catch' if leg_contact_at_end else 'fall'
    return reason


def analyse(cell_dir):
    cell_dir = Path(cell_dir)
    trace = np.load(cell_dir/'trace.npz')
    summary = json.loads((cell_dir/'summary.json').read_text(encoding='utf-8'))
    geometry = json.loads((cell_dir/'geometry.json').read_text(encoding='utf-8'))
    runtime = summary['runtime']; names = runtime['body_names']
    dt = float(trace['physics_dt']); policy_dt = float(trace['policy_dt'])
    m = trace['measurements']; forces = np.linalg.norm(trace['body_forces'], axis=-1)
    head = [i for i, n in enumerate(names) if n.startswith('Head')]
    legs = [i for i, n in enumerate(names) if re.fullmatch(r'.*_(thigh|calf)', n)]
    non_foot = [i for i, n in enumerate(names) if not n.endswith('_foot')]
    limits = joint_limits(summary['checkpoint'], runtime['joint_names'])
    sat = saturation(trace['policy_states'], limits)
    active = np.abs(trace['actions']).sum(-1) > 0
    cases = []
    for i, result in enumerate(summary['results']):
        end = min(len(m), max(1, int(round(result['elapsed_s']/dt))))
        g = geometry[i]; x = m[:end, i, X]
        edges = np.linspace(g['first_edge_x'], g['last_edge_x'], g['riser_count']) if g['riser_count'] else np.array([])
        moving = m[:end, i, CMD_X] > 0
        stalled = float(((m[:end, i, PLANAR_SPEED] < .05) & moving).sum()*dt)
        head_s = float((forces[:end, i][:, head].max(-1) > 1.).sum()*dt) if head else 0.
        window = slice(max(0, end-int(.5/dt)), end)
        mean_force = forces[window, i].mean(0)
        touching = [(names[k], float(mean_force[k])) for k in non_foot if mean_force[k] > 5.]
        leg_catch = any(mean_force[k] > 20. for k in legs)
        head_at_end = any(mean_force[k] > 5. for k in head)
        steps = active[:, i] & (np.arange(active.shape[0]) < int(round(result['elapsed_s']/policy_dt)))
        thigh = sat[steps, i, 4:8].mean()*100 if steps.any() else 0.
        calf = sat[steps, i, 8:12].mean()*100 if steps.any() else 0.
        cases.append(dict(case_id=result['case_id'], reason=result['reason'], elapsed_s=float(result['elapsed_s']),
                          risers_passed=int((edges < x.max()).sum()) if len(edges) else 0, riser_count=int(g['riser_count']),
                          stalled_s=stalled, head_contact_s=head_s,
                          touching_at_end=sorted(touching, key=lambda t: -t[1])[:4],
                          thigh_saturation_pct=float(thigh), calf_saturation_pct=float(calf),
                          mechanism=classify(result['reason'], stalled, head_s, leg_catch, head_at_end)))
    counts = {}
    for c in cases:
        counts[c['mechanism']] = counts.get(c['mechanism'], 0)+1
    return dict(cell=str(cell_dir), checkpoint=summary['checkpoint'].get('sha256'), profile=summary['checkpoint'].get('profile'),
                mechanisms=counts, cases=cases)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('cell_dir', type=Path)
    parser.add_argument('--json', type=Path)
    args = parser.parse_args(argv)
    report = analyse(args.cell_dir)
    print(f"{report['cell']}  profile={report['profile']}  mechanisms={report['mechanisms']}")
    for c in report['cases']:
        touch = ', '.join(f'{n}={f:.0f}N' for n, f in c['touching_at_end']) or '-'
        print(f"{c['case_id']:>34} {c['mechanism']:>21} t={c['elapsed_s']:5.1f}s risers {c['risers_passed']}/{c['riser_count']} "
              f"stall {c['stalled_s']:4.1f}s head {c['head_contact_s']:4.1f}s sat thigh/calf {c['thigh_saturation_pct']:3.1f}/{c['calf_saturation_pct']:3.1f}%  end contact: {touch}")
    if args.json:
        args.json.write_text(json.dumps(report, indent=2), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
