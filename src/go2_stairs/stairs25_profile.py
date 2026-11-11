"""STAIRS25 training profiles: create, validate, diff and bind to checkpoints. Never imports Isaac.

The reference profile STAIRS25_V1 (configs/stairs25-v1.json) is frozen: its SHA-256 is bound into the
reference checkpoints. Students copy it under a new profile name (see `new`) and change the copy.

    python -m go2_stairs.stairs25_profile new STUDENT_TEAM1_NO_HEAD
    python -m go2_stairs.stairs25_profile check configs/students/student_team1_no_head.json
"""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
import numpy as np

FAMILY = 'STAIRS25'
REFERENCE_PROFILE = 'STAIRS25_V1'
REFERENCE_SHA256 = '8f5f1556cbf5725c1427623a222c05743f5fae8fb8a8a4c4ed5f7cc3480859d9'
FIX_KEYS = ('actuators', 'head_termination', 'contact_penalties', 'foot_clearance', 'terrain', 'commands')
NATIVE_FAMILIES = ('pyramid_stairs', 'pyramid_stairs_inv', 'boxes', 'random_rough', 'hf_pyramid_slope', 'hf_pyramid_slope_inv')
STAIR_FAMILIES = ('pyramid_stairs', 'pyramid_stairs_inv')
JOINTS = tuple(f'{leg}_{part}_joint' for leg in ('FL', 'FR', 'RL', 'RR') for part in ('hip', 'thigh', 'calf'))
NAME = re.compile(r'[A-Z][A-Z0-9_]{2,63}')
TOP_REQUIRED = {'profile', 'public_sha256', 'fixes', 'actuators', 'rewards', 'terminations', 'terrain', 'commands', 'ppo'}
TOP_OPTIONAL = {'plan', 'authorization', 'evaluation', 'description', 'author', 'parent_profile'}
PPO_REQUIRED = {'schedule', 'desired_kl', 'public_init_learning_rate', 'scratch_learning_rate',
                'critic_warmup_updates_public_init', 'critic_warmup_learning_rate'}
PPO_OPTIONAL = {'finetune_learning_rate', 'critic_warmup_updates_finetune'}


class ProfileError(ValueError):
    pass


def repo_root():
    return Path(__file__).resolve().parents[2]


def reference_path():
    return repo_root()/'configs/stairs25-v1.json'


def students_dir():
    return repo_root()/'configs/students'


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value)


def _keys(errors, where, value, required, optional=frozenset()):
    if not isinstance(value, dict):
        errors.append(f'{where} must be an object'); return False
    missing = set(required)-set(value); unknown = set(value)-set(required)-set(optional)
    if missing:
        errors.append(f'{where}: missing {sorted(missing)}')
    if unknown:
        errors.append(f'{where}: unknown keys {sorted(unknown)} (typo? new keys need code support)')
    return not missing


def _extra_terms(errors, where, terms, reward):
    if not isinstance(terms, list):
        errors.append(f'{where} must be a list'); return
    names = set()
    for i, term in enumerate(terms):
        w = f'{where}[{i}]'
        if not _keys(errors, w, term, {'name', 'func', 'params'} | ({'weight'} if reward else set())):
            continue
        if not isinstance(term['name'], str) or not re.fullmatch(r'[a-z][a-z0-9_]{1,40}', term['name']) or term['name'] in names:
            errors.append(f'{w}.name must be a unique snake_case identifier')
        names.add(term['name'])
        if not isinstance(term['func'], str) or not re.fullmatch(r'[A-Za-z_][\w.]*:[A-Za-z_]\w*', term['func']):
            errors.append(f'{w}.func must look like "go2_stairs.student_terms.examples:function"')
        if reward and not _number(term['weight']):
            errors.append(f'{w}.weight must be a number')
        if not isinstance(term['params'], dict):
            errors.append(f'{w}.params must be an object')


def validate(profile):
    """Return the profile or raise ProfileError listing every problem found."""
    e = []
    if not _keys(e, 'profile file', profile, TOP_REQUIRED, TOP_OPTIONAL):
        raise ProfileError('; '.join(e))
    if not isinstance(profile['profile'], str) or not NAME.fullmatch(profile['profile']):
        e.append('profile name must match [A-Z][A-Z0-9_]{2,63}, e.g. STUDENT_TEAM1_NO_HEAD')
    fixes = profile['fixes']
    if _keys(e, 'fixes', fixes, set(FIX_KEYS)) and not all(isinstance(fixes[k], bool) for k in FIX_KEYS):
        e.append('fixes values must be true/false')
    a = profile['actuators']
    if _keys(e, 'actuators', a, {'stiffness', 'damping', 'groups'}, {'source'}):
        if not (_number(a['stiffness']) and a['stiffness'] > 0 and _number(a['damping']) and a['damping'] >= 0):
            e.append('actuators.stiffness must be >0 and damping >=0')
        owners = {j: 0 for j in JOINTS}
        for name, g in (a['groups'].items() if isinstance(a['groups'], dict) else []):
            if not _keys(e, f'actuators.groups.{name}', g, {'joints', 'effort_limit', 'velocity_limit'}):
                continue
            if not all(_number(g[k]) and g[k] > 0 for k in ('effort_limit', 'velocity_limit')):
                e.append(f'actuators.groups.{name}: limits must be positive numbers')
            for j in JOINTS:
                owners[j] += any(re.fullmatch(p, j) for p in g['joints'])
        if any(v != 1 for v in owners.values()):
            e.append(f'actuators.groups must cover each of the 12 joints exactly once: {owners}')
    r = profile['rewards']
    if _keys(e, 'rewards', r, {'feet_air_time_weight', 'leg_contact', 'feet_stumble'}, {'extra_terms'}):
        if not _number(r['feet_air_time_weight']):
            e.append('rewards.feet_air_time_weight must be a number')
        if _keys(e, 'rewards.leg_contact', r['leg_contact'], {'bodies', 'weight', 'threshold_N'}):
            if not (isinstance(r['leg_contact']['bodies'], list) and r['leg_contact']['bodies']):
                e.append('rewards.leg_contact.bodies must be a non-empty list of regexes')
            if not (_number(r['leg_contact']['weight']) and r['leg_contact']['weight'] <= 0 and _number(r['leg_contact']['threshold_N'])):
                e.append('rewards.leg_contact.weight must be <=0 (a penalty) and threshold_N a number')
        s = r['feet_stumble']
        if _keys(e, 'rewards.feet_stumble', s, {'weight', 'horizontal_to_vertical_ratio', 'min_horizontal_force_N'}):
            if not (_number(s['weight']) and s['weight'] <= 0 and _number(s['horizontal_to_vertical_ratio'])
                    and s['horizontal_to_vertical_ratio'] > 0 and _number(s['min_horizontal_force_N'])):
                e.append('rewards.feet_stumble: weight <=0, ratio >0, min force a number')
        if 'extra_terms' in r:
            _extra_terms(e, 'rewards.extra_terms', r['extra_terms'], True)
    t = profile['terminations']
    if _keys(e, 'terminations', t, {'head_contact'}, {'extra_terms'}):
        if _keys(e, 'terminations.head_contact', t['head_contact'], {'bodies', 'threshold_N'}):
            if not (_number(t['head_contact']['threshold_N']) and t['head_contact']['threshold_N'] > 0):
                e.append('terminations.head_contact.threshold_N must be >0')
        if 'extra_terms' in t:
            _extra_terms(e, 'terminations.extra_terms', t['extra_terms'], False)
    g = profile['terrain']
    if _keys(e, 'terrain', g, {'num_rows', 'num_cols', 'border_width_m', 'stair_step_height_range_m', 'stair_step_width_m', 'proportions'}):
        p = g['proportions']
        if not isinstance(p, dict) or set(p) != set(NATIVE_FAMILIES) or not all(_number(v) and v > 0 for v in p.values()):
            e.append(f'terrain.proportions must give a positive share for exactly {list(NATIVE_FAMILIES)}')
        elif abs(sum(p.values())-1) > 1e-6:
            e.append('terrain.proportions must sum to 1')
        elif isinstance(g['num_cols'], int) and len(set(column_families(p, g['num_cols']))) != len(p):
            e.append('terrain.num_cols too small: some terrain family would get no column')
        if not (isinstance(g['num_rows'], int) and 2 <= g['num_rows'] <= 20 and isinstance(g['num_cols'], int) and 6 <= g['num_cols'] <= 60):
            e.append('terrain.num_rows must be 2..20 and num_cols 6..60 (integers)')
        lo_hi = g['stair_step_height_range_m']
        if not (isinstance(lo_hi, list) and len(lo_hi) == 2 and all(map(_number, lo_hi)) and 0 <= lo_hi[0] < lo_hi[1] <= .35):
            e.append('terrain.stair_step_height_range_m must be [low, high] with 0 <= low < high <= 0.35')
        if not (_number(g['stair_step_width_m']) and .2 <= g['stair_step_width_m'] <= .5):
            e.append('terrain.stair_step_width_m must be 0.2..0.5')
        if not (_number(g['border_width_m']) and g['border_width_m'] >= 20):
            e.append('terrain.border_width_m must be >= 20 (height rays must not miss the map)')
    c = profile['commands']
    if _keys(e, 'commands', c, {'lin_vel_x', 'lin_vel_y', 'ang_vel_z'}):
        for k, v in c.items():
            if not (isinstance(v, list) and len(v) == 2 and all(map(_number, v)) and -2 <= v[0] < v[1] <= 2):
                e.append(f'commands.{k} must be [low, high] within -2..2')
    o = profile['ppo']
    if _keys(e, 'ppo', o, PPO_REQUIRED, PPO_OPTIONAL):
        if o['schedule'] not in ('adaptive', 'fixed') or not (_number(o['desired_kl']) and o['desired_kl'] > 0):
            e.append('ppo.schedule must be adaptive/fixed and desired_kl > 0')
        for k in ('public_init_learning_rate', 'scratch_learning_rate', 'critic_warmup_learning_rate', 'finetune_learning_rate'):
            if k in o and not (_number(o[k]) and 0 < o[k] <= 1e-2):
                e.append(f'ppo.{k} must be in (0, 0.01]')
        for k in ('critic_warmup_updates_public_init', 'critic_warmup_updates_finetune'):
            if k in o and not (isinstance(o[k], int) and not isinstance(o[k], bool) and 0 <= o[k] <= 1000):
                e.append(f'ppo.{k} must be an integer 0..1000')
    if e:
        raise ProfileError('; '.join(e))
    return profile


def parse_profile(raw):
    """bytes -> (profile, sha256). Rejects an edited copy that still claims the frozen reference name."""
    try:
        profile = json.loads(raw.decode('utf-8'))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProfileError(f'profile is not valid UTF-8 JSON: {error}') from error
    validate(profile)
    digest = hashlib.sha256(raw).hexdigest()
    if profile['profile'] == REFERENCE_PROFILE and digest != REFERENCE_SHA256:
        raise ProfileError('STAIRS25_V1 is the frozen reference profile; restore configs/stairs25-v1.json and put '
                           'changes in a copy with a new name: python -m go2_stairs.stairs25_profile new <NAME>')
    return profile, digest


def read_profile(path=None):
    """path -> (profile, sha256, utf-8 text)."""
    raw = Path(path or reference_path()).read_bytes()
    profile, digest = parse_profile(raw)
    return profile, digest, raw.decode('utf-8')


def load_profile(path=None):
    return read_profile(path)[0]


def profile_digest(path=None):
    return read_profile(path)[1]


def is_stairs25_info(info):
    return info.get('family') == FAMILY or info.get('profile') == REFERENCE_PROFILE


def profile_from_checkpoint_info(info):
    """Recover the exact training profile from a checkpoint's infos -> (profile, sha256, text)."""
    if 'profile_config' in info:
        raw = info['profile_config'].encode('utf-8')
        profile, digest = parse_profile(raw)
        if digest != info.get('profile_config_sha256') or profile['profile'] != info.get('profile'):
            raise ProfileError('embedded profile does not match the checkpoint record')
        return profile, digest, info['profile_config']
    if info.get('profile') == REFERENCE_PROFILE and info.get('profile_config_sha256') == REFERENCE_SHA256:
        # Reference checkpoints predate embedding; the frozen file must still be intact.
        return read_profile(reference_path())
    raise ProfileError('checkpoint does not carry a verifiable STAIRS25 profile')


def actuator_groups(profile):
    """Plain-data DC motor groups; saturation equals the joint's peak torque."""
    a = profile['actuators']
    return {name: dict(joint_names_expr=list(g['joints']), effort_limit=float(g['effort_limit']),
                       saturation_effort=float(g['effort_limit']), velocity_limit=float(g['velocity_limit']),
                       stiffness=float(a['stiffness']), damping=float(a['damping']), friction=0.)
            for name, g in a['groups'].items()}


def column_families(proportions, num_cols):
    """Same column rule as isaaclab TerrainGenerator._generate_curriculum_terrains."""
    names = list(proportions)
    cumulative = np.cumsum(np.array([proportions[n] for n in names], dtype=float)/sum(proportions.values()))
    return [names[int(np.min(np.where(i/num_cols+.001 < cumulative)[0]))] for i in range(num_cols)]


def stair_height_at_level(level, height_range, num_rows):
    """Nominal riser height at the centre of a curriculum row (the generator adds U(0,1) per row)."""
    low, high = height_range
    return low+(high-low)*(float(level)+.5)/num_rows


def diff(reference, candidate, prefix=''):
    """Flat list of (key, reference value, candidate value) for every changed leaf."""
    rows = []
    for key in sorted(set(reference) | set(candidate) if isinstance(reference, dict) and isinstance(candidate, dict) else []):
        a, b = reference.get(key, '<absent>'), candidate.get(key, '<absent>')
        path = f'{prefix}.{key}' if prefix else key
        if isinstance(a, dict) and isinstance(b, dict):
            rows += diff(a, b, path)
        elif a != b:
            rows.append((path, a, b))
    return rows


def new_profile(name, source=None, out_dir=None):
    """Copy a profile (reference by default) under a new name; never overwrites."""
    if not NAME.fullmatch(name) or name == REFERENCE_PROFILE:
        raise ProfileError('choose a new name matching [A-Z][A-Z0-9_]{2,63}, e.g. STUDENT_TEAM1_NO_HEAD')
    base = load_profile(source)
    profile = dict(profile=name, parent_profile=base['profile'], description='TODO: one sentence on what you changed and why',
                   **{k: v for k, v in base.items() if k not in ('profile', 'parent_profile', 'description', 'authorization', 'plan')})
    validate(profile)
    path = Path(out_dir or students_dir())/f'{name.lower()}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise ProfileError(f'{path} already exists')
    path.write_text(json.dumps(profile, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    p_new = sub.add_parser('new', help='copy a profile under a new name into configs/students/')
    p_new.add_argument('name'); p_new.add_argument('--from', dest='source', type=Path)
    p_check = sub.add_parser('check', help='validate a profile and show what differs from the reference')
    p_check.add_argument('path', type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == 'new':
            print(new_profile(args.name, args.source)); return 0
        profile, digest, _ = read_profile(args.path)
        print(f'OK  {profile["profile"]}  sha256={digest}')
        changes = [row for row in diff(load_profile(), profile) if row[0] not in ('profile', 'parent_profile', 'description', 'author', 'plan', 'authorization')]
        print(f'{len(changes)} change(s) vs {REFERENCE_PROFILE}:')
        for key, a, b in changes:
            print(f'  {key}: {json.dumps(a)} -> {json.dumps(b)}')
        return 0
    except ProfileError as error:
        print(f'INVALID: {error}', file=sys.stderr); return 2


if __name__ == '__main__':
    raise SystemExit(main())
