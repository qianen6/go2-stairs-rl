import json
import re
import tempfile
import unittest
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]


def fake_weights():
    weights = {'std': torch.full((12,), .5)}
    for name, dims in [('actor', [235, 512, 256, 128, 12]), ('critic', [235, 512, 256, 128, 1])]:
        for i, (a, b) in enumerate(zip(dims, dims[1:])):
            weights[f'{name}.{i*2}.weight'] = torch.zeros(b, a)
            weights[f'{name}.{i*2}.bias'] = torch.zeros(b)
    return weights


def legacy_infos(updates=3, **overrides):
    """Infos exactly as written by the reference STAIRS25_V1 runs (no embedded profile)."""
    from go2_stairs.stairs25_profile import REFERENCE_SHA256, load_profile
    from go2_stairs.checkpoints import capture_rng_state
    infos = dict(profile='STAIRS25_V1', profile_config_sha256=REFERENCE_SHA256, fixes=load_profile()['fixes'],
                 init_kind='public', origin_sha256=load_profile()['public_sha256'], updates=updates,
                 critic_warmup_updates=50, training_seed=0, num_envs=64, terrain_levels=[0]*64, rng=capture_rng_state())
    infos.update(overrides)
    return infos


def embedded_infos(text, updates=3, **overrides):
    import hashlib
    profile = json.loads(text)
    infos = legacy_infos(updates, family='STAIRS25', profile=profile['profile'], profile_config=text,
                         profile_config_sha256=hashlib.sha256(text.encode('utf-8')).hexdigest(), fixes=profile['fixes'])
    infos.update(overrides)
    return infos


def save_checkpoint(path, infos):
    torch.save(dict(model_state_dict=fake_weights(), optimizer_state_dict={'state': {}, 'param_groups': []},
                    iter=infos['updates']-1, infos=infos), path)


def student_text(name='STUDENT_TEST', **changes):
    from go2_stairs.stairs25_profile import load_profile
    profile = load_profile(); profile['profile'] = name; profile['parent_profile'] = 'STAIRS25_V1'
    for key, value in changes.items():
        section, field = key.split('__')
        profile[section][field] = value
    return json.dumps(profile, indent=2)


class ReferenceProfileTests(unittest.TestCase):
    def test_reference_is_frozen_and_valid(self):
        from go2_stairs.stairs25_profile import REFERENCE_SHA256, read_profile
        profile, digest, _ = read_profile()
        self.assertEqual((profile['profile'], digest), ('STAIRS25_V1', REFERENCE_SHA256))

    def test_actuator_limits_match_unitree_model(self):
        from go2_stairs.stairs25_profile import actuator_groups, load_profile
        tree = ET.parse(ROOT/'assets/go2/go2.xml')
        ranges = sorted({abs(float(m.get('ctrlrange').split()[1])) for m in tree.iter('motor') if m.get('ctrlrange')})
        self.assertEqual(ranges, [23.7, 45.43])
        groups = actuator_groups(load_profile())
        self.assertEqual((groups['calf']['effort_limit'], groups['hip_thigh']['effort_limit']), (45.43, 23.7))
        joints = [f'{leg}_{part}_joint' for leg in ('FL', 'FR', 'RL', 'RR') for part in ('hip', 'thigh', 'calf')]
        owners = Counter(name for j in joints for name, g in groups.items() if any(re.fullmatch(p, j) for p in g['joint_names_expr']))
        self.assertEqual(owners, Counter(hip_thigh=8, calf=4))

    def test_terrain_columns_and_target_heights(self):
        from go2_stairs.stairs25_profile import column_families, stair_height_at_level, load_profile
        t = load_profile()['terrain']
        self.assertEqual(Counter(column_families(t['proportions'], t['num_cols'])),
                         Counter(pyramid_stairs_inv=7, pyramid_stairs=5, boxes=3, random_rough=3, hf_pyramid_slope=1, hf_pyramid_slope_inv=1))
        self.assertGreater(stair_height_at_level(t['num_rows']-1, t['stair_step_height_range_m'], t['num_rows']), .25)

    def test_stumble_counts_only_mostly_horizontal_feet(self):
        from go2_stairs.stairs25 import stumble_count
        forces = torch.tensor([[[10., 0., 1.], [3., 0., 0.], [10., 0., 20.], [0., 8., -2.]]])
        self.assertEqual(stumble_count(forces, 2., 5.).tolist(), [2.])


class StudentProfileTests(unittest.TestCase):
    def test_new_profile_is_a_renamed_valid_copy(self):
        from go2_stairs.stairs25_profile import ProfileError, diff, load_profile, new_profile, read_profile
        with tempfile.TemporaryDirectory() as tmp:
            path = new_profile('STUDENT_TEAM1_NO_HEAD', out_dir=tmp)
            self.assertEqual(path.name, 'student_team1_no_head.json')
            profile, _, _ = read_profile(path)
            changed = {k for k, _, _ in diff(load_profile(), profile)}
            self.assertEqual(changed, {'profile', 'parent_profile', 'description', 'plan', 'authorization'})
            with self.assertRaises(ProfileError):
                new_profile('STUDENT_TEAM1_NO_HEAD', out_dir=tmp)
        with self.assertRaises(ProfileError):
            new_profile('STAIRS25_V1')

    def test_editing_the_reference_in_place_is_rejected(self):
        from go2_stairs.stairs25_profile import ProfileError, parse_profile, reference_path
        edited = reference_path().read_bytes().replace(b'"weight": -1.0', b'"weight": -2.0')
        with self.assertRaisesRegex(ProfileError, 'frozen reference'):
            parse_profile(edited)

    def test_validation_reports_typos_and_bad_values(self):
        from go2_stairs.stairs25_profile import ProfileError, parse_profile
        bad = json.loads(student_text(terrain__proportions={'pyramid_stairs': 1.0}))
        bad['rewards']['feet_air_time_weigth'] = .1
        bad['actuators']['groups'].pop('calf')
        with self.assertRaises(ProfileError) as ctx:
            parse_profile(json.dumps(bad).encode())
        message = str(ctx.exception)
        for fragment in ('feet_air_time_weigth', 'terrain.proportions', 'exactly once'):
            self.assertIn(fragment, message)

    def test_extra_terms_schema(self):
        from go2_stairs.stairs25_profile import ProfileError, parse_profile
        profile = json.loads(student_text())
        profile['rewards']['extra_terms'] = [dict(name='leg_force', func='go2_stairs.student_terms.examples:contact_force_excess',
                                                  weight=-.01, params={'threshold': 5.})]
        parse_profile(json.dumps(profile).encode())
        profile['rewards']['extra_terms'][0]['func'] = 'not a function'
        with self.assertRaises(ProfileError):
            parse_profile(json.dumps(profile).encode())

    def test_example_terms_import(self):
        from go2_stairs.stairs25 import resolve_function
        self.assertTrue(callable(resolve_function('go2_stairs.student_terms.examples:contact_force_excess')))


class CheckpointTests(unittest.TestCase):
    def test_reference_checkpoints_still_verify(self):
        from go2_stairs.stairs25 import verify_checkpoint
        from go2_stairs.guarded_finetune import verify_finetune_checkpoint
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'model_2.pt'; save_checkpoint(path, legacy_infos())
            meta = verify_checkpoint(path)
            self.assertEqual((meta['profile'], meta['finetune_updates'], meta['actuators']['calf']['effort_limit']), ('STAIRS25_V1', 3, 45.43))
            self.assertEqual(verify_finetune_checkpoint(path)['source_kind'], 'STAIRS25_TRAINED')

    def test_student_checkpoint_carries_its_own_profile(self):
        from go2_stairs.stairs25 import verify_checkpoint
        text = student_text()
        changed = json.loads(text); changed['actuators']['groups']['calf']['effort_limit'] = 40.
        text = json.dumps(changed, indent=2)
        with tempfile.TemporaryDirectory() as tmp:
            good = Path(tmp)/'good.pt'; save_checkpoint(good, embedded_infos(text))
            meta = verify_checkpoint(good)
            self.assertEqual((meta['profile'], meta['actuators']['calf']['effort_limit']), ('STUDENT_TEST', 40.))
            tampered = Path(tmp)/'tampered.pt'
            save_checkpoint(tampered, embedded_infos(text, profile_config=text.replace('40.0', '45.0')))
            with self.assertRaises(ValueError):
                verify_checkpoint(tampered)
            stale = Path(tmp)/'stale.pt'; save_checkpoint(stale, legacy_infos(profile_config_sha256='0'*64))
            with self.assertRaises(ValueError):
                verify_checkpoint(stale)


class InitializationTests(unittest.TestCase):
    def setUp(self):
        from go2_stairs.stairs25_profile import read_profile, parse_profile
        self.ref, self.ref_digest, _ = read_profile()
        self.student, self.student_digest = parse_profile(student_text(terrain__stair_step_width_m=.26).encode())

    def test_public_and_scratch(self):
        from go2_stairs.stairs25 import plan_initialization
        public = plan_initialization('public', self.ref, self.ref_digest, 'p')
        self.assertEqual((public['learning_rate'], public['critic_warmup'], public['post_warmup_learning_rate']), (1e-4, 50, 1e-4))
        scratch = plan_initialization('scratch', self.ref, self.ref_digest, 'p')
        self.assertEqual((scratch['learning_rate'], scratch['critic_warmup'], scratch['origin_sha256']), (1e-3, 0, None))

    def test_resume_requires_same_profile_and_continues_counters(self):
        from go2_stairs.stairs25 import plan_initialization
        info = legacy_infos(updates=600, num_envs=4096)
        plan = plan_initialization('resume', self.ref, self.ref_digest, 'p', info, 'sha', 4096)
        self.assertEqual((plan['base_updates'], plan['warmup_offset'], plan['post_warmup_learning_rate'], plan['restore_optimizer']), (600, 600, 1e-4, True))
        with self.assertRaises(ValueError):
            plan_initialization('resume', self.student, self.student_digest, 'p', info, 'sha', 4096)
        with self.assertRaises(ValueError):
            plan_initialization('resume', self.ref, self.ref_digest, 'p', info, 'sha', 1024)

    def test_finetune_restores_curriculum_only_for_identical_terrain(self):
        from go2_stairs.stairs25 import plan_initialization
        info = legacy_infos(updates=1500, num_envs=4096)
        same = json.loads(student_text()); same_digest = 'x'
        plan = plan_initialization('finetune', same, same_digest, 'p', info, 'sha', 4096)
        self.assertEqual((plan['base_updates'], plan['restore_curriculum'], plan['restore_optimizer'], plan['parent_profile']), (0, True, False, 'STAIRS25_V1'))
        self.assertFalse(plan_initialization('finetune', same, same_digest, 'p', info, 'sha', 1024)['restore_curriculum'])
        self.assertFalse(plan_initialization('finetune', self.student, self.student_digest, 'p', info, 'sha', 4096)['restore_curriculum'])


class CriticWarmupTests(unittest.TestCase):
    def test_warmup_freezes_actor_and_std_only(self):
        from go2_stairs.stairs25 import set_actor_trainable
        policy = torch.nn.Module()
        policy.actor = torch.nn.Sequential(torch.nn.Linear(4, 2)); policy.critic = torch.nn.Sequential(torch.nn.Linear(4, 1))
        policy.std = torch.nn.Parameter(torch.ones(2))
        before = {k: v.clone() for k, v in policy.state_dict().items()}
        optimizer = torch.optim.Adam(policy.parameters(), lr=.1)
        set_actor_trainable(policy, False)
        loss = policy.critic(torch.ones(1, 4)).sum()+policy.actor(torch.ones(1, 4)).sum()+policy.std.sum()
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        for k, v in policy.state_dict().items():
            self.assertEqual(torch.equal(before[k], v), not k.startswith('critic'), k)
        set_actor_trainable(policy, True)
        self.assertTrue(all(p.requires_grad for p in policy.parameters()))

    def test_schedule(self):
        from go2_stairs.stairs25 import warmup_schedule
        self.assertEqual([warmup_schedule(i, 2) for i in range(4)], ['critic_only', 'critic_only', 'full', 'full'])
        self.assertEqual(warmup_schedule(0, 0), 'full')


class EvaluationToolTests(unittest.TestCase):
    def test_case_seeds_are_disjoint_and_historical_seeds_unchanged(self):
        from go2_stairs.native_factory import native_cases
        self.assertEqual(native_cases(.2, 'up', 2)[0]['seed'], 62000)
        self.assertEqual(native_cases(.1, 'down', 2)[0]['seed'], 62000)
        seeds = {h: {c['seed'] for d in ('up', 'down') for c in native_cases(h, d, 32)} for h in (.05, .1, .15, .2, .25, .3)}
        new = [seeds[h] for h in (.05, .15, .25, .3)]
        self.assertEqual(len(set().union(*new)), sum(map(len, new)))
        self.assertFalse(set().union(*new) & (seeds[.1] | seeds[.2]))
        with self.assertRaises(ValueError):
            native_cases(.35, 'up', 4)

    def test_test_split_never_reuses_dev_seeds(self):
        from go2_stairs.fine_eval_factory import native_cases
        from go2_stairs.stairs25_matrix import SPLIT_SEED_OFFSET
        dev = {c['seed'] for h in (.1, .15, .2, .25) for d in ('up', 'down') for c in native_cases(h, d, 32)}
        dev |= {c['seed'] for c in native_cases(0., 'flat', 32)}
        self.assertFalse({s+SPLIT_SEED_OFFSET['test'] for s in dev} & dev)

    def test_matrix_cells_and_wilson(self):
        from go2_stairs.stairs25_matrix import matrix_cells, wilson
        self.assertEqual([c[0] for c in matrix_cells([.1, .25], 20)], ['flat', 'up10', 'down10', 'up25', 'down25'])
        self.assertEqual([c[0] for c in matrix_cells([.2, .25], 8, ['up20', 'up25'])], ['up20', 'up25'])
        with self.assertRaises(ValueError):
            matrix_cells([.2], 8, ['up30'])
        lo, hi = wilson(18, 20)
        self.assertTrue(.68 < lo < .7 and .97 < hi < .98)

    def test_fisher_exact(self):
        from go2_stairs.stairs25_compare import fisher_exact
        self.assertLess(fisher_exact(3, 20, 17, 20), 1e-3)
        self.assertAlmostEqual(fisher_exact(18, 20, 17, 20), 1.)
        self.assertAlmostEqual(fisher_exact(3, 20, 17, 20), fisher_exact(17, 20, 3, 20))

    def test_trace_classification_and_limits(self):
        from go2_stairs.trace_report import classify, joint_limits
        self.assertEqual(classify('timeout', 25., 20., False), 'stall_head_on_riser')
        self.assertEqual(classify('timeout', 25., 0., False), 'stall')
        self.assertEqual(classify('native_base_contact', 0., 0., True), 'fall_after_leg_catch')
        self.assertEqual(classify('native_base_contact', 0., 0., True, True), 'fall_after_head_contact')
        joints = [f'{leg}_{part}_joint' for part in ('hip', 'thigh', 'calf') for leg in ('FL', 'FR', 'RL', 'RR')]
        self.assertEqual(joint_limits({}, joints)[:, 0].tolist(), [23.5]*12)
        self.assertEqual(joint_limits({'actuator_profile': 'STAIRS25_V1'}, joints)[:, 0].tolist(), [23.7]*8+[45.43]*4)


if __name__ == '__main__':
    unittest.main()
