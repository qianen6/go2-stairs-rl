from collections import Counter
import importlib
import unittest


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        try:
            self.m = importlib.import_module('go2_stairs.evaluate')
        except ModuleNotFoundError:
            self.fail('Evaluation implementation is missing')

    def test_core_suite_has_all_120_cases_and_unique_seeds(self):
        cases = self.m.build_core_suite()
        self.assertEqual(len(cases), 120)
        counts = Counter((c['height'], c['depth'], c['steps'], c['direction']) for c in cases)
        self.assertEqual(len(counts), 24)
        self.assertEqual(set(counts.values()), {5})
        self.assertEqual(len({c['seed'] for c in cases}), 120)
        self.assertEqual({c['height'] for c in cases}, {.04, .06, .08})

    def rows(self):
        return [dict(case_id=c['case_id'], success=True, reason='success', elapsed_s=10.)
                for c in self.m.build_core_suite()]

    def test_all_pass_reports_separate_directions(self):
        report = self.m.summarize(self.m.build_core_suite(), self.rows())
        self.assertTrue(report['passed'])
        self.assertEqual(report['directions']['up']['episodes'], 60)
        self.assertEqual(report['directions']['down']['successes'], 60)

    def test_missing_and_error_cases_stay_in_denominator(self):
        rows = self.rows()[1:]
        rows[0].update(success=False, reason='runtime_error')
        report = self.m.summarize(self.m.build_core_suite(), rows)
        self.assertEqual(report['episodes'], 120)
        self.assertEqual(report['successes'], 118)
        self.assertEqual(report['failure_reasons']['missing_result'], 1)
        self.assertEqual(report['failure_reasons']['runtime_error'], 1)
        self.assertFalse(report['passed'])

    def test_bad_group_fails_even_if_aggregate_above_90(self):
        rows = self.rows()
        rows[0].update(success=False, reason='fall')
        rows[1].update(success=False, reason='fall')
        report = self.m.summarize(self.m.build_core_suite(), rows)
        self.assertGreater(report['successes'] / report['episodes'], .9)
        self.assertFalse(report['passed'])

    def test_duplicate_unknown_and_nonboolean_results_rejected(self):
        cases = self.m.build_core_suite()
        rows = self.rows()
        with self.assertRaises(ValueError):
            self.m.summarize(cases, rows + [rows[0]])
        rows[0]['case_id'] = 'unknown'
        with self.assertRaises(ValueError):
            self.m.summarize(cases, rows)
        rows = self.rows()
        rows[0]['success'] = 'false'
        with self.assertRaises(ValueError):
            self.m.summarize(cases, rows)

    def test_curriculum_uses_weaker_direction(self):
        self.assertEqual(self.m.next_level(1, .95, .4), 0)
        self.assertEqual(self.m.next_level(1, .95, .7), 1)
        self.assertEqual(self.m.next_level(1, .95, .9), 2)
        self.assertEqual(self.m.next_level(3, 1., 1.), 3)
        with self.assertRaises(ValueError):
            self.m.next_level(1, float('nan'), 1.)

    def test_truncated_manifest_never_passes_core_gate(self):
        core = self.m.build_core_suite()
        cases = [core[0], core[5]]
        rows = [dict(case_id=c['case_id'], success=True, reason='success', elapsed_s=1.) for c in cases]
        report = self.m.summarize(cases, rows)
        self.assertFalse(report['passed'])

    def test_changed_geometry_does_not_pass_frozen_core_gate(self):
        cases = self.m.build_core_suite()
        cases[0]['height'] = .01
        report = self.m.summarize(cases, self.rows())
        self.assertFalse(report['passed'])


if __name__ == '__main__':
    unittest.main()
