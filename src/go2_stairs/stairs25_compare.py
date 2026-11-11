"""Compare STAIRS25 evaluation matrices side by side, with Wilson intervals and Fisher exact tests.

    python -m go2_stairs.stairs25_compare reports/ref/MATRIX.json reports/mine/MATRIX.json --labels reference mine

With 20 cases per cell, differences of one or two successes are noise; the p-value column (first run vs
each other run, two-sided Fisher exact test on completed counts) makes that visible. Matrices from
different splits or case counts are flagged because they are not directly comparable.
"""
import argparse
import json
from math import comb
from pathlib import Path

from .stairs25_matrix import wilson


def fisher_exact(a_success, a_n, b_success, b_n):
    """Two-sided Fisher exact p-value for a 2x2 table of successes/failures."""
    total_success = a_success+b_success; total = a_n+b_n
    def probability(x):
        return comb(a_n, x)*comb(b_n, total_success-x)/comb(total, total_success)
    low, high = max(0, total_success-b_n), min(a_n, total_success)
    observed = probability(a_success)
    return min(1., sum(p for p in map(probability, range(low, high+1)) if p <= observed*(1+1e-9)))


def load(path):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    if data.get('status') != 'MATRIX_COMPLETE':
        raise ValueError(f'{path} is not a completed matrix (status {data.get("status")})')
    return data


def table(matrices, labels):
    cells = []
    for m in matrices:
        cells += [c for c in m['cells'] if c not in cells]
    head = '| cell | ' + ' | '.join(labels) + ''.join(f' | p ({labels[0]} vs {x})' for x in labels[1:]) + ' |'
    rows = [head, '|' + '---|'*(1+len(labels)+len(labels)-1)]
    for cell in cells:
        values, ps = [], []
        for m in matrices:
            c = m['cells'].get(cell)
            if c is None:
                values.append('—'); continue
            lo, hi = wilson(c['completed'], c['n'])
            values.append(f"{c['completed']}/{c['n']} ({lo:.2f}–{hi:.2f}) contact {c['head_or_body_contact']}")
        base = matrices[0]['cells'].get(cell)
        for m in matrices[1:]:
            other = m['cells'].get(cell)
            ps.append('—' if base is None or other is None else f"{fisher_exact(base['completed'], base['n'], other['completed'], other['n']):.3f}")
        rows.append(f'| {cell} | ' + ' | '.join(values) + ''.join(f' | {p}' for p in ps) + ' |')
    return '\n'.join(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('matrices', nargs='+', type=Path)
    parser.add_argument('--labels', nargs='+')
    args = parser.parse_args(argv)
    labels = args.labels or [p.parent.name for p in args.matrices]
    if len(labels) != len(args.matrices):
        raise SystemExit('give one label per matrix')
    matrices = [load(p) for p in args.matrices]
    splits = {m.get('split', 'dev') for m in matrices}; counts = {m.get('cases_per_cell') for m in matrices}
    if len(splits) > 1 or len(counts) > 1:
        print(f'WARNING: not directly comparable (splits {sorted(splits)}, cases per cell {sorted(counts)})\n')
    for label, m in zip(labels, matrices):
        meta = m['checkpoint_meta']
        print(f"- {label}: {meta['source_kind']} {meta.get('profile') or ''} sha256 {meta['sha256'][:12]} split={m.get('split', 'dev')}")
    print()
    print(table(matrices, labels))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
