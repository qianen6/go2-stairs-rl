"""STAIRS25 evaluation matrix: flat plus 10/15/20/25 cm up/down through the INTERIOR_STOP_V1 judge.

Never trains. Each cell runs go2_stairs.landing_eval in its own process. STAIRS25 checkpoints are
evaluated with their own training actuators; the public checkpoint keeps its release-compat profile,
as in the earlier landing confirmation. Completion counts come only from summary.json.

Two case splits share the protocol but use different initial conditions:
  dev   iterate and compare freely (the reference results in the lab manual are on this split);
  test  report once per final model; never tune on it.

    python -m go2_stairs.stairs25_matrix --checkpoint <model.pt> --output reports/x-dev
    python -m go2_stairs.stairs25_matrix --checkpoint <model.pt> --output reports/x-quick --cells up20 up25 --count 8
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

STANDARD_HEIGHTS = (.1, .15, .2, .25)
SPLIT_SEED_OFFSET = {'dev': 0, 'test': 100000}


def wilson(successes, n, z=1.96):
    if n == 0:
        return [0., 1.]
    p = successes/n
    centre = (p+z*z/(2*n))/(1+z*z/n)
    half = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return [max(0., centre-half), min(1., centre+half)]


def matrix_cells(heights, count, only=None):
    if count < 2 or count % 2 or count > 32:
        raise ValueError('cases per cell must be even and at most32')
    cells = [('flat', 'flat', 0.)]
    for height in heights:
        for direction in ('up', 'down'):
            cells.append((f'{direction}{round(height*100):02}', direction, float(height)))
    if only:
        unknown = set(only)-{c[0] for c in cells}
        if unknown:
            raise ValueError(f'unknown cells {sorted(unknown)}; available: {[c[0] for c in cells]}')
        cells = [c for c in cells if c[0] in only]
    return cells


def cell_metrics(results):
    n = len(results)
    completed = sum(bool(x['completed']) for x in results)
    reasons = {}
    for x in results:
        reasons[x['reason']] = reasons.get(x['reason'], 0)+1
    return dict(n=n, completed=completed, clean=sum(bool(x['clean']) for x in results),
                crossed=sum(bool(x['crossed']) for x in results), invalid=sum(not x['valid'] for x in results),
                head_or_body_contact=sum(bool(x['quality']['body_contact']) for x in results),
                completed_wilson95=wilson(completed, n), reasons=reasons)


def main(argv=None):
    from .guarded_finetune import verify_finetune_checkpoint
    from .machine import runtime_cwd
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split', choices=sorted(SPLIT_SEED_OFFSET), default='dev')
    parser.add_argument('--count', type=int, default=20, help='cases per cell, even, at most 32 (half at 0.8 m/s, half at 1.0 m/s)')
    parser.add_argument('--heights', type=float, nargs='+', help='default 0.10 0.15 0.20 0.25; 0.05 and 0.30 are also available')
    parser.add_argument('--cells', nargs='+', help='subset such as up20 up25 for a quick development loop')
    parser.add_argument('--duration', type=float, default=30.)
    parser.add_argument('--no-video', action='store_true')
    args = parser.parse_args(argv)
    cells = matrix_cells(args.heights or STANDARD_HEIGHTS, args.count, args.cells)
    checkpoint = args.checkpoint.resolve()
    meta = verify_finetune_checkpoint(checkpoint)
    flags = [] if meta['source_kind'] == 'STAIRS25_TRAINED' else ['--release-velocity-limits', '--release-stabilization']
    flags += ['--seed-offset', str(SPLIT_SEED_OFFSET[args.split])]
    root = args.output.resolve(); root.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[2]
    env = dict(os.environ, PYTHONPATH=str(repo/'src'), PYTHONIOENCODING='utf-8', PYTHONUNBUFFERED='1', OMNI_KIT_ACCEPT_EULA='YES')
    sources = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (repo/'src/go2_stairs').glob('*.py')}
    manifest = dict(status='RUNNING', training=False, protocol='INTERIOR_STOP_V1', split=args.split, checkpoint=str(checkpoint),
                    checkpoint_meta=meta, profile=meta.get('profile'), extra_flags=flags,
                    cases_per_cell=args.count, source_sha256=sources,
                    start_utc=datetime.now(timezone.utc).isoformat(), commands=[], cells={})
    def save():
        temp = root/'MATRIX.tmp'; temp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
        temp.replace(root/'MATRIX.json')
    save()
    try:
        for name, direction, height in cells:
            if {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (repo/'src/go2_stairs').glob('*.py')} != sources:
                raise ValueError('source changed during evaluation matrix')
            command = [sys.executable, '-m', 'go2_stairs.landing_eval', '--checkpoint', str(checkpoint),
                       '--output', str(root/name), '--direction', direction, '--riser-height', str(height),
                       '--count', str(args.count), '--duration', str(args.duration), '--headless', *flags]
            if not args.no_video and height >= .2:
                command += ['--record', '0']
            item = dict(cell=name, command=command, log=str(root/(name+'.log')), start_utc=datetime.now(timezone.utc).isoformat())
            manifest['commands'].append(item); save()
            with open(item['log'], 'w', encoding='utf-8') as stream:
                proc = subprocess.Popen(command, cwd=str(runtime_cwd()), env=env, stdout=stream, stderr=subprocess.STDOUT)
                try:
                    code = proc.wait(timeout=1200)
                except BaseException:
                    proc.kill(); proc.wait(); raise
            item.update(exit_status=code, end_utc=datetime.now(timezone.utc).isoformat()); save()
            summary = json.loads((root/name/'summary.json').read_text(encoding='utf-8'))
            if code or summary.get('status') != 'RETEST_COMPLETE':
                raise RuntimeError(f'evaluation failed: {name} (see {item["log"]})')
            if summary['checkpoint']['sha256'] != meta['sha256']:
                raise ValueError('evaluated checkpoint differs from requested checkpoint')
            manifest['cells'][name] = cell_metrics(summary['results']); save()
            print(name, json.dumps(manifest['cells'][name]), flush=True)
        manifest.update(status='MATRIX_COMPLETE', end_utc=datetime.now(timezone.utc).isoformat()); save()
        lines = [f"STAIRS25 evaluation matrix | {meta['source_kind']} {meta.get('profile') or ''} | {meta['sha256'][:12]} | "
                 f"split={args.split} | {args.count} cases/cell"]
        for name, m in manifest['cells'].items():
            lo, hi = m['completed_wilson95']
            lines.append(f"{name:>7}: completed {m['completed']:>2}/{m['n']} (95% {lo:.2f}-{hi:.2f})  clean {m['clean']:>2}  "
                         f"body/head contact {m['head_or_body_contact']:>2}  invalid {m['invalid']}  {m['reasons']}")
        (root/'MATRIX.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')
        print('\n'.join(lines), flush=True)
        return 0
    except BaseException as error:
        manifest.update(status='MATRIX_FAILED', error=repr(error), end_utc=datetime.now(timezone.utc).isoformat()); save()
        raise


if __name__ == '__main__':
    raise SystemExit(main())
