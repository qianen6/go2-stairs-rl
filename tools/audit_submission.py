"""Check student evaluation matrices against the kit: frozen sources, completed status, split and case counts.

    python tools/audit_submission.py <MATRIX.json> [<MATRIX.json> ...] [--kit <kit root>]

FAIL if any frozen file under src/go2_stairs/ (everything except student_terms/) differs from the kit's
KIT_SOURCE_SHA256.json, if the matrix is incomplete, or if the checkpoint named in it (when present on this
machine) no longer matches. Matrices made before source hashes were recorded are reported as WARN.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

STANDARD_CELLS = {'flat', 'up10', 'down10', 'up15', 'down15', 'up20', 'down20', 'up25', 'down25'}


def audit(matrix_path, kit_sources):
    problems, warnings = [], []
    m = json.loads(Path(matrix_path).read_text(encoding='utf-8'))
    if m.get('status') != 'MATRIX_COMPLETE':
        problems.append(f"status {m.get('status')}")
    recorded = m.get('source_sha256')
    if recorded is None:
        warnings.append('no source hashes recorded (matrix predates auditing)')
    else:
        changed = sorted(n for n, h in kit_sources.items() if recorded.get(n) != h)
        added = sorted(set(recorded)-set(kit_sources))
        if changed:
            problems.append(f'frozen source differs from kit: {changed}')
        if added:
            problems.append(f'extra core source files: {added}')
    cells = set(m.get('cells', {}))
    if cells != STANDARD_CELLS:
        warnings.append(f"non-standard cells {sorted(cells)} (fine for development, not for claims)")
    if m.get('cases_per_cell') != 20:
        warnings.append(f"{m.get('cases_per_cell')} cases per cell (claims need 20)")
    checkpoint = Path(m.get('checkpoint', ''))
    if checkpoint.is_file():
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        if digest != m['checkpoint_meta']['sha256']:
            problems.append('checkpoint file changed since evaluation')
    else:
        warnings.append('checkpoint not on this machine; hash not re-checked')
    return m, problems, warnings


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('matrices', nargs='+', type=Path)
    parser.add_argument('--kit', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)
    kit_sources = json.loads((args.kit/'KIT_SOURCE_SHA256.json').read_text(encoding='utf-8'))
    failed = False
    for path in args.matrices:
        m, problems, warnings = audit(path, kit_sources)
        meta = m.get('checkpoint_meta', {})
        label = f"{path}  [{meta.get('profile') or meta.get('source_kind')} {str(meta.get('sha256'))[:12]} split={m.get('split', 'dev')}]"
        print(('FAIL ' if problems else 'OK   ')+label)
        for p in problems:
            print('   problem:', p)
        for w in warnings:
            print('   warning:', w)
        failed |= bool(problems)
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
