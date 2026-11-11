"""Bounded mixed-terrain fine-tune/evaluation blocks, never an unbounded training loop."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def development_decision(baseline, candidate, exposure):
    if set(candidate) != set(baseline) or any(x['invalid'] for x in candidate.values()):
        return 'STOP_INVALID_EVALUATION'
    if any(candidate[k]['n'] != baseline[k]['n'] for k in baseline):
        return 'STOP_INVALID_EVALUATION'
    for k in ['flat', 'up10', 'down10', 'down20']:
        tolerance = 2 if k == 'down20' else 1
        if any(candidate[k][metric] < max(0, baseline[k][metric]-tolerance) for metric in ['completed', 'clean']):
            return 'STOP_BASIC_REGRESSION'
    for k in ['pyramid_stairs_h20', 'pyramid_stairs_inv_h20']:
        counts = exposure['groups'][k]
        if counts['actual_tile_samples'] <= 0 or counts['target_riser_zone_samples'] <= 0:
            return 'STOP_NO_TARGET_EXPOSURE'
    return 'CONTINUE'


def source_snapshot():
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}


def main():
    from .mixed_finetune import profile_path, verify_finetune_checkpoint
    from .reference_reproduction import PUBLIC_SHA
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() != PUBLIC_SHA:
        raise ValueError('this pilot must start from the unchanged NVIDIA public checkpoint')
    profile = json.loads(profile_path().read_text(encoding='utf-8'))
    root = args.output.resolve(); root.mkdir(parents=True, exist_ok=False)
    source = source_snapshot()
    status = dict(status='STARTING', training_authorized=True, origin_sha256=PUBLIC_SHA,
                  budget_finetune_updates=profile['budget_finetune_updates'], completed_finetune_updates=0,
                  num_envs=profile['num_envs'], seed=profile['seed'], blocks=[], commands=[],
                  source_sha256=source, acceptance='NOT_RUN', start_utc=datetime.now(timezone.utc).isoformat())
    environment = dict(os.environ, OMNI_KIT_ACCEPT_EULA='YES', PYTHONIOENCODING='utf-8', PYTHONUNBUFFERED='1')
    environment['PYTHONPATH'] = str(Path(__file__).resolve().parents[1])
    def record():
        status['updated_utc'] = datetime.now(timezone.utc).isoformat()
        temp = root/'status.tmp'; temp.write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding='utf-8')
        temp.replace(root/'status.json')
    def run(command, log, timeout):
        if source_snapshot() != source:
            raise ValueError('source changed before child launch')
        item = dict(command=command, log=str(log), cwd=r'C:\rl-runtimes', start_utc=datetime.now(timezone.utc).isoformat())
        status['commands'].append(item); record()
        with log.open('w', encoding='utf-8') as stream:
            process = subprocess.Popen(command, cwd=r'C:\rl-runtimes', env=environment, stdout=stream, stderr=subprocess.STDOUT)
            status['active_child_pid'] = process.pid; record()
            try:
                code = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait(); code = -9
            except BaseException:
                if process.poll() is None:
                    process.kill(); process.wait()
                raise
        item.update(exit_status=code, end_utc=datetime.now(timezone.utc).isoformat())
        status.pop('active_child_pid', None); record()
        if code or source_snapshot() != source:
            raise RuntimeError(f'child failed or source changed: exit={code}; log={log}')
    def evaluate(checkpoint, folder):
        folder.mkdir()
        expected = verify_finetune_checkpoint(checkpoint)['sha256']
        results = {}
        for name, direction, height, count in [('flat', 'flat', 0, 8), ('up10', 'up', .1, 4), ('down10', 'down', .1, 4),
                                              ('up20', 'up', .2, 8), ('down20', 'down', .2, 8)]:
            output = folder/name
            command = [sys.executable, '-m', 'go2_stairs.fine_tune_eval', '--checkpoint', str(checkpoint),
                       '--output', str(output), '--direction', direction, '--riser-height', str(height),
                       '--count', str(count), '--duration', '30', '--headless',
                       '--release-velocity-limits', '--release-stabilization']
            status['active_evaluation'] = name; record()
            run(command, folder/(name+'.log'), 900)
            summary = json.loads((output/'summary.json').read_text(encoding='utf-8'))
            if summary['status'] != 'RETEST_COMPLETE' or summary['checkpoint']['sha256'] != expected:
                raise ValueError('evaluation status or checkpoint mismatch')
            rows = summary['results']
            results[name] = dict(n=len(rows), completed=sum(r['completed'] for r in rows),
                                 clean=sum(r['clean'] for r in rows), invalid=sum(not r['valid'] for r in rows))
        (folder/'summary.json').write_text(json.dumps(dict(checkpoint=str(checkpoint), sha256=expected,
            cases=results, purpose='DEVELOPMENT_NOT_FINAL_ACCEPTANCE'), indent=2), encoding='utf-8')
        return results
    record()
    try:
        status['status'] = 'BASELINE_EVALUATION'; record()
        baseline = evaluate(args.checkpoint.resolve(), root/'baseline')
        if any(x['invalid'] for x in baseline.values()):
            raise ValueError('invalid baseline evaluation')
        status['baseline'] = baseline; record()
        parent = args.checkpoint.resolve()
        while status['completed_finetune_updates'] < profile['budget_finetune_updates']:
            start = status['completed_finetune_updates']
            count = min(profile['block_updates'], profile['budget_finetune_updates']-start)
            block = root/f'block-{start:06}'; block.mkdir()
            train = block/'train'
            status.update(status='TRAINING', active_block=str(block), active_training_status=str(train/'status.json'))
            status.pop('active_evaluation', None); record()
            command = [sys.executable, '-m', 'go2_stairs.mixed_finetune', '--checkpoint', str(parent), '--output', str(train),
                       '--updates', str(count), '--num-envs', str(profile['num_envs']), '--seed', str(profile['seed']), '--headless']
            run(command, block/'train.log', 1500)
            trained = json.loads((train/'status.json').read_text(encoding='utf-8'))
            if trained['status'] != 'TRAINING_BLOCK_COMPLETE_EVALUATION_PENDING' or trained['completed_finetune_updates'] != start+count:
                raise ValueError('training did not produce the requested verified update count')
            parent = Path(trained['checkpoint'])
            metadata = verify_finetune_checkpoint(parent)
            if metadata['sha256'] != trained['checkpoint_metadata']['sha256']:
                raise ValueError('checkpoint changed before evaluation')
            status.update(status='DEVELOPMENT_EVALUATION', completed_finetune_updates=start+count,
                          checkpoint=str(parent), checkpoint_sha256=metadata['sha256'])
            record()
            candidate = evaluate(parent, block/'development')
            decision = development_decision(baseline, candidate, trained['exposure'])
            result = dict(finetune_updates=start+count, checkpoint=str(parent), metrics=candidate, decision=decision)
            status['blocks'].append(result); record(); print(json.dumps(result), flush=True)
            if decision != 'CONTINUE':
                status.update(status=decision, end_utc=datetime.now(timezone.utc).isoformat()); record()
                return 0
        status.update(status='PILOT_BUDGET_COMPLETE', end_utc=datetime.now(timezone.utc).isoformat())
        record(); return 0
    except BaseException as error:
        status.update(status='JOB_FAILED', error=repr(error), end_utc=datetime.now(timezone.utc).isoformat()); record()
        raise


if __name__ == '__main__':
    raise SystemExit(main())
