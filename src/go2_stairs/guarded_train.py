"""Run verified train/replay blocks, never an unattended thousand-update blind run."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def source_snapshot():
    return {p.name:hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(__file__).parent.glob('*.py'))}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--iterations',type=int,default=1000)
    parser.add_argument('--block-updates',type=int,default=100)
    parser.add_argument('--num-envs',type=int,default=1024)
    parser.add_argument('--seed',type=int,default=0)
    args=parser.parse_args()
    if min(args.iterations,args.block_updates,args.num_envs)<1 or args.block_updates>100:
        parser.error('counts must be positive; evaluate at least every 100 updates')
    root=args.output.resolve()
    root.mkdir(parents=True,exist_ok=False)
    from .training_health import development_decision
    snapshot=source_snapshot()
    status=dict(status='STARTING',seed=args.seed,budget_updates=args.iterations,completed_updates=0,
                flat_gate_passed=False,blocks=[],source_sha256=snapshot,
                start_utc=datetime.now(timezone.utc).isoformat())
    def record():
        status['updated_utc']=datetime.now(timezone.utc).isoformat()
        temp=root/'status.tmp'
        temp.write_text(json.dumps(status,indent=2,allow_nan=False),encoding='utf-8')
        temp.replace(root/'status.json')
    def run(command,log):
        if source_snapshot()!=snapshot:
            raise RuntimeError('SOURCE_CHANGED: start a new job with the new source')
        environment=dict(os.environ,PYTHONIOENCODING='utf-8',PYTHONUNBUFFERED='1')
        with log.open('w',encoding='utf-8') as output:
            code=subprocess.run(command,stdout=output,stderr=subprocess.STDOUT,env=environment).returncode
        if source_snapshot()!=snapshot:
            raise RuntimeError('SOURCE_CHANGED during a child run')
        return code
    record()
    previous=None
    try:
        while status['completed_updates']<args.iterations:
            start=status['completed_updates']
            count=min(args.block_updates,args.iterations-start)
            block=root/f'block-{start:06}'
            block.mkdir(exist_ok=False)
            train=block/'train'
            command=[sys.executable,'-m','go2_stairs.train','--headless','--profile','bounded-v3',
                     '--tracking-std','.2','--num_envs',str(args.num_envs),'--seed',str(args.seed),
                     '--iterations',str(count),'--save_interval','50','--output',str(train)]
            if previous is not None:
                command+=['--resume',str(previous)]
            status.update(status='TRAINING',active_block=str(block),command=command)
            record()
            code=run(command,block/'train.log')
            result=json.loads((train/'status.json').read_text()) if (train/'status.json').is_file() else {}
            if code or result.get('status')!='PPO_JOB_COMPLETED':
                status.update(status='STOP_TRAINING_FAILURE',exit_code=code,child_status=result)
                record()
                return 2
            previous=Path(result['checkpoint']).resolve()
            if (not previous.is_relative_to(train.resolve()) or
                    hashlib.sha256(previous.read_bytes()).hexdigest()!=result['checkpoint_sha256'] or
                    result['completed_updates']!=start+count):
                raise ValueError('checkpoint path/hash/update counter mismatch')
            status.update(status='EVALUATING',completed_updates=start+count,checkpoint=str(previous))
            record()
            evaluation=block/'development'
            command=[sys.executable,'-m','go2_stairs.flat_development','--headless',
                     '--checkpoint',str(previous),'--output',str(evaluation)]
            code=run(command,block/'development.log')
            if code or not (evaluation/'summary.json').is_file():
                status.update(status='STOP_EVALUATION_FAILURE',exit_code=code)
                record()
                return 2
            replay=json.loads((evaluation/'summary.json').read_text())
            if replay['checkpoint_sha256']!=result['checkpoint_sha256']:
                raise ValueError('evaluation checkpoint identity mismatch')
            decision=development_decision(replay,status['completed_updates'])
            status['blocks'].append(dict(updates=status['completed_updates'],checkpoint=str(previous),
                                         evaluation=str(evaluation/'summary.json'),
                                         passed_count=replay['passed_count'],decision=decision))
            print(json.dumps(status['blocks'][-1]),flush=True)
            if decision=='FLAT_GATE_PASSED' or decision.startswith('STOP_'):
                status.update(status=decision,flat_gate_passed=decision=='FLAT_GATE_PASSED',
                              end_utc=datetime.now(timezone.utc).isoformat())
                record()
                return 0 if decision=='FLAT_GATE_PASSED' else 2
        status.update(status='BUDGET_EXHAUSTED',end_utc=datetime.now(timezone.utc).isoformat())
        record()
        return 0
    except Exception as exc:
        status.update(status='FAILED',error=f'{type(exc).__name__}: {exc}',end_utc=datetime.now(timezone.utc).isoformat())
        record()
        raise


if __name__=='__main__':
    raise SystemExit(main())
