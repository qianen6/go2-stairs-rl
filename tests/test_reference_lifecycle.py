import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch


class ReferenceLifecycleTests(unittest.TestCase):
    def test_parent_cancellation_cleans_actual_cpu_child_tree(self):
        from go2_stairs.reference import run_reference_child
        import psutil
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);marker=root/'child.json';pids=[]
            script=("import subprocess,sys,os,json,time;from pathlib import Path;"
                    "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']);"
                    f"Path({str(marker)!r}).write_text(json.dumps([os.getpid(),p.pid]));time.sleep(60)")
            original_wait=subprocess.Popen.wait
            fired=False
            def interrupted(proc,timeout=None):
                nonlocal fired
                if not fired:
                    fired=True
                    deadline=time.monotonic()+5
                    while not marker.exists() and time.monotonic()<deadline:
                        time.sleep(.02)
                    self.assertTrue(marker.exists())
                    pids.extend([proc.pid,*json.loads(marker.read_text())])
                    raise KeyboardInterrupt()
                return original_wait(proc,timeout=timeout)
            try:
                with patch.object(subprocess.Popen,'wait',interrupted):
                    with self.assertRaises(KeyboardInterrupt):
                        run_reference_child([sys.executable,'-c',script],root/'child.log',30)
                self.assertTrue(all(not psutil.pid_exists(pid) for pid in pids))
            finally:
                for pid in pids:
                    try:
                        psutil.Process(pid).kill()
                    except psutil.NoSuchProcess:
                        pass

    def test_cleanup_failure_is_reported_without_unbounded_wait(self):
        from go2_stairs.reference import run_reference_child
        process=SimpleNamespace(pid=12345,wait=lambda timeout:(_ for _ in ()).throw(KeyboardInterrupt()))
        with tempfile.TemporaryDirectory() as folder, \
             patch('go2_stairs.reference.subprocess.Popen',return_value=process), \
             patch('go2_stairs.reference.terminate_child_tree',side_effect=RuntimeError('fixture cleanup failure')):
            with self.assertRaisesRegex(RuntimeError,'CHILD_CLEANUP_FAILED'):
                run_reference_child(['fixture'],Path(folder)/'log',1)

    def test_post_replay_source_drift_never_publishes_evaluated(self):
        from go2_stairs.reference import main
        from test_reference import ReferenceTests
        from flat_fixture import flat_report
        import hashlib
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'job';drift=False
            class Child:
                def __init__(self,cmd,**kwargs):
                    self.cmd=cmd
                def __enter__(self):return self
                def __exit__(self,*args):return False
                def wait(self,timeout=None):
                    nonlocal drift
                    cmd=self.cmd
                    if '--max_iterations' in cmd:
                        native=root/'native'/'run';(native/'params').mkdir(parents=True)
                        torch.save(ReferenceTests().checkpoint(),native/'model_1.pt')
                        (native/'params/env.yaml').write_text('fixture: 1')
                        (native/'params/agent.yaml').write_text('fixture: 1')
                    else:
                        out=Path(cmd[cmd.index('--output')+1]);out.mkdir()
                        checkpoint=Path(cmd[cmd.index('--checkpoint')+1])
                        data=flat_report(True);data['checkpoint_sha256']=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                        (out/'summary.json').write_text(json.dumps(data));drift=True
                    return 0
            with patch.object(sys,'argv',['reference','--output',str(root),'--updates','2']), \
                 patch('go2_stairs.reference.source_identity',side_effect=lambda:{'commit':'changed' if drift else 'pinned'}), \
                 patch('go2_stairs.reference.adapter_identity',return_value={'adapter':'pinned'}), \
                 patch('go2_stairs.assets.locked_go2_assets',return_value={'sha256':'a'*64}), \
                 patch('go2_stairs.reference.subprocess.Popen',Child):
                with self.assertRaises(ValueError):main()
            self.assertEqual(json.loads((root/'status.json').read_text())['status'],'REFERENCE_FAILED')

    def test_policy_receives_current_phase_command_not_previous_frame(self):
        from go2_stairs.reference_eval import reference_policy_step
        command=SimpleNamespace(time_left=torch.zeros(1),is_standing_env=torch.ones(1,dtype=torch.bool),
                                vel_command_b=torch.ones(1,3))
        seen=[]
        env=SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda name:command))
        wrapper=SimpleNamespace(unwrapped=env,get_observations=lambda:{'command':command.vel_command_b.clone()},
                                step=lambda action:('next',0,0,{}))
        policy=SimpleNamespace(act_inference=lambda obs:(seen.append(obs['command'].clone()) or torch.zeros(1,12)))
        for step,expected in [(0,[0,0,0]),(50,[.1,0,0]),(550,[0,0,.2]),(700,[0,0,0])]:
            reference_policy_step(wrapper,policy,step,torch.tensor([.1]))
            self.assertTrue(torch.allclose(seen[-1],torch.tensor([expected],dtype=torch.float32)))
