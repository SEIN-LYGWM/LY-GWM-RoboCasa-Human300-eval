"""Submit eight single-GPU workers once; each runs several of the 50 tasks."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess


def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['full']);parser.parse_args()
    root=Path(__file__).resolve().parent
    table=json.loads((root/'job_table.json').read_text())['full']
    pools=json.loads((root/'pool_table.json').read_text())
    assert len(pools)==8 and sorted(i for pool in pools for i in pool)==list(range(50))
    assert len(table)==50 and all(j['mode']=='learned_reward' for j in table)
    assert not list((Path(table[0]['run'])/'tasks').glob('task_*/episodes/*.json')), 'Results already exist'
    intent=root/'full_submission_intent.json'
    with intent.open('x') as f:
        json.dump(dict(phase='full',status='submitting',jobs=8,tasks=50,total_episodes=2500),f)
        f.flush();os.fsync(f.fileno())
    cmd=['sbatch','--parsable','--job-name=m489_s42_full','--partition=gpu','--qos=gpugpu',
         '--nodes=1','--ntasks=1','--gres=gpu:1','--array=0-7%8',
         '--exclude=paraai-n32-h-01-agent-4','--time=24:00:00','--export=ALL',
         '--chdir='+str(root),'--output='+str(root/'full-%A_%a.out'),
         '--error='+str(root/'full-%A_%a.out'),str(root/'full_dispatch.sh')]
    print('Submitting 8 GPU workers for 50 tasks:', ' '.join(cmd),flush=True)
    result=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True)
    (root/'full_submission.out').write_text(result.stdout)
    (root/'full_submission.err').write_text(result.stderr)
    if result.returncode:raise RuntimeError('Submission rejected; no automatic retry: '+result.stderr)
    match=re.fullmatch(r'(\d+)(?:;[^\s;]+)?',result.stdout.strip())
    if not match:raise RuntimeError('Uncertain job ID; inspect scheduler before any retry')
    jobid=match.group(1)
    intent.write_text(json.dumps(dict(phase='full',status='submitted',array_job_id=jobid,
        jobs=8,tasks=50,total_episodes=2500,dispatch='eight_sequential_workers'),indent=2))
    print('S42_SUBMITTED=True array_job_id='+jobid+' phase=full',flush=True)

if __name__=='__main__':main()
