"""One allocated GPU sequentially evaluates its assigned tasks."""
import json
import os
from pathlib import Path
import subprocess

root=Path(__file__).resolve().parent
pool_index=int(os.environ['SLURM_ARRAY_TASK_ID'])
pools=json.loads((root/'pool_table.json').read_text())
table=json.loads((root/'job_table.json').read_text())['full']
assert 0<=pool_index<8
failed=[]
for index in pools[pool_index]:
    job=table[index]
    assert job['task_index']==index and job['mode']=='learned_reward'
    print('S42_TASK_BEGIN pool=%d task=%d'%(pool_index,index),flush=True)
    env=dict(os.environ,SLURM_ARRAY_TASK_ID=str(index))
    rc=subprocess.call(['bash',str(Path(job['run'])/'job.sh'),job['run']],env=env)
    print('S42_TASK_END pool=%d task=%d rc=%d'%(pool_index,index,rc),flush=True)
    if rc:failed.append(index)
print('S42_POOL_COMPLETE '+json.dumps(dict(pool=pool_index,failed_tasks=failed)),flush=True)
raise SystemExit(1 if failed else 0)
