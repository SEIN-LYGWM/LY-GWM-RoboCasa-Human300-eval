import json,os,subprocess,sys
from pathlib import Path
r=Path(__file__).resolve().parent
j=json.loads((r/'job_table.json').read_text())[sys.argv[1]][int(os.environ['SLURM_ARRAY_TASK_ID'])]
env=dict(os.environ,SLURM_ARRAY_TASK_ID=str(j['task_index']))
raise SystemExit(subprocess.call(['bash',str(Path(j['run'])/'job.sh'),j['run']],env=env))
