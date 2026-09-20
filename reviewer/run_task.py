"""Run one 50-episode task on an already allocated Linux GPU. Does not submit jobs."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parent))
from prepare_run import sha, read, need

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True);p.add_argument('--task-index',type=int,required=True,choices=range(50))
    p.add_argument('--gpu',type=int,help='Physical GPU id outside Slurm; do not override a scheduler allocation')
    a=p.parse_args(); need(os.name=='posix','Run inside Linux with the two prepared Python environments')
    run=Path(a.run).resolve(); c=read(run/'reviewer_runtime.json')
    need(c['run_dir']==str(run),'Runtime moved; prepare a fresh run')
    for rel,h in c['runtime_files_sha256'].items():need(sha(run/rel)==h,'Runtime code changed: '+rel)
    for path,h in c['runtime_assets_sha256'].items():need(sha(path)==h,'Runtime asset changed: '+path)
    spec=importlib.util.spec_from_file_location('reviewer_joint_audit',run/'joint_audit.py')
    audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
    audit.verify_assets(run,read(run/'protocol.json'))
    e=os.environ.copy()
    need(not (a.gpu is not None and e.get('SLURM_JOB_ID')),'Use the scheduler-provided GPU mapping')
    if a.gpu is not None:e['CUDA_VISIBLE_DEVICES']=str(a.gpu)
    need(e.get('CUDA_VISIBLE_DEVICES','').isdigit(),'Exactly one numeric CUDA_VISIBLE_DEVICES is required by the evaluated EGL mapping')
    e['SLURM_JOB_ID']=e.get('SLURM_JOB_ID') or str(time.time_ns())
    e['SLURM_ARRAY_TASK_ID']=str(a.task_index)
    for key,value in {'PYTHONNOUSERSITE':'1','PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1',
        'OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1',
        'TOKENIZERS_PARALLELISM':'false','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
        'HF_DATASETS_OFFLINE':'1','HF_HUB_DISABLE_TELEMETRY':'1','NO_ALBUMENTATIONS_UPDATE':'1',
        'MUJOCO_GL':'egl','PYOPENGL_PLATFORM':'egl','PYTHONOPTIMIZE':'0'}.items():e[key]=value
    for key,folder in {'TMPDIR':'tmp','HF_HOME':'hf_cache','HF_MODULES_CACHE':'hf_modules',
                       'NUMBA_CACHE_DIR':'numba_cache','TRITON_CACHE_DIR':'triton_cache'}.items():
        path=run/folder;path.mkdir(exist_ok=True);e[key]=str(path)
    e.pop('TRANSFORMERS_CACHE',None)
    for key in ('CC','CXX','CFLAGS','CXXFLAGS','CPPFLAGS','LDFLAGS'):e.pop(key,None)
    import shutil
    if shutil.which('gcc'):e['CC']=shutil.which('gcc')
    if shutil.which('g++'):e['CXX']=e['CUDAHOSTCXX']=shutil.which('g++')
    if c['library_dir']:e['LD_LIBRARY_PATH']=c['library_dir']+os.pathsep+e.get('LD_LIBRARY_PATH','')
    e['PATH']=str(Path(c['policy_env'])/'bin')+os.pathsep+e.get('PATH','')
    python=str(Path(c['policy_env'])/'bin/python')
    return subprocess.call([python,'-I','-B','-u',str(run/'worker.py'),'orchestrate'],cwd=run,env=e)
if __name__=='__main__':raise SystemExit(main())
