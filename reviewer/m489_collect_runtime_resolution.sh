#!/usr/bin/env bash
set +e
set +u
set +o pipefail 2>/dev/null || true
(
/home/bingxing2/apps/anaconda/2021.11/envs/py310torch251cu121/bin/python -I -B -u - <<'PY'
import datetime, hashlib, json, os, shutil, subprocess, tempfile, zipfile
from pathlib import Path
ROOT=Path('/home/bingxing2/home/scx9fvq/m489_robocasa365_gr00t_n1_5')
BASE=Path('/home/bingxing2/apps/anaconda/2021.11/envs/py310torch251cu121')
THIRD=Path('/home/bingxing2/home/scx9fvq/LY-GWM-RC/third_party')
parent=ROOT/'06_release';parent.mkdir(exist_ok=True)
out=Path(tempfile.mkdtemp(prefix='runtime_resolution_',dir=parent))
errors=[]
probe=r'''import importlib.metadata as m,importlib.util,json,sys,platform,re
from pathlib import Path
norm=lambda x:re.sub(r'[-_.]+','-',x).lower()
all_dist={}
for d in m.distributions():
 n=norm(d.metadata.get('Name','unknown'))
 all_dist.setdefault(n,[]).append({'version':d.version,'location':str(d.locate_file(''))})
selected={}
for n in sorted(all_dist):
 try:
  d=m.distribution(n);selected[n]={'version':d.version,'location':str(d.locate_file(''))}
 except Exception as e:selected[n]={'error':str(e)}
modules={}
for n in ('numpy','torch','transformers','mujoco','gymnasium','flash_attn'):
 try:
  s=importlib.util.find_spec(n);modules[n]=None if s is None else {'origin':s.origin,'search_locations':list(s.submodule_search_locations or [])}
 except Exception as e:modules[n]={'error':str(e)}
# Only NumPy is imported. Metadata selection alone is not proof of import identity.
try:
 import numpy as np
 numpy_actual={'version':np.__version__,'file':np.__file__}
except Exception as e:numpy_actual={'error':str(e)}
print(json.dumps({'python':sys.version,'prefix':sys.prefix,'sys_path':sys.path,
 'machine':platform.machine(),'metadata_selected':selected,
 'duplicate_metadata':{n:v for n,v in all_dist.items() if len(v)>1},
 'module_locations':modules,'numpy_import':numpy_actual}))'''
report={'collected_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'scope':'Current environment resolution and current simulator code snapshot; not historical commit recovery',
 'environments':{},'external_sources':{},'errors':errors,'gpu_job_submitted':False,
 'training_run':False,'rollout_run':False,'packages_installed':False,'original_files_modified':False}
for tag,name,expected in [('policy','gr00t_n15_py310_t251_cu121','1.26.4'),('simulation','robocasa101_sim_py310_t251_cu121','2.2.5')]:
 try:
  env=os.environ.copy();env['LD_LIBRARY_PATH']=str(BASE/'lib')+os.pathsep+env.get('LD_LIBRARY_PATH','');env['OPENBLAS_NUM_THREADS']='1';env['OMP_NUM_THREADS']='1'
  r=subprocess.run([str(ROOT/'02_env'/name/'bin/python'),'-I','-B','-c',probe],env=env,capture_output=True,text=True,timeout=60,check=True)
  d=json.loads(r.stdout);report['environments'][tag]=d
  if d['numpy_import'].get('version')!=expected:errors.append({'phase':tag,'error':'Imported NumPy differs from intended environment','observed':d['numpy_import'],'expected':expected})
 except Exception as e:errors.append({'phase':tag,'error':str(e)})
# Text/code only. Large simulation assets and datasets remain external.
suffixes={'.py','.json','.yaml','.yml','.toml','.cfg','.ini','.txt','.md','.rst','.sh','.xml','.in'}
exclude={'assets','datasets','videos','demos','checkpoints','trained_models','__pycache__','node_modules'}
manifest={};total=0
if shutil.disk_usage(parent).free<1024**3:raise RuntimeError('Need at least 1 GiB free disk')
for name in ('robocasa','robosuite','robomimic'):
 source=THIRD/name;entry={'path':str(source),'skipped_large':[],'skipped_symlinks':[],'files':0};report['external_sources'][name]=entry
 if not source.is_dir():errors.append({'phase':name,'error':'Missing source'});continue
 for field,args in [('git_commit',['rev-parse','HEAD']),('git_status',['status','--porcelain','--untracked-files=no'])]:
  try:
   r=subprocess.run(['git','-C',str(source)]+args,capture_output=True,text=True,timeout=30)
   entry[field]={'returncode':r.returncode,'stdout':r.stdout.strip(),'stderr':r.stderr.strip()[:2000]}
  except Exception as e:entry[field]={'error':str(e)}
 for directory,dirs,files in os.walk(source,followlinks=False):
  dirs[:]=sorted(x for x in dirs if not x.startswith('.') and x not in exclude and not (Path(directory)/x).is_symlink())
  for filename in sorted(files):
   f=Path(directory)/filename;rel=f.relative_to(source)
   if filename.startswith('.'):continue
   if f.suffix.lower() not in suffixes and not filename.upper().startswith(('LICENSE','NOTICE','COPYING')):continue
   if f.is_symlink():entry['skipped_symlinks'].append(str(rel));continue
   st=f.stat()
   if st.st_size>16*1024**2:entry['skipped_large'].append(str(rel));continue
   if total+st.st_size>256*1024**2:raise RuntimeError('Text/source export exceeds 256 MiB; no automatic expansion')
   target=out/'external_source_snapshot'/name/rel;target.parent.mkdir(parents=True,exist_ok=True)
   shutil.copyfile(f,target)
   end=f.stat()
   if (st.st_size,st.st_mtime_ns)!=(end.st_size,end.st_mtime_ns):raise RuntimeError('Source changed during copy: '+str(f))
   key=str(Path(name)/rel);h=hashlib.sha256(target.read_bytes()).hexdigest()
   if hashlib.sha256(f.read_bytes()).hexdigest()!=h:raise RuntimeError('Copy mismatch: '+str(f))
   manifest[key]={'bytes':st.st_size,'sha256':h};entry['files']+=1;total+=st.st_size
 print('source_copied='+name+' files='+str(entry['files']),flush=True)
 if not entry['files']:errors.append({'phase':name,'error':'No source files copied'})
manifest_text=json.dumps(manifest,sort_keys=True,indent=2)+'\n'
(out/'source_file_manifest.json').write_text(manifest_text)
report['source_manifest_sha256']=hashlib.sha256(manifest_text.encode()).hexdigest()
report['source_bytes']=total;report['source_files']=len(manifest)
report['collection_accepted']=not errors
(out/'runtime_resolution.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
archive=out.with_suffix('.zip')
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
 for f in sorted(out.rglob('*')):
  if f.is_file():z.write(f,str(f.relative_to(out)))
print('##### RUNTIME_RESOLUTION_REPORT #####')
print(json.dumps({'collection_accepted':not errors,'zip_path':str(archive),'zip_bytes':archive.stat().st_size,
 'zip_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'source_files':len(manifest),
 'numpy_imports':{k:v['numpy_import'] for k,v in report['environments'].items()},'errors':errors,
 'training_run':False,'rollout_run':False,'gpu_job_submitted':False},ensure_ascii=False,indent=2))
raise SystemExit(0 if not errors else 1)
PY
rc=$?
echo "runtime_resolution_rc=$rc"
exit "$rc"
)
