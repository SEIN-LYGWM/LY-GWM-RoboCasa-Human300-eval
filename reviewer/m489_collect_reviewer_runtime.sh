#!/usr/bin/env bash
set +e
set +u
set +o pipefail 2>/dev/null || true
(
/home/bingxing2/apps/anaconda/2021.11/envs/py310torch251cu121/bin/python -I -B -u - <<'PY'
import datetime, hashlib, json, os, platform, shutil, subprocess, tempfile, zipfile
from pathlib import Path
ROOT=Path('/home/bingxing2/home/scx9fvq/m489_robocasa365_gr00t_n1_5')
THIRD=Path('/home/bingxing2/home/scx9fvq/LY-GWM-RC/third_party')
SOURCE=ROOT/'01_source/Isaac-GR00T-9d7d7a9eb7ad30bd8ce30448d9ab53a918b45b10_s5_source_recovery_20260909_140603'
BASE=ROOT/'00_offline_assets/inbox/r1_gr00t_n15_base/payload'
parent=ROOT/'06_release';parent.mkdir(exist_ok=True)
out=Path(tempfile.mkdtemp(prefix='reviewer_runtime_',dir=parent))
errors=[]
probe='''import importlib.metadata as m,json,sys,platform
items=sorted([{'name':d.metadata.get('Name','unknown'),'version':d.version} for d in m.distributions()],key=lambda x:(x['name'].lower(),x['version']))
print(json.dumps({'python':sys.version,'prefix':sys.prefix,'machine':platform.machine(),'packages':items}))'''
report={'collected_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'scope':'Current read-only environment inventory; not proof of historical environment identity',
        'training_run':False,'rollout_run':False,'gpu_job_submitted':False,'packages_installed':False,
        'environments':{},'external_sources':{},'copied_files':{},'errors':errors}
for tag,env in [('policy',ROOT/'02_env/gr00t_n15_py310_t251_cu121'),('simulation',ROOT/'02_env/robocasa101_sim_py310_t251_cu121')]:
    try:
        r=subprocess.run([str(env/'bin/python'),'-I','-B','-c',probe],capture_output=True,text=True,timeout=60,check=True)
        data=json.loads(r.stdout);report['environments'][tag]=data
        (out/(tag+'-packages.txt')).write_text('\n'.join(x['name']+'=='+x['version'] for x in data['packages'])+'\n')
    except Exception as e:errors.append({'phase':tag,'error':str(e)})
for tag,path in [('gr00t',SOURCE)]+[(n,THIRD/n) for n in ('robocasa','robosuite','robomimic')]:
    entry={'path':str(path),'exists':path.is_dir()};report['external_sources'][tag]=entry
    if not path.is_dir():errors.append({'phase':tag,'error':'Missing source directory'});continue
    for name,args in [('git_commit',['rev-parse','HEAD']),('git_status',['status','--porcelain','--untracked-files=no'])]:
        try:
            r=subprocess.run(['git','-C',str(path)]+args,capture_output=True,text=True,timeout=30)
            entry[name]=r.stdout.strip() if r.returncode==0 else None
            entry[name+'_available']=r.returncode==0
        except Exception as e:entry[name]=None;entry[name+'_error']=str(e)
    # Selected dependency declarations and license files only; no git credentials.
    for pattern in ('LICENSE*','NOTICE*','COPYING*','pyproject.toml','setup.py','setup.cfg','requirements*.txt'):
        for f in sorted(path.glob(pattern)):
            if f.is_file() and not f.is_symlink() and f.stat().st_size<2*1024**2:
                dest=out/'source_records'/tag/f.name;dest.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(f,dest);report['copied_files'][str(dest.relative_to(out))]={'source':str(f),'sha256':hashlib.sha256(dest.read_bytes()).hexdigest()}
for f in sorted(BASE.glob('LICENSE*')):
    if f.is_file() and f.stat().st_size<2*1024**2:
        dest=out/'base_model_notices'/f.name;dest.parent.mkdir(exist_ok=True)
        shutil.copyfile(f,dest);report['copied_files'][str(dest.relative_to(out))]={'source':str(f),'sha256':hashlib.sha256(dest.read_bytes()).hexdigest()}
report['base_model_license_found']=bool(list((out/'base_model_notices').glob('*')))
processor=SOURCE/'gr00t/model/backbone/eagle2_hg_model'
report['processor_files']=[{'name':f.name,'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()}
 for f in sorted(processor.glob('*')) if f.is_file() and not f.is_symlink() and f.stat().st_size<32*1024**2]
report['inventory_completed']=not errors
(out/'runtime_inventory.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
archive=out.with_suffix('.zip')
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
    for f in sorted(out.rglob('*')):
        if f.is_file():z.write(f,str(f.relative_to(out)))
print('##### REVIEWER_RUNTIME_INVENTORY #####')
print(json.dumps({'inventory_completed':not errors,'output_dir':str(out),'zip_path':str(archive),
 'zip_bytes':archive.stat().st_size,'zip_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
 'base_model_license_found':report['base_model_license_found'],'errors':errors,
 'gpu_job_submitted':False,'training_run':False,'rollout_run':False},ensure_ascii=False,indent=2))
raise SystemExit(0 if not errors else 1)
PY
rc=$?
echo "reviewer_runtime_inventory_rc=$rc"
exit "$rc"
)
