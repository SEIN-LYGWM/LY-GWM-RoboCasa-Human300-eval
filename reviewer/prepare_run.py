"""Prepare an isolated, path-relocated copy; never run training or a rollout."""
import argparse
import ast
import copy
import hashlib
import json
import shutil
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
MANIFEST_SHA = '3e30861783695e833bb7fe3e3a38e6e9e6410b659f5bce858eee96b048fcf8fc'
PROTOCOL_SHA = 'ae04489ecf03b1eca3250d1302c866d1f5f8657e4b9be1d5d3f772ae12c0c371'

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(8*1024**2), b''): h.update(block)
    return h.hexdigest()

def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p, x): Path(p).write_text(json.dumps(x, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
def digest(x): return hashlib.sha256(json.dumps(x, sort_keys=True).encode()).hexdigest()
def need(ok, message):
    if not ok: raise RuntimeError(message)

def safe_join(root, rel):
    p = Path(rel)
    need(not p.is_absolute() and '..' not in p.parts, 'Unsafe relative path')
    out = root/p
    need(out.resolve().is_relative_to(root.resolve()), 'Path escapes root: '+str(out))
    return out

def relocate_worker(text, paths):
    tree = ast.parse(text); lines = text.splitlines(keepends=True); replacements=[]; counts={k:0 for k in paths}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in paths:
                replacements.append((node.lineno-1, node.end_lineno, name+' = Path('+repr(str(paths[name]))+')\n'))
                counts[name]+=1
    need(counts=={'POLICY':2,'SIM':2,'THIRD':2,'SOURCE':2,'CHECKPOINT':1}, 'Worker layout changed')
    for start,end,value in reversed(replacements): lines[start:end]=[value]
    patched=''.join(lines)
    # Apart from these nine path assignments, the entire AST must be identical.
    normalized=ast.parse(patched)
    for item in (tree,normalized):
        for node in item.body:
            if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id in paths:
                node.value=ast.Constant(value='PATH_RELOCATION')
    need(ast.dump(tree)==ast.dump(normalized), 'Non-path worker change')
    return patched

def prepare(a):
    out=Path(a.output).expanduser().absolute()
    need(not out.exists(), 'Output must be a new directory; existing results will not be reused')
    need(not out.is_relative_to(PACKAGE), 'Choose output outside the evidence package')
    weights=Path(a.weights).expanduser().resolve()
    source=Path(a.source).expanduser().resolve() if a.source else PACKAGE/'evaluated_snapshot/gr00t_source'
    paths={'POLICY':Path(a.policy_env).expanduser().resolve(),'SIM':Path(a.sim_env).expanduser().resolve(),
           'THIRD':Path(a.third_party).expanduser().resolve(),'SOURCE':source,'CHECKPOINT':weights/'gr00t'}
    for key in ('POLICY','SIM'): need((paths[key]/'bin/python').is_file(), 'Missing environment: '+key)
    for name in ('robocasa','robosuite','robomimic'): need((paths['THIRD']/name).is_dir(), 'Missing external source: '+name)
    manifest_path=PACKAGE/'provenance/file_manifest.json'
    need(sha(manifest_path)==MANIFEST_SHA, 'W8 manifest mismatch')
    sealed=read(manifest_path)
    # Check every exported snapshot file against the original sealed manifest.
    inventory=read(PACKAGE/'export_inventory.json')
    for rel,meta in inventory.items():
        f=safe_join(PACKAGE,rel)
        need(f.stat().st_size==meta['bytes'] and sha(f)==meta['sha256'], 'Export file mismatch: '+rel)
        if rel.startswith('evaluated_snapshot/'):
            original=rel.removeprefix('evaluated_snapshot/')
            need(sealed[original]['sha256']==meta['sha256'], 'Export/sealed inventory mismatch')
    ckpt=paths['CHECKPOINT']
    index=ckpt/'model.safetensors.index.json'
    shards=set(read(index)['weight_map'].values()) if index.exists() else {'model.safetensors'}
    needed=shards|{'config.json','_COMPLETE.json','experiment_cfg/metadata.json'}
    if index.exists(): needed.add(index.name)
    checked={}
    for rel in sorted(needed):
        f=safe_join(ckpt,rel); expected=sealed['gr00t_checkpoint_120000/'+rel]['sha256']
        need(sha(f)==expected, 'GR00T asset mismatch: '+rel); checked[str(f)]=expected
    ly=weights/'lygwm/best.pt'; expected=sealed['joint_s32/lygwm-best.pt']['sha256']
    need(sha(ly)==expected, 'LY-GWM checkpoint mismatch'); checked[str(ly)]=expected
    # Verify all packaged source files at the selected runtime source location.
    for rel,meta in sealed.items():
        if rel.startswith('gr00t_source/'):
            f=safe_join(source,rel.removeprefix('gr00t_source/'))
            need(sha(f)==meta['sha256'], 'GR00T runtime source mismatch: '+str(f))
            checked[str(f)]=meta['sha256']
    original=PACKAGE/'evaluated_snapshot/joint_s32'
    protocol=read(original/'protocol.json'); need(digest(protocol)==PROTOCOL_SHA, 'Original protocol mismatch')
    relocated=copy.deepcopy(protocol)
    relocated['checkpoint_path']=str(ckpt)
    relocated['joint_model']['lygwm_checkpoint']=str(out/'lygwm-best.pt')
    old_sources=protocol['joint_model']['gr00t_source_sha256']
    new_sources={}
    for old,expected in old_sources.items():
        parts=Path(old).parts; offset=parts.index('gr00t')
        new_sources[str(source.joinpath(*parts[offset:]))]=expected
    relocated['joint_model']['gr00t_source_sha256']=new_sources
    before=copy.deepcopy(protocol); after=copy.deepcopy(relocated)
    for x in (before,after):
        x['checkpoint_path']='PATH'; x['joint_model']['lygwm_checkpoint']='PATH'
        x['joint_model']['gr00t_source_sha256']=sorted(x['joint_model']['gr00t_source_sha256'].values())
    need(before==after, 'Non-path protocol change')
    patched=relocate_worker((original/'worker.py').read_text(),paths)
    out.mkdir(parents=True)
    originals=read(original/'build_report.json')
    for name in originals['generated_file_sha256']:
        if name in ('worker.py','protocol.json','lygwm-best.pt'): continue
        shutil.copyfile(original/name,out/name)
    shutil.copyfile(ly,out/'lygwm-best.pt')
    (out/'worker.py').write_text(patched,encoding='utf-8')
    write(out/'protocol.json',relocated)
    # This receipt describes the newly prepared copy, not a new accepted rollout.
    build={'build_accepted':True,'build_scope':'path_relocation_static_only',
           'rollout_verified':False,'original_protocol_sha256':PROTOCOL_SHA,
           'protocol_sha256':digest(relocated),
           'generated_file_sha256':{name:sha(out/name) for name in originals['generated_file_sha256']}}
    write(out/'build_report.json',build)
    config={'run_dir':str(out),'policy_env':str(paths['POLICY']),'sim_env':str(paths['SIM']),
            'library_dir':str(Path(a.library_dir).expanduser().resolve()) if a.library_dir else None,
            'runtime_assets_sha256':checked,'runtime_files_sha256':dict(build['generated_file_sha256'],**{'build_report.json':sha(out/'build_report.json')}),
            'original_protocol_sha256':PROTOCOL_SHA,'relocated_protocol_sha256':digest(relocated),
            'training_run':False,'rollout_run':False,'original_snapshot_modified':False,
            'worker_changes':'Only POLICY, SIM, THIRD, SOURCE and CHECKPOINT top-level paths',
            'historical_episode_results_copied':False}
    write(out/'reviewer_runtime.json',config)
    print(json.dumps({'prepare_accepted':True,'run_dir':str(out),'gpu_validation_run':False,
                      'original_protocol_sha256':PROTOCOL_SHA,'relocated_protocol_sha256':digest(relocated)},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('weights','policy-env','sim-env','third-party','output'):p.add_argument('--'+n,required=True)
    p.add_argument('--source');p.add_argument('--library-dir')
    prepare(p.parse_args())
