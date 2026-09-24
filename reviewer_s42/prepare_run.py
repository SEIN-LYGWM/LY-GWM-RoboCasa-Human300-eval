"""Prepare S42 in a new directory; no training, installation, or GPU execution."""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import shutil

PACKAGE = Path(__file__).resolve().parents[1]
ORIGINAL = PACKAGE / 'evaluated_snapshot/s42/s42_original/full/learned_reward'
PROTOCOL_SHA = '4bdd5abfabe2674caba78c1f7b637f3c141d5550a09b55e065020c8a9c069e71'

def need(ok, message):
    if not ok:
        raise RuntimeError(message)

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')

def safe_join(root, rel):
    p = Path(rel)
    need(not p.is_absolute() and '..' not in p.parts, 'Unsafe relative path: '+str(rel))
    result = root / p
    need(result.resolve().is_relative_to(root.resolve()), 'Path escapes root')
    return result

def relocate_worker(text, paths):
    tree = ast.parse(text)
    lines = text.splitlines(keepends=True)
    counts = {key: 0 for key in paths}
    changes = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            key = node.targets[0].id
            if key in paths:
                changes.append((node.lineno-1, node.end_lineno, key+' = Path('+repr(str(paths[key]))+')\n'))
                counts[key] += 1
    need(counts == {'POLICY':2, 'SIM':2, 'THIRD':2, 'SOURCE':2, 'CHECKPOINT':1}, 'Worker path layout changed')
    for start, end, value in reversed(changes):
        lines[start:end] = [value]
    changed = ''.join(lines)
    normalized = ast.parse(changed)
    for item in (tree, normalized):
        for node in item.body:
            if isinstance(node, ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id in paths:
                node.value = ast.Constant(value='PATH_RELOCATION')
    need(ast.dump(tree) == ast.dump(normalized), 'Non-path worker change')
    return changed

def relocate_protocol(protocol, out, paths, weights):
    result = copy.deepcopy(protocol)
    result['checkpoint_path'] = str(paths['CHECKPOINT'])
    j = result['joint_model']
    j['lygwm_checkpoint'] = str(weights / 'lygwm/best.pt')
    j['scorer_checkpoint'] = str(weights / 's42/scorer-best.pt')
    j['scorer_config_path'] = str(weights / 's42/scorer-training-config.json')
    j['gr00t_source_sha256'] = {
        str(paths['SOURCE'] / 'gr00t' / old.split('/gr00t/', 1)[1]): h
        for old, h in j['gr00t_source_sha256'].items()
    }
    result['diagnostic_source_sha256'] = {
        str(paths['THIRD'] / old.split('/third_party/', 1)[1]): h
        for old, h in result['diagnostic_source_sha256'].items()
    }
    def normalize(p):
        p = copy.deepcopy(p)
        p['checkpoint_path'] = 'PATH'
        for key in ('lygwm_checkpoint','scorer_checkpoint','scorer_config_path'):
            p['joint_model'][key] = 'PATH'
        p['joint_model']['gr00t_source_sha256'] = sorted(p['joint_model']['gr00t_source_sha256'].values())
        p['diagnostic_source_sha256'] = sorted(p['diagnostic_source_sha256'].values())
        return p
    need(normalize(protocol) == normalize(result), 'Non-path protocol change')
    return result

def prepare(args):
    need(__debug__, 'Do not disable assertions with -O')
    out = Path(args.output).expanduser().resolve()
    need(not out.exists(), 'Output must not exist; historical results are never resumed/copied')
    need(not out.is_relative_to(PACKAGE.resolve()), 'Output must be outside the repository')
    weights = Path(args.weights).expanduser().resolve()
    paths = {
        'POLICY':Path(args.policy_env).expanduser().resolve(),
        'SIM':Path(args.sim_env).expanduser().resolve(),
        'THIRD':Path(args.third_party).expanduser().resolve(),
        'SOURCE':Path(args.source).expanduser().resolve() if args.source else PACKAGE / 'evaluated_snapshot/gr00t_source',
        'CHECKPOINT':weights / 'gr00t',
    }
    for key in ('POLICY','SIM'):
        need((paths[key]/'bin/python').is_file(), 'Missing environment: '+key)
    for name in ('robocasa','robosuite','robomimic'):
        need((paths['THIRD']/name).is_dir(), 'Missing simulator source/assets: '+name)
    manifest = read(PACKAGE/'releases/s42/package_manifest.json')
    for rel, meta in manifest['files'].items():
        f = safe_join(PACKAGE, rel)
        need(f.stat().st_size == meta['bytes'] and sha(f) == meta['sha256'], 'Packaged S42 file mismatch: '+rel)
    p = read(ORIGINAL/'protocol.json')
    b = read(ORIGINAL/'build_report.json')
    need(digest(p) == b['protocol_sha256'] == PROTOCOL_SHA, 'Original S42 protocol mismatch')
    for name, h in b['generated_file_sha256'].items():
        need(sha(safe_join(ORIGINAL, name)) == h, 'Frozen S42 source changed: '+name)
    assets = {}
    ckpt = paths['CHECKPOINT']
    for name, field in (('_COMPLETE.json','checkpoint_commit_sha256'),
                        ('config.json','checkpoint_config_sha256'),
                        ('experiment_cfg/metadata.json','checkpoint_metadata_sha256')):
        assets[str(ckpt/name)] = p[field]
    for name, h in b['policy_weight_sha256'].items():
        assets[str(safe_join(ckpt, name))] = h
    relocated = relocate_protocol(p, out, paths, weights)
    j = relocated['joint_model']
    for field, hfield in (('lygwm_checkpoint','lygwm_checkpoint_sha256'),
                          ('scorer_checkpoint','scorer_checkpoint_sha256'),
                          ('scorer_config_path','scorer_config_sha256')):
        assets[j[field]] = j[hfield]
    assets.update(j['gr00t_source_sha256'])
    assets.update(relocated['diagnostic_source_sha256'])
    # Preserve and validate all packaged GR00T source/processor files, not only
    # the nine files directly pinned by the S42 protocol.
    inventory = read(PACKAGE/'export_inventory.json')
    prefix = 'evaluated_snapshot/gr00t_source/'
    for rel, meta in inventory.items():
        if rel.startswith(prefix):
            assets[str(safe_join(paths['SOURCE'], rel[len(prefix):]))] = meta['sha256']
    for path, h in assets.items():
        need(sha(path) == h, 'Runtime asset mismatch: '+path)
    patched = relocate_worker((ORIGINAL/'worker.py').read_text(encoding='utf-8'), paths)
    out.mkdir(parents=True)
    for name in b['generated_file_sha256']:
        if name not in ('worker.py','protocol.json'):
            shutil.copyfile(ORIGINAL/name, out/name)
    (out/'worker.py').write_text(patched, encoding='utf-8')
    write(out/'protocol.json', relocated)
    build = dict(build_accepted=True, build_scope='S42_path_relocation_static_only',
        gpu_validated=False, original_protocol_sha256=PROTOCOL_SHA,
        protocol_sha256=digest(relocated), policy_weight_sha256=b['policy_weight_sha256'],
        generated_file_sha256={name:sha(out/name) for name in b['generated_file_sha256']})
    write(out/'build_report.json', build)
    write(out/'reviewer_runtime.json', dict(
        run_dir=str(out), policy_env=str(paths['POLICY']), sim_env=str(paths['SIM']),
        library_dir=str(Path(args.library_dir).expanduser().resolve()) if args.library_dir else None,
        original_protocol_sha256=PROTOCOL_SHA, relocated_protocol_sha256=digest(relocated),
        runtime_assets_sha256=assets,
        runtime_files_sha256=dict(build['generated_file_sha256'], **{'build_report.json':sha(out/'build_report.json')}),
        historical_episode_results_copied=False, training_run=False, rollout_run=False,
        worker_changes='Only nine top-level path assignments; remaining AST identical',
        mode='learned_reward', zero_bounds_gate=False))
    print(json.dumps(dict(prepare_accepted=True, run_dir=str(out), gpu_validation_run=False,
        original_protocol_sha256=PROTOCOL_SHA, relocated_protocol_sha256=digest(relocated)), indent=2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('weights','policy-env','sim-env','third-party','output'):
        parser.add_argument('--'+key, required=True)
    parser.add_argument('--source')
    parser.add_argument('--library-dir')
    prepare(parser.parse_args())
