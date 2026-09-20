
import ast
import collections
import datetime
import fcntl
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path('/home/bingxing2/home/scx9fvq/m489_robocasa365_gr00t_n1_5')
RUN = ROOT / '05_lygwm/w8_seal_audit_v1'
BASELINE = ROOT / '03_contract/s31_pretrain50_selftrained'
JOINT = ROOT / '03_contract/s32_pretrain50_lygwm_shadow'
W5 = ROOT / '05_lygwm/w5_graph_training_v1'
W6 = ROOT / '05_lygwm/w6_joint_shadow_probe'
BASE_SHA = 'ee15b8a67247a3f37f6e137c222e0b79cc1801c6d0ab539ec6db247e32bbc943'
JOINT_SHA = 'ae04489ecf03b1eca3250d1302c866d1f5f8657e4b9be1d5d3f772ae12c0c371'
LY_SHA = 'a964d19a0078cc6282d5da79f06f7d7ead3137799b8f6e969ccff5e0643be9f1'
EXPECTED_OUTCOMES = dict(both_success=123, both_failure=2242,
                        baseline_success_joint_failure=58, baseline_failure_joint_success=77)


def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))


def digest(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True).encode()).hexdigest()


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        before = os.fstat(f.fileno())
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
        after = os.fstat(f.fileno())
    assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), 'File changed during hashing: ' + str(p)
    return h.hexdigest()


def save(p, value):
    temp = p.with_name(p.name + '.tmp')
    with temp.open('w', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    temp.replace(p)


REPORT = dict(w8_accepted=False, status='starting', errors=[], gpu_job_submitted=False,
              training_run=False, rollout_run=False, original_files_modified=False,
              leaderboard_submission_run=False, root_cause_identified=False)


def progress(phase, **fields):
    REPORT.update(phase=phase, updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(), **fields)
    save(RUN / 'progress.json', REPORT)
    print(json.dumps(dict(phase=phase, **fields), ensure_ascii=False), flush=True)


def compare_records(manifest, left, right):
    task_rows, all_rows, changed = [], [], []
    counts = collections.Counter()
    contract_fields = ('task_index', 'episode_index', 'task_name', 'task_group',
                       'constructor_seed', 'max_episode_steps', 'n_action_steps')
    for task in manifest['tasks']:
        index = task['task_index']
        per_task = collections.Counter()
        step_changes = request_changes = 0
        for e in range(50):
            a, b = left[index][e], right[index][e]
            for key in contract_fields:
                assert a[key] == b[key], 'Episode contract differs: %s/%s/%s' % (index, e, key)
            kind = ('both_success' if a['task_success'] and b['task_success'] else
                    'both_failure' if not a['task_success'] and not b['task_success'] else
                    'baseline_success_joint_failure' if a['task_success'] else
                    'baseline_failure_joint_success')
            row = {k: a[k] for k in contract_fields}
            row['outcome'] = kind
            fields = ('task_success', 'env_step_count', 'policy_request_count', 'terminated', 'truncated',
                      'outside_predicted_bounds_count', 'outside_executed_bounds_count', 'elapsed_seconds', 'job_id', 'node')
            row['baseline'] = {k: a.get(k) for k in fields}
            row['joint'] = {k: b.get(k) for k in fields}
            row['executed_step_count_changed'] = a['env_step_count'] != b['env_step_count']
            row['policy_request_count_changed'] = a['policy_request_count'] != b['policy_request_count']
            step_changes += int(row['executed_step_count_changed'])
            request_changes += int(row['policy_request_count_changed'])
            counts[kind] += 1; per_task[kind] += 1
            all_rows.append(row)
            if a['task_success'] != b['task_success']:
                changed.append(row)
        losses = per_task['baseline_success_joint_failure']
        gains = per_task['baseline_failure_joint_success']
        task_rows.append(dict(task_index=index, task_name=task['task_name'], task_group=task['task_group'],
                              baseline_successes=per_task['both_success'] + losses,
                              joint_successes=per_task['both_success'] + gains,
                              lost_successes=losses, gained_successes=gains, net_success_change=gains-losses,
                              changed_outcome_episodes=losses+gains,
                              executed_step_count_changes=step_changes, policy_request_count_changes=request_changes,
                              first_outcome_change_episode=min((r['episode_index'] for r in changed if r['task_index']==index), default=None)))
    task_rows.sort(key=lambda x: (-x['changed_outcome_episodes'], x['task_index']))
    return dict(counts), task_rows, all_rows, changed


def load_results(directory, manifest, protocol_sha):
    results = {}
    groups = {}
    for task in manifest['tasks']:
        index = task['task_index']; results[index] = {}
        paths = sorted((directory / 'tasks' / ('task_%03d' % index) / 'episodes').glob('episode_*.json'))
        assert len(paths) == 50, 'Missing/extra episode records: ' + str(index)
        group = groups.setdefault(task['task_group'], dict(episodes=0, successes=0))
        for e, p in enumerate(paths):
            item = read(p)
            assert p.name == 'episode_%03d.json' % e
            assert item['task_index'] == index and item['episode_index'] == e
            assert item['task_name'] == task['task_name'] and item['task_group'] == task['task_group']
            assert item['constructor_seed'] == 489000 + index*1000 + e
            assert item['protocol_sha256'] == protocol_sha and item['split'] == 'pretrain'
            assert item['engineering_valid'] is True and item['episode_completed'] is True
            assert type(item['task_success']) is bool
            assert 0 < item['env_step_count'] <= task['max_episode_steps']
            assert item['policy_request_count'] > 0
            assert item['extra_action_clipping_applied'] is False
            assert item['extra_action_denormalization_applied'] is False
            results[index][e] = item
            group['episodes'] += 1; group['successes'] += int(item['task_success'])
    return results, groups


def snapshot_plan(protocol):
    plan = {}
    def add(source, relative):
        assert source.is_file(), 'Missing source: ' + str(source)
        assert not source.is_symlink(), 'Unexpected evidence symlink: ' + str(source)
        relative = str(relative)
        assert relative not in plan, 'Duplicate archive path: ' + relative
        plan[relative] = dict(source=str(source), bytes=source.stat().st_size, mtime_ns=source.stat().st_mtime_ns)
    for tag, directory in (('baseline_s31', BASELINE), ('joint_s32', JOINT)):
        for name in ('protocol.json', 'target50_manifest.json', 'build_report.json', 'overall_results.json',
                     'worker.py', 'job.sh', 'auto_submit.py', 'auto_submit_state.json', 'auto_submit.log'):
            add(directory/name, Path(tag)/name)
        for pattern in ('slurm-*.out', 'submission.*'):
            for p in sorted(directory.glob(pattern)):
                if p.is_file(): add(p, Path(tag)/p.name)
        for task in range(50):
            td = directory/'tasks'/('task_%03d'%task)
            for pattern in ('*.json', '*.log', 'episodes/episode_*.json'):
                for p in sorted(td.glob(pattern)):
                    add(p, Path(tag)/p.relative_to(directory))
    for name in ('joint_audit.py', 'shadow_adapter.py', 'graph_model.py', 'baseline_episode_outcomes.json', 'lygwm-best.pt'):
        add(JOINT/name, Path('joint_s32')/name)
    ckpt = Path(protocol['checkpoint_path'])
    for p in sorted(ckpt.rglob('*')):
        if p.is_file(): add(p, Path('gr00t_checkpoint_120000')/p.relative_to(ckpt))
    for tag, directory, required in (
        ('gr00t_training', ROOT/'04_train_repro/r5_formal_train', ('formal_config.json', 'formal_results.json')),
        ('w5_training', W5, ('config.json', 'training_results.json', 'graph_model.py')),
        ('w6_probe', W6, ('report.json', 'w6_contract.json', 'worker.py', 'shadow_adapter.py', 'graph_model.py')),
    ):
        for name in required: add(directory/name, Path(tag)/name)
    # Archive implementation files; simulator assets and Python environments remain external.
    source = Path(next(iter(protocol['joint_model']['gr00t_source_sha256'])))
    source = next(p for p in source.parents if p.name.startswith('Isaac-GR00T-'))
    for p in sorted(source.rglob('*')):
        if any(s in ('.git', '__pycache__') for s in p.relative_to(source).parts): continue
        if p.is_file() and not p.is_symlink() and (p.suffix in ('.py', '.json', '.yaml', '.yml', '.toml', '.sh', '.md', '.txt') or p.name.startswith('LICENSE')):
            assert p.stat().st_size < 100 * 1024**2, 'Unexpectedly large source/config: ' + str(p)
            add(p, Path('gr00t_source')/p.relative_to(source))
    add(Path(__file__), Path('audit_tool')/'worker.py')
    add(RUN/'joint_audit_reference.py', Path('audit_tool')/'joint_audit_reference.py')
    return plan


def copy_verified(plan):
    payload = RUN / 'sealed_payload'; payload.mkdir(exist_ok=True)
    manifest_path = RUN/'file_manifest.json'
    existing = read(manifest_path) if manifest_path.exists() else {}
    required = sum(v['bytes'] for k,v in plan.items() if not (payload/k).exists())
    free = shutil.disk_usage(RUN).free
    assert free >= required + 2*1024**3, 'Insufficient free disk for independent archive copy'
    progress('copy_and_hash', archive_file_count=len(plan), archive_total_gib=round(sum(v['bytes'] for v in plan.values())/1024**3,3),
             required_additional_gib=round(required/1024**3,3), free_disk_gib=round(free/1024**3,3))
    receipts = {}
    completed_bytes = 0
    for i,(relative, info) in enumerate(sorted(plan.items()),1):
        source=Path(info['source']); target=payload/relative
        st=source.stat()
        assert (st.st_size,st.st_mtime_ns)==(info['bytes'],info['mtime_ns']), 'Source changed since planning: '+str(source)
        if i == 1 or i % 100 == 0 or info['bytes'] >= 64*1024**2:
            progress('copy_and_hash', current_file=relative, files_verified=i-1, verified_bytes=completed_bytes)
        target.parent.mkdir(parents=True,exist_ok=True)
        h=hashlib.sha256()
        if target.exists():
            source_sha=sha(source)
            assert target.stat().st_size==info['bytes'] and sha(target)==source_sha, 'Existing archive file differs: '+relative
        else:
            tmp=target.with_name(target.name+'.copying')
            with source.open('rb') as src, tmp.open('wb') as dst:
                for block in iter(lambda:src.read(8*1024*1024),b''):
                    h.update(block); dst.write(block)
                dst.flush(); os.fsync(dst.fileno())
            source_sha=h.hexdigest()
            assert sha(tmp)==source_sha and tmp.stat().st_size==info['bytes'], 'Copy readback mismatch'
            tmp.replace(target)
        end=source.stat()
        assert (end.st_size,end.st_mtime_ns)==(info['bytes'],info['mtime_ns']), 'Source changed during copy'
        target.chmod(0o444)
        receipt=dict(info, sha256=source_sha)
        if relative in existing: assert existing[relative]==receipt, 'Sealed receipt changed'
        receipts[relative]=receipt
        completed_bytes+=info['bytes']
        if i%100==0 or i==len(plan): save(manifest_path,receipts)
    assert set(existing).issubset(receipts)
    return receipts


def main():
    if os.environ.get('M489_W8_LOCK_FD'):
        lock=os.fdopen(int(os.environ['M489_W8_LOCK_FD']),'a')
    else:
        lock=(RUN/'work.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (RUN/'SEALED.json').exists(), 'Already sealed; use monitor to read final report'
    progress('preflight')
    q=subprocess.run(['squeue','-h','-r','-u','scx9fvq','-o','%i|%Z'],capture_output=True,text=True,timeout=60,check=True)
    assert not any(l.rsplit('|',1)[-1].strip() in (str(BASELINE),str(JOINT)) for l in q.stdout.splitlines()), 'Evaluation still active'
    for directory in (BASELINE,JOINT):
        assert read(directory/'auto_submit_state.json')['status']=='completed'
    bp=read(BASELINE/'protocol.json'); jp=read(JOINT/'protocol.json')
    assert digest(bp)==BASE_SHA and digest(jp)==JOINT_SHA
    assert {k:v for k,v in jp.items() if k!='joint_model'}==bp
    assert bp['manifest']==read(BASELINE/'target50_manifest.json')==read(JOINT/'target50_manifest.json')
    assert bp['split']==bp['manifest']['split']=='pretrain'
    assert jp['joint_model']['lygwm_checkpoint_sha256']==LY_SHA
    sys.path.insert(0,str(RUN))
    import joint_audit_reference as audit
    audit.verify_assets(JOINT,jp)
    progress('recompute_episode_audit')
    left,lg=load_results(BASELINE,bp['manifest'],BASE_SHA)
    right,rg=load_results(JOINT,jp['manifest'],JOINT_SHA)
    for index,records in left.items():
        for job in {str(x['job_id']) for x in records.values()}:
            assert job.isdigit()
            td=BASELINE/'tasks'/('task_%03d'%index)
            reports=[read(td/(prefix+job+'.json')) for prefix in ('server-report-','simulation-report-','report-')]
            for r in reports:
                assert str(r['job_id'])==job and r['protocol_sha256']==BASE_SHA and r['accepted'] is True
            assert reports[0]['successful_requests']==reports[1]['successful_requests']>0
            assert reports[0]['handler_errors']==0
    for saved in read(JOINT/'baseline_episode_outcomes.json')['episodes']:
        actual=left[saved['task_index']][saved['episode_index']]
        assert all(actual[key]==value for key,value in saved.items()), 'Baseline snapshot differs from original episode'
    for directory, groups, total in ((BASELINE,lg,181),(JOINT,rg,200)):
        final=read(directory/'overall_results.json')
        assert final['completed_tasks']==50 and final['valid_episodes']==2500 and final['all_2500_completed'] is True
        assert final['successes']==sum(x['successes'] for x in groups.values())==total
        for name,value in groups.items():
            assert final['groups'][name]['episodes']==value['episodes'] and final['groups'][name]['successes']==value['successes']
    evidence=audit.final_evidence(JOINT,right,jp)
    stored=read(JOINT/'overall_results.json')
    for key,value in evidence.items(): assert stored[key]==value, 'W7 final evidence mismatch: '+key
    assert evidence['shadow_execution_audit']['shadow_forward_count']==56760
    assert evidence['shadow_execution_audit']['shadow_prediction_samples']==283800
    counts, tasks, all_pairs, changed=compare_records(bp['manifest'],left,right)
    assert counts==EXPECTED_OUTCOMES and len(changed)==135
    # Compare simulation source structurally, ignoring formatting/comments.
    def sim_ast(p):
        nodes=[n for n in ast.parse(p.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='simulation']
        assert len(nodes)==1
        return ast.dump(nodes[0],include_attributes=False)
    assert sim_ast(BASELINE/'worker.py')==sim_ast(JOINT/'worker.py')
    progress('plan_archive', changed_outcome_episodes=135, tasks_with_outcome_changes=sum(t['changed_outcome_episodes']>0 for t in tasks))
    plan=snapshot_plan(jp)
    if (RUN/'copy_plan.json').exists(): assert read(RUN/'copy_plan.json')==plan, 'Source inventory changed; do not mix snapshots'
    else:save(RUN/'copy_plan.json',plan)
    receipts=copy_verified(plan)
    # Ensure all planned source entries still have the same sizes/times at closeout.
    for info in plan.values():
        st=Path(info['source']).stat()
        assert (st.st_size,st.st_mtime_ns)==(info['bytes'],info['mtime_ns']), 'Source changed before closeout'
    analysis=RUN/'analysis';analysis.mkdir(exist_ok=True)
    save(analysis/'all_2500_paired_episodes.json',all_pairs)
    save(analysis/'changed_135_episodes.json',changed)
    save(analysis/'task_comparison.json',tasks)
    summary=dict(
        w8_accepted=True, status='completed', split='pretrain', model_name=stored['model_name'],
        baseline_successes=181,joint_successes=200,baseline_success_rate=0.0724,joint_success_rate=0.08,
        change_percentage_points=0.76,paired_outcomes=counts,
        changed_outcome_episodes=135,tasks_with_outcome_changes=sum(t['changed_outcome_episodes']>0 for t in tasks),
        tasks_ranked_by_outcome_changes=tasks[:10],
        episodes_with_executed_step_count_changes=sum(x['executed_step_count_changed'] for x in all_pairs),
        episodes_with_policy_request_count_changes=sum(x['policy_request_count_changed'] for x in all_pairs),
        simulation_function_unchanged=True,shadow_outputs_influence_actions=False,
        baseline_component_reports_accepted=True,
        shadow_execution_audit=evidence['shadow_execution_audit'],
        archive_file_count=len(receipts),archive_total_bytes=sum(x['bytes'] for x in receipts.values()),
        archive_path=str(RUN/'sealed_payload'),file_manifest_sha256=sha(RUN/'file_manifest.json'),
        all_archive_file_copies_hash_verified=True,independent_copies=True,
        root_cause_identified=False,
        evidence_limits=['No per-step observation/action/RNG traces in the episode summaries; first trajectory divergence is not established',
                         'Same constructor seeds and unchanged simulation function do not establish bitwise trajectory reproducibility',
                         'Current shadow predictions do not select or modify actions; the score difference is not established LY-GWM benefit'],
        external_dependencies_not_bundled=['POLICY and SIM Python environments','RoboCasa/robosuite/robomimic source and simulator assets','offline Eagle processor/cache dependencies'],
        standalone_reproduction_bundle=False,leaderboard_submission_run=False,
        gpu_job_submitted=False,training_run=False,rollout_run=False,original_files_modified=False,errors=[])
    save(RUN/'audit_report.json',summary)
    seal=dict(status='sealed',created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              baseline_protocol_sha256=BASE_SHA,joint_protocol_sha256=JOINT_SHA,
              file_manifest_sha256=sha(RUN/'file_manifest.json'),audit_report_sha256=sha(RUN/'audit_report.json'),
              analysis_sha256={p.name:sha(p) for p in sorted(analysis.glob('*.json'))},
              note='Independent verified snapshot; source files remain unchanged; archive files are read-only, not immutable storage')
    save(RUN/'SEALED.json',seal)
    for p in list(analysis.glob('*.json'))+[RUN/'file_manifest.json',RUN/'audit_report.json',RUN/'SEALED.json',RUN/'copy_plan.json']:
        p.chmod(0o444)
    progress('completed',status='completed',w8_accepted=True,archive_path=summary['archive_path'],file_manifest_sha256=seal['file_manifest_sha256'])
    print('##### COMPACT_REPORT_BEGIN: M489_LYGWM_W8 #####',flush=True)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    print('##### COMPACT_REPORT_END: M489_LYGWM_W8 #####',flush=True)


if __name__=='__main__':
    try:main()
    except Exception as exc:
        traceback.print_exc()
        progress('failed',status='failed',errors=[dict(type=type(exc).__name__,error=str(exc))])
        raise SystemExit(1)
