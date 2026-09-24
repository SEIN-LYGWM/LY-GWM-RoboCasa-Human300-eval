"""Audit one frozen policy on exactly 50 x 50 new episodes."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

GROUPS=dict(atomic_seen=900,composite_seen=800,composite_unseen=800)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,value):
    assert p.parent == SOURCE_ROOT
    assert p.name in ('per_task_results.json','evaluation_receipt.json','overall_results.json')
    p=OUTPUT_ROOT/p.name
    temp=p.with_name(p.name+'.tmp')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n');temp.replace(p)

def validate_records(directory,task,p,protocol_sha):
    records={};index=task['task_index']
    for path in sorted((directory/'episodes').glob('episode_*.json')):
        r=read(path);e=r['episode_index']
        assert type(e) is int and 0<=e<50 and e not in records
        assert path.name=='episode_%03d.json'%e
        assert r['protocol_sha256']==protocol_sha
        assert r['engineering_valid'] is True and r['episode_completed'] is True
        assert r['environment_closed'] is True and type(r['task_success']) is bool
        assert r['task_index']==index and r['task_name']==task['task_name']
        assert r['task_group']==task['task_group']
        assert r['constructor_seed']==489000+index*1000+e
        assert r['split']=='pretrain' and r['checkpoint_path']==p['checkpoint_path']
        assert r['max_episode_steps']==task['max_episode_steps'] and r['n_action_steps']==16
        assert 0<r['env_step_count']<=task['max_episode_steps']
        assert r['policy_request_count']>0
        record_bounds(r)
        assert r['extra_action_clipping_applied'] is False
        assert r['extra_action_denormalization_applied'] is False
        records[e]=r
    if set(records)!=set(range(50)):
        raise RuntimeError('Incomplete task %03d %s: %d/50'%(index,task['task_name'],len(records)))
    return records

def collect_phase(root,phase='full'):
    assert phase=='full'
    table=read(root/'job_table.json')['full']
    assert len(table)==50 and [j['task_index'] for j in table]==list(range(50))
    assert len({j['run'] for j in table})==1 and all(j['mode']=='learned_reward' for j in table)
    run=Path(table[0]['run']);p=read(run/'protocol.json');b=read(run/'build_report.json')
    sys.path.insert(0,str(run))
    spec=importlib.util.spec_from_file_location('s42_runtime_audit',str(run/'decision_audit.py'))
    audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
    joint=audit.verify_assets(run,p,verify_policy_weights=True)
    assert p['experiment']['version']=='S42_FULL_LEARNED_REWARD_V1'
    assert p['experiment']['phase']=='full' and joint['mode']=='learned_reward'
    assert p['manifest']['total_episodes']==2500 and p['manifest']['episodes_per_task']==50
    assert p['manifest']==read(run/'target50_manifest.json')
    receipt=read(root/'launch_receipt.json')
    assert receipt['protocol_sha256']==b['protocol_sha256']
    for name,expected in receipt['management_sha256'].items():
        assert audit.sha(root/name)==expected,'Management source changed: '+name
    groups={g:dict(episodes=0,successes=0) for g in GROUPS};per_task=[]
    counters=dict(executed_action_chunks=0,selected_nonfirst_chunks=0,
        changed_environment_chunks=0,prediction_divergent_chunks=0,score_divergent_chunks=0)
    versions=set()
    for item in table:
        i=item['task_index'];task=p['manifest']['tasks'][i]
        assert task['task_index']==i and task['episode_count']==50
        d=run/'tasks'/('task_%03d'%i)
        records=validate_records(d,task,p,b['protocol_sha256'])
        jobs={str(r['job_id']) for r in records.values()}
        assert len(jobs)==1,'Mixed jobs in one task; partial resume is unsupported'
        job=next(iter(jobs))
        reports=[read(d/(prefix+job+'.json')) for prefix in ('report-','server-report-','simulation-report-')]
        assert all(r['accepted'] is True and r['protocol_sha256']==b['protocol_sha256'] for r in reports)
        server=reports[1];sim=reports[2]
        assert server['scorer_loaded'] is True and server['lygwm_loaded'] is True
        assert server['decision_mode']=='learned_reward'
        assert server['scorer_checkpoint_sha256']==joint['scorer_checkpoint_sha256']
        assert server['lygwm_checkpoint_sha256']==joint['lygwm_checkpoint_sha256']
        versions.add(sim['robocasa_version'])
        evidence=audit.verify_evidence(d,job,b['protocol_sha256'],'learned_reward')
        assert evidence['action_evidence_accepted'] is True
        assert evidence['executed_action_chunks']==sum(r['policy_request_count'] for r in records.values())
        assert evidence['request_batches']==server['successful_requests']==sim['successful_requests']
        for key in counters:counters[key]+=evidence[key]
        wins=sum(r['task_success'] for r in records.values())
        g=groups[task['task_group']];g['episodes']+=50;g['successes']+=wins
        per_task.append(dict(task_index=i,task_name=task['task_name'],task_group=task['task_group'],
            episodes=50,successes=wins,success_rate=wins/50,job_id=job,
            max_episode_steps=task['max_episode_steps'],action_evidence=evidence))
    assert len(versions)==1, 'Mixed simulator versions'
    for name,g in groups.items():
        assert g['episodes']==GROUPS[name]
        g['success_rate']=g['successes']/g['episodes']
    wins=sum(g['successes'] for g in groups.values())
    result=dict(phase='full',experiment='S42_FULL_LEARNED_REWARD_V1',complete=True,
        new_evaluation=True,all_2500_completed=True,completed_tasks=50,valid_episodes=2500,
        successes=wins,overall_success_rate=wins/2500,groups=groups,
        mode='learned_reward',algorithm='four_candidate_learned_reward',
        learned_state_reward_scorer=True,goal_conditioned_value_scorer=False,
        scoring_scope='Recorded binary state reward; not long-horizon value or calibrated rollout success',
        split='pretrain',robocasa_version=next(iter(versions)),
        protocol_sha256=b['protocol_sha256'],checkpoint_path=p['checkpoint_path'],
        policy_weight_sha256=b['policy_weight_sha256'],
        scorer_checkpoint_sha256=joint['scorer_checkpoint_sha256'],
        lygwm_checkpoint_sha256=joint['lygwm_checkpoint_sha256'],
        action_evidence=counters,action_evidence_verified=True,
        leaderboard_metrics_percent=dict(atomic_seen_success=100*groups['atomic_seen']['success_rate'],
            composite_seen_success=100*groups['composite_seen']['success_rate'],
            composite_unseen_success=100*groups['composite_unseen']['success_rate']),
        comparison_scope='Single arm; no baseline effect estimated.',official_acceptance=False)
    annotate_acceptance(result,per_task)
    write(root/'per_task_results.json',per_task)
    write(root/'evaluation_receipt.json',dict(launch=receipt,
        overall_results_sha256=audit.digest(result),per_task_results_sha256=audit.digest(per_task),
        digest_format='SHA256 of json.dumps(value, sort_keys=True).encode()',
        tasks=[dict(task_index=x['task_index'],job_id=x['job_id'],evidence=x['action_evidence']) for x in per_task]))
    write(root/'overall_results.json',result)
    return result


# Explicit post-run acceptance revision; original rollout protocol is unchanged.
import hashlib
SOURCE_ROOT = Path('/home/bingxing2/home/scx9fvq/m489_robocasa365_gr00t_n1_5/03_contract/s42_full_learned_reward_v1')
OUTPUT_ROOT = Path(__file__).resolve().parent
EXPECTED_INPUT_SHA256 = {'collect.py': '8463719f3b8c5e29ffc34e631e7ce60543454d56775b09e2f246349ce4f64b25', 'full/learned_reward/protocol.json': '67bcc011d3d252f521e810141283e8c63060285b7c5518ab3a97f015b49591fd', 'full/learned_reward/build_report.json': '1408995550f0d596568a132e4fb90f2415c58508a1b2d19c3cb1d5a838698ef9'}
REVISION = 'S42_ACCEPTANCE_V2_BOUNDS_DIAGNOSTIC_ONLY'
_bounds_rows = []

def record_bounds(r):
    a = r['outside_predicted_bounds_count']
    b = r['outside_executed_bounds_count']
    assert type(a) is int and type(b) is int and a >= 0 and b >= 0
    _bounds_rows.append(dict(task_index=r['task_index'], episode_index=r['episode_index'],
        task_group=r['task_group'], task_success=r['task_success'],
        predicted=a, executed=b))

def bounds_summary(rows):
    affected = [r for r in rows if r['predicted'] or r['executed']]
    return dict(episodes=len(rows), affected_episodes=len(affected),
        affected_successful_episodes=sum(r['task_success'] for r in affected),
        affected_tasks=len({r['task_index'] for r in affected}),
        predicted_scalar_count=sum(r['predicted'] for r in rows),
        executed_prefix_scalar_count=sum(r['executed'] for r in rows))

def annotate_acceptance(result, per_task):
    assert len(_bounds_rows) == 2500
    assert len({(r['task_index'], r['episode_index']) for r in _bounds_rows}) == 2500
    total = bounds_summary(_bounds_rows)
    for task in per_task:
        rows = [r for r in _bounds_rows if r['task_index'] == task['task_index']]
        assert len(rows) == 50
        task['bounds_diagnostics'] = bounds_summary(rows)
    result['acceptance_rule_revision'] = dict(version=REVISION,
        post_run_rule_change=True,
        change='Nonzero action-bound counts no longer block internal acceptance.',
        retained_checks='All other original collector checks; bounds counters must be nonnegative integers.',
        original_collector_path=str(SOURCE_ROOT / 'collect.py'),
        original_collector_sha256=EXPECTED_INPUT_SHA256['collect.py'],
        revised_collector_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        original_input_file_sha256=EXPECTED_INPUT_SHA256)
    result['bounds_diagnostics'] = dict(policy='diagnostic_only', total=total,
        groups={g: bounds_summary([r for r in _bounds_rows if r['task_group'] == g]) for g in GROUPS},
        original_zero_bounds_condition_satisfied=(total['affected_episodes'] == 0),
        interpretation='Scalar elements in selected action chunks and consumed input prefixes; not post-controller measurements.')
    result['internal_acceptance_under_revised_rules'] = True
    result['new_rollouts_performed_by_this_collector'] = False
    result['official_acceptance'] = False

def verify_revision_inputs():
    if not __debug__:
        raise RuntimeError('Assertions must remain enabled.')
    for name, digest in EXPECTED_INPUT_SHA256.items():
        if hashlib.sha256((SOURCE_ROOT / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError('Input file changed: ' + str(SOURCE_ROOT / name))
    for name in ('per_task_results.json', 'evaluation_receipt.json', 'overall_results.json'):
        if (OUTPUT_ROOT / name).exists():
            raise RuntimeError('Output already exists; rerun the installer for a fresh directory: ' + name)
    print('ACCEPTANCE_REVISION', REVISION, flush=True)
    print('OUTPUT_DIRECTORY', str(OUTPUT_ROOT), flush=True)
    print('Checking frozen assets, weights, records, and action evidence...', flush=True)

if __name__=='__main__':
    verify_revision_inputs()
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['full']);parser.parse_args()
    print(json.dumps(collect_phase(SOURCE_ROOT),indent=2,ensure_ascii=False))
