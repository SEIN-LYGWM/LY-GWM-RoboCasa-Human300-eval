"""Audit one frozen policy on exactly 50 x 50 new episodes."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

GROUPS=dict(atomic_seen=900,composite_seen=800,composite_unseen=800)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,value):
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
        assert r['outside_predicted_bounds_count']==r['outside_executed_bounds_count']==0
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
    write(root/'per_task_results.json',per_task)
    write(root/'evaluation_receipt.json',dict(launch=receipt,
        overall_results_sha256=audit.digest(result),per_task_results_sha256=audit.digest(per_task),
        digest_format='SHA256 of json.dumps(value, sort_keys=True).encode()',
        tasks=[dict(task_index=x['task_index'],job_id=x['job_id'],evidence=x['action_evidence']) for x in per_task]))
    write(root/'overall_results.json',result)
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['full']);parser.parse_args()
    print(json.dumps(collect_phase(Path(__file__).resolve().parent),indent=2,ensure_ascii=False))
