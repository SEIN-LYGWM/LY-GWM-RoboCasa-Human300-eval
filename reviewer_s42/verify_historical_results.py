"""Read-only verification of exported records. Full historical traces are not bundled."""
import json
from pathlib import Path
from prepare_run import PACKAGE, ORIGINAL, read, sha, digest, need, safe_join
from collect_results import validate_records

def verify():
    need(__debug__, 'Do not use -O')
    manifest=read(PACKAGE/'releases/s42/package_manifest.json')
    for rel,m in manifest['files'].items():
        p=safe_join(PACKAGE,rel)
        need(p.stat().st_size==m['bytes'] and sha(p)==m['sha256'], 'Snapshot mismatch: '+rel)
    source=PACKAGE/'evaluated_snapshot/s42'
    result=read(source/'acceptance_v2/overall_results.json')
    tasks=read(source/'acceptance_v2/per_task_results.json')
    receipt=read(source/'acceptance_v2/evaluation_receipt.json')
    need(digest(result)==receipt['overall_results_sha256'], 'Result receipt mismatch')
    need(digest(tasks)==receipt['per_task_results_sha256'], 'Task receipt mismatch')
    p=read(ORIGINAL/'protocol.json'); b=read(ORIGINAL/'build_report.json')
    need(digest(p)==b['protocol_sha256']==result['protocol_sha256'], 'Protocol mismatch')
    for rel,h in b['generated_file_sha256'].items():
        need(sha(ORIGINAL/rel)==h, 'Original source changed: '+rel)
    for rel,h in receipt['launch']['management_sha256'].items():
        need(sha(source/'s42_original'/rel)==h, 'Management source changed: '+rel)
    need(len(tasks)==50 and [t['task_index'] for t in tasks]==list(range(50)), 'Task coverage')
    groups={g:dict(episodes=0,successes=0) for g in result['groups']}
    bounds=dict(episodes=0,affected_episodes=0,affected_successful_episodes=0,affected_tasks=0,predicted_scalar_count=0,executed_prefix_scalar_count=0)
    counts={k:0 for k in result['action_evidence']}
    for task, saved in zip(p['manifest']['tasks'],tasks):
        d=ORIGINAL/'tasks'/('task_%03d'%task['task_index'])
        records=validate_records(d,task,p,b['protocol_sha256'])
        jobs={str(r['job_id']) for r in records.values()}
        need(jobs=={saved['job_id']}, 'Job identity mismatch')
        wins=sum(r['task_success'] for r in records.values())
        need(wins==saved['successes'] and saved['episodes']==50, 'Task score mismatch')
        g=groups[task['task_group']];g['episodes']+=50;g['successes']+=wins
        evidence=saved['action_evidence']
        need(evidence['action_evidence_accepted'] is True, 'Original evidence report failed')
        need(evidence['executed_action_chunks']==sum(r['policy_request_count'] for r in records.values()), 'Chunk counts')
        for k in counts:counts[k]+=evidence[k]
        reports=[read(d/(prefix+saved['job_id']+'.json')) for prefix in ('report-','server-report-','simulation-report-')]
        need(all(r['accepted'] is True and r['protocol_sha256']==b['protocol_sha256'] for r in reports),'Report mismatch')
        server=reports[1];sim=reports[2]
        need(server['scorer_loaded'] is True and server['lygwm_loaded'] is True and server['decision_mode']=='learned_reward','Model report')
        need(server['scorer_checkpoint_sha256']==result['scorer_checkpoint_sha256'] and server['lygwm_checkpoint_sha256']==result['lygwm_checkpoint_sha256'],'Model hash record')
        need(evidence['request_batches']==server['successful_requests']==sim['successful_requests'],'Request counts')
        need(sim['robocasa_version']==result['robocasa_version'],'Simulator version')
        affected=0
        for r in records.values():
            x=r['outside_predicted_bounds_count'];y=r['outside_executed_bounds_count'];flag=x>0 or y>0
            bounds['episodes']+=1;bounds['affected_episodes']+=flag;affected+=flag
            bounds['affected_successful_episodes']+=flag and r['task_success']
            bounds['predicted_scalar_count']+=x;bounds['executed_prefix_scalar_count']+=y
        bounds['affected_tasks']+=affected>0
    for g in groups.values():g['success_rate']=g['successes']/g['episodes']
    need(groups==result['groups'], 'Group result mismatch')
    need(bounds==result['bounds_diagnostics']['total'], 'Bounds diagnostics mismatch')
    need(counts==result['action_evidence'], 'Action report totals mismatch')
    need(sum(g['successes'] for g in groups.values())==result['successes']==207,'Overall result')
    return dict(exported_file_hashes_verified=True,episode_records_verified=2500,successes=207,
        receipt_digests_verified=True,recorded_action_evidence_totals=counts,
        original_zero_bounds_condition_satisfied=False,internal_acceptance_under_revised_rules=True,
        historical_full_action_trace_audit_replayed=False,weights_loaded=False,gpu_rollouts_run=False,
        official_acceptance=False)
if __name__=='__main__':print(json.dumps(verify(),indent=2))
