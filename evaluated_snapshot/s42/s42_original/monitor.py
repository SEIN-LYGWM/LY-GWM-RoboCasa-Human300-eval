"""Read-only progress; audit automatically when all parent jobs complete."""
import argparse
import datetime
import json
from pathlib import Path
import subprocess
import time

def read(path):
    for attempt in range(3):
        try:return json.loads(path.read_text(encoding='utf-8-sig'))
        except (OSError,ValueError):
            if attempt==2:raise
            time.sleep(0.3)

def snapshot(root):
    intent=read(root/'full_submission_intent.json')
    job=intent.get('array_job_id')
    if not job:raise RuntimeError('No confirmed job ID; inspect full_submission.out/.err before any retry')
    print('\n===== S42_STATUS %s ====='%datetime.datetime.now().isoformat(' ',timespec='seconds'),flush=True)
    print('SUBMISSION '+json.dumps(intent))
    q=subprocess.run(['squeue','-h','-j',job,'-o','%i %T %M %R'],
        stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=30)
    active=bool(q.stdout.strip())
    print('QUEUE '+(q.stdout.strip() or '(no active entries)'))
    if q.returncode:print('QUEUE_QUERY_NOTE '+q.stderr.strip())
    table=read(root/'job_table.json')['full'];completed=0;successes=0;accepted=0;done=0;details=[];unreadable=[]
    for item in table:
        i=item['task_index'];d=Path(item['run'])/'tasks'/('task_%03d'%i);p=d/'task_summary.json'
        if not p.exists():continue
        try:
            s=read(p);n=s['completed_episodes'];completed+=n;successes+=s['successes'];done+=int(n==50)
            parent=d/('report-'+str(s['last_job_id'])+'.json')
            r=read(parent) if parent.exists() else {}
            ok=n==50 and r.get('accepted') is True;accepted+=int(ok)
            if not ok:details.append(dict(task_index=i,task=s['task_name'],completed=n,expected=50,
                successes=s['successes'],error=s.get('last_error') or r.get('error')))
        except (OSError,ValueError,KeyError) as e:unreadable.append(dict(task_index=i,error=str(e)))
    print(json.dumps(dict(completed_tasks=done,total_tasks=50,accepted_tasks=accepted,
        completed_episodes=completed,expected_episodes=2500,successes_so_far=successes,
        progress_percent=round(completed/25,2),success_rate_so_far_percent=round(100*successes/completed,3) if completed else None,
        scores_are_provisional=True),ensure_ascii=False))
    for detail in details:print(json.dumps(detail,ensure_ascii=False))
    for detail in unreadable:print('READ_RETRY_NEEDED '+json.dumps(detail))
    if accepted==50 and not unreadable:
        print('All 2500 episodes reported; verifying weights, records and action traces...',flush=True)
        from collect import collect_phase
        result=collect_phase(root)
        print('S42_COMPLETE=True\n'+json.dumps(result,indent=2,ensure_ascii=False),flush=True)
        return 'complete'
    if active or q.returncode and 'Invalid job id' not in q.stderr:return 'running'
    return 'inactive'

def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['full'])
    parser.add_argument('--watch',action='store_true');args=parser.parse_args()
    root=Path(__file__).resolve().parent;inactive=0
    while True:
        status=snapshot(root)
        if status=='complete':break
        inactive=inactive+1 if status=='inactive' else 0
        if inactive>=3 or status=='inactive' and not args.watch:
            job=read(root/'full_submission_intent.json')['array_job_id']
            print('Full evaluation is not verified complete. No job will be resubmitted automatically.')
            print('Inspect: sacct -j '+job+' --format=JobID,State,ExitCode,Elapsed,NodeList')
            print('Also inspect full-<array_job_id>_<task_index>.out and task reports.');break
        if not args.watch:break
        time.sleep(30)

if __name__=='__main__':
    try:main()
    except KeyboardInterrupt:print('\nMonitor stopped; GPU jobs continue.')
