"""Experiment integrity and action evidence, independent of historical scores."""
import hashlib
import json
from pathlib import Path
from selection_rule import select_medoid, select_reward


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def digest(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True).encode()).hexdigest()


def action_sha(action):
    h = hashlib.sha256()
    for key,value in sorted(action.items()):
        h.update(json.dumps([key,list(value.shape)],separators=(',',':')).encode())
        h.update(value.astype('<f4').tobytes(order='C'))
    return h.hexdigest()


def verify_assets(run, protocol, verify_policy_weights=True):
    run = Path(run)
    build = read(run/'build_report.json')
    assert build['build_accepted'] is True
    assert digest(protocol) == build['protocol_sha256'], 'Protocol changed'
    for name,expected in build['generated_file_sha256'].items():
        assert sha(run/name)==expected, 'Experiment source changed: '+name
    joint = protocol['joint_model']
    assert joint['mode'] in ('first_candidate','learned_reward')
    assert joint['candidate_count'] == 4
    assert joint['selection_rule'] == 'argmax_s38_reward_logit_of_lygwm_predicted_features_and_state'
    assert sha(joint['lygwm_checkpoint']) == joint['lygwm_checkpoint_sha256']
    assert sha(joint['scorer_checkpoint']) == joint['scorer_checkpoint_sha256']
    assert sha(joint['scorer_config_path']) == joint['scorer_config_sha256']
    for path,expected in joint['gr00t_source_sha256'].items():
        assert sha(path)==expected, 'Policy source changed: '+path
    for path,expected in protocol['diagnostic_source_sha256'].items():
        assert sha(Path(path))==expected, 'Diagnostic label source changed: '+path
    ckpt = Path(protocol['checkpoint_path'])
    for name,field in (
        ('_COMPLETE.json','checkpoint_commit_sha256'),
        ('config.json','checkpoint_config_sha256'),
        ('experiment_cfg/metadata.json','checkpoint_metadata_sha256'),
    ):
        assert sha(ckpt/name)==protocol[field], 'Policy metadata changed'
    # Verify checkpoint tensor files too, not just its config.
    if verify_policy_weights:
        for name,expected in build['policy_weight_sha256'].items():
            assert sha(ckpt/name)==expected, 'Policy weight changed: '+name
    return joint


def verify_evidence(directory, job, protocol_sha, mode):
    directory = Path(directory)
    selection = directory/('selection-'+job+'.jsonl')
    execution = directory/('execution-'+job+'.jsonl')
    expected = {}
    calls = 0
    changed = 0
    nonfirst = 0
    divergent = 0
    score_divergent = 0
    with selection.open() as f:
        for line in f:
            event=json.loads(line)
            calls += 1
            assert event['protocol_sha256']==protocol_sha
            assert event['call']==calls and event['mode']==mode
            assert event['candidate_count']==4 and event['finite'] is True
            assert event['scorer_checkpoint_sha256']=='5f9ad4b52eddfc2ca3f1ac06efcae3f20a02a764857987fe447810d5cde68a8b'
            assert len(event['rows'])==event['batch_size']
            if calls==1:
                assert event['cached_head_replay_max_abs'] is not None
                assert event['cached_head_replay_max_abs']<=1e-6
            for row in event['rows']:
                _,_,spread=select_medoid(row['predicted_states'])
                assert spread==row['prediction_spread']
                best=select_reward(row['scores'])
                assert row['reward_logit_spread']==max(row['scores'])-min(row['scores'])
                score_divergent+=int(row['reward_logit_spread']>0)
                selected=best if mode=='learned_reward' else 0
                assert row['reward_best_index']==best and row['selected_index']==selected
                assert row['selected_action_sha256']==row['candidate_action_sha256'][selected]
                assert len(row['candidate_action_sha256'])==4
                assert row['selected_minus_first_logit']==row['scores'][selected]-row['scores'][0]
                if mode=='learned_reward':
                    assert row['selected_minus_first_logit']>=0
                assert row['environment_action_changed']==(row['environment_action_sha256']!=row['first_environment_action_sha256'])
                context=row['context']
                key=(context['episode_index'],context['request_index'])
                assert key not in expected
                expected[key]=row['environment_action_sha256']
                nonfirst += int(selected!=0)
                changed += int(row['environment_action_changed'])
                divergent += int(spread>1e-12)
                if mode=='first_candidate':
                    assert row['environment_action_changed'] is False
    matched=0
    with execution.open() as f:
        for line in f:
            event=json.loads(line)
            assert event['protocol_sha256']==protocol_sha
            key=(event['episode_index'],event['request_index'])
            assert expected.pop(key)==event['received_action_sha256']
            assert 0<event['executed_steps']<=16
            matched+=1
    assert not expected and calls>0 and matched>0
    return dict(request_batches=calls,executed_action_chunks=matched,
                selected_nonfirst_chunks=nonfirst,changed_environment_chunks=changed,
                prediction_divergent_chunks=divergent,score_divergent_chunks=score_divergent,
                action_evidence_accepted=True,
                selection_trace_sha256=sha(selection),execution_trace_sha256=sha(execution))
