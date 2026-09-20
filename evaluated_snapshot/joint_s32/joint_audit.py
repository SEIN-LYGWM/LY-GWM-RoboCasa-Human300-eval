"""Read-only provenance checks and matched-episode accounting for W7."""
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def verify_assets(run, protocol):
    run = Path(run)
    build = read(run / 'build_report.json')
    assert build['build_accepted'] is True
    assert digest(protocol) == build['protocol_sha256'], 'Protocol changed'
    for name, expected in build['generated_file_sha256'].items():
        assert sha(run / name) == expected, 'Joint file changed: ' + name
    joint = protocol['joint_model']
    assert joint['mode'] == 'learned_graph_dynamics_shadow_prediction'
    assert joint['shadow_outputs_influence_actions'] is False
    assert sha(joint['lygwm_checkpoint']) == joint['lygwm_checkpoint_sha256']
    for path, expected in joint['gr00t_source_sha256'].items():
        assert sha(path) == expected, 'GR00T source changed: ' + path
    ckpt = Path(protocol['checkpoint_path'])
    for name, field in (
        ('_COMPLETE.json', 'checkpoint_commit_sha256'),
        ('config.json', 'checkpoint_config_sha256'),
        ('experiment_cfg/metadata.json', 'checkpoint_metadata_sha256'),
    ):
        assert sha(ckpt / name) == protocol[field], 'GR00T checkpoint metadata changed'
    return joint


def verify_task_reports(run, results, indices, protocol):
    """Every episode is tied to an accepted server AND simulator invocation."""
    run = Path(run)
    protocol_sha = digest(protocol)
    weight_sha = protocol['joint_model']['lygwm_checkpoint_sha256']
    forward_count = 0
    prediction_samples = 0
    for index in indices:
        episodes = results[index]
        assert len(episodes) == 50, 'Incomplete task: ' + str(index)
        jobs = {str(item['job_id']) for item in episodes.values()}
        for job in jobs:
            assert job.isdigit(), 'Invalid episode job identity'
            directory = run / 'tasks' / ('task_%03d' % index)
            server = read(directory / ('server-report-' + job + '.json'))
            sim = read(directory / ('simulation-report-' + job + '.json'))
            combined = read(directory / ('report-' + job + '.json'))
            for report in (server, sim, combined):
                assert str(report['job_id']) == job
                assert report['protocol_sha256'] == protocol_sha
                assert report['accepted'] is True
            assert server['lygwm_loaded'] is True
            assert server['lygwm_checkpoint_sha256'] == weight_sha
            assert server['shadow_outputs_influence_actions'] is False
            assert server['handler_errors'] == 0
            count = server['successful_requests']
            assert count > 0
            assert count == server['shadow_forward_count'] == sim['successful_requests']
            assert combined['joint_shadow_accepted'] is True
            sample_count = sum(
                item['policy_request_count'] for item in episodes.values()
                if str(item['job_id']) == job
            )
            assert sample_count == server['shadow_prediction_samples'], 'Shadow/episode request coverage differs'
            assert server['first_shadow_prediction']['finite'] is True
            assert server['last_shadow_prediction']['finite'] is True
            forward_count += count
            prediction_samples += sample_count
    return dict(shadow_forward_count=forward_count,
                shadow_prediction_samples=prediction_samples,
                episode_request_coverage_accepted=True)


def final_evidence(run, results, protocol):
    audit = verify_task_reports(run, results, range(50), protocol)
    baseline = read(Path(run) / 'baseline_episode_outcomes.json')
    assert baseline['protocol_sha256'] == protocol['joint_model']['baseline_protocol_sha256']
    counts = dict(both_success=0, both_failure=0,
                  baseline_success_joint_failure=0, baseline_failure_joint_success=0)
    assert len(baseline['episodes']) == 2500
    seen = set()
    for old in baseline['episodes']:
        key = (old['task_index'], old['episode_index'])
        assert key not in seen
        seen.add(key)
        new = results[key[0]][key[1]]
        for field in ('task_name', 'task_group', 'constructor_seed', 'max_episode_steps', 'n_action_steps'):
            assert new[field] == old[field], 'Paired evaluation mismatch: ' + field
        a, b = old['task_success'], new['task_success']
        name = ('both_success' if a and b else 'both_failure' if not a and not b
                else 'baseline_success_joint_failure' if a else 'baseline_failure_joint_success')
        counts[name] += 1
    old_success = counts['both_success'] + counts['baseline_success_joint_failure']
    new_success = counts['both_success'] + counts['baseline_failure_joint_success']
    assert old_success == 181
    return dict(
        w7_rollout_accepted=True,
        model_name=protocol['joint_model']['model_name'],
        joint_mode=protocol['joint_model']['mode'],
        shadow_outputs_influence_actions=False,
        lygwm_checkpoint_sha256=protocol['joint_model']['lygwm_checkpoint_sha256'],
        gr00t_parameters=2724163520, lygwm_parameters=10237204,
        combined_parameters=2734400724,
        shadow_execution_audit=audit,
        baseline_comparison=dict(
            paired_episodes=2500, baseline_successes=old_success,
            joint_successes=new_success, baseline_success_rate=old_success / 2500,
            joint_success_rate=new_success / 2500,
            success_rate_difference_percentage_points=(new_success - old_success) / 25,
            observed_success_count_not_lower=new_success >= old_success,
            all_paired_success_outcomes_identical=(counts['both_success'] + counts['both_failure'] == 2500),
            outcomes=counts,
            scope='Same manifest, seeds and environment protocol; descriptive comparison, not proof of general non-inferiority or LY-GWM benefit',
        ),
    )
