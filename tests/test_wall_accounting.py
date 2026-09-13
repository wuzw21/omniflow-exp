import asyncio
import json
from pathlib import Path

import pytest

from omniflow.runtime.timing import TimingLedger, measure


def _paired_samples():
    from src.experiment.performance_metrics import PAIR_IDENTITY_FIELDS
    result = []
    for pair_id, ref, cand, ref_success, outcome in [
        ('a', 10, 7, True, 'actions_succeeded'),
        ('b', 20, 40, True, 'actions_failed'),
        ('c', 1, 100, False, 'unknown'),
    ]:
        identity = {field: 'frozen' for field in PAIR_IDENTITY_FIELDS}
        identity.update(task_name=pair_id, evaluation_seed=113, task_parameters={'value': 'new'})
        for condition, elapsed, success in [('memory_off', ref, ref_success), ('full', cand, True)]:
            result.append({'pair_id': pair_id, 'condition': condition, 'identity': dict(identity),
                           'official_success': success, 'execution_duration_ms': elapsed,
                           'replay_outcome': outcome})
    return result


def test_paired_latency_includes_recovered_success_and_separates_success_sets():
    from src.experiment.performance_metrics import summarize_paired_experiments
    report = summarize_paired_experiments(_paired_samples(), expected_pair_ids=['a', 'b', 'c'])
    from jsonschema import validate
    schema = json.loads((Path(__file__).parents[1] / 'schemas/omniflow_paired_experiments.v1.json').read_text())
    validate(report, schema)
    assert report['own_success_sets']['full']['success_mean_execution_ms'] == 49
    assert report['own_success_sets']['memory_off']['success_rate'] == pytest.approx(2/3)
    assert report['paired_full_system']['pair_count'] == 2
    assert report['paired_full_system']['candidate_mean_execution_ms'] == 23.5
    assert report['paired_full_system']['mean_delta_ms'] == 8.5  # recovery erases fast-path gain
    assert report['paired_fast_path']['mean_delta_ms'] == -3
    assert report['paired_recovered_success']['mean_delta_ms'] == 20
    assert report['paired_full_system']['task_cluster_bootstrap_delta_ci95_ms'] is not None
    assert report['paired_fast_path']['task_cluster_bootstrap_delta_ci95_ms'] is None


@pytest.mark.parametrize('field', ['model', 'device_model', 'task_parameters', 'evaluation_seed',
                                 'store_sha256', 'transfer_states_sha256',
                                  'endpoint_id', 'code_commit', 'transfer_sha256', 'source_sha256'])
def test_pairing_rejects_changed_experimental_controls(field):
    from src.experiment.performance_metrics import summarize_paired_experiments
    rows = _paired_samples()
    rows[1]['identity'][field] = 'different'
    with pytest.raises(ValueError, match='pair_identity_mismatch'):
        summarize_paired_experiments(rows, expected_pair_ids=['a', 'b', 'c'])


def test_missing_failed_and_unknown_runs_do_not_become_success_or_zero_latency():
    from src.experiment.performance_metrics import summarize_paired_experiments
    rows = _paired_samples()[:-1]
    rows[1]['execution_duration_ms'] = None
    report = summarize_paired_experiments(rows, expected_pair_ids=['a', 'b', 'c'])
    assert report['missing_pair_ids'] == ['c']
    assert report['own_success_sets']['full']['success_rate'] is None
    assert report['official_success_intersection_count'] == 2
    assert report['paired_full_system']['pair_count'] == 1
    assert report['paired_fast_path']['candidate_mean_execution_ms'] is None
    with pytest.raises(ValueError, match='duplicate_pair_condition'):
        summarize_paired_experiments(rows + [rows[0]], expected_pair_ids=['a', 'b', 'c'])


@pytest.mark.parametrize('elapsed', [float('nan'), float('inf'), -1, True])
def test_pairing_never_accepts_invalid_time_values(elapsed):
    from src.experiment.performance_metrics import summarize_paired_experiments
    rows = _paired_samples()
    rows[0]['execution_duration_ms'] = elapsed
    with pytest.raises(ValueError, match='invalid_execution_duration_ms'):
        summarize_paired_experiments(rows, expected_pair_ids=['a', 'b', 'c'])


@pytest.mark.parametrize('events,outcome', [
    ([{'success': False}, {'success': True}], 'actions_failed'),
    ([{'success': True}], 'actions_succeeded'),
    ([], 'not_attempted'), (None, 'unknown'),
])
def test_explicit_runlog_loader_keeps_earlier_failure_and_unknown_usage(tmp_path, events, outcome):
    from src.experiment.performance_metrics import paired_sample_from_runlog
    log = tmp_path / 'run_log.json'
    run = {'schema_version': 'omniflow.run_log.v1', 'task_name': 'example',
           'validator': {'official': True, 'success': True},
           'diagnostics': {'function_resume': {'events': events},
                           'wall_accounting': {'covered_wall_ms': 3, 'owner_delta_ms': 0.1,
                               'components': {'transfer': {'exclusive_ms': 1}, 'other': {'exclusive_ms': 2}}}}}
    log.write_text(json.dumps(run))
    row = {'task_name': 'example', 'task_random_seed': 113, 'task_params': {'value': 'new'},
           'official_validator_used': True, 'success': True, 'execution_duration_ms': 3.1,
           'duration_ms': 100, 'model': 'frozen', 'model_base_url': 'endpoint'}
    log.with_name('task_results.jsonl').write_text(json.dumps(row) + '\n')
    sample = paired_sample_from_runlog(log, pair_id='one', condition='full',
                                      environment_identity={'model': 'frozen', 'endpoint_id': 'endpoint'})
    assert sample['execution_duration_ms'] == 3.1
    assert sample['replay_outcome'] == outcome
    assert sample['model_calls'] is sample['total_tokens'] is None
    assert sample['component_status'] == 'reconciled'
    assert len(sample['run_log_sha256']) == 64
    run['diagnostics']['wall_accounting']['covered_wall_ms'] = 9
    log.write_text(json.dumps(run))
    with pytest.raises(ValueError, match='does_not_reconcile'):
        paired_sample_from_runlog(log, pair_id='one', condition='full',
                                 environment_identity={'model': 'frozen', 'endpoint_id': 'endpoint'})


def test_absent_component_is_zero_only_with_a_complete_ledger():
    from src.experiment.performance_metrics import summarize_paired_experiments
    rows = _paired_samples()
    rows[1].update(component_status='reconciled', component_wall_ms={'checker': 4})
    rows[3].update(component_status='reconciled', component_wall_ms={'other': 40})
    report = summarize_paired_experiments(rows, expected_pair_ids=['a', 'b', 'c'], bootstrap_repeats=0)
    checker = report['paired_full_system']['candidate_resources']['components']['checker']
    assert checker == {'observed_count': 2, 'mean_exclusive_ms': 2}


def test_androidworld_common_step_owner_accounts_failures_and_reset():
    from types import SimpleNamespace
    from src.integrations.android_world.run_episode import _ExperimentAgentAdapter
    class Agent:
        def reset(self):
            pass
        def step(self, goal):
            with measure('host.test'):
                raise RuntimeError('failed step')
    adapter = _ExperimentAgentAdapter(Agent(), recording_session=SimpleNamespace(
        env=SimpleNamespace(), start_episode=lambda: None))
    with pytest.raises(RuntimeError, match='failed step'):
        adapter.step('goal')
    report = adapter.execution_timing.report()
    assert report['components']['execution.other']['failed_calls'] == 1
    assert report['components']['host.test']['calls'] == 1
    assert 0 < report['covered_wall_ms'] <= adapter.execution_duration_ms
    adapter.reset()
    assert adapter.execution_timing.report()['covered_wall_ms'] == 0


def test_nested_wall_time_reconciles_without_double_counting():
    clock = [0]
    ledger = TimingLedger(clock=lambda: clock[0] * 1_000_000)
    with ledger.span('execution.other'):
        clock[0] = 2
        with measure('transfer'):
            clock[0] = 7
        clock[0] = 10
    report = ledger.report()
    from jsonschema import validate
    validate(report, json.loads((Path(__file__).parents[1]/'schemas/omniflow_wall_accounting.v1.json').read_text()))
    assert report['covered_wall_ms'] == report['accounted_wall_ms'] == 10
    assert report['components']['transfer']['exclusive_ms'] == 5
    assert report['components']['execution.other']['exclusive_ms'] == 5


def test_lifecycle_and_execution_are_separate_accounting_scopes():
    clock = [0]
    lifecycle = TimingLedger(clock=lambda: clock[0] * 1_000_000)
    execution = TimingLedger(clock=lambda: clock[0] * 1_000_000)
    with lifecycle.span('lifecycle.other'):
        with measure('setup'):
            clock[0] = 2
        with measure('lifecycle.execution'):
            with execution.span('execution.other'):
                with measure('completion_checker'):
                    clock[0] = 5
                clock[0] = 7
        with measure('official_validator'):
            clock[0] = 9
        clock[0] = 10
    report = lifecycle.report(owner_wall_ms=10.1)
    assert report['covered_wall_ms'] == report['accounted_wall_ms'] == 10
    assert report['owner_delta_ms'] == pytest.approx(.1)
    assert report['components']['lifecycle.execution']['exclusive_ms'] == 5
    assert 'completion_checker' not in report['components']
    assert execution.report()['covered_wall_ms'] == 5


def test_parallel_tasks_and_cancelled_span_remain_accounted():
    async def scenario():
        clock = [0]
        ledger = TimingLedger(clock=lambda: clock[0] * 1_000_000)
        entered, finish = asyncio.Event(), asyncio.Event()
        async def work():
            with measure('device'):
                entered.set()
                await finish.wait()
        with ledger.span('execution.other'):
            task = asyncio.create_task(work())
            await entered.wait()
            clock[0] = 2
            with measure('logging'):
                clock[0] = 5
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            clock[0] = 6
        report = ledger.report()
        assert report['covered_wall_ms'] == report['accounted_wall_ms'] == 6
        assert report['components']['concurrent']['exclusive_ms'] == 3
        assert report['components']['device']['failed_calls'] == 1
    asyncio.run(scenario())


def test_async_cached_observation_is_awaited_and_included_in_post_action_timing():
    from omniflow.runtime.core import execute_action
    from omniflow.core.config import PluginSet
    from omniflow.core.model import Action, ActionResult, Observation
    clock = [0]
    ledger = TimingLedger(clock=lambda: clock[0] * 1_000_000)
    after = Observation(xml='<after/>')
    class Host:
        async def act(self, action):
            return ActionResult(True)
        async def take_after_action_observation(self):
            clock[0] = 5
            return after
        async def observe(self, **kwargs):
            raise AssertionError('cached state must avoid another observe')
    with ledger.span('execution.other'):
        result = asyncio.run(execute_action(Action('wait', {'duration_ms': 1}),
            observation=Observation(xml='<before/>'), host=Host(), plugins=PluginSet()))
    assert result.after is after
    assert ledger.report()['components']['observe.post_action']['exclusive_ms'] == 5


@pytest.mark.parametrize('fails', [False, True])
def test_model_stream_first_event_and_later_processing_are_distinct(fails):
    from omniflow.vlm.planner import normalize_openai_model_turn_response
    clock = [0]
    ledger = TimingLedger(clock=lambda: clock[0] * 1_000_000)
    def stream():
        clock[0] = 5
        if fails:
            raise TimeoutError('first event timeout')
        yield {'model': 'test', 'choices': []}
        clock[0] = 9
    with ledger.span('model.planner'):
        if fails:
            with pytest.raises(TimeoutError):
                normalize_openai_model_turn_response(stream(), requested_model='test')
        else:
            assert normalize_openai_model_turn_response(stream(), requested_model='test')['tool_calls'] == []
    report = ledger.report()
    components = report['components']
    assert components['model.planner.first_stream_event']['exclusive_ms'] == 5
    assert components['model.planner.first_stream_event']['failed_calls'] == int(fails)
    assert components['model.planner.stream']['exclusive_ms'] == (0 if fails else 4)
    assert report['accounted_wall_ms'] == report['covered_wall_ms']
