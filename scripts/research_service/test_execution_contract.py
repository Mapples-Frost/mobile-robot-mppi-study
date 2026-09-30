"""Control-plane tests use temporary synthetic evidence, never simulation or APIs."""
import copy
import hashlib
import json
import os
import pathlib
import tempfile
import unittest
from unittest.mock import patch
import execution_contract as ec


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = pathlib.Path(self.tmp.name) / 'repo'; self.root.mkdir()
        self.state = pathlib.Path(self.tmp.name) / 'state'; self.state.mkdir()
        self.script = 'experiments/specific_diagnostic_task_v0.py'
        ec.write(self.root / self.script, {'synthetic': True})
        self.zero = {k: 0 for k in ec.UNITS}
        self.task = dict(task_id='B6', description='Synthetic control-plane test',
            script_patterns=[self.script], method='IMPROVED diagnostic', split='diagnostic', seeds=['0'],
            config_constraints={'threshold': 0.000001, 'scenario': 'dev242'},
            training_budget={'gradient_steps': 0}, validation_budget={'solver_cap': 6},
            test_budget={'sealed_test_episodes': 0}, resource_limits=dict(self.zero, solver_calls=6),
            max_attempts=4, max_zero_usage_repairs=3, timeout_seconds=900,
            dependencies=[], pass_conditions={'G2': True}, continue_without_review=True)
        self.plan = {'schema_version': 1, 'request_id': 'request-one', 'tasks': [self.task]}
        self.ready = self.publish(self.plan)
        self.args = dict(task_id='B6', script=self.script, config={'threshold': 0.000001, 'scenario': 'dev242'},
            method=self.task['method'], split='diagnostic', seed='0', timeout_seconds=900,
            training_budget=self.task['training_budget'], validation_budget=self.task['validation_budget'],
            test_budget=self.task['test_budget'], resource_request=dict(self.zero, solver_calls=6))

    def publish(self, plan, name='one'):
        report = self.root / ('docs/' + name + '.md'); report.parent.mkdir(exist_ok=True)
        report.write_text('Lead synthetic report')
        path = self.root / ('docs/' + name + '.execution_plan.json'); ec.write(path, plan)
        return dict(request_id=plan['request_id'], primary_analyst='claude-opus-5-5', audit_id=name,
            report=str(report.relative_to(self.root)), report_sha256=hashlib.sha256(report.read_bytes()).hexdigest(),
            execution_plan=str(path.relative_to(self.root)), execution_plan_sha256=hashlib.sha256(path.read_bytes()).hexdigest())

    def start(self, eid='e1', args=None, ready=None):
        return ec.authorize(self.root, self.state, ready or self.ready, args or self.args, eid, self.state / (eid + '.json'))

    def outcome(self, snapshot, outcome='scientific_result', used=None, evidence=None, error=None, exit_status=0):
        path = self.state / (snapshot['experiment_id'] + '-outcome.json')
        ec.write(path, dict(experiment_id=snapshot['experiment_id'], task_id=snapshot['task']['task_id'],
            snapshot_sha256=snapshot['snapshot_sha256'], outcome=outcome,
            resources=self.zero if used is None else used,
            evidence={'G2': True} if evidence is None else evidence, engineering_error=error))
        return ec.finish(self.state, snapshot, path, exit_status)

    def test_launched_snapshot_survives_new_publication_and_missing_latest_index(self):
        s = self.start()
        next_plan = copy.deepcopy(self.plan); next_plan['request_id'] = 'request-two'
        self.publish(next_plan, 'two')
        got = ec.verify_snapshot(self.root, self.state / 'e1.json', 'request-one')
        self.assertEqual(got, s)

    def test_report_tampering_rejected(self):
        (self.root / self.ready['report']).write_text('modified')
        with self.assertRaises(ValueError): self.start()

    def test_plan_tampering_rejected(self):
        (self.root / self.ready['execution_plan']).write_text('{}')
        with self.assertRaises(ValueError): self.start()

    def test_snapshot_tampering_rejected(self):
        s = self.start(); s['resource_request']['solver_calls'] = 999
        ec.write(self.state / 'e1.json', s)
        with self.assertRaises(ValueError): ec.verify_snapshot(self.root, self.state / 'e1.json')

    def test_source_tampering_rejected(self):
        self.start(); (self.root / self.script).write_text('changed')
        with self.assertRaises(ValueError): ec.verify_snapshot(self.root, self.state / 'e1.json')

    def test_wrong_task_request_seed_split_scope_and_config_rejected(self):
        for key, value in [('task_id','unknown'),('seed','2'),('split','test'),('script','experiments/unrelated_diagnostic.py'),('config', {'threshold': 0.001, 'scenario': 'dev242'})]:
            args = copy.deepcopy(self.args); args[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): self.start(args=args)

    def test_missing_receipt_retains_reservation_and_requires_review(self):
        s = self.start(); d = ec.finish(self.state, s, self.state / 'missing', 1)
        self.assertEqual(d['handoff'], 'lead_review')
        with ec.ledger(self.state) as c:
            r = c.execute('select reservation,actual from attempts').fetchone()
        self.assertEqual(json.loads(r[0])['solver_calls'], 6); self.assertIsNone(r[1])
        with self.assertRaises(ValueError): self.start('e2')

    def test_only_proven_zero_usage_operational_failure_gets_bounded_repair(self):
        for i in range(4):
            s = self.start('e' + str(i))
            d = self.outcome(s, 'engineering_failure', evidence={'no_scientific_outcome': True}, error='loader', exit_status=1)
            self.assertEqual(d['handoff'], 'bounded_engineering_repair' if i < 3 else 'lead_review')
        with self.assertRaises(ValueError): self.start('e5')

    def test_consuming_failure_never_uses_engineering_shortcut(self):
        d = self.outcome(self.start(), 'engineering_failure', used=dict(self.zero, solver_calls=1), evidence={'no_scientific_outcome': True}, error='loader', exit_status=1)
        self.assertEqual(d['handoff'], 'lead_review')

    def test_unknown_counter_never_means_zero(self):
        d = self.outcome(self.start(), 'engineering_failure', used={'solver_calls': 0}, evidence={'no_scientific_outcome': True}, error='loader', exit_status=1)
        self.assertEqual(d['handoff'], 'lead_review'); self.assertFalse(d['receipt_valid'])

    def test_failed_scientific_gate_requires_review_even_exit_zero(self):
        d = self.outcome(self.start(), evidence={'G2': False})
        self.assertEqual(d['handoff'], 'lead_review'); self.assertEqual(d['scientific_acceptance'], 'task_gates_failed')

    def test_passing_task_gate_is_not_reproduction_acceptance(self):
        d = self.outcome(self.start(), used=dict(self.zero, solver_calls=4))
        self.assertEqual(d['handoff'], 'preauthorized_continuation'); self.assertEqual(d['scientific_acceptance'], 'task_gates_passed')
        with self.assertRaises(ValueError): self.start('duplicate')

    def test_overrun_requires_review_and_is_accounted(self):
        d = self.outcome(self.start(), used=dict(self.zero, solver_calls=7))
        self.assertEqual(d['handoff'], 'lead_review')
        with ec.ledger(self.state) as c:
            self.assertEqual(json.loads(c.execute('select actual from attempts').fetchone()[0])['solver_calls'], 7)

    def test_requested_over_budget_rejected_before_reservation(self):
        args = copy.deepcopy(self.args); args['resource_request']['solver_calls'] = 7
        with self.assertRaises(ValueError): self.start(args=args)
        with ec.ledger(self.state) as c:
            self.assertEqual(c.execute('select count(*) from attempts').fetchone()[0], 0)

    def test_dependency_cannot_launch_until_prerequisite_gates_pass(self):
        dependent = copy.deepcopy(self.task); dependent.update(task_id='C0', dependencies=['B6'])
        self.plan['tasks'].append(dependent); self.ready = self.publish(self.plan)
        args = copy.deepcopy(self.args); args['task_id'] = 'C0'
        with self.assertRaises(ValueError): self.start('dependent', args=args)
        self.outcome(self.start('prerequisite'))
        self.assertEqual(self.start('dependent', args=args)['task']['task_id'], 'C0')

    def test_bool_counter_rejected_and_no_test_authorization(self):
        for change in ('bool','test','cycle','scope'):
            p = copy.deepcopy(self.plan); t = p['tasks'][0]
            if change == 'bool': t['resource_limits']['solver_calls'] = False
            if change == 'test': t['test_budget']['sealed_test_episodes'] = 1
            if change == 'cycle': t['dependencies'] = ['B6']
            if change == 'scope': t['script_patterns'] = ['experiments/**/*.py']
            with self.subTest(change=change), self.assertRaises(ValueError): ec.validate_plan(p)

    def test_receipt_identity_mismatch_requires_review(self):
        s = self.start(); path = self.state / 'wrong.json'
        ec.write(path, {'experiment_id':'another'})
        self.assertEqual(ec.finish(self.state, s, path, 0)['handoff'], 'lead_review')

    def test_runtime_helper_records_canonical_receipt(self):
        self.start(); target = self.state / 'runtime.json'
        with patch.dict(os.environ, BOHN_EXECUTION_SNAPSHOT=str(self.state/'e1.json'), BOHN_OUTCOME_RECEIPT=str(target)):
            ec.record_outcome(self.root, 'scientific_result', self.zero, {'G2': True})
        self.assertEqual(ec.read(target)['experiment_id'], 'e1')


if __name__ == '__main__': unittest.main()
