"""Lead-authored task contracts and conservative budget accounting (Python 3.7+).

No API access or credentials. Publication and receipts are immutable evidence;
unknown usage retains its reservation. This module never grants test access.
"""
import fnmatch
import hashlib
import json
import os
import pathlib
import sqlite3

VERSION = 1
UNITS = ('solver_calls', 'plant_steps', 'training_steps', 'validation_episodes', 'test_episodes')
ENGINEERING_ERRORS = ('dependency', 'missing_file', 'loader', 'shape', 'authorization', 'startup')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def read(path):
    return json.loads(pathlib.Path(path).read_text())


def write(path, value):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.new')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True))
    temp.replace(path)


def repo_path(root, name):
    root = pathlib.Path(root).resolve()
    path = (root / name).resolve()
    if path == root or root not in path.parents or any(x in ('.git', '.secrets') for x in path.parts):
        raise ValueError('Invalid contract evidence path')
    return path


def resources(value):
    if not isinstance(value, dict) or set(value) != set(UNITS):
        raise ValueError('All resource counters are required; unknown is not zero')
    if any(type(v) is not int or v < 0 for v in value.values()):
        raise ValueError('Resource counters must be nonnegative integers')
    return value


def validate_plan(plan):
    if not isinstance(plan, dict) or set(plan) != {'schema_version', 'request_id', 'tasks'}:
        raise ValueError('Execution plan fields mismatch')
    if plan.get('schema_version') != VERSION or not isinstance(plan.get('tasks'), list) or not 1 <= len(plan['tasks']) <= 32:
        raise ValueError('Expected nonempty versioned execution plan')
    if not isinstance(plan.get('request_id'), str) or not plan['request_id']:
        raise ValueError('Plan request ID required')
    ids = set()
    for task in plan['tasks']:
        required = {'task_id', 'description', 'script_patterns', 'split', 'method', 'seeds', 'config_constraints',
                    'training_budget', 'validation_budget', 'test_budget', 'resource_limits',
                    'max_attempts', 'max_zero_usage_repairs', 'timeout_seconds', 'dependencies',
                    'pass_conditions', 'continue_without_review'}
        if set(task) != required:
            raise ValueError('Task fields mismatch: ' + str(sorted(required.symmetric_difference(task))))
        tid = task['task_id']
        if not isinstance(tid, str) or not tid or tid in ids:
            raise ValueError('Unique stable task IDs required')
        ids.add(tid)
        for field in ('description', 'split', 'method'):
            if not isinstance(task[field], str) or not task[field]:
                raise ValueError('Nonempty task ' + field + ' required')
        patterns = task['script_patterns']
        if not isinstance(patterns, list) or not patterns:
            raise ValueError('Bounded script patterns required')
        for pattern in patterns:
            if not isinstance(pattern, str) or not pattern.startswith('experiments/') or '..' in pattern or not pattern.endswith('.py') or any(x in pattern.rsplit('/', 1)[0] for x in '*?[') or len(pattern.rsplit('/', 1)[-1].split('*')[0]) < 12:
                raise ValueError('Task ' + tid + ': script_patterns must be repository-relative Python paths such as experiments/bohn2021_aws/specific_diagnostic_v0*.py; bare filenames/stems are not paths. Received: ' + str(pattern))
        for field in ('config_constraints', 'training_budget', 'validation_budget', 'test_budget', 'pass_conditions'):
            if not isinstance(task[field], dict):
                raise ValueError('Object required: ' + field)
        resources(task['resource_limits'])
        if not isinstance(task['seeds'], list) or not task['seeds'] or any(not isinstance(x, str) for x in task['seeds']):
            raise ValueError('Explicit seed strings required')
        if task['split'].lower() in ('test', 'final', 'sealed', 'sealed_test', 'final_test', 'sealed-test', 'final-test'):
            raise ValueError('Final/sealed split is unavailable to structured development plans')
        # This first deployment supports development work only; the existing final gate stays separate.
        if task['resource_limits']['test_episodes'] or any(isinstance(v, (int, float)) and v > 0 for v in task['test_budget'].values()):
            raise ValueError('Structured development plans cannot authorize final/sealed test')
        if not isinstance(task['dependencies'], list) or any(not isinstance(x, str) for x in task['dependencies']):
            raise ValueError('Dependency IDs required')
        for field, cap in (('max_attempts', 20), ('max_zero_usage_repairs', 3), ('timeout_seconds', 14400)):
            v = task[field]
            if type(v) is not int or v < (10 if field == 'timeout_seconds' else 0) or v > cap:
                raise ValueError('Invalid bounded ' + field)
        if task['max_attempts'] < 1 or type(task['continue_without_review']) is not bool:
            raise ValueError('Positive attempts and explicit continuation boolean required')
    graph = {t['task_id']: t['dependencies'] for t in plan['tasks']}
    def visit(tid, stack):
        if tid not in graph or tid in stack:
            raise ValueError('Unknown or cyclic task dependency')
        for dep in graph[tid]:
            visit(dep, stack + [tid])
    for tid in graph:
        visit(tid, [])
    return plan


def plan_from_ready(root, ready):
    report = repo_path(root, ready['report'])
    if hashlib.sha256(report.read_bytes()).hexdigest() != ready.get('report_sha256'):
        raise ValueError('Lead report digest mismatch')
    if not ready.get('execution_plan'):
        return None
    path = repo_path(root, ready['execution_plan'])
    if hashlib.sha256(path.read_bytes()).hexdigest() != ready.get('execution_plan_sha256'):
        raise ValueError('Execution plan digest mismatch')
    plan = validate_plan(read(path))
    if plan['request_id'] != ready['request_id'] or ready.get('primary_analyst') != 'claude-opus-5-5':
        raise ValueError('Execution plan authority/request mismatch')
    return plan


def ledger(state):
    c = sqlite3.connect(str(pathlib.Path(state) / 'task_control.sqlite'), timeout=30)
    c.execute('pragma journal_mode=WAL')
    c.execute('create table if not exists attempts(eid text primary key, plan text, task text, status text, reservation text, actual text, passed integer default 0, repair integer default 0)')
    return c


def budget_context(state):
    with ledger(state) as c:
        rows = c.execute('select eid,plan,task,status,reservation,actual,passed,repair from attempts order by rowid desc').fetchall()
    charged = {k: sum(json.loads(r[5] or r[4])[k] for r in rows) for k in UNITS}
    return dict(attempt_count=len(rows), known_usage_count=sum(r[5] is not None for r in rows),
                accounted_resources_including_unknown_reservations=charged,
                recent_attempts=[dict(zip(('experiment_id','plan_sha256','task_id','status','reserved_resources','actual_resources','passed','repair'),r)) for r in rows[:30]],
                rule='Task budgets are cumulative within a published plan. New plans are explicit lead reauthorization, not free retries; disclose cumulative cross-plan usage. Unknown counters retain reservations.')


def match_config(actual, constraints):
    for key, expected in constraints.items():
        if key not in actual:
            raise ValueError('Missing frozen config field: ' + key)
        if isinstance(expected, dict):
            if not isinstance(actual[key], dict):
                raise ValueError('Frozen config type mismatch: ' + key)
            match_config(actual[key], expected)
        elif actual[key] != expected or type(actual[key]) is not type(expected):
            raise ValueError('Frozen config value mismatch: ' + key)


def authorize(root, state, ready, args, eid, snapshot_path):
    plan = plan_from_ready(root, ready)
    if plan is None:
        return None
    task = next((t for t in plan['tasks'] if t['task_id'] == args.get('task_id')), None)
    if task is None:
        raise ValueError('Use an exact task_id from the current lead execution plan')
    if not any(fnmatch.fnmatchcase(args['script'], p) for p in task['script_patterns']):
        raise ValueError('Script outside approved task scope')
    for field in ('split', 'method', 'training_budget', 'validation_budget', 'test_budget'):
        if args.get(field) != task[field]:
            raise ValueError('Frozen task field mismatch: ' + field)
    if args.get('seed') not in task['seeds']:
        raise ValueError('Seed outside approved task scope')
    match_config(args.get('config', {}), task['config_constraints'])
    if args.get('timeout_seconds', 900) > task['timeout_seconds']:
        raise ValueError('Task wall-time allowance exceeded')
    request = resources(args.get('resource_request'))
    plan_id = ready['execution_plan_sha256']
    with ledger(state) as c:
        c.execute('begin immediate')
        rows = c.execute('select reservation,actual,passed,repair from attempts where plan=? and task=? order by rowid', (plan_id, task['task_id'])).fetchall()
        if len(rows) >= task['max_attempts'] or any(r[2] for r in rows):
            raise ValueError('Task completed or attempt allowance exhausted')
        if rows and not rows[-1][3]:
            raise ValueError('Previous substantive/unknown result requires a new lead plan')
        for dep in task['dependencies']:
            if not c.execute('select 1 from attempts where plan=? and task=? and passed=1', (plan_id, dep)).fetchone():
                raise ValueError('Task dependency has not passed: ' + dep)
        used = {k: sum(json.loads(r[1] or r[0])[k] for r in rows) for k in UNITS}
        if any(used[k] + request[k] > task['resource_limits'][k] for k in UNITS):
            raise ValueError('Cumulative task resource allowance exceeded')
        snapshot = dict(schema_version=VERSION, ready=ready, task=task, experiment_id=eid,
                        plan_task_ids=[t['task_id'] for t in plan['tasks']],
                        script=args['script'], script_sha256=hashlib.sha256(repo_path(root, args['script']).read_bytes()).hexdigest(),
                        resource_request=request, execution_args_sha256=digest(args))
        snapshot['snapshot_sha256'] = digest(snapshot)
        write(snapshot_path, snapshot)
        c.execute('insert into attempts(eid,plan,task,status,reservation) values(?,?,?,?,?)',
                  (eid, plan_id, task['task_id'], 'reserved', json.dumps(request)))
    return snapshot


def verify_snapshot(root, path, expected_request=None):
    snapshot = read(path)
    body = dict(snapshot); declared = body.pop('snapshot_sha256', None)
    if digest(body) != declared:
        raise ValueError('Execution snapshot digest mismatch')
    ready = snapshot['ready']
    plan = plan_from_ready(root, ready)
    if plan is None or snapshot['task'] not in plan['tasks']:
        raise ValueError('Snapshot task missing from lead publication')
    if snapshot.get('plan_task_ids') != [t['task_id'] for t in plan['tasks']]:
        raise ValueError('Snapshot task sequence differs from lead publication')
    if expected_request is not None and expected_request != ready['request_id']:
        raise ValueError('Snapshot expected request mismatch')
    if hashlib.sha256(repo_path(root, snapshot['script']).read_bytes()).hexdigest() != snapshot['script_sha256']:
        raise ValueError('Frozen executable source changed')
    return snapshot


def runtime_snapshot(root, expected_request=None):
    path = os.environ.get('BOHN_EXECUTION_SNAPSHOT')
    return verify_snapshot(root, path, expected_request) if path else None


def record_outcome(root, outcome, used, evidence, engineering_error=None):
    """Called by an instrumented experiment only after counters/gates are known."""
    snapshot = runtime_snapshot(root)
    if snapshot is None:
        raise ValueError('No authorized execution snapshot')
    receipt = dict(experiment_id=snapshot['experiment_id'], task_id=snapshot['task']['task_id'],
                   snapshot_sha256=snapshot['snapshot_sha256'], outcome=outcome,
                   resources=resources(used), evidence=evidence, engineering_error=engineering_error)
    write(os.environ['BOHN_OUTCOME_RECEIPT'], receipt)


def finish(state, snapshot, receipt_path, exit_status):
    """Missing, invalid or over-budget receipts always require lead review."""
    if snapshot is None:
        return dict(handoff='legacy_review', scientific_acceptance='unverified')
    task = snapshot['task']; eid = snapshot['experiment_id']
    decision = dict(handoff='lead_review', scientific_acceptance='unverified', receipt_valid=False)
    actual = None; passed = False; repair = False
    try:
        receipt = read(receipt_path)
        if (receipt.get('experiment_id'), receipt.get('task_id'), receipt.get('snapshot_sha256')) != (eid, task['task_id'], snapshot['snapshot_sha256']):
            raise ValueError('Outcome receipt identity mismatch')
        actual = resources(receipt.get('resources'))
        decision['receipt_valid'] = True
        decision['actual_resources'] = actual
        if any(actual[k] > snapshot['resource_request'][k] for k in UNITS):
            raise ValueError('Recorded usage exceeded reservation')
        if not isinstance(receipt.get('evidence'), dict):
            raise ValueError('Outcome evidence required')
        if receipt.get('outcome') == 'engineering_failure' and receipt.get('engineering_error') in ENGINEERING_ERRORS and all(v == 0 for v in actual.values()) and receipt['evidence'].get('no_scientific_outcome') is True:
            with ledger(state) as c:
                n = c.execute('select count(*) from attempts where plan=? and task=? and repair=1', (snapshot['ready']['execution_plan_sha256'], task['task_id'])).fetchone()[0]
            repair = n < task['max_zero_usage_repairs']
            if repair:
                decision['handoff'] = 'bounded_engineering_repair'
        elif exit_status == 0 and receipt.get('outcome') == 'scientific_result':
            # A nonempty lead-defined gate map is mandatory for automatic continuation.
            passed = bool(task['pass_conditions']) and all(receipt['evidence'].get(k) == v and type(receipt['evidence'].get(k)) is type(v) for k, v in task['pass_conditions'].items())
            decision['scientific_acceptance'] = 'task_gates_passed' if passed else 'task_gates_failed'
            if passed and task['continue_without_review']:
                decision['handoff'] = 'preauthorized_continuation'
    except (OSError, ValueError, TypeError, KeyError) as error:
        decision['receipt_error'] = type(error).__name__ + ': ' + str(error)
    with ledger(state) as c:
        c.execute('update attempts set status=?,actual=?,passed=?,repair=? where eid=?',
                  (decision['handoff'], json.dumps(actual) if actual is not None else None, int(passed), int(repair), eid))
        if decision['handoff'] == 'preauthorized_continuation':
            completed = {r[0] for r in c.execute('select distinct task from attempts where plan=? and passed=1', (snapshot['ready']['execution_plan_sha256'],))}
            if set(snapshot['plan_task_ids']).issubset(completed):
                decision['handoff'] = 'plan_completed_review'
                c.execute('update attempts set status=? where eid=?', (decision['handoff'], eid))
    return decision
