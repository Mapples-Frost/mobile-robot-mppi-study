# Structured scientific execution handoff, version 1

This is an operational change. It does not change the Bohn research question,
scientific thresholds, train/validation/test split rules, model assignments or effort.
Opus remains scientific lead, Astra independently reviews, and GPT-5.5 implements.

## Authority and launch

Opus uses the native `submit_execution_plan` tool to submit a draft for its current
request. Publication writes the report, transcript, manifest and execution-plan JSON,
then atomically replaces `PLAN_READY.json` containing both file digests. `LATEST.md`
is a navigation index, never an authorization gate in structured mode. Legacy plans
remain compatible until Opus publishes the first structured plan; no permissions are
inferred from legacy prose during migration.

Each task names a stable ID, narrowly scoped experiment filenames, exact method,
split and allowed seed strings, exact training/validation/test budget declarations,
frozen scientific config constraints, cumulative resource limits, attempt/repair/wall
bounds, dependency IDs and explicit machine-readable pass conditions. A task pass is
local acceptance, never reproduction success. Structured version 1 cannot authorize
sealed/final tests. Existing final-test rules remain in force.

The scheduler reserves resources in `state/task_control.sqlite` before launch and
records an immutable digest-bound snapshot under `state/execution_snapshots/`.
The child receives `BOHN_EXECUTION_SNAPSHOT` and `BOHN_OUTCOME_RECEIPT`; it receives
no API credentials. Its experiment ID, task, plan, main script hash and full launch
argument digest are linked in the registry. Verify the snapshot before consuming
resources. Later publications cannot invalidate a launched snapshot. Modified report,
plan, snapshot or executable files fail verification.

## Outcome receipts and bounded continuation

Instrumented scripts import `execution_contract` through the supplied `PYTHONPATH`:

```python
snapshot = execution_contract.runtime_snapshot(ROOT)
# Use real counters from this run; missing or uncertain values must not become zero.
execution_contract.record_outcome(
    ROOT, 'scientific_result', measured_resources,
    {'G2': measured_g2_pass, 'other_lead_gate': measured_other_gate})
```

All five integer counters are mandatory: `solver_calls`, `plant_steps`,
`training_steps` (optimizer/gradient updates), `validation_episodes`, `test_episodes`.
If additional units such as episodes, selector fits or simulation draws matter,
freeze and disclose them in the task budget/config and raw gates too; the five-counter
ledger is not a replacement for task-specific fair-budget accounting.

Every substantive failure, unknown/missing/invalid receipt, overrun or failed gate
requests Opus review. Exit zero alone cannot grant continuation. Only a receipt with
all counters explicitly zero, `outcome=engineering_failure`, an enumerated operational
error and `evidence.no_scientific_outcome=true` can grant a repair. At most three such
repairs are permitted, and total attempts/wall/resource limits still apply. Unknown
usage retains the entire reservation. An interrupted experiment is never replayed;
its durable experiment record and existing tool receipts remain authoritative.

Successful automatic continuation additionally requires nonempty lead-defined pass
conditions and `continue_without_review=true`. Dependency launches verify the passed
prerequisite in the ledger. Scripts should not unconditionally create another review
request: let the scheduler decide from the validated receipt. Independent new scientific
questions still require the lead. New plans are explicit reauthorization; cumulative
cross-plan usage is supplied to Opus and must remain disclosed.

Wildcard artifact inventory now expands matching files/directories and attaches newly
updated outputs rather than treating a glob as a nonexistent literal path. Historical
matches are excluded. Missing outputs and all failed evidence remain visible.

## Verification and rollback

`python3 -m unittest -v test_execution_contract` verifies authorization integrity,
snapshot stability, resource accounting, unknown-use conservatism, bounded repairs,
scientific failures and dependencies. The same module passes on the legacy Python 3.7
used by scientific experiments. `test_flow_integration` verifies a real isolated child,
receipt/registry handoff, wildcard inventory, native tool schema and publication.
Tests use temporary synthetic evidence and zero simulation calls; no training or test
data is accessed. Production source backups are retained under the server's state
deployment archive, and normal code/evidence external backups remain enabled.
