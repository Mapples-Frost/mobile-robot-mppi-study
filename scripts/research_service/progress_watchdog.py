"""Persistent evidence-progress supervisor; no model/API calls or experiment execution."""
import datetime as dt
import json
import pathlib
import sqlite3
import time

BASE=pathlib.Path('/data/openai-agent')
ROOT=BASE/'mobile-robot-mppi-study'
STATE=BASE/'state'
UNITS=('solver_calls','plant_steps','training_steps','validation_episodes','test_episodes')


def read(path,default=None):
    try:return json.loads(pathlib.Path(path).read_text())
    except (OSError,ValueError):return {} if default is None else default


def save(path,value):
    path=pathlib.Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.new');temp.write_text(json.dumps(value,indent=2));temp.replace(path)


def timestamp(value):
    try:return dt.datetime.fromisoformat(value.replace('Z','+00:00'))
    except (ValueError,TypeError,AttributeError):return None


def evaluate(rows,now,active,backup):
    """Receipt counters, not process completion or prose, determine measurement progress."""
    cutoff=now-dt.timedelta(hours=2)
    totals={k:0 for k in UNITS};recent=[];last_solver=None;last_transition=None;unknown=0
    for meta in rows:
        t=timestamp(meta.get('timestamp'))
        receipt=(meta.get('coordination') or {})
        values=receipt.get('actual_resources')
        known=receipt.get('receipt_valid') is True and isinstance(values,dict) and all(type(values.get(k)) is int and values[k]>=0 for k in UNITS)
        if known and values['solver_calls']>0:last_solver=max(last_solver or t,t) if t else last_solver
        if known and (values['plant_steps']>0 or values['training_steps']>0):last_transition=max(last_transition or t,t) if t else last_transition
        if t and t>=cutoff:
            recent.append(meta)
            if known:
                for k in UNITS:totals[k]+=values[k]
            else:unknown+=1
    active_run=active.get('status')=='running'
    recent.sort(key=lambda x:timestamp(x.get('timestamp')) or cutoff)
    inactive_measurement_window=not active_run and len(recent)>=3 and all(totals[k]==0 for k in ('solver_calls','plant_steps','training_steps'))
    no_transitions=not active_run and len(recent)>=6 and totals['plant_steps']==0 and totals['training_steps']==0
    alert=inactive_measurement_window or no_transitions
    failed=sum(x.get('status') in ('failed','interrupted') for x in recent)
    recent_ids=[x.get('experiment_id') for x in recent[-6:]]
    issue='no_measurement_progress' if inactive_measurement_window else 'no_closed_loop_or_training_progress' if no_transitions else None
    incident='progress:'+str(last_transition or 'no-recorded-transition')+':'+str(last_solver or 'no-recorded-solve')+':'+str(issue)
    instructions=''
    if alert:
        instructions='Research-progress alarm: service uptime, token usage, metadata receipts and process exit0 are not experimental progress. Continue work; do not pause the project or alter scientific thresholds/splits. Diagnose the exact remaining operational blocker from raw errors and fix its shared cause, rather than adding another wrapper or metadata-only gate. Preserve all failure evidence and cumulative resource limits. Prefer the smallest already-authorized useful solver/plant/training measurement after genuine prerequisites pass. Use up to three sequential registered runs within a bounded solo cycle for approved repair/measurement feedback; never run in parallel. '
        if backup.get('status')=='verified':
            instructions+='External backup currently reports verified. If an experiment still reports backup missing, reconcile its per-version gate with the verified content/source evidence. Do not equate a dirty live research log with unbacked immutable inputs. Read backup_contract.py and PROGRESS_SUPERVISION_20261001.md; keep real recoverability verification. '
        else:instructions+='Backup is not currently verified. Repair that shared infrastructure once; avoid repeatedly running status-only experiments against unchanged evidence. '
    return dict(time=now.isoformat(),window_hours=2,window_registered_runs=len(recent),window_failed_runs=failed,
        measured_resources=totals,unknown_receipt_runs=unknown,last_measured_solver_utc=last_solver.isoformat() if last_solver else None,
        last_plant_or_training_utc=last_transition.isoformat() if last_transition else None,
        active_experiment=active.get('experiment_id') if active_run else None,status='active_scientific_run' if active_run else 'stagnation_detected' if alert else 'observing',
        alert_active=alert,alert_id=incident if alert else None,recent_experiment_ids=recent_ids,backup_status=backup.get('status'),instructions=instructions,
        caveat='Counters are measured reported usage, not success or independent scientific acceptance. Unknown receipts are not assumed zero.')


def capture():
    now=dt.datetime.now(dt.timezone.utc)
    with sqlite3.connect('file:'+str(STATE/'research.sqlite')+'?mode=ro',uri=True,timeout=10) as c:
        rows=[dict(experiment_id=x[0],timestamp=x[1],status=x[2],coordination=json.loads(x[3]) if x[3] else {}) for x in c.execute("select id,timestamp,status,json_extract(metadata,'$.coordination') from experiments order by timestamp desc limit 300")]
    result=evaluate(rows,now,read(STATE/'active_experiment.json'),read(STATE/'backup_status.json'))
    previous=read(STATE/'progress_watchdog.json');save(STATE/'progress_watchdog.json',result)
    if result['alert_id']!=previous.get('alert_id') or result['status']!=previous.get('status'):
        with (STATE/'progress_watchdog_events.jsonl').open('a') as f:f.write(json.dumps(result)+'\n')
    return result


def main():
    while True:
        try:capture()
        except Exception as error:
            save(STATE/'progress_watchdog_error.json',dict(time=dt.datetime.now(dt.timezone.utc).isoformat(),error_type=type(error).__name__,message=str(error)[:500]))
        time.sleep(60)


if __name__=='__main__':main()
