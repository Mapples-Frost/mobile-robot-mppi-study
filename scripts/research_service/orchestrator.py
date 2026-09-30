"""Bounded, durable GPT-5.5 research worker. Python 3.12, stdlib only."""
import csv, datetime as dt, fcntl, hashlib, json, os, pathlib, signal, socket, sqlite3, subprocess, threading, time, urllib.request, urllib.error, uuid, shutil, resource
BASE=pathlib.Path('/data/openai-agent'); ROOT=BASE/'mobile-robot-mppi-study'; STATE=BASE/'state'
STATE.mkdir(exist_ok=True); SERVICE=ROOT/'scripts/research_service'
from resource_monitor import ResourceSampler, link_run
from research_memory import registry_context
import working_language
STOP=False

def now(): return dt.datetime.now(dt.timezone.utc).isoformat()
def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.new'); temp.write_text(json.dumps(value,indent=2,ensure_ascii=False,default=str)); temp.replace(path)
def load(path,default=None):
    try:return json.loads(path.read_text())
    except FileNotFoundError:return {} if default is None else default

def secrets():
    out={}
    for line in (BASE/'.secrets/agent.env').read_text().splitlines():
        if '=' in line:
            k,v=line.split('=',1);out[k]=v
    return out
SECRET=secrets(); REDACT=[SECRET['OPENAI_API_KEY']]
if (BASE/'.secrets/github.token').exists():REDACT.append((BASE/'.secrets/github.token').read_text().strip())
def redact(text):
    for value in REDACT:
        if value: text=text.replace(value,'[REDACTED]')
    return text

def event(kind,**data):
    with (STATE/'events.jsonl').open('a') as f:f.write(redact(json.dumps(dict(time=now(),kind=kind,**data),default=str))+'\n')

def db():
    con=sqlite3.connect(STATE/'research.sqlite',timeout=30)
    con.execute('pragma journal_mode=WAL')
    con.execute('create table if not exists calls(id text primary key, timestamp text, model text, effort text, status text, usage text, duration real)')
    con.execute('create table if not exists experiments(id text primary key, timestamp text, status text, metadata text)')
    return con

def safe_path(value):
    p=(ROOT/value).resolve()
    if not p.is_relative_to(ROOT.resolve()):raise ValueError('Path outside repository')
    if any(x in ('.secrets','.git') for x in p.parts) or p.name in ('agent.env','github.token'):raise ValueError('Protected path')
    return p

def usage_today():
    with db() as c: rows=c.execute('select usage from calls where timestamp like ?', (now()[:10]+'%',)).fetchall()
    return len(rows),sum(json.loads(r[0]).get('total_tokens',0) for r in rows)

def clean_env():
    e={k:v for k,v in os.environ.items() if k in ('PATH','LANG','LC_ALL','HOME','TMPDIR')}
    e.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',TF_NUM_INTRAOP_THREADS='1',TF_NUM_INTEROP_THREADS='1',TF_CPP_MIN_LOG_LEVEL='3',MPLBACKEND='Agg',PYTHONUNBUFFERED='1')
    return e

def identity():
    def git(args):return subprocess.run(['git',*args],cwd=ROOT,capture_output=True,text=True,timeout=20).stdout.strip()
    diff=git(['diff','--binary']);tree={}
    for folder in (ROOT/'experiments',ROOT/'scripts'):
        for p in sorted(folder.rglob('*.py')):tree[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    return dict(source_tree_sha256=hashlib.sha256(json.dumps(tree,sort_keys=True).encode()).hexdigest(),source_files=tree,commit_sha=git(['rev-parse','HEAD']),dirty_state=git(['status','--porcelain']),dirty_diff_sha256=hashlib.sha256(diff.encode()).hexdigest())

def execute(args):
    execution_lock=(STATE/'experiment.lock').open('a');fcntl.flock(execution_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    path=safe_path(args['script']);argv=args.get('args',[])
    if load(STATE/'backup_status.json').get('status')=='failed' and args.get('split') not in ('smoke','diagnostic','audit','train_select'):
        raise ValueError('Backup unavailable: only reversible diagnosis/smoke/audit allowed')
    if path.suffix!='.py' or not path.is_file():raise ValueError('Existing Python script required')
    if not all(isinstance(x,str) for x in argv):raise ValueError('String arguments required')
    fingerprint=hashlib.sha256((hashlib.sha256(path.read_bytes()).hexdigest()+json.dumps([argv,args.get('config',{})],sort_keys=True)).encode()).hexdigest()
    retry_state=load(STATE/'experiment_retries.json')
    if retry_state.get(fingerprint,0)>=3:raise ValueError('Three identical failures: change or diagnose implementation before retrying this experiment')
    if any('test'==x.lower() for x in argv) and not load(STATE/'research_state.json').get('final_test_authorized'):
        raise ValueError('Final test gate is closed')
    timeout=min(max(int(args.get('timeout_seconds',900)),10),14400)
    py='/home/mapples/.local/share/bohn2021-python37/bin/python' if args.get('interpreter')=='legacy' else str(ROOT/'.venv/bin/python')
    eid=dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S')+'_'+uuid.uuid4().hex[:8]
    dest=ROOT/'research_artifacts/aws_runs'/eid;dest.mkdir(parents=True)
    meta=dict(script_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),experiment_id=eid,timestamp=now(),**identity(),**args,environment=dict(host=socket.gethostname(),python=py),stdout=str(dest/'stdout.log'),stderr=str(dest/'stderr.log'),status='running',failure_reason=None)
    meta['supervisor_call_id']=load(STATE/'active_tool.json').get('call_id')
    dump(dest/'registry.json',meta)
    with db() as c:c.execute('insert into experiments values(?,?,?,?)',(eid,meta['timestamp'],'running',json.dumps(meta)))
    dump(STATE/'active_experiment.json',meta);started=time.monotonic();peak=0;cpu_before=resource.getrusage(resource.RUSAGE_CHILDREN)
    with (dest/'stdout.log').open('w') as out,(dest/'stderr.log').open('w') as err:
        proc=subprocess.Popen([py,'-u',str(pathlib.Path('/home/mapples/projects/mobile-robot-mppi-study')/path.relative_to(ROOT)),*argv],cwd='/home/mapples/projects/mobile-robot-mppi-study',env=clean_env(),stdout=out,stderr=err,start_new_session=True)
        meta['pid']=proc.pid;meta['process_started_utc']=now();dump(STATE/'active_experiment.json',meta)
        sampler=ResourceSampler(proc.pid,dest/'cpu_samples.jsonl')
        while proc.poll() is None:
            sampler.sample()
            if time.monotonic()-started>timeout or sum((dest/n).stat().st_size for n in ('stdout.log','stderr.log'))>128*1024**2:
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(15)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                meta['failure_reason']='wall_or_log_volume_limit';break
            try:
                status=pathlib.Path('/proc/%d/status'%proc.pid).read_text()
                peak=max(peak,int(next(x.split()[1] for x in status.splitlines() if x.startswith('VmRSS:'))))
            except (OSError,StopIteration):pass
            time.sleep(2)
    process_ended=now();command_wall=time.monotonic()-started
    cpu_after=resource.getrusage(resource.RUSAGE_CHILDREN)
    retry_state[fingerprint]=0 if proc.returncode==0 else retry_state.get(fingerprint,0)+1;dump(STATE/'experiment_retries.json',retry_state)
    meta.update(retry_fingerprint=fingerprint,cpu_seconds=cpu_after.ru_utime+cpu_after.ru_stime-cpu_before.ru_utime-cpu_before.ru_stime,exit_status=proc.returncode,runtime_seconds=time.monotonic()-started,peak_process_rss_kb=peak,status='complete' if proc.returncode==0 else 'failed',ended=now())
    meta['full_command_wall_seconds']=command_wall
    meta['mean_process_tree_cpu_percent_instance']=100*meta['cpu_seconds']/command_wall/os.cpu_count()
    meta['process_ended_utc']=process_ended
    meta['resource_monitoring']=sampler.summary()
    meta['cloudwatch']=link_run(dest,meta['process_started_utc'],process_ended)
    if proc.returncode and not meta['failure_reason']:meta['failure_reason']='nonzero_exit'
    for log in ('stdout.log','stderr.log'):
        p=dest/log
        # Children get no credentials. Still scrub exact known secrets before exposure.
        if p.stat().st_size<64*1024*1024:p.write_text(redact(p.read_text(errors='replace')))
    meta['artifact_inventory']=[]
    for artifact in args.get('artifacts',[]):
        p=safe_path(artifact);meta['artifact_inventory'].append(dict(path=artifact,exists=p.exists(),bytes=p.stat().st_size if p.is_file() else None,sha256=hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() and p.stat().st_size<128*1024**2 else None))
    dump(dest/'registry.json',meta);dump(STATE/'active_experiment.json',dict(status='idle',last_experiment=eid))
    with db() as c:c.execute('update experiments set status=?,metadata=? where id=?',(meta['status'],json.dumps(meta),eid))
    registry=ROOT/'EXPERIMENT_REGISTRY.csv';fresh=not registry.exists()
    with registry.open('a',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['experiment_id','timestamp','method','seed','split','commit_sha','status','exit_status','runtime_seconds','peak_process_rss_kb','record'])
        if fresh:writer.writeheader()
        writer.writerow({**{k:meta.get(k,'') for k in writer.fieldnames if k!='record'},'record':str(dest/'registry.json')})
    state=load(STATE/'research_state.json');state['last_completed_experiment']=meta
    state['consecutive_failures']=0 if proc.returncode==0 else state.get('consecutive_failures',0)+1
    if state['consecutive_failures']>=5:state['phase']='diagnosis'
    dump(STATE/'research_state.json',state)
    event('experiment',experiment_id=eid,status=meta['status'],runtime_seconds=meta['runtime_seconds'])
    # Automatically hand off substantive outcomes, including failures. GPT-5.5
    # cannot skip the primary analyst merely by forgetting to write a request.
    script_name=path.name.lower()
    if not any(word in script_name for word in ('backup','preflight','smoke','readiness','status_capture','inventory','registry','token_usage','gate_recheck','gate_preflight','pending_evidence')):
        evidence=[a['path'] for a in meta.get('artifact_inventory',[]) if a.get('exists') and a.get('path','').endswith(('summary.md','raw.json','completed.json'))]
        dump(ROOT/'docs/bohn2021_takeover/astra_reviews/NEXT_REVIEW_REQUEST.json',dict(request_id='execution-result:'+eid,trigger='execution_result_handoff',created=now(),experiment_id=eid,status='analysis_requested',purpose=meta.get('purpose'),execution_status=meta['status'],question='Active scientific lead: interpret this outcome, verify current raw evidence and select the next scientific action for GPT-5.5 implementation. Distinguish hypotheses from verified causes and account for fair baselines, terminal confounds and source-level generalization.',evidence_paths=evidence+[str((dest/'registry.json').relative_to(ROOT))]))
    return dict(record=str(dest.relative_to(ROOT)/'registry.json'),exit_status=proc.returncode,runtime_seconds=meta['runtime_seconds'],stdout_tail=(dest/'stdout.log').read_text(errors='replace')[-12000:],stderr_tail=(dest/'stderr.log').read_text(errors='replace')[-8000:])

def call_tool(name,args):
    if name=='list_files':
        p=safe_path(args.get('path','.'));pattern=args.get('pattern','*')
        paths=sorted(p.glob(pattern));return [dict(path=str(x.relative_to(ROOT)),bytes=x.stat().st_size if x.is_file() else None) for x in paths[:500] if '.secrets' not in x.parts and '.git' not in x.parts]
    if name=='read_file':
        p=safe_path(args['path']);text=p.read_text(errors='replace');start=max(0,int(args.get('offset',0)));return dict(content=redact(text[start:start+24000]),total_characters=len(text),offset=start)
    if name=='write_file':
        p=safe_path(args['path'])
        if str(p).startswith(str(SERVICE)):raise ValueError('Supervisor code is protected; write research extension scripts elsewhere')
        analyst_root=ROOT/'docs/bohn2021_takeover/astra_reviews'
        opus_root=ROOT/'docs/bohn2021_takeover/opus_lead'
        if opus_root in p.parents or (analyst_root in p.parents and not (p.parent==analyst_root and p.name in ('NEXT_REVIEW_REQUEST.json','RESPONSE_LOG.md'))):
            raise ValueError('Analyst publications are protected; use the shared analysis request and RESPONSE_LOG handoff')
        content=args['content']
        if any(s and s in content for s in REDACT):raise ValueError('Secret content forbidden')
        if p.exists():
            hist=STATE/'edit_history'/uuid.uuid4().hex;hist.parent.mkdir(exist_ok=True);shutil.copy2(p,hist)
        p.parent.mkdir(parents=True,exist_ok=True);p.write_text(content);return dict(written=str(p.relative_to(ROOT)),sha256=hashlib.sha256(content.encode()).hexdigest())
    if name=='run_experiment':return execute(args)
    if name=='update_state':
        state=load(STATE/'research_state.json');patch=args['state']
        if patch.get('final_test_authorized'):
            gate=load(ROOT/'final_test_gate.json')
            if not gate.get('validation_passed') or not gate.get('frozen_commit') or not gate.get('model_hashes') or not gate.get('independent_audit_passed'):raise ValueError('Incomplete final test gate')
        state.update(patch);state['updated']=now();dump(STATE/'research_state.json',state);return state
    raise ValueError('Unknown tool')

TOOLS=[]
def tool(name,description,properties,required):TOOLS.append(dict(type='function',name=name,description=description,parameters=dict(type='object',properties=properties,required=required,additionalProperties=False)))
S={'type':'string'};I={'type':'integer'}
tool('list_files','List repository files. Use bounded patterns.',{'path':S,'pattern':S},['path'])
tool('read_file','Read at most 24000 characters of a repository text file. Test outcomes remain sealed.',{'path':S,'offset':I},['path'])
tool('write_file','Write research code, protocol, or report. Old file is archived; never silently change frozen sources.',{'path':S,'content':S},['path','content'])
tool('run_experiment','Run exactly one bounded Python script. All execution is registered, including smoke and diagnostics. No shell.',{'script':{'type':'string','description':'Repository-relative path to an existing .py file, for example experiments/bohn2021_aws/fit_population_diagnosis.py. Never inline source code.'},'args':{'type':'array','items':S},'interpreter':{'type':'string','enum':['modern','legacy']},'timeout_seconds':I,'method':S,'seed':S,'split':S,'purpose':S,'config':{'type':'object'},'training_budget':{'type':'object'},'validation_budget':{'type':'object'},'test_budget':{'type':'object'},'artifacts':{'type':'array','items':S}},['script','interpreter','method','seed','split','purpose','config','training_budget','validation_budget','test_budget','artifacts'])
tool('update_state','Persist phase, hypothesis, next_experiment, queue, blockers, failures and scientific decisions.',{'state':{'type':'object'}},['state'])

def api(items, force_state=False):
    # User explicitly removed all daily API/token budgets. Usage remains audited.
    effort=load(STATE/'api_smoke.json')['selected_effort']
    assert SECRET['OPENAI_MODEL']=='gpt-5.5' and effort=='xhigh'
    body=dict(model='gpt-5.5',reasoning={'effort':effort},input=working_language.responses_input(items),tools=TOOLS,store=False)
    if force_state:body['tool_choice']={'type':'function','name':'update_state'}
    req=urllib.request.Request(SECRET['OPENAI_BASE_URL'].rstrip('/')+'/responses',data=json.dumps(body).encode(),headers={'Authorization':'Bearer '+SECRET['OPENAI_API_KEY'],'Content-Type':'application/json'})
    cid=uuid.uuid4().hex;started=time.monotonic();usage={};status='error'
    try:
        with urllib.request.urlopen(req,timeout=900) as r:answer=json.load(r)
        usage=answer.get('usage',{});status=answer.get('status','unknown')
        if not str(answer.get('model','')).startswith('gpt-5.5'):raise RuntimeError('Unexpected returned model; no fallback permitted')
        return answer
    except urllib.error.HTTPError as error:
        detail=redact(error.read().decode(errors='replace'))[:2000]
        raise RuntimeError('API_HTTP_'+str(error.code)+': '+detail) from None
    finally:
        with db() as c:c.execute('insert into calls values(?,?,?,?,?,?,?)',(cid,now(),'gpt-5.5',effort,status,json.dumps(usage),time.monotonic()-started))
        event('api',call_id=cid,status=status,usage=usage)

def heartbeat():
    while True:
        dump(STATE/'heartbeat.json',dict(pid=os.getpid(),time=now(),state='running'))
        address=os.environ.get('NOTIFY_SOCKET')
        if address:
            try:
                with socket.socket(socket.AF_UNIX,socket.SOCK_DGRAM) as s:s.connect(address.replace('@','\0',1));s.sendall(b'WATCHDOG=1')
            except OSError:pass
        time.sleep(30)

def current_roles():
    roles=load(STATE/'research_roles.json')
    if roles.get('status')=='active' and roles.get('active_lead')=='claude-opus-5-5':
        return roles
    return dict(active_lead='gpt-6-astra',independent_reviewer='gpt-6-astra',executor='gpt-5.5',
                lead_ready_path='docs/bohn2021_takeover/astra_reviews/ANALYSIS_READY.json',
                lead_index_path='docs/bohn2021_takeover/astra_reviews/LATEST.md',status=roles.get('status','legacy'))

def awaiting_astra_analysis():
    # Retain the established function name for recovery compatibility; gate the active lead.
    out=ROOT/'docs/bohn2021_takeover/astra_reviews'
    try:
        roles=current_roles();request=load(out/'NEXT_REVIEW_REQUEST.json')
        if not request.get('request_id'):return False
        ready=load(safe_path(roles['lead_ready_path']))
        if ready.get('request_id') != request['request_id'] and request['request_id'] not in ready.get('supersedes_request_ids',[]):return True
        if ready.get('primary_analyst') != roles['active_lead']:return True
        report=safe_path(ready.get('report',''))
        expected=ready.get('report_sha256')
        return not (report.is_file() and expected and hashlib.sha256(report.read_bytes()).hexdigest()==expected)
    except (OSError,ValueError,TypeError,KeyError):
        return True

def role_context():
    roles=current_roles();ready=load(safe_path(roles['lead_ready_path']))
    result=dict(roles=roles,latest_plan=ready)
    if ready.get('report'):
        try:result['plan_content']=redact(safe_path(ready['report']).read_text())[:40000]
        except (OSError,ValueError):pass
    return result

def durable_tool_call(item):
    receipt=STATE/'tool_receipts'/(hashlib.sha256(item['call_id'].encode()).hexdigest()+'.json')
    previous=load(receipt)
    if previous.get('status')=='complete':return previous['result']
    # A completed/interrupted registered experiment must never be launched a second
    # time because its model-facing response was lost during a supervisor restart.
    if item['name']=='run_experiment':
        with db() as c:
            row=c.execute("select id,status,metadata from experiments where json_extract(metadata,'$.supervisor_call_id')=? order by timestamp desc limit 1",(item['call_id'],)).fetchone()
        if row:
            eid,status,metadata=row;meta=json.loads(metadata)
            result=dict(record='research_artifacts/aws_runs/'+eid+'/registry.json',exit_status=meta.get('exit_status'),runtime_seconds=meta.get('runtime_seconds'),recovered_registered_status=status,replayed_without_execution=True)
            for key in ('stdout','stderr'):
                path=meta.get(key)
                if path:
                    log=safe_path(path)
                    if log.exists():result[key+'_tail']=redact(log.read_text(errors='replace')[-12000:])
            dump(receipt,dict(call_id=item['call_id'],name=item['name'],status='complete',result=result,time=now()))
            return result
    dump(receipt,dict(call_id=item['call_id'],name=item['name'],status='pending',time=now()))
    dump(STATE/'active_tool.json',dict(call_id=item['call_id'],name=item['name'],time=now()))
    try:result=call_tool(item['name'],json.loads(item['arguments']))
    except Exception as e:
        result={'error':type(e).__name__,'message':redact(str(e))[:2000]}
        event('tool_error',name=item['name'],error=result)
    dump(receipt,dict(call_id=item['call_id'],name=item['name'],status='complete',result=result,time=now()))
    return result

def iteration():
    checkpoint=STATE/'active_iteration.json';live=load(checkpoint)
    if live.get('status')=='in_progress':
        items=live['items'];outputs=live['outputs'];executions=live['executions'];start_turn=live['turn']
        iteration_id=live['iteration_id'];pending=live.get('pending_calls',[])
    else:
        state=load(STATE/'research_state.json');recent=load(STATE/'last_iteration.json')
        if 'last_completed_experiment' in state:
            old=state['last_completed_experiment'];state['last_completed_experiment']={k:old.get(k) for k in ('experiment_id','status','purpose','exit_status','stdout','stderr','runtime_seconds')}
        if recent:recent['tool_tail']=[dict(call_id=x.get('call_id'),output=x.get('output','')[-2000:]) for x in recent.get('tool_tail',[])]
        facts=registry_context(STATE)
        state['last_experiment']=facts['latest_registered']
        state['last_completed_experiment']=facts['latest_process_complete']
        context=dict(registry_evidence=facts,state=state,last_iteration=recent,repository=str(ROOT),available_disk_gb=shutil.disk_usage(BASE).free/1e9,backup=load(STATE/'backup_status.json'))
        items=[dict(role='system',content=(SERVICE/'MISSION.md').read_text()),dict(role='user',content='Continue authorized research with concrete actions. Inspect evidence; preserve state for next iteration. Current supervisor context:\n'+json.dumps(context,default=str)[-28000:])]
        outputs=[];executions=0;start_turn=0;iteration_id=uuid.uuid4().hex;pending=[]
    # Replace stale saved mission/role instructions while preserving tool receipts and evidence.
    items[0]=dict(role='system',content=working_language.system_text((SERVICE/'MISSION.md').read_text()))
    roles=role_context();role_key=json.dumps([roles['roles'].get('active_lead'),roles['latest_plan'].get('audit_id'),roles['latest_plan'].get('report_sha256')])
    if live.get('role_context_key')!=role_key:
        items.append(dict(role='user',content='Current authorized agent roles and scientific plan; supersedes earlier role assignments. Implement approved tasks, operational repairs and measurements. Scientific causal analysis/direction belongs to the active lead; Astra provides independent critique.\n'+json.dumps(roles,ensure_ascii=False)))
    def persist(turn,pending_calls):
        dump(checkpoint,dict(status='in_progress',iteration_id=iteration_id,turn=turn,items=items,outputs=outputs,executions=executions,pending_calls=pending_calls,role_context_key=role_key,updated=now()))
    registry_refresh_due=(live.get('status')=='in_progress')
    persist(start_turn,pending)
    dump(STATE/'registry_context.json',registry_context(STATE))
    dump(STATE/'supervisor_status.json',dict(time=now(),phase='implementing_scientific_plan',active_lead=current_roles()['active_lead'],api_loop_suspended=False))
    for turn in range(start_turn,12):
        if pending:
            calls=pending;pending=[]
        else:
            if registry_refresh_due:
                items.append(dict(role='user',content='Authoritative current registry facts for this resumed cycle; supersede stale experiment pointers in narrative memory. They are evidence data, not new scientific instructions.\n'+json.dumps(registry_context(STATE),ensure_ascii=False)))
                registry_refresh_due=False
            if turn==10:items.append(dict(role='user',content='Two calls remain in this bounded cycle. Prefer a concrete bounded diagnostic now if inputs suffice. Avoid re-reading evidence already inspected.'))
            if turn==11:items.append(dict(role='user',content='Final call of this cycle: use update_state to persist a concise cumulative research memory: findings with paths, files already inspected, precise next action, hypothesis, queue and unresolved issues. Do not mark research complete merely because this cycle ends. Next cycle must continue instead of repeating this audit.'))
            persist(turn,[])
            answer=api(items,force_state=(turn==11));calls=[]
            for output in answer.get('output',[]):
                if output['type']=='function_call':
                    items.append({k:output[k] for k in ('type','call_id','name','arguments')});calls.append(output)
                elif output['type']=='message':
                    message=''.join(c.get('text','') for c in output.get('content',[]));items.append(dict(role='assistant',content=message));outputs.append(message)
            persist(turn,calls)
            if not calls:break
        for index,item in enumerate(calls):
            if item['name']=='run_experiment' and executions>=1:
                result={'error':'ValueError','message':'One experiment per iteration: persist next action for the next bounded cycle'}
            else:
                result=durable_tool_call(item)
                if item['name']=='run_experiment':executions+=1
            text=redact(json.dumps(result,default=str))[:30000]
            items.append(dict(type='function_call_output',call_id=item['call_id'],output=text))
            event('tool',name=item['name'],arguments={k:v for k,v in json.loads(item['arguments']).items() if k!='content'},result_summary=text[:2000])
            remaining=calls[index+1:];persist(turn if remaining else turn+1,remaining)
    record=dict(time=now(),outputs=outputs,tool_tail=[x for x in items if x.get('type')=='function_call_output'][-4:])
    record['inspected_files']=[json.loads(x['arguments']).get('path') for x in items if x.get('type')=='function_call' and x.get('name')=='read_file']
    archive=STATE/'iterations'/(iteration_id+'.json');dump(archive,items);dump(STATE/'last_iteration.json',record)
    with (ROOT/'RESEARCH_LOG.md').open('a') as f:f.write('\n\n## '+now()+'\n'+redact('\n'.join(outputs))+'\n')
    dump(checkpoint,dict(status='completed',iteration_id=iteration_id,updated=now()))

def housekeeping():
    # Stable commit and upload handled separately; no API calls or training overlap.
    previous=load(STATE/'backup_status.json')
    if previous.get('status') in ('verified','partial') and (dt.datetime.now(dt.timezone.utc)-dt.datetime.fromisoformat(previous['time'])).total_seconds()<300:return
    p=subprocess.run(['/usr/bin/python3',str(SERVICE/'backup.py')],cwd=ROOT,env=clean_env(),capture_output=True,text=True,timeout=14400)
    event('backup_process',exit=p.returncode,tail=redact(p.stdout[-1000:]+p.stderr[-1000:]))
    if p.returncode:raise RuntimeError('Backup not verified; diagnose infrastructure before unique formal work')

def main():
    lock=(STATE/'orchestrator.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    threading.Thread(target=heartbeat,daemon=True).start();failures=0
    # Reconcile supervisor interruption; never infer a stale running row completed.
    with db() as c:c.execute("update experiments set status='interrupted' where status='running'")
    while True:
        try:
            if not (STATE/'migration_complete.json').exists():event('waiting_for_migration');time.sleep(60);continue
            current=load(STATE/'research_state.json')
            reasons={'research_goal_change','new_paid_resource','irreversible_data_loss','missing_user_only_information','major_scientific_fork'}
            if current.get('awaiting_user_reason') in reasons:
                housekeeping();dump(STATE/'supervisor_status.json',dict(time=now(),phase='awaiting_user',reason=current['awaiting_user_reason']));time.sleep(3600);continue
            acceptance=load(ROOT/'final_acceptance.json')
            if current.get('project_complete') and acceptance.get('independent_audit_passed') and acceptance.get('all_seeds_reported') and acceptance.get('raw_results_archived'):
                housekeeping();dump(STATE/'supervisor_status.json',dict(time=now(),phase='final_delivery_complete'));time.sleep(3600);continue
            if shutil.disk_usage(BASE).free<15*1024**3:event('disk_guard');time.sleep(300);continue
            if dt.datetime.now(dt.timezone.utc)>=dt.datetime(2026,10,25,8,tzinfo=dt.timezone.utc):
                housekeeping();dump(STATE/'deadline_status.json',dict(time=now(),status='expiry_archive_only'));time.sleep(3600);continue
            if not (STATE/'last_iteration.json').exists():
                # First bounded reasoning cycle starts after verified migration and local audit.
                # Existing historical raw data also remain in the external WSL source.
                iteration()
            try:housekeeping()
            except Exception as backup_error:
                state=load(STATE/'research_state.json');state.update(phase='infrastructure_diagnosis',backup_error=str(backup_error),next_experiment='Diagnose and repair backup; no new formal experiments until verified');dump(STATE/'research_state.json',state)
            if awaiting_astra_analysis():
                # No new research direction is delegated to GPT-5.5 while the active lead
                # analyzes it. Keep heartbeat/backups and automatically resume.
                dump(STATE/'supervisor_status.json',dict(time=now(),phase='awaiting_scientific_lead',active_lead=current_roles()['active_lead'],executor_role='implementation_and_experiments',api_loop_suspended=True))
                time.sleep(30)
                continue
            iteration();failures=0
            dump(STATE/'supervisor_status.json',dict(time=now(),phase='between_iterations',api_calls_today=usage_today()[0]))
            time.sleep(5)
        except Exception as e:
            message=redact(str(e))[:1500]
            failures+=1
            event('iteration_error',error_type=type(e).__name__,message=message,consecutive_failures=failures)
            state=load(STATE/'research_state.json');state.update(supervisor_error=message,api_error_count=failures)
            if failures>=5:state['phase']='diagnosis'
            dump(STATE/'research_state.json',state)
            time.sleep(min(900,30*2**min(failures,5)))

if __name__=='__main__':main()
