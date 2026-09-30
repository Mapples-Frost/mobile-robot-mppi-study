"""Persistent primary-preferred routing for the same Astra model and effort."""
import contextlib
import datetime as dt
import email.utils
import fcntl
import http.client
import json
import os
import pathlib
import re
import sqlite3
import threading
import time
import urllib.error
import urllib.request
import uuid

BASE = pathlib.Path('/data/openai-agent')
STATE = BASE / 'state'
WORK = STATE / 'astra_reviewer'
ROUTE = STATE / 'astra_router'
MODEL = 'gpt-6-astra'
EFFORT = 'max'
PRIMARY_PROBE_SECONDS = 120
IDLE_PRIMARY_PROBE_SECONDS = 300
_wake = threading.Event()
_monitor_started = False

def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()

def load(path, default=None):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {} if default is None else default

def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex[:8] + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    tmp.chmod(0o600)
    tmp.replace(path)

def redact(value):
    text = str(value)
    for p in (BASE / '.secrets').glob('*'):
        if p.is_file():
            for line in p.read_text(errors='replace').splitlines():
                secret = line.split('=', 1)[-1].strip()
                if len(secret) >= 16:
                    text = text.replace(secret, '[REDACTED]')
    return re.sub(r'\bsk-[A-Za-z0-9_-]{16,}', '[REDACTED_KEY]', text)

def event(kind, **fields):
    ROUTE.mkdir(parents=True, exist_ok=True)
    with (ROUTE / 'events.jsonl').open('a') as f:
        f.write(redact(json.dumps(dict(time=now(), kind=kind, **fields), ensure_ascii=False)) + '\n')

@contextlib.contextmanager
def state_lock():
    ROUTE.mkdir(parents=True, exist_ok=True)
    with (ROUTE / 'state.lock').open('a') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        yield

def default_state():
    return dict(active_endpoint='primary', primary_probe_interval_seconds=PRIMARY_PROBE_SECONDS,
                idle_primary_probe_interval_seconds=IDLE_PRIMARY_PROBE_SECONDS,
                endpoints={'primary': {}, 'backup': {}}, updated=now())

def state():
    with state_lock():
        return load(ROUTE / 'status.json', default_state())

def endpoint_config(endpoint):
    filename = 'reviewer.env' if endpoint == 'primary' else 'astra_backup.env'
    return dict(x.split('=', 1) for x in (BASE / '.secrets' / filename).read_text().splitlines() if '=' in x)

class EndpointError(RuntimeError):
    def __init__(self, message, *, retryable=True, retry_after=0):
        super().__init__(redact(message))
        self.retryable = retryable
        self.retry_after = retry_after
        self.observed_at = time.time()

class EndpointBusy(EndpointError):
    """Local serialization is not evidence of a provider outage."""

def failure(endpoint, error):
    observed_at = getattr(error, 'observed_at', time.time())
    with state_lock():
        s = load(ROUTE / 'status.json', default_state())
        d = s.setdefault('endpoints', {}).setdefault(endpoint, {})
        if d.get('last_success_epoch', 0) > observed_at:
            event('stale_failure_ignored', endpoint=endpoint)
            return
        d.update(healthy=False, last_failure=now(), last_error=redact(error)[:1400],
                 last_failure_epoch=observed_at,
                 consecutive_failures=d.get('consecutive_failures', 0) + 1)
        if endpoint == 'primary':
            s['active_endpoint'] = 'backup'
            s['next_primary_probe_epoch'] = time.time() + max(PRIMARY_PROBE_SECONDS, getattr(error, 'retry_after', 0))
        s['updated'] = now()
        save(ROUTE / 'status.json', s)
    event('endpoint_failure', endpoint=endpoint, error=str(error), retryable=getattr(error, 'retryable', True))

def success(endpoint, purpose, observed_at=None):
    observed_at = time.time() if observed_at is None else observed_at
    recovered = False
    with state_lock():
        s = load(ROUTE / 'status.json', default_state())
        previous = s.get('active_endpoint', 'primary')
        d = s.setdefault('endpoints', {}).setdefault(endpoint, {})
        if d.get('last_failure_epoch', 0) > observed_at:
            event('stale_success_ignored', endpoint=endpoint)
            return
        d.update(healthy=True, last_success=now(), last_success_epoch=observed_at, consecutive_failures=0)
        if endpoint == 'primary':
            s['active_endpoint'] = 'primary'
            s['next_primary_probe_epoch'] = time.time() + IDLE_PRIMARY_PROBE_SECONDS
            recovered = previous != 'primary'
        # A backup response must not undo a concurrent successful primary recovery probe.
        s['updated'] = now()
        save(ROUTE / 'status.json', s)
    if recovered:
        _wake.set()
        event('primary_recovered', purpose=purpose, switching='next_model_call')

def retry_after_seconds(value):
    if not value:
        return 0
    try:
        return max(0, min(3600, float(value)))
    except ValueError:
        try:
            return max(0, min(3600, email.utils.parsedate_to_datetime(value).timestamp() - time.time()))
        except (ValueError, TypeError):
            return 0

def validate(answer):
    if not isinstance(answer, dict):
        raise EndpointError('Invalid response JSON object')
    returned = str(answer.get('model', ''))
    if returned != MODEL and not returned.startswith(MODEL + '-'):
        raise EndpointError('Returned model mismatch; no model fallback allowed')
    reasoning = answer.get('reasoning') or {}
    effort = reasoning.get('effort') if isinstance(reasoning, dict) else None
    if effort not in ('max', 'xhigh'):
        raise EndpointError('Returned effort below verified capability: ' + str(effort))
    if answer.get('status') != 'completed':
        raise EndpointError('Incomplete response: ' + str(answer.get('status')))
    return effort

def parse_stream(response, record=lambda event: None, timeout=600):
    """Accept only a full response.completed payload, never a partial text stream."""
    began = time.monotonic()
    pending = []
    max_event_bytes = 32 * 1024 * 1024
    pending_bytes = 0

    def dispatch():
        if not pending:
            return None
        payload = '\n'.join(pending)
        pending.clear()
        if payload.strip() == '[DONE]':
            return None
        try:
            item = json.loads(payload)
        except ValueError:
            raise EndpointError('Malformed JSON in Responses event stream') from None
        if not isinstance(item, dict):
            raise EndpointError('Invalid Responses stream event object')
        record(item)
        kind = item.get('type')
        if kind == 'response.completed':
            answer = item.get('response')
            validate(answer)
            return answer
        if kind in ('error', 'response.failed', 'response.incomplete'):
            detail = item.get('error') or (item.get('response') or {}).get('error') or item.get('message') or kind
            raise EndpointError('Responses stream failure: ' + redact(detail)[:1000])
        return None

    for raw in response:
        if time.monotonic() - began > timeout:
            raise EndpointError('Responses stream exceeded per-call wall-time limit')
        if len(raw) > max_event_bytes:
            raise EndpointError('Responses stream event exceeded memory limit')
        try:
            line = raw.decode('utf-8').rstrip('\r\n')
        except UnicodeError:
            raise EndpointError('Invalid UTF-8 in Responses event stream') from None
        if not line:
            result = dispatch()
            pending_bytes = 0
            if result is not None:
                return result
        elif line.startswith('data:'):
            pending.append(line[5:].lstrip(' '))
            pending_bytes += len(raw)
            if pending_bytes > max_event_bytes:
                raise EndpointError('Responses stream event exceeded memory limit')
        # SSE comments, event names and sequence IDs do not change output semantics.
    result = dispatch()
    if result is not None:
        return result
    raise EndpointError('Responses stream ended before response.completed; partial output discarded')

def once(body, endpoint, purpose='review', timeout=600):
    if endpoint not in ('primary', 'backup'):
        raise ValueError('Unknown Astra endpoint')
    if body.get('model') != MODEL or (body.get('reasoning') or {}).get('effort') != EFFORT:
        raise ValueError('Pinned Astra model/effort changed; prohibited')
    ROUTE.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    cid = uuid.uuid4().hex
    began = time.monotonic()
    status, actual_effort, usage = 'error', 'unverified', {}
    registered = False
    lock = (ROUTE / (endpoint + '_inflight.lock')).open('a')
    try:
        # Serialize requests to each account, including its recovery probes.
        deadline = time.monotonic() + (0.2 if purpose == 'primary_recovery_probe' else 5)
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() > deadline:
                    raise EndpointBusy('Endpoint already has an in-flight local request')
                time.sleep(0.1)
        try:
            cfg = endpoint_config(endpoint)
            if not cfg.get('REVIEWER_API_KEY') or not cfg.get('REVIEWER_BASE_URL', '').startswith('https://'):
                raise ValueError('Missing or invalid HTTPS endpoint configuration')
        except (OSError, ValueError) as error:
            raise EndpointError('Endpoint configuration unavailable: ' + type(error).__name__) from None
        with sqlite3.connect(STATE / 'research.sqlite', timeout=30) as c:
            c.execute('insert into calls values(?,?,?,?,?,?,?)', (cid, now(), MODEL, EFFORT,
                      'running', '{}', 0))
        registered = True
        save(ROUTE / 'inflight' / (cid + '.json'), dict(call_id=cid, endpoint=endpoint,
             purpose=purpose, pid=os.getpid(), process_start=pathlib.Path('/proc/self/stat').read_text().split()[21], started=now()))
        event('api_started', call_id=cid, endpoint=endpoint, purpose=purpose, requested_effort=EFFORT)
        wire_body = dict(body, stream=True)
        headers = {'Authorization': 'Bearer ' + cfg['REVIEWER_API_KEY'],
                   'Content-Type': 'application/json', 'Accept': 'text/event-stream',
                   'User-Agent': 'BohnResearchAgent/1.0'}
        request = urllib.request.Request(cfg['REVIEWER_BASE_URL'].rstrip('/') + '/responses',
                                         data=json.dumps(wire_body).encode(), headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                content_type = response.headers.get('Content-Type', '')
                event('response_headers', call_id=cid, endpoint=endpoint, purpose=purpose,
                      transport='sse' if 'text/event-stream' in content_type else 'json',
                      time_to_headers_seconds=time.monotonic()-began)
                if 'text/event-stream' in content_type:
                    stream_path = WORK / 'streams' / (cid + '.metadata.jsonl')
                    stream_path.parent.mkdir(parents=True, exist_ok=True)
                    last_progress = 0
                    observed_events = 0
                    with stream_path.open('w') as log:
                        def record(item):
                            nonlocal usage, last_progress, observed_events
                            observed_events += 1
                            snapshot = item.get('response')
                            if isinstance(snapshot, dict) and isinstance(snapshot.get('usage'), dict):
                                usage = snapshot['usage']
                            kind = item.get('type')
                            elapsed = time.monotonic() - began
                            if observed_events == 1 or elapsed-last_progress >= 5 or kind in ('response.completed','response.failed','response.incomplete','error'):
                                log.write(json.dumps(dict(time=now(), type=kind, events_seen=observed_events,
                                           elapsed_seconds=elapsed, usage=usage))+'\n')
                                log.flush()
                                last_progress = elapsed
                                with sqlite3.connect(STATE / 'research.sqlite', timeout=30) as c:
                                    c.execute('update calls set usage=?,duration=? where id=?', (json.dumps(usage),elapsed,cid))
                        answer = parse_stream(response, record, timeout=max(1,timeout-(time.monotonic()-began)))
                else:
                    # Compatibility with an endpoint that ignores stream=True; completion checks still apply.
                    answer = json.load(response)
        except urllib.error.HTTPError as error:
            detail = redact(error.read().decode(errors='replace'))[:1400]
            # Payload/parameter mistakes are engineering failures, not endpoint outages.
            recoverable = error.code in (401, 403, 404, 408, 409, 425, 429) or error.code >= 500
            raise EndpointError('API_HTTP_' + str(error.code) + ': ' + detail,
                                retryable=recoverable,
                                retry_after=retry_after_seconds(error.headers.get('Retry-After'))) from None
        except (urllib.error.URLError, TimeoutError, OSError, ValueError, http.client.HTTPException) as error:
            raise EndpointError(type(error).__name__ + ': ' + redact(error)[:1000]) from None
        usage = answer.get('usage') or {} if isinstance(answer, dict) else {}
        if not isinstance(usage, dict):
            usage = {}
        actual_effort = validate(answer)
        answer['_research_route'] = dict(endpoint=endpoint, purpose=purpose, call_id=cid,
                                         requested_effort=EFFORT, returned_effort=actual_effort,
                                         observed_at_epoch=time.time())
        save(WORK / 'responses' / (cid + '.json'), answer)
        status = 'completed'
        return answer
    except BaseException as error:
        if not isinstance(error, Exception):
            status = 'interrupted'
        raise
    finally:
        lock.close()
        elapsed = time.monotonic() - began
        if registered:
            with sqlite3.connect(STATE / 'research.sqlite', timeout=30) as c:
                c.execute('update calls set timestamp=?,effort=?,status=?,usage=?,duration=? where id=?',
                          (now(), actual_effort, status, json.dumps(usage), elapsed, cid))
            event('api', call_id=cid, endpoint=endpoint, purpose=purpose, status=status,
                  requested_effort=EFFORT, returned_effort=actual_effort, usage=usage, duration=elapsed)
            (ROUTE / 'inflight' / (cid + '.json')).unlink(missing_ok=True)

def request(body, purpose='review'):
    if body.get('model') != MODEL or (body.get('reasoning') or {}).get('effort') != EFFORT:
        raise ValueError('Pinned Astra model/effort changed; prohibited')
    chosen = state().get('active_endpoint', 'primary')
    order = ['primary', 'backup'] if chosen == 'primary' else ['backup', 'primary']
    errors = []
    for endpoint in order:
        try:
            answer = once(body, endpoint, purpose)
        except EndpointBusy as error:
            errors.append(endpoint + ': ' + str(error))
            event('local_endpoint_busy', endpoint=endpoint, purpose=purpose)
            continue
        except EndpointError as error:
            errors.append(endpoint + ': ' + str(error))
            if not error.retryable:
                raise
            failure(endpoint, error)
            continue
        success(endpoint, purpose, answer['_research_route']['observed_at_epoch'])
        return answer
    raise EndpointError('Both Astra endpoints unavailable: ' + ' | '.join(errors))

def probe_primary():
    body = dict(model=MODEL, reasoning={'effort': EFFORT}, store=False,
                max_output_tokens=4096,
                input=[dict(role='user', content='Endpoint health check only. Reply exactly OK; no tools.')])
    try:
        answer = once(body, 'primary', 'primary_recovery_probe', timeout=120)
        text = ''.join(x.get('text', '') for item in answer.get('output', [])
                       if item.get('type') == 'message' for x in item.get('content', []))
        if text.strip() != 'OK':
            raise EndpointError('Primary health probe returned unexpected text')
    except EndpointBusy:
        with state_lock():
            s = load(ROUTE / 'status.json', default_state())
            s['next_primary_probe_epoch'] = time.time() + 30
            save(ROUTE / 'status.json', s)
        event('primary_probe_deferred', reason='local_request_in_flight')
        return None
    except EndpointError as error:
        failure('primary', error)
        return False
    success('primary', 'primary_recovery_probe', answer['_research_route']['observed_at_epoch'])
    return True

def monitor():
    while dt.datetime.now(dt.timezone.utc) < dt.datetime(2026, 10, 25, 8, tzinfo=dt.timezone.utc):
        try:
            s = state()
            with state_lock():
                latest = load(ROUTE / 'status.json', default_state())
                latest['monitor_heartbeat'] = now()
                latest['monitor_pid'] = os.getpid()
                save(ROUTE / 'status.json', latest)
            last = s.get('endpoints', {}).get('primary', {}).get('last_success_epoch', 0)
            due = s.get('next_primary_probe_epoch', 0)
            if time.time() >= due and (s.get('active_endpoint') == 'backup' or time.time() - last >= IDLE_PRIMARY_PROBE_SECONDS):
                probe_primary()
        except Exception as error:
            event('monitor_error', error_type=type(error).__name__, message=redact(error)[:1000])
            with state_lock():
                s = load(ROUTE / 'status.json', default_state())
                s['next_primary_probe_epoch'] = time.time() + PRIMARY_PROBE_SECONDS
                save(ROUTE / 'status.json', s)
        time.sleep(10)

def start_monitor():
    global _monitor_started
    if _monitor_started:
        return
    ROUTE.mkdir(parents=True, exist_ok=True)
    # Reconcile only this router's abandoned requests; unknown provider usage stays unknown.
    for path in (ROUTE / 'inflight').glob('*.json'):
        old = load(path)
        try:
            live_start = pathlib.Path('/proc/%s/stat' % int(old['pid'])).read_text().split()[21]
            live = live_start == old.get('process_start')
        except (OSError, KeyError, ValueError):
            live = False
        if not live and old.get('call_id'):
            with sqlite3.connect(STATE / 'research.sqlite', timeout=30) as c:
                c.execute("update calls set status='interrupted_usage_unknown' where id=? and status='running'", (old['call_id'],))
            event('abandoned_request_reconciled', **old, usage_known=False)
            path.unlink(missing_ok=True)
    with state_lock():
        s = load(ROUTE / 'status.json', default_state())
        s.setdefault('next_primary_probe_epoch', time.time() + PRIMARY_PROBE_SECONDS)
        save(ROUTE / 'status.json', s)
    _monitor_started = True
    threading.Thread(target=monitor, name='astra-primary-health', daemon=True).start()
    event('monitor_started', primary_probe_interval_seconds=PRIMARY_PROBE_SECONDS,
          idle_primary_probe_interval_seconds=IDLE_PRIMARY_PROBE_SECONDS)

def wait_retry(seconds):
    # A successful primary recovery probe wakes an otherwise long retry cooldown.
    _wake.wait(timeout=seconds)
    _wake.clear()
