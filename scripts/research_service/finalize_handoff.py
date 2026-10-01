import sys,json,datetime as dt
from pathlib import Path
ROOT=Path('/data/openai-agent/mobile-robot-mppi-study');STATE=ROOT.parent/'state';OUT=ROOT/'docs/bohn2021_takeover/local_opus_handoff_20261001'
sys.path.insert(0,str(ROOT/'scripts/research_service'))
import orchestrator as worker
import author_handoff as author
checkpoint=json.loads((STATE/'handoff_author_checkpoint.json').read_text());items=checkpoint['items']
worker.TOOLS[:]=[]
worker.tool('write_handoff','Deliver the comprehensive English handoff report and structured resume memory now.',{'report':{'type':'string'},'resume_memory':{'type':'object'},'evidence_paths':{'type':'array','items':{'type':'string'}}},['report','resume_memory','evidence_paths'])
request=worker.urllib.request.Request
def forced_request(url,data=None,**kwargs):
 if str(url).endswith('/responses') and data:
  body=json.loads(data);body['tool_choice']={'type':'function','name':'write_handoff'};data=json.dumps(body).encode()
 return request(url,data=data,**kwargs)
worker.urllib.request.Request=forced_request
items.append(dict(role='user',content='The repository audit has read enough actual evidence. Additional source reading is now prohibited for this finite handoff. Deliver write_handoff now. Be useful and candid, mark uncertain/uninspected claims as limitations, cite exact source paths actually read, summarize historic completed training and current failed engineering separately. Frozen-server status and packaging are operational facts, not completed science. Provide a thorough report for a new local Opus single-agent researcher.'))
for attempt in range(3):
 author.save(STATE/'handoff_author_status.json',dict(status='running',phase='writing_report',audit_calls=24,finalization_attempt=attempt+1,updated=author.now()))
 answer=worker.api(items)
 for output in answer.get('output',[]):
  if output['type']=='function_call':
   items.append({k:output[k] for k in ('type','call_id','name','arguments')})
   try:result=author.call(output['name'],json.loads(output['arguments']))
   except Exception as e:result=dict(error=type(e).__name__,message=worker.redact(str(e))[:1000])
   items.append(dict(type='function_call_output',call_id=output['call_id'],output=json.dumps(result)))
  elif output['type']=='message':items.append(dict(role='assistant',content=''.join(c.get('text','') for c in output.get('content',[]))))
 author.save(OUT/'HANDOFF_TOOL_TRANSCRIPT.json',items)
 if (OUT/'GPT55_HANDOFF_READY.json').exists():
  author.save(STATE/'handoff_author_status.json',dict(status='completed',calls=25+attempt,completed=author.now(),report=str((OUT/'GPT55_HANDOFF_REPORT.md').relative_to(ROOT))));print('GPT55_HANDOFF_READY',flush=True);break
else:
 author.save(STATE/'handoff_author_status.json',dict(status='failed',error='Finite finalization failed',updated=author.now()));raise SystemExit(1)
