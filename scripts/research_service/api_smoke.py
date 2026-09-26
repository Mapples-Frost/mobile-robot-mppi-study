import json, os, urllib.request, urllib.error, pathlib, time
root=pathlib.Path('/data/openai-agent')
env={}
for line in (root/'.secrets/agent.env').read_text().splitlines():
    if '=' in line:
        k,v=line.split('=',1); env[k]=v
attempts=[]
for effort in ['xhigh','high','medium','low','none']:
    body={'model':'gpt-5.5','reasoning':{'effort':effort},'input':'Reply exactly OK.','max_output_tokens':64,'store':False}
    req=urllib.request.Request(env['OPENAI_BASE_URL'].rstrip('/')+'/responses',data=json.dumps(body).encode(),headers={'Authorization':'Bearer '+env['OPENAI_API_KEY'],'Content-Type':'application/json'})
    try:
        with urllib.request.urlopen(req,timeout=120) as r: data=json.load(r)
        record={'effort':effort,'status':'accepted','model':data.get('model'),'usage':data.get('usage'),'response_status':data.get('status')}
        attempts.append(record)
        (root/'state/api_smoke.json').write_text(json.dumps({'attempts':attempts,'selected_effort':effort,'model':'gpt-5.5'},indent=2))
        print(json.dumps(record)); break
    except urllib.error.HTTPError as e:
        raw=e.read().decode(errors='replace').replace(env['OPENAI_API_KEY'],'[REDACTED]')
        attempts.append({'effort':effort,'http_status':e.code,'message':raw[:1200]})
        (root/'state/api_smoke.json').write_text(json.dumps({'attempts':attempts},indent=2))
        print(json.dumps(attempts[-1]))
        if e.code not in [400,422] or not any(s in raw.lower() for s in ['reasoning','effort']): break
    except Exception as e:
        print(json.dumps({'error_type':type(e).__name__})); break
