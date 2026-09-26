import subprocess, pathlib, json, datetime
root=pathlib.Path(r'D:\Projects\mobile-robot-mppi-study')
out=root/'docs/bohn2021_takeover'; out.mkdir(exist_ok=True)
for name,args in {'status':['status','--porcelain=v1'],'branch':['branch','-avv'],'log':['log','-20','--format=%H %aI %s'],'remote':['remote','-v'],'diff':['diff','--binary']}.items():
    p=subprocess.run(['git',*args],cwd=root,capture_output=True)
    (out/('windows_'+name+'.txt')).write_bytes(p.stdout)
paths=[]
for p in root.rglob('*'):
    if p.is_file() and ('bohn' in str(p).lower() or 'pendulum' in p.name.lower() or 'cartpole' in p.name.lower()):
        if not any(x in p.parts for x in ['.git','.venv','.venv-cuda','__pycache__']): paths.append({'path':str(p.relative_to(root)),'bytes':p.stat().st_size})
(out/'windows_inventory.json').write_text(json.dumps(paths,indent=2))
print('Windows evidence files:',len(paths))
