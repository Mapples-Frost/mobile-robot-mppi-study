import argparse, collections, hashlib, json, os, pathlib, shutil, sqlite3, stat, tarfile, tempfile, urllib.request

def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()

def download(url,path,expected):
 if path.exists() and digest(path)==expected:return
 tmp=path.with_name(path.name+'.download')
 request=urllib.request.Request(url,headers={'User-Agent':'bohn-local-handoff-restore'})
 with urllib.request.urlopen(request,timeout=180) as response,tmp.open('wb') as out:shutil.copyfileobj(response,out,1024*1024)
 if digest(tmp)!=expected:tmp.unlink();raise RuntimeError('Downloaded SHA256 mismatch: '+path.name)
 tmp.replace(path)

def safe_target(dest,key):
 raw=pathlib.PurePosixPath(key)
 if raw.is_absolute() or '..' in raw.parts or '\\' in key:raise ValueError('Unsafe evidence path')
 p=dest.joinpath(*raw.parts).resolve()
 if os.path.commonpath((str(dest),str(p)))!=str(dest):raise ValueError('Path escapes destination')
 return p

def restore_archive(archive,dest,rows):
 wanted={r['path']:r for r in rows};seen=set()
 with tarfile.open(archive,'r|gz') as tar:
  for member in tar:
   if member.name not in wanted:continue
   if not member.isfile():raise ValueError('Expected regular evidence file: '+member.name)
   r=wanted[member.name];p=safe_target(dest,member.name);p.parent.mkdir(parents=True,exist_ok=True)
   stream=tar.extractfile(member);tmp=p.with_name(p.name+'.handoff-part');h=hashlib.sha256();size=0
   with tmp.open('wb') as out:
    for block in iter(lambda:stream.read(1024*1024),b''):out.write(block);h.update(block);size+=len(block)
   if h.hexdigest()!=r['sha256'] or size!=r['bytes']:tmp.unlink();raise RuntimeError('Evidence file SHA256 mismatch: '+member.name)
   tmp.replace(p);os.chmod(p,stat.S_IMODE(member.mode)&0o777)
   if r.get('mtime_ns'):os.utime(p,ns=(r['mtime_ns'],r['mtime_ns']))
   seen.add(member.name)
 if seen!=set(wanted):raise RuntimeError('Archive does not contain all expected paths: '+archive.name)

def main():
 parser=argparse.ArgumentParser(description='Restore exact final AWS handoff evidence by verified per-file index. Code is cloned separately. Does not start research or call models.')
 parser.add_argument('--index',required=True,help='Local HANDOFF_INDEX.json.gz or its public GitHub download URL')
 parser.add_argument('--destination',required=True,help='New isolated transfer base; contains mobile-robot-mppi-study/ and state/')
 parser.add_argument('--plan-only',action='store_true');parser.add_argument('--verify-only',action='store_true');parser.add_argument('--keep-downloads',action='store_true')
 args=parser.parse_args();dest=pathlib.Path(args.destination).expanduser().resolve();dest.mkdir(parents=True,exist_ok=True)
 cache=dest/'.handoff_downloads';cache.mkdir(exist_ok=True);index_path=pathlib.Path(args.index)
 if args.index.startswith('https://'):
  index_path=cache/'HANDOFF_INDEX.json.gz'
  with urllib.request.urlopen(urllib.request.Request(args.index,headers={'User-Agent':'bohn-local-handoff-restore'}),timeout=180) as response,index_path.open('wb') as out:shutil.copyfileobj(response,out,1024*1024)
 import gzip
 opener=gzip.open if str(index_path).endswith('.gz') else open
 with opener(index_path,'rt',encoding='utf-8') as f:index=json.load(f)
 grouped=collections.defaultdict(list)
 for row in index['files']:grouped[row['asset']].append(row)
 assets={a['name']:a for a in index['assets']};size=sum(assets[name]['bytes'] for name in grouped)
 print(json.dumps(dict(files=len(index['files']),packages=len(grouped),uncompressed_bytes=sum(r['bytes'] for r in index['files']),download_bytes=size,code_commit=index['code_commit'],destination=str(dest)),indent=2),flush=True)
 if args.plan_only:return
 marker=dest/'HANDOFF_RESTORE_CHECKPOINT.json'
 checkpoint=json.loads(marker.read_text()) if marker.exists() else {'index_id':index['handoff_id'],'completed_assets':[]}
 if checkpoint['index_id']!=index['handoff_id']:raise ValueError('Different handoff previously restored here; use a fresh destination')
 done=set(checkpoint['completed_assets'])
 for number,name in enumerate(sorted(grouped),1):
  rows=grouped[name]
  if args.verify_only or name in done:
   missing=[r['path'] for r in rows if not safe_target(dest,r['path']).is_file() or digest(safe_target(dest,r['path']))!=r['sha256']]
   if not missing:continue
   if args.verify_only:raise RuntimeError('Missing/changed restored evidence: '+str(missing[:5]))
  asset=assets[name];archive=cache/name
  for attempt in range(4):
   try:download(asset['url'],archive,asset['sha256']);break
   except Exception:
    if attempt==3:raise
    import time;time.sleep(min(2**attempt*3,24))
  restore_archive(archive,dest,rows);done.add(name);checkpoint['completed_assets']=sorted(done)
  temp=marker.with_suffix('.new');temp.write_text(json.dumps(checkpoint,indent=2));temp.replace(marker)
  if not args.keep_downloads:archive.unlink()
  print('verified',number,'/',len(grouped),name,flush=True)
 receipt=dict(handoff_id=index['handoff_id'],code_commit=index['code_commit'],verified_files=len(index['files']),verified_packages=len(grouped),does_not_claim_reproduction=True,all_expected_file_hashes_verified=True)
 (dest/'HANDOFF_RESTORE_VERIFIED.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt),flush=True)
if __name__=='__main__':main()
