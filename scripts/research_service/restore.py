import pathlib,tarfile,hashlib,json,argparse
parser=argparse.ArgumentParser(description='Verify and restore downloaded Bohn GitHub release packages; does not fetch credentials.')
parser.add_argument('assets');parser.add_argument('destination');args=parser.parse_args()
assets=pathlib.Path(args.assets);dest=pathlib.Path(args.destination).resolve();dest.mkdir(parents=True,exist_ok=True)
for manifest in sorted(assets.glob('*.manifest.json')):
 data=json.loads(manifest.read_text());archive=assets/data['asset']['name'];h=hashlib.sha256()
 with archive.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 assert h.hexdigest()==data['asset']['sha256'],archive
 with tarfile.open(archive) as tar:
  for member in tar.getmembers():
   target=(dest/member.name).resolve()
   if not target.is_relative_to(dest) or member.issym() or member.islnk():raise ValueError('Unsafe archive member '+member.name)
  tar.extractall(dest,filter='data')
 for row in data['entries']:
  p=dest/row['path'];h=hashlib.sha256()
  with p.open('rb') as f:
   for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
  assert h.hexdigest()==row['sha256'],p
 print('verified',archive.name)
