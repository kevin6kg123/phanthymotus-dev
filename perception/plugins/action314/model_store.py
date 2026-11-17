"""Fetch only the hash-pinned task model asset; no private bundles or images."""
import hashlib,json,os,tarfile,tempfile,threading,urllib.request
from pathlib import Path

_LOCK=threading.Lock()
HERE=Path(__file__).resolve().parent

def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
 return h.hexdigest()

def ensure_models():
 spec=json.loads((HERE/'asset.json').read_text());root=Path(os.environ.get('ACTION314_MODEL_DIR','/models/action314-cpu-v1')).resolve()
 with _LOCK:
  root.mkdir(parents=True,exist_ok=True)
  def complete():return all((root/n).is_file() and digest(root/n)==h for n,h in spec['files'].items())
  if complete():return root
  if os.environ.get('ACTION314_OFFLINE')=='1':raise RuntimeError('Required action314 model files missing or hash mismatch in '+str(root))
  with tempfile.TemporaryDirectory(prefix='.action314-',dir=root) as temp:
   temp=Path(temp);archive=temp/'models.tar.gz';request=urllib.request.Request(spec['url'],headers={'User-Agent':'PhanthyMotus-Action314/1'})
   with urllib.request.urlopen(request,timeout=45) as source,archive.open('wb') as destination:
    count=0
    while True:
     block=source.read(1024*1024)
     if not block:break
     count+=len(block)
     if count>spec['max_download_bytes']:raise ValueError('Model asset exceeds declared maximum size')
     destination.write(block)
   if digest(archive)!=spec['archive_sha256']:raise ValueError('Downloaded model asset SHA256 mismatch')
   with tarfile.open(archive,'r:gz') as source:
    members={m.name:m for m in source.getmembers()}
    if set(members)!=set(spec['files']):raise ValueError('Unexpected model archive members')
    for name,expected in spec['files'].items():
     member=members[name]
     if Path(name).name!=name or not member.isfile() or member.size>30_000_000:raise ValueError('Unsafe model archive member')
     payload=source.extractfile(member)
     if payload is None:raise ValueError('Unreadable model asset member')
     with (temp/name).open('wb') as target:
      while True:
       block=payload.read(1024*1024)
       if not block:break
       target.write(block)
     if digest(temp/name)!=expected:raise ValueError('Model member SHA256 mismatch: '+name)
   for name in spec['files']:os.replace(temp/name,root/name)
  if not complete():raise ValueError('Model installation incomplete')
  return root
