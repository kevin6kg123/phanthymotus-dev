"""Locate the hash-pinned action314 models.

The models are bundled in the repository (plugins/action314/models) so the
evaluation device never needs network access at start-up. A download from the
GitHub Release is only attempted when ACTION314_ALLOW_DOWNLOAD=1.
"""
import hashlib,json,os,tarfile,tempfile,threading,urllib.request
from pathlib import Path

_LOCK=threading.Lock()
HERE=Path(__file__).resolve().parent
BUNDLED=HERE/'models'
_VERIFIED=set()

def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
 return h.hexdigest()

def _complete(root,spec):
 key=str(root)
 if key in _VERIFIED:return True
 ok=all((root/n).is_file() and digest(root/n)==h for n,h in spec['files'].items())
 if ok:_VERIFIED.add(key)
 return ok

def ensure_models():
 spec=json.loads((HERE/'asset.json').read_text())
 with _LOCK:
  candidates=[BUNDLED]
  if os.environ.get('ACTION314_MODEL_DIR'):candidates.insert(0,Path(os.environ['ACTION314_MODEL_DIR']).resolve())
  for root in candidates:
   if _complete(root,spec):return root
  if os.environ.get('ACTION314_ALLOW_DOWNLOAD')!='1':
   raise RuntimeError('Bundled action314 models missing or hash mismatch in '+str(BUNDLED))
  root=Path(os.environ.get('ACTION314_MODEL_DIR','/tmp/action314-models')).resolve();root.mkdir(parents=True,exist_ok=True)
  with tempfile.TemporaryDirectory(prefix='.action314-',dir=root) as temp:
   temp=Path(temp);archive=temp/'models.tar.gz';request=urllib.request.Request(spec['url'],headers={'User-Agent':'PhanthyMotus-Action314/1'})
   with urllib.request.urlopen(request,timeout=20) as source,archive.open('wb') as destination:
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
     with (temp/name).open('wb') as target:target.write(payload.read())
     if digest(temp/name)!=expected:raise ValueError('Model member SHA256 mismatch: '+name)
   for name in spec['files']:os.replace(temp/name,root/name)
  if not _complete(root,spec):raise ValueError('Model installation incomplete')
  return root
