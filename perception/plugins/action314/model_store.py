"""Locate the hash-pinned action314 models (v13: models are NOT stored in git).

Every file listed in asset.json (ONNX models, runtime.json, TensorRT engines per TensorRT version) is fetched
from the first reachable base URL (default: the internal JuiceFS HTTP mirror), verified by SHA256 and kept in a
host-shared cache under /dev/shm so that the ten instances on one Jetson download once (flock).
ACTION314_MODEL_DIR may point at a pre-provisioned directory; ACTION314_MODEL_URLS overrides the base URLs.
"""
import fcntl, hashlib, json, os, threading, time, urllib.request
from pathlib import Path

_LOCK = threading.Lock()
HERE = Path(__file__).resolve().parent
_VERIFIED = {}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''): h.update(block)
    return h.hexdigest()


def spec():
    return json.loads((HERE / 'asset.json').read_text())


def _ok(path, sha):
    key = (str(path), sha)
    if _VERIFIED.get(key): return True
    good = Path(path).is_file() and digest(path) == sha
    if good: _VERIFIED[key] = True
    return good


def _cache_root(s):
    tag = hashlib.sha256(json.dumps(s['files'], sort_keys=True).encode()).hexdigest()[:12]
    d = Path(os.environ.get('ACTION314_MODEL_CACHE', '/dev/shm/action314_models')) / tag
    d.mkdir(parents=True, exist_ok=True)
    try: os.chmod(d.parent, 0o777); os.chmod(d, 0o777)
    except OSError: pass
    return d


def _fetch(name, sha, size, dest, s):
    urls = [u for u in os.environ.get('ACTION314_MODEL_URLS', '').split(',') if u] or s['base_urls']
    errors = []
    for base in urls:
        url = base.rstrip('/') + '/' + name
        tmp = dest.with_name('.%s.%d' % (dest.name, os.getpid()))
        try:
            t = time.time()
            req = urllib.request.Request(url, headers={'User-Agent': 'PhanthyMotus-Action314/13'})
            with urllib.request.urlopen(req, timeout=float(os.environ.get('ACTION314_DOWNLOAD_TIMEOUT', '30'))) as r, tmp.open('wb') as f:
                n = 0
                while True:
                    b = r.read(1 << 20)
                    if not b: break
                    n += len(b)
                    if n > size: raise ValueError('larger than declared size')
                    f.write(b)
            if digest(tmp) != sha: raise ValueError('sha256 mismatch')
            os.replace(tmp, dest)
            return '%s (%.1f MB, %.1fs)' % (url, size / 1e6, time.time() - t)
        except Exception as e:
            errors.append('%s: %s' % (url, str(e)[:120]))
            try: tmp.unlink()
            except FileNotFoundError: pass
    raise RuntimeError('cannot download %s: %s' % (name, ' | '.join(errors)))


def ensure_files(names=None, log=None):
    """Make sure the named files (default: all non-optional) are present and verified. Returns the root dir."""
    s = spec(); files = s['files']
    if names is None: names = [n for n, v in files.items() if not v.get('lazy')]
    if os.environ.get('ACTION314_MODEL_DIR'):
        root = Path(os.environ['ACTION314_MODEL_DIR']).resolve()
        if all(_ok(root / n, files[n]['sha256']) for n in names): return root
    root = _cache_root(s)
    with _LOCK:
        if all(_ok(root / n, files[n]['sha256']) for n in names): return root
        with open(root / '.lock', 'a+') as lk:
            try: os.chmod(root / '.lock', 0o666)
            except OSError: pass
            fcntl.flock(lk, fcntl.LOCK_EX)
            for n in names:
                dest = root / n
                if _ok(dest, files[n]['sha256']): continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                try: os.chmod(dest.parent, 0o777)
                except OSError: pass
                msg = _fetch(n, files[n]['sha256'], files[n]['size'], dest, s)
                if log: log('downloaded ' + msg)
    return root


def ensure_models(log=None):
    return ensure_files(None, log)
