"""Start-up and periodic environment diagnostics for action314 (printed into the container log so the
leaderboard's inference log shows the real evaluation environment)."""
import json, logging, os, platform, socket, threading, time, urllib.request
from pathlib import Path

log = logging.getLogger(__name__)


def _read(path, n=400):
    try: return Path(path).read_text(errors='replace').strip()[:n]
    except Exception as e: return 'n/a (%s)' % type(e).__name__


def _meminfo():
    out = {}
    try:
        for line in open('/proc/meminfo'):
            k, v = line.split(':', 1)
            if k in ('MemTotal', 'MemAvailable', 'SwapTotal', 'SwapFree', 'Shmem'): out[k] = int(v.split()[0]) // 1024
    except Exception:
        pass
    return out


def _gpu_load():
    for p in ('/sys/devices/platform/gpu.0/load', '/sys/devices/gpu.0/load', '/sys/devices/platform/17000000.gpu/load'):
        v = _read(p, 20)
        if not v.startswith('n/a'): return v
    return 'n/a'


def _probe(url, timeout=5):
    t = time.time()
    try:
        req = urllib.request.Request(url, method='HEAD', headers={'User-Agent': 'action314-diag'})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return 'HTTP %s %.2fs len=%s' % (r.status, time.time() - t, r.headers.get('Content-Length'))
    except Exception as e:
        return 'FAIL %.2fs %s' % (time.time() - t, str(e)[:120])


def startup_report(asset):
    env_keys = sorted(k for k in os.environ if k.startswith(('JP_', 'MCP_', 'WS_', 'NVIDIA_', 'CUDA', 'ROS_', 'ACTION314_', 'PHANTHY', 'DARVIN', 'LD_PRELOAD', 'OMP_')))
    try:
        from importlib.metadata import version
        pk = {}
        for n in ('tensorrt', 'onnxruntime-gpu', 'onnxruntime', 'numpy', 'opencv-python', 'torch'):
            try: pk[n] = version(n)
            except Exception: pass
    except Exception:
        pk = {}
    cg = _read('/sys/fs/cgroup/memory.max', 40)
    if cg.startswith('n/a'): cg = _read('/sys/fs/cgroup/memory/memory.limit_in_bytes', 40)
    try:
        st = os.statvfs('/dev/shm'); shm = '%d MB free / %d MB' % (st.f_bavail * st.f_frsize >> 20, st.f_blocks * st.f_frsize >> 20)
    except Exception as e:
        shm = 'n/a'
    rep = {
        'host': socket.gethostname(), 'arch': platform.machine(), 'python': platform.python_version(),
        'device_model': _read('/proc/device-tree/model', 120).replace('\x00', ''),
        'l4t': _read('/etc/nv_tegra_release', 160).split('\n')[0],
        'jetpack_env': _read('/etc/jetpack.env', 200).replace('\n', ' '),
        'nproc': os.cpu_count(), 'mem_mb': _meminfo(), 'cgroup_mem_limit': cg, 'shm': shm,
        'packages': pk, 'env': {k: os.environ[k][:120] for k in env_keys},
        'gpu_load': _gpu_load(), 'dev_nv': sorted(p.name for p in Path('/dev').glob('nv*'))[:12],
    }
    log.warning('[action314][DIAG] env %s', json.dumps(rep, ensure_ascii=False))
    urls = [u for u in os.environ.get('ACTION314_MODEL_URLS', '').split(',') if u] or asset.get('base_urls', [])
    for u in urls:
        log.warning('[action314][DIAG] mirror %s -> %s', u, _probe(u.rstrip('/') + '/runtime.json'))
    log.warning('[action314][DIAG] github -> %s', _probe('https://github.com', 5))


def periodic(get_state, every=60.0):
    def run():
        while True:
            time.sleep(every)
            try:
                m = _meminfo()
                log.warning('[action314][DIAG] t=%s memavail=%sMB gpu_load=%s %s', time.strftime('%H:%M:%S'), m.get('MemAvailable'),
                            _gpu_load(), json.dumps(get_state(), ensure_ascii=False)[:600])
            except Exception as e:
                log.warning('[action314][DIAG] periodic failed: %s', e)
    threading.Thread(target=run, daemon=True, name='action314-diag').start()
