"""Host-wide shared GPU server for action314 (pose + classifiers on TensorRT).

All evaluator instances on a Jetson share the host /dev/shm (--ipc=host). The first instance to take the flock
loads (or builds) one FP16 TensorRT engine per model and serves them on a Unix socket under /dev/shm; every
instance, the owner included, is a client. One CUDA context and one copy of each engine per host instead of
one per instance, so the 8 GB Orin is not exhausted and inference really runs on the GPU.

Wire protocol (little endian)
  pose request : b'P' + u32 n + n*416*416*3 uint8 (RGB letterboxed)          -> (n, 56, A)
  model request: b'M' + u8 idx + u32 nbytes + float32 payload (engine shape)  -> (1, rows, cols)
  response     : b'O' + u32 d0 + u32 d1 + u32 d2 + float16 data | b'E' + u32 len + utf8
"""
import fcntl, logging, os, queue, shutil, socket, struct, subprocess, sys, threading, time
from pathlib import Path
import numpy as np

log = logging.getLogger(__name__)
SIZE = 416
FRAME_BYTES = SIZE * SIZE * 3
MAX_BATCH = int(os.environ.get('ACTION314_GPU_MAX_BATCH', '8'))
DEFER = os.environ.get('ACTION314_DEFER_BUILD', '1') == '1'   # serve pose + shipped engines first, build the rest in the background
MIN_BUILD_MB = int(os.environ.get('ACTION314_TRT_MIN_MB', '900'))
DEFER_FETCH = set(filter(None, os.environ.get('ACTION314_DEFER_FETCH', 'clip').split(',')))


def _mem_avail_mb():
    try:
        for line in open('/proc/meminfo'):
            if line.startswith('MemAvailable:'): return int(line.split()[1]) // 1024
    except Exception: pass
    return 1 << 30


def trt_version():
    """TensorRT version without importing tensorrt, so clients never create a CUDA context."""
    try:
        from importlib.metadata import version
        for name in ('tensorrt', 'tensorrt-cu12', 'tensorrt_cu12'):
            try: return version(name)
            except Exception: pass
    except Exception:
        pass
    import tensorrt as trt
    return trt.__version__


def shm_dir(tag):
    d = Path(os.environ.get('ACTION314_TRT_CACHE', '/dev/shm/action314_gpu')) / ('%s-%s' % (trt_version(), tag))
    d.mkdir(parents=True, exist_ok=True)
    try: os.chmod(d, 0o777)
    except OSError: pass
    return d


def _recvall(s, n):
    buf = bytearray(n); view = memoryview(buf); got = 0
    while got < n:
        k = s.recv_into(view[got:], n - got)
        if k == 0: raise ConnectionError('socket closed')
        got += k
    return bytes(buf)


def build_engine(onnx_path, plan_path, opt_level=3, workspace_mb=512):
    """FP16 engine; dynamic batch 1..MAX_BATCH when the ONNX input batch is symbolic."""
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.WARNING)
    b = trt.Builder(logger)
    flags = 0 if int(trt.__version__.split('.')[0]) >= 10 else 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    net = b.create_network(flags); parser = trt.OnnxParser(net, logger)
    if not parser.parse(Path(onnx_path).read_bytes()):
        raise RuntimeError('onnx parse failed: ' + '; '.join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    cfg = b.create_builder_config(); cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb << 20)
    cfg.set_flag(trt.BuilderFlag.FP16)
    if hasattr(cfg, 'builder_optimization_level'): cfg.builder_optimization_level = opt_level
    inp = net.get_input(0); shape = tuple(inp.shape)
    if any(d < 0 for d in shape):
        rest = tuple(d if d > 0 else SIZE for d in shape[1:])
        prof = b.create_optimization_profile()
        prof.set_shape(inp.name, (1,) + rest, (4,) + rest, (MAX_BATCH,) + rest); cfg.add_optimization_profile(prof)
    plan = b.build_serialized_network(net, cfg)
    if plan is None: raise RuntimeError('engine build returned None: %s' % onnx_path)
    tmp = str(plan_path) + '.%d' % os.getpid()
    with open(tmp, 'wb') as f: f.write(bytes(plan))
    os.replace(tmp, plan_path)


def _reply(c, y):
    y = np.asarray(y, np.float32)
    if y.ndim == 2: y = y[None]
    c.sendall(b'O' + struct.pack('<III', *y.shape) + y.astype(np.float16).tobytes())


class Server:
    def __init__(self, pose_plan, extra_plans, sock_path):
        try:
            from utils.tensorrt_runtime import TensorRTEngine
        except ImportError:
            from tensorrt_runtime import TensorRTEngine
        self.pose = TensorRTEngine(pose_plan)
        x = np.zeros((1, 3, SIZE, SIZE), np.float32)
        for bs in (1, 2, 4, MAX_BATCH): self.pose.infer(np.repeat(x, bs, 0))
        self.extra = []
        self._engine_cls = TensorRTEngine
        for p in extra_plans: self.extra.append(self._load(p) if p is not None else None)
        self.extra_lock = threading.Lock()
        self.q = queue.Queue(); self.sock_path = str(sock_path)
        self.stats = {'pose_batches': 0, 'pose_frames': 0, 'pose_gpu_ms': 0.0, 'cls_calls': 0, 'cls_gpu_ms': 0.0, 'clients': 0}
        try: os.unlink(self.sock_path)
        except FileNotFoundError: pass
        self.srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        tmp = self.sock_path + '.%d' % os.getpid(); self.srv.bind(tmp); os.chmod(tmp, 0o777); os.replace(tmp, self.sock_path)
        self.srv.listen(64)
        threading.Thread(target=self._accept, daemon=True, name='a314-gpu-accept').start()
        threading.Thread(target=self._batcher, daemon=True, name='a314-gpu-batch').start()

    def _load(self, p):
        e = self._engine_cls(p); shp = tuple(int(v) for v in e.input_shape)
        e.infer(np.zeros(shp, np.float32)); return (e, shp)

    def add_extra(self, idx, p):
        ent = self._load(p)
        with self.extra_lock: self.extra[idx] = ent

    def _accept(self):
        while True:
            c, _ = self.srv.accept(); self.stats['clients'] += 1
            threading.Thread(target=self._client, args=(c,), daemon=True).start()

    def _client(self, c):
        try:
            while True:
                kind = _recvall(c, 1)
                try:
                    if kind == b'P':
                        n = struct.unpack('<I', _recvall(c, 4))[0]
                        if not 1 <= n <= MAX_BATCH: raise ValueError('bad batch size %d' % n)
                        frames = np.frombuffer(_recvall(c, n * FRAME_BYTES), np.uint8).reshape(n, SIZE, SIZE, 3)
                        ev = threading.Event(); slot = {}; self.q.put((frames, ev, slot)); ev.wait()
                        if 'err' in slot: raise RuntimeError(slot['err'])
                        _reply(c, slot['out'])
                    elif kind == b'M':
                        idx = _recvall(c, 1)[0]; nbytes = struct.unpack('<I', _recvall(c, 4))[0]
                        buf = _recvall(c, nbytes); ent = self.extra[idx] if idx < len(self.extra) else None
                        if ent is None: raise LookupError('engine %d not loaded yet' % idx)
                        e, shp = ent
                        x = np.frombuffer(buf, np.float32).reshape(shp)
                        with self.extra_lock:
                            t = time.perf_counter(); y = np.asarray(e.infer(np.ascontiguousarray(x))[0], np.float32)
                            self.stats['cls_gpu_ms'] += (time.perf_counter() - t) * 1000; self.stats['cls_calls'] += 1
                        _reply(c, y)
                    else:
                        raise ConnectionError('bad request tag %r' % kind)
                except (ConnectionError, OSError):
                    raise
                except Exception as e:
                    m = (type(e).__name__ + ': ' + str(e)).encode()[:2000]; c.sendall(b'E' + struct.pack('<I', len(m)) + m)
        except (ConnectionError, OSError):
            pass
        finally:
            c.close()

    def _batcher(self):
        while True:
            items = [self.q.get()]; deadline = time.monotonic() + 0.004
            while sum(len(i[0]) for i in items) < MAX_BATCH:
                try: items.append(self.q.get(timeout=max(0.0, deadline - time.monotonic())))
                except queue.Empty: break
            groups, cur = [], []
            for it in items:
                if cur and sum(len(i[0]) for i in cur) + len(it[0]) > MAX_BATCH: groups.append(cur); cur = []
                cur.append(it)
            if cur: groups.append(cur)
            for group in groups:
                try:
                    x = np.concatenate([g[0] for g in group]).transpose(0, 3, 1, 2).astype(np.float32) * np.float32(1 / 255.0)
                    t = time.perf_counter(); y = np.asarray(self.pose.infer(np.ascontiguousarray(x))[0], np.float32)
                    self.stats['pose_gpu_ms'] += (time.perf_counter() - t) * 1000; self.stats['pose_batches'] += 1; self.stats['pose_frames'] += len(x)
                    k = 0
                    for frames, ev, slot in group:
                        slot['out'] = y[k:k + len(frames)]; k += len(frames); ev.set()
                except Exception as e:
                    for _, ev, slot in group: slot['err'] = type(e).__name__ + ': ' + str(e)[:500]; ev.set()


class Client:
    def __init__(self, sock_path, timeout=10.0):
        self.sock_path = str(sock_path); self.timeout = timeout; self.s = None; self.lock = threading.Lock()

    def _connect(self):
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM); s.settimeout(self.timeout); s.connect(self.sock_path); self.s = s

    def _call(self, payload):
        with self.lock:
            for attempt in (0, 1):
                try:
                    if self.s is None: self._connect()
                    self.s.sendall(payload)
                    tag = _recvall(self.s, 1)
                    if tag == b'E':
                        n = struct.unpack('<I', _recvall(self.s, 4))[0]; raise RuntimeError('gpu server: ' + _recvall(self.s, n).decode())
                    if tag != b'O': raise ConnectionError('bad reply tag')
                    d = struct.unpack('<III', _recvall(self.s, 12))
                    return np.frombuffer(_recvall(self.s, d[0] * d[1] * d[2] * 2), np.float16).astype(np.float32).reshape(d)
                except (ConnectionError, OSError, socket.timeout):
                    try: self.s.close()
                    except Exception: pass
                    self.s = None
                    if attempt: raise

    def __call__(self, frames_u8):
        """(n, 416, 416, 3) uint8 RGB letterboxed -> (n, 56, A) float32 raw pose output"""
        f = np.ascontiguousarray(frames_u8, np.uint8)
        return self._call(b'P' + struct.pack('<I', len(f)) + f.tobytes())

    def model(self, idx, x):
        """extra engine idx on float32 x of its static input shape -> output[0] (leading 1 dropped)"""
        b = np.ascontiguousarray(x, np.float32).tobytes()
        return self._call(b'M' + bytes([idx]) + struct.pack('<I', len(b)) + b)[0]


def _build_child(onnx, plan, status):
    t0 = time.time()
    status('building GPU engine %s (TensorRT %s)' % (Path(onnx).name, trt_version()))
    r = subprocess.run([sys.executable, os.path.abspath(__file__), 'build', str(onnx), str(plan), os.environ.get('ACTION314_TRT_OPT_LEVEL', '1')],
                       capture_output=True, text=True, timeout=float(os.environ.get('ACTION314_TRT_BUILD_TIMEOUT', '1800')))
    if r.returncode != 0 or not Path(plan).exists():
        raise RuntimeError('engine build failed rc=%s: %s' % (r.returncode, (r.stderr or r.stdout)[-400:]))
    status('built %s in %.1fs' % (Path(plan).name, time.time() - t0))


def start_or_join(models, tag, on_ready, on_error, status=None):
    """models: [(name, get_onnx_path, get_shipped_plan)], callables returning a Path or None (lazy download).
    The first is the pose model (dynamic batch), the rest are static extra engines. Runs in the background; on_ready(Client, role) once the host server answers."""
    status = status or (lambda m: log.info('[action314] %s', m))

    def run():
        try:
            d = shm_dir(tag); sock = d / 'gpu.sock'; lockf = open(d / '.lock', 'a+')
            try:
                try:
                    from . import model_store as _ms
                except Exception:
                    import model_store as _ms
                gone = _ms.clean_stale(d.parent, keep=d.name)
                if gone: status('removed stale engine dirs %s' % gone)
            except Exception as e: status('stale cleanup failed: %s' % str(e)[:120])
            try: os.chmod(d / '.lock', 0o666)
            except OSError: pass
            while True:
                if sock.exists():
                    try:
                        cl = Client(sock, timeout=float(os.environ.get('ACTION314_GPU_TIMEOUT', '10')))
                        cl(np.zeros((1, SIZE, SIZE, 3), np.uint8)); on_ready(cl, 'client'); return
                    except Exception as e:
                        status('gpu socket not usable yet: %s' % str(e)[:120])
                try:
                    fcntl.flock(lockf, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    time.sleep(0.5); continue
                # owner: the flock stays held for this process's lifetime and is released by the kernel on exit
                plans = [d / (name + '.plan') for name, _, _ in models]
                for attempt in (0, 1):
                    for (name, get_onnx, get_plan), plan in zip(models, plans):
                        if not plan.exists() and attempt == 0 and name in DEFER_FETCH and plan != plans[0]:
                            status('deferring engine %s until the server is up' % name); continue
                        if not plan.exists() and attempt == 0:
                            try: shipped = get_plan()
                            except Exception as e: shipped = None; status('shipped engine %s unavailable: %s' % (name, str(e)[:200]))
                            if shipped and Path(shipped).is_file():
                                try: os.replace(shipped, plan)  # same /dev/shm filesystem: move, no second copy in RAM
                                except OSError: shutil.copyfile(shipped, str(plan) + '.tmp'); os.replace(str(plan) + '.tmp', plan)
                                status('using shipped engine %s' % name)
                        if not plan.exists() and (plan == plans[0] or not DEFER): _build_child(get_onnx(), plan, status)
                    try:
                        server = Server(plans[0], [p if p.exists() else None for p in plans[1:]], sock); break
                    except Exception as e:
                        status('engine load failed (%s: %s); rebuilding on this host' % (type(e).__name__, str(e)[:200]))
                        for p in plans:
                            try: p.unlink()
                            except FileNotFoundError: pass
                        if attempt: raise
                start_or_join.server = server
                if os.environ.get('ACTION314_SHM_HYGIENE', '1') == '1':
                    freed = 0
                    for i, p in enumerate(plans):
                        if (i == 0 or server.extra[i - 1] is not None) and p.exists():
                            try: freed += p.stat().st_size; p.unlink()
                            except OSError: pass
                    status('freed %.0f MB of loaded engine files from /dev/shm' % (freed / 1e6))
                status('GPU server serving %s' % sock)
                on_ready(Client(sock), 'owner')
                for i, ((name, get_onnx, get_plan), plan) in enumerate(zip(models[1:], plans[1:])):
                    if server.extra[i] is not None: continue
                    try:
                        for _w in range(600):
                            if _mem_avail_mb() >= MIN_BUILD_MB: break
                            time.sleep(1.0)
                        else:
                            status('deferred engine %s skipped: low memory' % name); continue
                        if not plan.exists():
                            try: shipped = get_plan()
                            except Exception as e: shipped = None; status('shipped engine %s unavailable: %s' % (name, str(e)[:200]))
                            if shipped and Path(shipped).is_file():
                                try: os.replace(shipped, plan)
                                except OSError: shutil.copyfile(shipped, str(plan) + '.tmp'); os.replace(str(plan) + '.tmp', plan)
                                status('fetched deferred engine %s' % name)
                        if not plan.exists(): _build_child(get_onnx(), plan, status)
                        server.add_extra(i, plan); status('hot-loaded engine %s' % name)
                        if os.environ.get('ACTION314_SHM_HYGIENE', '1') == '1':
                            try: plan.unlink()
                            except OSError: pass
                    except Exception as e:
                        status('deferred engine %s failed: %s' % (name, str(e)[:200]))
                return
        except Exception as e:
            on_error(e)
    threading.Thread(target=run, daemon=True, name='action314-gpu').start()


if __name__ == '__main__':
    if sys.argv[1] == 'build':
        try: os.nice(-5)
        except Exception: pass
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        t = time.time(); build_engine(sys.argv[2], sys.argv[3], int(sys.argv[4])); print('built %.1fs' % (time.time() - t))
