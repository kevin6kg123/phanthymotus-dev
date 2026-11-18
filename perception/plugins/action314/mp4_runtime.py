"""Action314 inference runtime (v13): YOLO11m-pose on a host-shared GPU TensorRT server + ST-GCN++ classifiers.

Pose: every instance sends letterboxed frames to one TensorRT engine per host (gpu_server.py). The engine is
shipped for the official TensorRT build (trt/<TensorRT version>/*.plan on the model mirror); if it does not deserialize on the
host it is rebuilt once. Until the server answers (a few seconds at start-up), a small YOLO11n CPU model keeps
the instance responsive; `wait_gpu()` lets the warm-up call block until the GPU path is live.
Classifiers run as static TensorRT engines on the same server (ONNX Runtime CPU only if the server is down).
"""
import hashlib, json, logging, math, os, threading, time
from pathlib import Path
import cv2, numpy as np
try:
    from .lite_pose import prepare_frame, decode_pose
    from . import gpu_server, model_store
except ImportError:
    from lite_pose import prepare_frame, decode_pose
    import gpu_server, model_store
log = logging.getLogger(__name__)
SIZE = 416


class _OrtModel:
    def __init__(self, path, threads=1):
        import onnxruntime as ort
        o = ort.SessionOptions(); o.inter_op_num_threads = 1; o.intra_op_num_threads = threads
        self.s = ort.InferenceSession(str(path), sess_options=o, providers=['CPUExecutionProvider'])
        self.inp = self.s.get_inputs()[0].name

    def __call__(self, x):
        return self.s.run(None, {self.inp: x})[0]


def letterbox_u8(frame, size=SIZE):
    h, w = frame.shape[:2]; gain = min(size / w, size / h); nw, nh = round(w * gain), round(h * gain)
    left = (size - nw) // 2; top = (size - nh) // 2
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[top:top + nh, left:left + nw] = cv2.resize(frame, (nw, nh))[:, :, ::-1]
    return canvas, {'width': w, 'height': h, 'gain': gain, 'left': left, 'top': top}


class MP4Recognizer:
    def __init__(self, root, config='runtime.json'):
        self.root = Path(root).resolve(); self.config = c = json.loads((self.root / config).read_text())
        cv2.setNumThreads(1)  # files were SHA256-verified by model_store
        self.device_errors = {}; self.gpu = None; self.gpu_role = None; self.gpu_ready = threading.Event()
        self.counters = {'gpu_frames': 0, 'cpu_frames': 0, 'gpu_errors': 0}; self.gpu_log = []
        threads = int(os.environ.get('ACTION314_CPU_THREADS', str(c.get('cpu_threads', 1))))
        self.bodies = [_OrtModel(self.root / x['file'], 1) for x in c['classifiers']]
        self.cpu_front = _OrtModel(self.root / c['pose_model_cpu'], threads) if c.get('pose_model_cpu') else None
        self._check_bodies()
        labels = [x['activity'] for x in c['labels']]
        self.none_index = labels.index('none') if 'none' in labels and os.environ.get('ACTION314_ALLOW_NONE') != '1' else None
        if os.environ.get('ACTION314_DEVICE', 'gpu').lower() != 'cpu': self._join_gpu()
        else: self.device_errors['gpu'] = 'disabled by ACTION314_DEVICE=cpu'

    # ---- GPU attachment -------------------------------------------------------------------
    def _status(self, msg):
        self.gpu_log.append('%s %s' % (time.strftime('%H:%M:%S'), msg)); self.gpu_log[:] = self.gpu_log[-20:]
        log.info('[action314] %s', msg)

    def _join_gpu(self):
        c = self.config
        names = [c['pose_model']] + [x['file'] for x in c['classifiers']]
        tag = hashlib.sha256(''.join(c['model_sha256'][n] for n in names).encode()).hexdigest()[:12]
        listed = model_store.spec()['files']
        try:
            ver = gpu_server.trt_version()
        except Exception as e:
            ver = None; self.device_errors['trt_version'] = str(e)[:200]

        def fetch(rel):   # lazily download one listed file; None if not published for this host
            if rel not in listed: return None
            return model_store.ensure_files([rel], self._status) / rel
        models = [(Path(n).stem, (lambda n=n: fetch(n)), (lambda n=n: fetch('trt/%s/%s.plan' % (ver, Path(n).stem)) if ver else None))
                  for n in names]

        def ready(client, role):
            self.gpu, self.gpu_role = client, role; self.gpu_ready.set(); self.device_errors.pop('gpu', None)
            self._status('backend=gpu-trt (%s)' % role)

        def error(e):
            self.device_errors['gpu'] = type(e).__name__ + ': ' + str(e)[:300]
            log.error('[action314] GPU server unavailable: %s', self.device_errors['gpu'])
        gpu_server.start_or_join(models, tag, ready, error, self._status)

    def wait_gpu(self, timeout):
        return self.gpu_ready.wait(timeout)

    @property
    def device(self):
        return 'gpu-trt' if self.gpu is not None else ('cpu-warmup' if 'gpu' not in self.device_errors else 'cpu')

    def _check_bodies(self):
        for b in self.bodies:
            lg = b(np.zeros((1, self.config.get('pose_frames', 100), 17, 3), np.float32))
            if lg.shape != (1, 126) or not np.isfinite(lg).all(): raise RuntimeError('bad classifier output %s' % (lg.shape,))

    # ---- inference ------------------------------------------------------------------------
    def pose(self, frame):
        g = self.gpu
        if g is not None:
            try:
                x, geo = letterbox_u8(frame)
                out = g(x[None]); self.counters['gpu_frames'] += 1
                return decode_pose(out, geo)
            except Exception as e:
                self.counters['gpu_errors'] += 1; self.device_errors['gpu_last'] = type(e).__name__ + ': ' + str(e)[:200]
                log.warning('[action314] GPU pose call failed, re-attaching: %s', e)
                self.gpu = None; self.gpu_ready.clear(); self._join_gpu()
        if self.cpu_front is None: raise RuntimeError('GPU not ready and no CPU warm-up model')
        x, geo = prepare_frame(frame); self.counters['cpu_frames'] += 1
        return decode_pose(self.cpu_front(x), geo)

    def probs(self, poses):
        c = self.config; v = np.asarray(poses, dtype=np.float32)
        v = v[np.linspace(0, len(v) - 1, c.get('pose_frames', 100)).round().astype(int)][None]
        prob = np.zeros(126, np.float64); g = self.gpu
        for i, (b, spec) in enumerate(zip(self.bodies, c['classifiers'])):
            z = None
            if g is not None:
                try: z = g.model(i, v)[0].astype(np.float64); self.counters['gpu_cls'] = self.counters.get('gpu_cls', 0) + 1
                except Exception as e: self.device_errors['gpu_cls'] = type(e).__name__ + ': ' + str(e)[:200]
            if z is None:
                z = b(v)[0].astype(np.float64); self.counters['cpu_cls'] = self.counters.get('cpu_cls', 0) + 1
            z = z / spec.get('temperature', 1.0); z -= z.max(); p = np.exp(z); p /= p.sum(); prob += spec['weight'] * p
        if self.none_index is not None: prob[self.none_index] = 0.0
        return prob / max(prob.sum(), 1e-12)

    def label(self, prob):
        i = int(prob.argmax()); lab = self.config['labels'][i]
        return {'id': 1, 'position': [0.0, 0.0], 'activity': lab['activity'], 'activity_zh': lab['activity_zh'], 'activity_score': float(prob[i])}

    def classify(self, path):
        start = time.perf_counter(); c = self.config
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened(): raise ValueError('Cannot open video')
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps = cap.get(cv2.CAP_PROP_FPS)
        if total < 1 or not math.isfinite(fps) or fps <= 0: cap.release(); raise ValueError('Invalid video metadata')
        count = min(total, c.get('max_pose_frames', 100), max(2, math.ceil(total / fps * c.get('pose_hz', 8))))
        wanted = set(np.linspace(0, total - 1, count).round().astype(int).tolist()); poses = []; i = 0
        try:
            while True:
                ok, fr = cap.read()
                if not ok: break
                if i in wanted: poses.append(self.pose(fr)[0])
                i += 1
        finally:
            cap.release()
        if not poses: raise ValueError('No frames decoded')
        prob = self.probs(poses); el = (time.perf_counter() - start) * 1000
        return ({'timestamp': time.time(), 'count': 1, 'persons': [self.label(prob)], 'latency_ms': el},
                {'processed_frames': len(poses), 'prediction': int(prob.argmax()), 'latency_ms': el, 'probabilities': prob.tolist()})

    def gpu_info(self):
        srv = getattr(gpu_server.start_or_join, 'server', None)
        return {'device': self.device, 'role': self.gpu_role, 'counters': dict(self.counters),
                'server_stats': dict(srv.stats) if srv else None, 'log': self.gpu_log[-6:]}
