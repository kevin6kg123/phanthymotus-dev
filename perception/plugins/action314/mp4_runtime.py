"""Action314 inference runtime: YOLO11n-pose + ST-GCN++ classifiers.

Backends, in order: TensorRT engines shipped for this TensorRT family (models/trt/<family>/),
ONNX Runtime CUDA, ONNX Runtime CPU. Every backend is warmed up before it is accepted; any
failure falls through to the next one. ACTION314_DEVICE=trt|cuda|cpu pins the first choice.
"""
import fcntl, hashlib, json, logging, math, os, threading, time
from pathlib import Path
import cv2, numpy as np
try:
    from .lite_pose import prepare_frame, decode_pose
except ImportError:
    from lite_pose import prepare_frame, decode_pose
log = logging.getLogger(__name__)


class _OrtModel:
    def __init__(self, path, kind):
        import onnxruntime as ort
        o = ort.SessionOptions(); o.inter_op_num_threads = 1
        if kind == 'cuda':
            o.intra_op_num_threads = 1
            prov = [('CUDAExecutionProvider', {'device_id': 0, 'gpu_mem_limit': int(os.environ.get('ACTION314_GPU_MEM_MB', '160')) << 20,
                     'arena_extend_strategy': 'kSameAsRequested', 'cudnn_conv_algo_search': 'HEURISTIC'}), 'CPUExecutionProvider']
        else:
            o.intra_op_num_threads = int(os.environ.get('ACTION314_CPU_THREADS', '1')); prov = ['CPUExecutionProvider']
        self.s = ort.InferenceSession(str(path), sess_options=o, providers=prov)
        if kind == 'cuda' and self.s.get_providers()[0] != 'CUDAExecutionProvider':
            raise RuntimeError('CUDA provider inactive')
        self.inp = self.s.get_inputs()[0].name

    def __call__(self, x):
        return self.s.run(None, {self.inp: x})[0]


class _TrtModel:
    def __init__(self, path):
        try:
            from utils.tensorrt_runtime import TensorRTEngine
        except ImportError:
            from tensorrt_runtime import TensorRTEngine
        self.e = TensorRTEngine(path)

    def __call__(self, x):
        return np.asarray(self.e.infer(np.ascontiguousarray(x, dtype=np.float32))[0], dtype=np.float32)


class MP4Recognizer:
    def __init__(self, root, config='runtime.json'):
        self.root = Path(root).resolve(); self.config = c = json.loads((self.root / config).read_text())
        cv2.setNumThreads(1)
        names = [c['pose_model']] + [x['file'] for x in c['classifiers']]
        for n in names:
            p = (self.root / n).resolve()
            if hashlib.sha256(p.read_bytes()).hexdigest() != c['model_sha256'][n]:
                raise ValueError('Model hash mismatch: ' + n)
        want = os.environ.get('ACTION314_DEVICE', 'trt').lower()
        # onnxruntime-CUDA is opt-in only: ten concurrent CUDA/cuDNN processes exhausted the 8 GB Orin in the official run.
        order = {'trt': ['trt', 'cpu'], 'cuda': ['cuda', 'cpu'], 'cpu': ['cpu']}.get(want, ['trt', 'cpu'])
        self.device, self.device_errors = None, {}
        for kind in order:
            try:
                self._build(kind); self.device = kind; break
            except Exception as e:
                self.device_errors[kind] = type(e).__name__ + ': ' + str(e)[:240]
                log.warning('[action314] backend %s unavailable: %s', kind, self.device_errors[kind])
        if self.device is None:
            raise RuntimeError('no usable backend: %s' % self.device_errors)
        log.info('[action314] backend=%s', self.device)
        if self.device != 'trt' and os.environ.get('ACTION314_TRT_BUILD', '1') == '1':
            threading.Thread(target=self._upgrade, daemon=True, name='action314-trt').start()
        labels = [x['activity'] for x in c['labels']]
        self.none_index = labels.index('none') if 'none' in labels and os.environ.get('ACTION314_ALLOW_NONE') != '1' else None

    def _build(self, kind):
        c = self.config; files = [c['pose_model']] + [x['file'] for x in c['classifiers']]
        if kind == 'trt':
            try:
                from utils.tensorrt_runtime import tensorrt_family
            except ImportError:
                from tensorrt_runtime import tensorrt_family
            d = self.root / 'trt' / tensorrt_family()
            plans = [d / (Path(f).stem + '.plan') for f in files]
            missing = [str(p) for p in plans if not p.is_file()]
            if missing: raise FileNotFoundError('missing engines: ' + ', '.join(missing))
            models = [_TrtModel(p) for p in plans]
        else:
            if kind == 'cuda':
                import onnxruntime as ort
                if 'CUDAExecutionProvider' not in ort.get_available_providers(): raise RuntimeError('no CUDA provider')
            models = [_OrtModel(self.root / f, kind) for f in files]
        front, bodies = models[0], models[1:]
        out = front(np.zeros((1, 3, 416, 416), np.float32))
        if np.asarray(out).shape[1] != 56: raise RuntimeError('unexpected pose output %s' % (np.asarray(out).shape,))
        for b in bodies:
            lg = b(np.zeros((1, c.get('pose_frames', 100), 17, 3), np.float32))
            if lg.shape != (1, 126) or not np.isfinite(lg).all(): raise RuntimeError('bad classifier output %s' % (lg.shape,))
        self.front, self.bodies = front, bodies

    def _plans_dir(self):
        import tensorrt as trt
        return Path(os.environ.get('ACTION314_TRT_CACHE', '/dev/shm/action314_trt')) / trt.__version__

    def _upgrade(self):
        """Build TensorRT plans once per host (shared /dev/shm + file lock), then hot-swap from CPU."""
        try:
            import tensorrt as trt
            d = self._plans_dir(); d.mkdir(parents=True, exist_ok=True)
            try:
                idx = max(0, (int(os.environ.get('MCP_PORT', '15720')) - 15720) // 100)
            except ValueError:
                idx = 0
            time.sleep(min(idx, 9) * float(os.environ.get('ACTION314_TRT_STAGGER_S', '3')))
            c = self.config; names = [Path(c['pose_model']).stem] + [Path(x['file']).stem for x in c['classifiers']]
            with open(d / '.lock', 'w') as lk:
                fcntl.flock(lk, fcntl.LOCK_EX)
                if not (d / 'done').exists():
                    avail = 0
                    for line in open('/proc/meminfo'):
                        if line.startswith('MemAvailable:'): avail = int(line.split()[1]) // 1024
                    if avail < int(os.environ.get('ACTION314_TRT_MIN_MB', '1200')):
                        raise RuntimeError('not enough memory to build engines (%d MB available)' % avail)
                    t0 = time.time(); logger = trt.Logger(trt.Logger.ERROR)
                    for n in names:
                        b = trt.Builder(logger); net = b.create_network(0); parser = trt.OnnxParser(net, logger)
                        if not parser.parse((self.root / (n + '.onnx')).read_bytes()): raise RuntimeError('onnx parse failed: ' + n)
                        cfg = b.create_builder_config(); cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 256 << 20)
                        cfg.set_flag(trt.BuilderFlag.FP16); cfg.builder_optimization_level = int(os.environ.get('ACTION314_TRT_OPT_LEVEL', '1'))
                        plan = b.build_serialized_network(net, cfg)
                        if plan is None: raise RuntimeError('engine build failed: ' + n)
                        tmp = d / ('.%s.%d' % (n, os.getpid())); tmp.write_bytes(bytes(plan)); os.replace(tmp, d / (n + '.plan'))
                        del plan, cfg, parser, net, b
                    (d / 'done').write_text('%.1f' % (time.time() - t0))
                    log.info('[action314] TensorRT %s plans built in %.1fs', trt.__version__, time.time() - t0)
            models = [_TrtModel(d / (n + '.plan')) for n in names]
            front, bodies = models[0], models[1:]
            out = np.asarray(front(np.zeros((1, 3, 416, 416), np.float32)))
            if out.shape[1] != 56: raise RuntimeError('unexpected pose output %s' % (out.shape,))
            for b in bodies:
                lg = b(np.zeros((1, self.config.get('pose_frames', 100), 17, 3), np.float32))
                if lg.shape != (1, 126) or not np.isfinite(lg).all(): raise RuntimeError('bad classifier output')
            self.front, self.bodies, self.device = front, bodies, 'trt'
            log.info('[action314] backend=trt (runtime-built plans, hot swap)')
        except Exception as e:
            self.device_errors['trt_runtime'] = type(e).__name__ + ': ' + str(e)[:240]
            log.warning('[action314] runtime TensorRT upgrade failed, staying on %s: %s', self.device, self.device_errors['trt_runtime'])

    def pose(self, frame):
        x, g = prepare_frame(frame)
        return decode_pose(self.front(x), g)

    def probs(self, poses):
        c = self.config; v = np.asarray(poses, dtype=np.float32)
        v = v[np.linspace(0, len(v) - 1, c.get('pose_frames', 100)).round().astype(int)][None]
        prob = np.zeros(126, np.float64)
        for b, spec in zip(self.bodies, c['classifiers']):
            z = b(v)[0].astype(np.float64) / spec.get('temperature', 1.0); z -= z.max(); p = np.exp(z); p /= p.sum(); prob += spec['weight'] * p
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
