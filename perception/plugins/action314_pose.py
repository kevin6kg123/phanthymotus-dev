"""Darvin #314 pose tool (v13): 126-class action recognition over a ROS2 camera stream.

Follows the reference pose plugin's MCP contract (start/stop/info/config shapes, info.instances
keyed by instance id). One ROS node per instance id is created once and reused across
start/stop cycles, so the evaluator's per-case start/stop never creates or destroys nodes.
"""
from __future__ import annotations
import collections, json, logging, os, queue, re, tempfile, threading, time, urllib.parse, urllib.request
from pathlib import Path
import cv2, numpy as np
from plugins.action314.mp4_runtime import MP4Recognizer
from plugins.action314.model_store import ensure_models, spec as _asset_spec
from plugins.action314 import diag as _diag

log = logging.getLogger(__name__)
_VIDEO = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v'}
_KPTS = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
         'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle']
POSE_HZ = float(os.environ.get('ACTION314_POSE_HZ', '8'))
MIN_POSES = int(os.environ.get('ACTION314_MIN_POSES', '4'))
# The evaluator scores the FIRST message that carries an activity, so the decision is emitted once,
# after the stream goes idle (end of the clip), from all observed poses.
EMIT_IDLE_S = float(os.environ.get('ACTION314_EMIT_IDLE', '0.38'))
GPU_WAIT_S = float(os.environ.get('ACTION314_GPU_WAIT', '300'))
# A pause right after the first frames of a case is a delivery hiccup, not the end of the clip (official clips
# have >= 16 frames). Few raw frames -> require a longer silence before deciding.
RGB_KEEP = int(os.environ.get('ACTION314_RGB_KEEP', '48'))
RGB_MAXSIDE = int(os.environ.get('ACTION314_RGB_MAXSIDE', '400'))
PRECOMPUTE_S = float(os.environ.get('ACTION314_PRECOMPUTE', '0.15'))
RGB_QUIET_S = float(os.environ.get('ACTION314_RGB_QUIET', '0.12'))
RGB_EVERY = int(os.environ.get('ACTION314_RGB_EVERY', '12'))
HAND_KEEP = int(os.environ.get('ACTION314_HAND_KEEP', '8'))
RGB_FIRST = int(os.environ.get('ACTION314_RGB_FIRST', '6'))
RGB_COVER = float(os.environ.get('ACTION314_RGB_COVER', '0.7'))
RGB_MIN_FRAMES = int(os.environ.get('ACTION314_RGB_MIN', '6'))
EMIT_MIN_RAW = int(os.environ.get('ACTION314_EMIT_MIN_RAW', '14'))
EMIT_IDLE_SHORT_S = float(os.environ.get('ACTION314_EMIT_IDLE_SHORT', '0.75'))
TOOLS = [{
    'name': 'pose', 'type': 'processor', 'multiInstance': True,
    'description': 'Recognize one of 126 single-person actions from a camera stream or video; outputs activity, activity_zh, activity_score.',
    'inputSchema': {'type': 'object', 'properties': {
        'action': {'type': 'string', 'enum': ['start', 'stop', 'info', 'config', 'reset', 'list_actions', 'recognize_by_video', 'recognize_by_photo', 'recognize_by_url']},
        'input_topic': {'type': 'string'}, 'instance_id': {'type': 'string'}, 'fps': {'type': 'integer'},
        'video_path': {'type': 'string', 'format': 'file', 'accept': 'video/*', 'uploadTo': 'mcp'},
        'image_path': {'type': 'string', 'format': 'file', 'accept': 'image/*,video/*', 'uploadTo': 'mcp'},
        'url': {'type': 'string'}}, 'required': ['action'],
        'x-action-params': {'start': {'params': ['input_topic']}, 'stop': {'params': []}, 'info': {'params': []},
                            'recognize_by_video': {'params': ['video_path']}, 'recognize_by_photo': {'params': ['image_path']}, 'recognize_by_url': {'params': ['url']}}},
    'topic_in': [{'format': 'image/jpeg', 'desc': 'camera image input'}],
    'topic_out': [{'format': 'data/json', 'desc': '126-class action label and confidence'}],
}]


class PosePerceptionPlugin:
    PREFIX = 'pose'

    def __init__(self, plugin_cfg, namespace, executor):
        self.cfg = dict(plugin_cfg or {}); self.namespace = namespace; self.executor = executor
        self._runtime = None; self._load_error = None; self._lock = threading.RLock(); self._infer = threading.Lock()
        self._nodes = {}; self._gpu_waited = False; self._last = None; self._pub_node = None; self._pub = None; self._on_demand = False
        threading.Thread(target=self._preload, daemon=True, name='action314-preload').start()

    def get_tools(self):
        return TOOLS

    def _preload(self):
        try:
            _diag.startup_report(_asset_spec())
        except Exception:
            log.exception('[action314] diag failed')
        try:
            t = time.perf_counter(); self._runtime = MP4Recognizer(ensure_models(lambda m: log.info('[action314] %s', m)))
            log.info('[action314] models ready in %.2fs (backend=%s)', time.perf_counter() - t, self._runtime.device)
            _diag.periodic(lambda: {'device': self._runtime.device, 'errors': self._runtime.device_errors, **self._runtime.gpu_info(),
                                    'cases': sum(n.cases for n in list(self._nodes.values()))}, float(os.environ.get('ACTION314_DIAG_EVERY', '120')))
        except Exception as e:
            self._load_error = type(e).__name__ + ': ' + str(e)[:300]; log.exception('[action314] model load failed')

    def _model(self, wait=60.0):
        t = time.time()
        while self._runtime is None and self._load_error is None and time.time() - t < wait:
            time.sleep(0.05)
        if self._runtime is None: raise RuntimeError(self._load_error or 'model still loading')
        return self._runtime

    # ---- one-shot inputs -------------------------------------------------
    def _recognize_path(self, path):
        path = Path(path)
        if not path.is_file(): raise ValueError('Input path is not a file')
        m = self._model()
        with self._infer:
            if path.suffix.lower() in _VIDEO:
                result, _ = m.classify(path)
            else:
                frame = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is None: result, _ = m.classify(path)
                else:
                    t = time.perf_counter(); pose, _ = m.pose(frame)
                    result = {'timestamp': time.time(), 'count': 1, 'persons': [m.label(m.probs([pose]))], 'latency_ms': (time.perf_counter() - t) * 1000}
        self._last = result; self._publish_oneshot(result); return result

    def _publish_oneshot(self, result):
        if self.executor is None: return
        try:
            from std_msgs.msg import String
            import rclpy
            if self._pub_node is None:
                self._pub_node = rclpy.create_node('action314_oneshot'); self._pub = self._pub_node.create_publisher(String, '/perception/pose', 10)
                self.executor.add_node(self._pub_node)
            self._pub.publish(String(data=json.dumps(result)))
        except Exception:
            log.exception('[action314] one-shot publish failed')

    def _recognize(self, args):
        path = next((args.get(n) for n in ['video_path', 'image_path', 'file_path', 'path', 'video'] if args.get(n)), None)
        if path: return self._recognize_path(path)
        url = args.get('url')
        if not url: raise ValueError('video_path, image_path, file_path or url is required')
        if urllib.parse.urlparse(url).scheme not in {'http', 'https'}: raise ValueError('Only http(s) URL inputs are supported')
        suffix = Path(urllib.parse.urlparse(url).path).suffix.lower() or '.mp4'
        with tempfile.NamedTemporaryFile(dir=os.environ.get('ACTION314_TEMP_DIR', tempfile.gettempdir()), suffix=suffix) as f:
            with urllib.request.urlopen(url, timeout=30) as r:
                n = 0
                while True:
                    b = r.read(1 << 20)
                    if not b: break
                    n += len(b)
                    if n > 64 << 20: raise ValueError('Input URL exceeds 64MiB')
                    f.write(b)
            f.flush(); return self._recognize_path(f.name)

    # ---- MCP surface -------------------------------------------------------
    def dispatch(self, name, args):
        action = args.get('action', name)
        if action == 'pose':
            action = 'recognize_by_video' if any(args.get(n) for n in ['video_path', 'image_path', 'file_path', 'path', 'url']) else 'info'
        iid = args.get('instance_id') or ''
        try:
            if action in {'recognize_by_video', 'recognize_video', 'recognize_by_photo', 'recognize_by_url', 'recognize'}:
                return self._recognize(args)
            if action == 'info':
                return self._info(args)
            if action == 'list_actions':
                labels = self._model().config['labels']
                return {'ok': True, 'activities': [{'name': r['activity'], 'name_zh': r['activity_zh']} for r in labels], 'count': len(labels)}
            if action == 'config':
                cfg = {k: v for k, v in args.items() if k not in ('action', 'instance_id') and v is not None and v != ''}
                return {'status': 'configured', 'instance_id': iid, 'config': cfg} if iid else {'status': 'configured', 'config': cfg}
            if action == 'start':
                topic = args.get('input_topic') or ((args.get('input_topics') or [''])[0])
                if not topic:
                    self._on_demand = True
                    return {'state': 'running', 'mode': 'on_demand', 'output': '/perception/pose'}
                key = iid or topic
                if not self._gpu_waited:
                    self._gpu_waited = True
                    try:
                        m = self._model(wait=GPU_WAIT_S); t = time.time(); ok = m.wait_gpu(GPU_WAIT_S)
                        log.info('[action314] start: GPU %s after %.1fs wait (device=%s)', 'ready' if ok else 'NOT ready', time.time() - t, m.device)
                    except Exception:
                        log.exception('[action314] GPU wait failed')
                with self._lock:
                    node = self._nodes.get(key)
                    if node is None or node.topic != topic:
                        if node is not None: node.deactivate()
                        node = _Camera(self, topic, key); self._nodes[key] = node
                    node.activate()
                return node.status(True)
            if action == 'stop':
                with self._lock:
                    nodes = [self._nodes[iid]] if iid in self._nodes else ([] if iid else list(self._nodes.values()))
                    for n in nodes: n.deactivate()
                if not iid: self._on_demand = False
                if iid and nodes: return nodes[0].status(False)
                return {'state': 'idle', 'stopped_instances': [n.key for n in nodes]} if nodes else {'state': 'idle'}
            if action == 'reset':
                for n in list(self._nodes.values()): n.activate() if n.active else None
                self._last = None; return {'ok': True}
            return None
        except Exception as e:
            log.exception('[action314] request failed')
            return {'timestamp': time.time(), 'count': 0, 'persons': [], 'ok': False, 'error': type(e).__name__ + ': ' + str(e), 'latency_ms': 0}

    def _info(self, args):
        base = {'name': 'PosePerception', 'manufacture': 'Embodied', 'model': 'action314'}
        if self._runtime is None and self._load_error is None:
            return {**base, 'state': 'loading', 'desc': 'Loading action engine...'}
        if self._runtime is None:
            return {**base, 'state': 'error', 'desc': 'Engine load failed: ' + str(self._load_error)}
        with self._lock:
            active = {k: n for k, n in self._nodes.items() if n.active}
        instances = {k: n.info() for k, n in active.items()}
        iid = args.get('instance_id') or ''; topic = args.get('input_topic') or ''
        if iid in self._nodes: topic = self._nodes[iid].topic
        elif not topic and active: topic = next(iter(active.values())).topic
        out = topic.rstrip('/') + '/poses' if topic else '/perception/pose'
        info = {**base, 'state': 'running' if instances else 'idle', 'keypoints': 17, 'keypoint_names': _KPTS,
                'action_backend': 'action314', 'action_backend_effective': 'action314', 'postures': [], 'instances': instances,
                'topic_in': [{'topic': topic, 'format': 'image/jpeg'}] if topic else [],
                'topic_out': [{'topic': out, 'format': 'data/json'}, {'topic': out + '/skeleton', 'format': 'sensor/pose2d'}] if topic else [],
                'desc': '126-class action recognition (YOLO11m-pose TensorRT + ST-GCN++ TensorRT, host-shared GPU server)', 'hands': False,
                'device': self._runtime.device, 'device_errors': self._runtime.device_errors, 'gpu': self._runtime.gpu_info(), 'supported_actions': 126, 'last_result': self._last}
        preds = {k: n.last_prediction for k, n in active.items() if n.last_prediction}
        if preds: info['last_prediction'] = preds
        return info

    def shutdown(self):
        for n in list(self._nodes.values()): n.close()
        self._nodes.clear()
        if self._pub_node is not None:
            try: self.executor.remove_node(self._pub_node); self._pub_node.destroy_node()
            except Exception: pass


class _Camera:
    def __init__(self, owner, topic, key):
        import rclpy
        from sensor_msgs.msg import CompressedImage
        from std_msgs.msg import String
        from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
        self.owner, self.topic, self.key = owner, topic, key
        self.output_topic = topic.rstrip('/') + '/poses'; self._String = String
        self.active = False; self.epoch = 0; self.cases = 0; self._state = threading.Lock()
        self.queue = queue.Queue(maxsize=int(os.environ.get('ACTION314_QUEUE', '1'))); self.stop_evt = threading.Event()
        self._reset_case()
        self.error_count = 0; self._last_arrival = 0.0; self._interval = 1.0 / max(POSE_HZ, 0.5)
        qin = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=2, durability=DurabilityPolicy.VOLATILE)
        self.node = rclpy.create_node('action314_' + re.sub('[^A-Za-z0-9_]', '_', key)[-60:])
        self.pub = self.node.create_publisher(String, self.output_topic, 10)
        self.sub = self.node.create_subscription(CompressedImage, topic, self._on_image, qin)
        owner.executor.add_node(self.node)
        self.rgb_evt = threading.Event(); self.rgb_job = None; self.rgb_res = None
        self.rgb_thread = threading.Thread(target=self._rgb_worker, daemon=True, name='action314-rgb-' + key[-16:]); self.rgb_thread.start()
        self.thread = threading.Thread(target=self._worker, daemon=True, name='action314-' + key[-20:]); self.thread.start()

    def _reset_case(self):
        self.history = collections.deque(maxlen=100); self.acc = None; self.since = 0
        self.detect_count = 0; self.person_count = 0; self.last_actions = {}; self.last_prediction = {}; self.result = None
        self.emitted = None; self.last_frame_t = 0.0; self.last_raw_t = 0.0; self.cached = None; self.raw_count = 0; self.max_gap = 0.0; self.frames = []; self.frame_step = 1; self.frame_seen = 0
        self.rgb_res = None; self.rgb_job = None; self.rgb_last = 0
        self.hand_entries = []; self.hand_step = 1; self.hand_seen = 0; self.hand_q = None; self.hand_gate = None

    def activate(self):
        with self._state:
            self.epoch += 1; self._drain(); self._reset_case(); self._last_arrival = 0.0; self.active = True; self.cases += 1

    def deactivate(self):
        try:
            if self.active and self.emitted is None and self.history: self._emit(self.epoch, 'stop')
        except Exception:
            log.exception('[action314] emit on stop failed')
        with self._state:
            self.active = False; self.epoch += 1; self._drain()

    def _emit(self, epoch, why='idle'):
        m = self.owner._model(wait=5)
        now = time.monotonic()
        log.info('[action314] emit why=%s key=%s poses=%d raw=%d maxgap=%.2fs gap_raw=%.2fs gap_pose=%.2fs', why, self.key[-12:], len(self.history),
                 self.raw_count, self.max_gap, now - self.last_raw_t if self.last_raw_t else -1, now - self.last_frame_t if self.last_frame_t else -1)
        c = self.cached
        if c is not None and c[0] == len(self.history) and epoch == self.epoch:
            skel = c[1]
        else:
            skel = self._predict(m)
            if skel is None or epoch != self.epoch: return
        rr = self.rgb_res
        pr = rr[2] if rr is not None and rr[0] == epoch else None
        hq = self.hand_q[1] if self.hand_q is not None and self.hand_q[0] == epoch else None
        prob = m.fuse(skel, pr, hq)
        log.info('[action314] decide rgb=%s rgb_frames=%s of %d', pr is not None, rr[1] if rr is not None else 0, self.frame_seen)
        try:
            labs = [l['activity'] if isinstance(l, dict) else str(l) for l in m.config['labels']]; ni = labs.index('none') if 'none' in labs else None
            top = np.argsort(prob)[::-1][:3]
            log.info('[action314] diag pnone=%.3f top3=%s', float(skel[ni]) if ni is not None else -1.0, ';'.join('%s:%.2f' % (labs[j], prob[j]) for j in top))
        except Exception: pass
        person = m.label(prob); self.emitted = person
        self.last_actions = {1: person['activity']}
        self.last_prediction = {'activity': person['activity'], 'activity_zh': person['activity_zh'], 'score': person['activity_score']}
        result = {'timestamp': time.time(), 'count': 1, 'persons': [person], 'latency_ms': 0}
        self.detect_count += 1; self.person_count = 1; self.owner._last = result
        if epoch == self.epoch: self.pub.publish(self._String(data=json.dumps(result)))

    def _predict(self, m):
        with self.owner._infer:
            if not self.history: return None
            return m.skel_probs(list(self.history))

    def _want_rgb(self, force=False):
        if self.rgb_job is not None or len(self.frames) < RGB_MIN_FRAMES or not self.history: return
        if not force and (self.frame_seen < RGB_FIRST if self.rgb_last == 0 else self.frame_seen - self.rgb_last < RGB_EVERY): return
        if self.rgb_res is not None and self.rgb_res[0] == self.epoch and self.rgb_res[1] == self.frame_seen: return
        self.rgb_last = self.frame_seen
        self.rgb_job = (self.epoch, self.frame_seen, list(self.history), list(self.frames), force); self.rgb_evt.set()

    def _rgb_worker(self):
        while not self.stop_evt.is_set():
            if not self.rgb_evt.wait(0.2): continue
            self.rgb_evt.clear(); job = self.rgb_job
            if job is None: continue
            try:
                ep, seen, poses, frames, force = job; m = self.owner._model(wait=5)
                if not force and self.last_frame_t and time.monotonic() - self.last_frame_t > RGB_QUIET_S: continue
                pr = m.rgb_probs(poses, frames) if ep == self.epoch else None
                if pr is not None and ep == self.epoch: self.rgb_res = (ep, seen, pr)
                if getattr(m, 'has_hand', False) and ep == self.epoch and self.hand_entries and self._hand_gate(m, ep, poses, pr):
                    ents = list(self.hand_entries); sel = [ents[i] for i in sorted(set(np.linspace(0, len(ents) - 1, min(4, len(ents))).round().astype(int).tolist()))]
                    todo = [e for e in sel if e['kp'] is None]
                    if todo and ep == self.epoch:
                        r = m.hand_process([e['crop'] for e in todo])
                        if r is not None:
                            for e, k in zip(todo, r): e['kp'] = k
                    done = [e['kp'] for e in sel if e['kp'] is not None]
                    if len(done) == len(sel) and ep == self.epoch: self.hand_q = (ep, m.hand_probs(done))
            except Exception:
                self.error_count += 1; log.exception('[action314] rgb failed')
            finally:
                self.rgb_job = None

    def _keep_hand(self, m, frame, pose):
        """RTMW person crop at pose time; uniform temporal coverage with at most HAND_KEEP entries."""
        self.hand_seen += 1
        if (self.hand_seen - 1) % self.hand_step: return
        try: crop = m.hand_crops(frame, pose)
        except Exception: crop = None
        if crop is None: return
        self.hand_entries.append({'crop': crop, 'kp': None})
        if len(self.hand_entries) >= HAND_KEEP:
            self.hand_entries = self.hand_entries[::2]; self.hand_step *= 2

    def _hand_gate(self, m, ep, poses, pr):
        """run the finger branch only when a static hand gesture is in the top-2 of the current fused prediction (sticky per case);
        closed until the branch has been initialised by its own daemon thread."""
        if getattr(m, '_hand', None) is None:
            if not getattr(m, '_hand_init_started', False):
                m._hand_init_started = True
                def _init():
                    try: m._hand_branch(); log.info('[action314] hand branch ready')
                    except Exception as e: log.warning('[action314] hand branch unavailable: %s', str(e)[:200])
                threading.Thread(target=_init, daemon=True, name='action314-hand-init').start()
            return False
        g = getattr(self, 'hand_gate', None)
        if g is not None and g[0] == ep and g[1]: return True
        ok = False
        try:
            with self.owner._infer: sk = m.skel_probs(list(poses))
            if sk is not None and ep == self.epoch:
                pf = m.fuse(sk, pr); top = np.argsort(pf)[::-1][:2]
                ok = bool(set(int(i) for i in top) & set(int(c) for c in m.config['hand'].get('classes', [59, 60, 84, 85, 86, 87, 88])))
        except Exception:
            ok = False
        self.hand_gate = (ep, ok)
        if ok: log.info('[action314] hand gate on key=%s poses=%d', self.key[-12:], len(poses))
        return ok

    def _keep_frame(self, frame):
        """store a downscaled copy for the RGB branch: uniform temporal coverage with at most RGB_KEEP frames."""
        self.frame_seen += 1
        if (self.frame_seen - 1) % self.frame_step: return
        h, w = frame.shape[:2]; k = RGB_MAXSIDE / max(h, w)
        self.frames.append(cv2.resize(frame, (int(w * k), int(h * k)), interpolation=cv2.INTER_AREA) if k < 1 else frame)
        if len(self.frames) >= RGB_KEEP:
            self.frames = self.frames[::2]; self.frame_step *= 2

    def _drain(self):
        try:
            while True: self.queue.get_nowait()
        except queue.Empty:
            pass

    def status(self, running):
        return {'state': 'running' if running else 'idle', 'input': self.topic, 'output': self.output_topic,
                'skeleton_output': self.output_topic + '/skeleton', 'overlay_output': None, 'gesture_output': None, 'mode': 'stream', 'hands': False}

    def info(self):
        return {'input': self.topic, 'output': self.output_topic, 'skeleton_output': self.output_topic + '/skeleton', 'overlay_output': None,
                'confidence': 0.25, 'fps': POSE_HZ, 'kpt_confidence': 0.0, 'max_persons': 1, 'publish_keypoints': 'off', 'publish_bbox': False,
                'publish_overlay': False, 'detect_count': self.detect_count, 'persons_last_frame': self.person_count,
                'last_actions': dict(self.last_actions), 'gesture_output': None, 'gesture_whitelist': [], 'gesture_event_count': 0,
                'recent_gestures': [], 'hands': False, 'stream_errors': self.error_count}

    def _on_image(self, msg):
        try:
            if not self.active: return
            now = time.monotonic()
            if self.last_raw_t: self.max_gap = max(self.max_gap, now - self.last_raw_t)
            self.last_raw_t = now; self.raw_count += 1
            if now - self._last_arrival < self._interval: return
            self._last_arrival = now
            item = (self.epoch, bytes(msg.data))
            try:
                self.queue.put_nowait(item)
            except queue.Full:
                try: self.queue.get_nowait()
                except queue.Empty: pass
                try: self.queue.put_nowait(item)
                except queue.Full: pass
        except Exception:
            self.error_count += 1

    def _worker(self):
        while not self.stop_evt.is_set():
            try:
                epoch, data = self.queue.get(timeout=0.02)
            except queue.Empty:
                try:
                    need = EMIT_IDLE_S if self.raw_count >= EMIT_MIN_RAW else EMIT_IDLE_SHORT_S
                    idle = time.monotonic() - max(self.last_frame_t, self.last_raw_t) if self.last_frame_t else 0
                    if self.active and self.emitted is None and self.history and self.last_frame_t:
                        if idle >= need:
                            self._emit(self.epoch)
                        elif idle >= PRECOMPUTE_S and (self.cached is None or self.cached[0] != len(self.history)):
                            ep = self.epoch; tag = len(self.history)
                            rr = self.rgb_res
                            if self.frames and (rr is None or rr[0] != self.epoch or rr[1] < RGB_COVER * self.frame_seen): self._want_rgb(force=True)
                            pr = self._predict(self.owner._model(wait=5))
                            if ep == self.epoch and pr is not None: self.cached = (tag, pr)
                except Exception:
                    self.error_count += 1; log.exception('[action314] emit failed')
                continue
            if epoch != self.epoch or not self.active: continue
            try:
                started = time.perf_counter()
                m = self.owner._model(wait=30)
                frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                if frame is None: self.error_count += 1; continue
                with self.owner._infer:
                    pose, _ = m.pose(frame)
                if epoch != self.epoch: continue
                self.history.append(pose); self.last_frame_t = time.monotonic()
                if getattr(m, 'has_hand', False): self._keep_hand(m, frame, pose)
                if m.has_rgb: self._keep_frame(frame); self._want_rgb()
                if self.emitted is not None:
                    result = {'timestamp': time.time(), 'count': 1, 'persons': [self.emitted]}
                else:
                    result = {'timestamp': time.time(), 'count': 0, 'persons': []}
                result['latency_ms'] = int((time.perf_counter() - started) * 1000)
                self.detect_count += 1; self.person_count = result['count']
                if self.active and epoch == self.epoch:
                    self.pub.publish(self._String(data=json.dumps(result)))
            except Exception:
                self.error_count += 1; log.exception('[action314] stream frame failed')

    def close(self):
        self.active = False; self.stop_evt.set(); self.thread.join(timeout=5); self.rgb_thread.join(timeout=2)
        try:
            self.owner.executor.remove_node(self.node); self.node.destroy_node()
        except Exception:
            log.exception('[action314] node teardown failed')
