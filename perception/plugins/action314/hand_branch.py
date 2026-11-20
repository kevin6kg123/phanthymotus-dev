"""Hand-gesture branch for action314: MediaPipe palm detector + hand-landmark model (OpenCV model zoo, Apache-2.0; the
pre/post-processing in mp_palmdet.py / mp_handpose.py is vendored unchanged) run on full-resolution wrist crops, followed by
a small finger-geometry MLP. Crop rule, landmark normalisation and MLP are identical to the offline training pipeline
(hand_lm_cache.py / hand_cls.py). The two networks are served as TensorRT engines by the shared GPU server; each engine
returns one packed tensor (palm: (1,2016,19) regressors|score, hand: (1,128) landmarks|conf|handedness|world)."""
import math
import numpy as np, cv2
try:
    from .mp_palmdet import MPPalmDet
    from .mp_handpose import MPHandPose
except ImportError:
    from mp_palmdet import MPPalmDet
    from mp_handpose import MPHandPose

GESTURES = [59, 60, 84, 85, 86, 87, 88]   # ball up paper, play magic cube, thumb up/down, OK, victory, snap fingers
CROP = 256


class _Net:
    """stands in for the cv.dnn network used by the vendored classes."""
    def __init__(self, fn, split): self.fn, self.split, self.x = fn, split, None
    def setInput(self, x): self.x = np.ascontiguousarray(x, np.float32)
    def getUnconnectedOutLayersNames(self): return None
    def forward(self, _=None): return self.split(np.asarray(self.fn(self.x), np.float32))


class _Palm(MPPalmDet):
    def __init__(self, fn):
        self.nms_threshold, self.score_threshold, self.topK = 0.3, 0.4, 5000
        self.input_size = np.array([192, 192]); self.anchors = self._load_anchors()
        self.model = _Net(fn, lambda y: [y.reshape(1, 2016, 19)[..., :18], y.reshape(1, 2016, 19)[..., 18:]])


class _Hand(MPHandPose):
    def __init__(self, fn):
        self.conf_threshold = 0.5; self.input_size = np.array([224, 224])
        self.PALM_LANDMARK_IDS = [0, 5, 9, 13, 17, 1, 2]; self.PALM_LANDMARKS_INDEX_OF_PALM_BASE = 0; self.PALM_LANDMARKS_INDEX_OF_MIDDLE_FINGER_BASE = 2
        self.PALM_BOX_PRE_SHIFT_VECTOR = [0, 0]; self.PALM_BOX_PRE_ENLARGE_FACTOR = 4; self.PALM_BOX_SHIFT_VECTOR = [0, -0.4]
        self.PALM_BOX_ENLARGE_FACTOR = 3; self.HAND_BOX_SHIFT_VECTOR = [0, -0.1]; self.HAND_BOX_ENLARGE_FACTOR = 1.65
        self.model = _Net(fn, lambda y: [y.reshape(1, 128)[:, :63], y.reshape(1, 128)[:, 63:64], y.reshape(1, 128)[:, 64:65], y.reshape(1, 128)[:, 65:128]])


def wrist_crops(frame, pose):
    """per arm (left, right COCO wrist/elbow): (crop 256x256 BGR, centre, forearm, scale, x0, y0) or None. pose (17,3) in [-1,1]."""
    h, w = frame.shape[:2]; p = np.asarray(pose, np.float32); out = [None, None]
    x = (p[:, 0] + 1) * w / 2; y = (p[:, 1] + 1) * h / 2
    for a, (wi, ei) in enumerate(((9, 7), (10, 8))):
        if p[wi, 2] < 0.15 or p[ei, 2] < 0.15: continue
        wr, el = np.array([x[wi], y[wi]]), np.array([x[ei], y[ei]]); fore = max(float(np.linalg.norm(wr - el)), 0.06 * min(w, h))
        c = wr + 0.35 * (wr - el); side = max(2.2 * fore, 0.15 * min(w, h)); s = int(side); x0, y0 = int(c[0] - side / 2), int(c[1] - side / 2)
        sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(w, x0 + s), min(h, y0 + s)
        if sx1 <= sx0 or sy1 <= sy0 or s < 8: continue
        pad = np.zeros((s, s, 3), np.uint8); pad[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = frame[sy0:sy1, sx0:sx1]
        k = CROP / float(s); img = cv2.resize(pad, (CROP, CROP), interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_LINEAR)
        out[a] = (img, c.astype(np.float32), fore, k, x0, y0)
    return out


def _gelu(x):
    return 0.5 * x * (1.0 + np.vectorize(math.erf, otypes=[np.float32])(x / math.sqrt(2.0)))


class HandBranch:
    def __init__(self, mlp_npz, palm_fn, hand_fn, none_index=37):
        z = np.load(mlp_npz); self.W = {k: z[k].astype(np.float32) for k in z.files}; self.none = none_index
        self.palm, self.hand = _Palm(palm_fn), _Hand(hand_fn)

    def process(self, crops):
        """crops from wrist_crops -> per arm (world lm (21,3), image lm normalised (21,2), conf, handedness) or None."""
        res = [None, None]
        for a, e in enumerate(crops):
            if e is None: continue
            img, c, fore, k, x0, y0 = e
            palms = self.palm.infer(img)
            if palms is None or len(palms) == 0: continue
            r = self.hand.infer(img, palms[np.argmax(palms[:, -1])])
            if r is None: continue
            r = np.asarray(r, np.float32); im = r[4:67].reshape(21, 3)[:, :2] / k + np.array([x0, y0], np.float32)
            res[a] = (r[67:130].reshape(21, 3), (im - c) / fore, float(r[131]), float(r[130]))
        return res

    @staticmethod
    def features(results, T=8):
        idx = np.linspace(0, len(results) - 1, T).round().astype(int)
        LM = np.zeros((T, 2, 21, 3), np.float32); IM = np.zeros((T, 2, 21, 2), np.float32); CF = np.zeros((T, 2), np.float32); HD = np.zeros((T, 2), np.float32)
        for t, j in enumerate(idx):
            for a in (0, 1):
                r = results[j][a]
                if r is not None: LM[t, a], IM[t, a], CF[t, a], HD[t, a] = r
        wv = LM - LM[:, :, :1]; wv = wv / (np.linalg.norm(wv[:, :, 9], axis=-1, keepdims=True)[..., None] + 1e-6)
        sv = IM - IM[:, :, :1]; sv = sv / (np.linalg.norm(sv[:, :, 9], axis=-1, keepdims=True)[..., None] + 1e-6)
        f = np.concatenate([wv.reshape(T, 2, -1), sv.reshape(T, 2, -1), IM[:, :, 0], HD[..., None] * 2 - 1], -1)
        m = (CF > 0).astype(np.float32)
        return (f * m[..., None]).astype(np.float32), m

    def _logits(self, x, m):
        W = self.W
        h = _gelu(x @ W['f_0_weight'].T + W['f_0_bias']); h = _gelu(h @ W['f_3_weight'].T + W['f_3_bias']) + W['arm']
        mm = m[..., None]; cnt = mm.sum((0, 1))
        mean = (h * mm).sum((0, 1)) / max(float(cnt[0]), 1.0); mx = (h + (mm - 1) * 1e4).max((0, 1)) * float(cnt[0] > 0)
        z = np.concatenate([mean, mx, m.mean(0)]) @ W['out_1_weight'].T + W['out_1_bias']; z[self.none] = -1e4
        z = z - z.max(); p = np.exp(z); return p / p.sum()

    def probs(self, results):
        """results: list (time order) of process() outputs -> (126,) probabilities or None when no hand was seen."""
        if not results: return None
        x, m = self.features(results)
        if m.sum() == 0: return None
        xm = x[:, [1, 0]].copy(); mm = m[:, [1, 0]].copy()
        XI = list(range(0, 63, 3)) + list(range(63, 105, 2)) + [105]
        xm[..., XI] = -xm[..., XI]; xm[..., 107] = -xm[..., 107]
        return (self._logits(x, m) + self._logits(xm, mm)) / 2


def redistribute(prob, q, b, classes=GESTURES):
    """keep the fused probability mass of the gesture group, re-split it with weight b according to the hand model."""
    if q is None or b <= 0: return prob
    c = np.asarray(classes, int); tot = float(prob[c].sum()); qs = float(q[c].sum())
    if tot <= 0 or qs <= 0: return prob
    out = prob.copy(); out[c] = tot * ((1 - b) * prob[c] / tot + b * q[c] / qs); return out
