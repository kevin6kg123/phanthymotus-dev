"""Whole-body finger branch for action314 (v43): RTMW-l (MMPose, Apache-2.0; 133 COCO-WholeBody keypoints, 384x288 SimCC) on
person crops taken at pose time, followed by a temporal finger model whose gesture-class probabilities re-split the fused
gesture mass. The crop rule (person box from the yolo pose, 1.25 expansion, aspect 288:384), the features and the model are
identical to the offline pipeline (wb_extract.py / wb_seq.py FEAT=hand, 4 frames). The network is served by the shared GPU
server as one TensorRT engine (fixed batch 4) whose ONNX graph already ends in argmax/max over the SimCC bins (output
(4,133,3): x_bin, y_bin, score). Features are invariant to translation and uniform scale, so keypoints stay in crop coordinates."""
import math
import numpy as np, cv2

GESTURES = [59, 60, 84, 85, 86, 87, 88]
SZ = (288, 384)
MEAN = np.array([123.675, 116.28, 103.53], np.float32); STD = np.array([58.395, 57.12, 57.375], np.float32)


def person_crop(frame, pose):
    """uint8 RGB crop (384,288,3) of the person given the BGR frame and the yolo pose (17,3) in [-1,1], or None."""
    p = np.asarray(pose, np.float32); v = p[:, 2] > 0.2
    if v.sum() < 3: return None
    H, W = frame.shape[:2]; xs = (p[v, 0] + 1) * W / 2; ys = (p[v, 1] + 1) * H / 2
    x1, y1, x2, y2 = xs.min(), ys.min() - 0.1 * (ys.max() - ys.min()), xs.max(), ys.max()
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2; w, h = (x2 - x1) * 1.25, (y2 - y1) * 1.25; ar = SZ[0] / SZ[1]
    if w > h * ar: h = w / ar
    else: w = h * ar
    if w < 4: return None
    s = SZ[0] / w; M = np.array([[s, 0, SZ[0] / 2 - s * cx], [0, s, SZ[1] / 2 - s * cy]], np.float32)
    return cv2.warpAffine(frame, M, SZ, flags=cv2.INTER_LINEAR)[:, :, ::-1].copy()


def frame_features(f):
    """(133,3) -> (118,) per-frame hand features (elbows/wrists + both hands, wrist-relative, shoulder-normalised) or None."""
    if f is None or f[:, 2].max() == 0: return None
    sh = (f[5, :2] + f[6, :2]) / 2; sw = np.linalg.norm(f[5, :2] - f[6, :2]) + 1e-6
    def hand(h):
        xy = h[:, :2] - h[0, :2]; s = np.linalg.norm(xy, axis=1).max() + 1e-6; xy = xy / s; tips = [4, 8, 12, 16, 20]
        d = [np.linalg.norm(xy[a] - xy[b]) for i, a in enumerate(tips) for b in tips[i + 1:]]
        return np.concatenate([xy.ravel(), d, (h[0, :2] - sh) / sw, [h[:, 2].mean()]])
    return np.concatenate([((f[7:11, :2] - sh) / sw).ravel(), hand(f[91:112]), hand(f[112:133])]).astype(np.float32)


def _gelu(x):
    return 0.5 * x * (1.0 + np.vectorize(math.erf, otypes=[np.float32])(x / math.sqrt(2.0)))


def _conv(x, w, b, dil):
    """x (T,Cin), w (Cout,Cin,3), same padding -> (T,Cout)"""
    T = x.shape[0]; xp = np.pad(x, ((dil, dil), (0, 0))); return sum(xp[k * dil:k * dil + T] @ w[:, :, k].T for k in range(3)) + b


class RTMWBranch:
    """temporal finger model (3 conv layers, masked mean+max pooling) trained on all classes; only its gesture-class
    probabilities are used (redistribute)."""
    def __init__(self, npz, net_fn):
        z = np.load(npz); self.W = {k: z[k].astype(np.float32) for k in z.files}; self.fn = net_fn; self.nc = self.W['fcb_0'].shape[0]
        self.seeds = int(z['n_seeds'][0])

    def process(self, crops):
        """crops: list of 1..4 (384,288,3) uint8 RGB -> list of (133,3) keypoints (crop pixels, score). Engine batch is fixed at 4."""
        k = len(crops); c = list(crops) + [crops[-1]] * (4 - k)
        x = ((np.stack(c).astype(np.float32) - MEAN) / STD).transpose(0, 3, 1, 2)
        y = np.asarray(self.fn(np.ascontiguousarray(x)), np.float32).reshape(4, 133, 3)[:k].copy()
        y[..., 0] /= 2.0; y[..., 1] /= 2.0; return list(y)

    def probs(self, kps):
        """kps: time-ordered list of (133,3) -> (n_classes,) probabilities or None."""
        fs = [frame_features(f) for f in kps]; m = np.array([f is not None for f in fs], np.float32)
        if m.sum() == 0: return None
        D = self.W['mu'].shape[0]; x = np.stack([f if f is not None else np.zeros(D, np.float32) for f in fs])
        x = ((x - self.W['mu']) / self.W['sd'] * m[:, None]).astype(np.float32); out = np.zeros(self.nc, np.float32)
        for s in range(self.seeds):
            W = lambda n: self.W['%s_%d' % (n, s)]
            h = _gelu(_conv(x, W('c1w'), W('c1b'), 1)); h = h + _gelu(_conv(h, W('c2w'), W('c2b'), 1)); h = h + _gelu(_conv(h, W('c3w'), W('c3b'), 2))
            mean = (h * m[:, None]).sum(0) / max(float(m.sum()), 1.0); mx = (h + (m[:, None] - 1) * 1e4).max(0)
            z = np.concatenate([mean, mx]) @ W('fcw').T + W('fcb'); z = z - z.max(); p = np.exp(z); out += p / p.sum()
        return out / self.seeds


def redistribute(prob, q, b, classes=GESTURES):
    """keep the fused probability mass of the gesture group, re-split it with weight b according to the finger model."""
    if q is None or b <= 0: return prob
    c = np.asarray(classes, int); tot = float(prob[c].sum()); qs = float(q[c].sum())
    if tot <= 0 or qs <= 0: return prob
    out = prob.copy(); out[c] = tot * ((1 - b) * prob[c] / tot + b * q[c] / qs); return out
