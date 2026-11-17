"""Build FP16 TensorRT plans for the action314 ONNX models with the TensorRT of the current
environment (run at image build time so plan and runtime versions always match).
usage: python3 action314_build_engines.py MODELS_DIR [--check]
Never fails the caller: any problem is printed and the runtime falls back to CPU."""
import os, sys, time, traceback
from pathlib import Path
sys.path.insert(0, '/work')
root = Path(sys.argv[1])
try:
    import numpy as np
    import tensorrt as trt
    try:
        from utils.tensorrt_runtime import tensorrt_family, TensorRTEngine
    except ImportError:
        from tensorrt_runtime import tensorrt_family, TensorRTEngine
    fam = tensorrt_family(); out = root / 'trt' / fam; out.mkdir(parents=True, exist_ok=True)
    logger = trt.Logger(trt.Logger.WARNING)
    print('[action314] TensorRT', trt.__version__, 'family', fam, flush=True)
    for name in ['pose', 'joint', 'bone']:
        t = time.time()
        builder = trt.Builder(logger); net = builder.create_network(0); parser = trt.OnnxParser(net, logger)
        if not parser.parse((root / (name + '.onnx')).read_bytes()):
            raise RuntimeError('parse failed %s: %s' % (name, [str(parser.get_error(i)) for i in range(parser.num_errors)]))
        cfg = builder.create_builder_config(); cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 256 << 20)
        cfg.set_flag(trt.BuilderFlag.FP16)
        cfg.builder_optimization_level = int(os.environ.get('ACTION314_TRT_OPT_LEVEL', '2'))
        plan = builder.build_serialized_network(net, cfg)
        if plan is None: raise RuntimeError('build failed ' + name)
        (out / (name + '.plan')).write_bytes(bytes(plan))
        print('[action314]', name, 'plan bytes', (out / (name + '.plan')).stat().st_size, 'build %.1fs' % (time.time() - t), flush=True)
    if '--check' in sys.argv:
        x = np.zeros((1, 3, 416, 416), np.float32); e = TensorRTEngine(out / 'pose.plan'); y = e.infer(x)[0]
        c = TensorRTEngine(out / 'joint.plan').infer(np.zeros((1, 100, 17, 3), np.float32))[0]
        print('[action314] check pose', getattr(y, 'shape', None), 'joint', getattr(c, 'shape', None), flush=True)
except Exception:
    print('[action314] engine build skipped:', traceback.format_exc().splitlines()[-1], flush=True)
    for p in (root / 'trt').glob('*/*.plan') if (root / 'trt').exists() else []:
        p.unlink()
