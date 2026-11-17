"""Build FP16 TensorRT plans for the action314 ONNX models inside the deployment image.
usage: python3 build_engines.py MODELS_DIR  -> MODELS_DIR/trt/<family>/{pose,joint,bone}.plan"""
import sys, time
from pathlib import Path
sys.path.insert(0, '/work')
import tensorrt as trt
try:
    from utils.tensorrt_runtime import tensorrt_family
except ImportError:
    from tensorrt_runtime import tensorrt_family
root = Path(sys.argv[1]); fam = tensorrt_family(); out = root / 'trt' / fam; out.mkdir(parents=True, exist_ok=True)
logger = trt.Logger(trt.Logger.WARNING)
print('TensorRT', trt.__version__, 'family', fam, flush=True)
for name in ['pose', 'joint', 'bone']:
    t = time.time()
    builder = trt.Builder(logger); net = builder.create_network(0); parser = trt.OnnxParser(net, logger)
    if not parser.parse((root / (name + '.onnx')).read_bytes()):
        raise SystemExit('parse failed %s: %s' % (name, [str(parser.get_error(i)) for i in range(parser.num_errors)]))
    cfg = builder.create_builder_config(); cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 256 << 20)
    cfg.set_flag(trt.BuilderFlag.FP16)
    plan = builder.build_serialized_network(net, cfg)
    if plan is None: raise SystemExit('build failed ' + name)
    (out / (name + '.plan')).write_bytes(bytes(plan))
    print(name, 'plan bytes', (out / (name + '.plan')).stat().st_size, 'build %.1fs' % (time.time() - t), flush=True)
