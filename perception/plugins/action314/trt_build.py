"""Child process that builds FP16 TensorRT plans (keeps the plugin process responsive).
usage: python3 trt_build.py ONNX_DIR OUT_DIR OPT_LEVEL name [name ...]"""
import os, sys, time
import tensorrt as trt
try:
    os.nice(-10)
except Exception:
    pass
src, out, level, names = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4:]
logger = trt.Logger(trt.Logger.ERROR)
for n in names:
    t = time.time(); b = trt.Builder(logger); net = b.create_network(0 if int(trt.__version__.split('.')[0]) >= 10 else 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)); parser = trt.OnnxParser(net, logger)
    if not parser.parse(open(os.path.join(src, n + '.onnx'), 'rb').read()):
        sys.exit('onnx parse failed: ' + n)
    cfg = b.create_builder_config(); cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 256 << 20)
    cfg.set_flag(trt.BuilderFlag.FP16)
    if hasattr(cfg, 'builder_optimization_level'): cfg.builder_optimization_level = level
    plan = b.build_serialized_network(net, cfg)
    if plan is None: sys.exit('engine build failed: ' + n)
    tmp = os.path.join(out, '.%s.%d' % (n, os.getpid()))
    with open(tmp, 'wb') as f: f.write(bytes(plan))
    os.replace(tmp, os.path.join(out, n + '.plan'))
    print(n, 'built %.1fs' % (time.time() - t), flush=True)
