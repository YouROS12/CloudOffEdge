"""TensorRT 8.2 engine runner for the Jetson Nano (Python 3.6, pycuda).

Jetson-only: imports tensorrt + pycuda. One TrtModel per backbone engine;
all share one CUDA context, pushed/popped around each call so the models
can be used from any thread (FastAPI runs sync endpoints in a threadpool).
"""

import ctypes
import glob
import os
import threading

import numpy as np
import pycuda.driver as cuda
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
_ctx = None
_lock = threading.Lock()

# One activation scratch buffer shared by every execution context. Engines
# run strictly one at a time (under _lock), so they never need their scratch
# simultaneously; per-context buffers cost ~35 MB x 12 chunks on the Nano.
_scratch = {"buf": None, "size": 0, "contexts": []}

# TensorRT 8.2's Python binding takes the device_memory pointer as a PyCapsule.
ctypes.pythonapi.PyCapsule_New.restype = ctypes.py_object
ctypes.pythonapi.PyCapsule_New.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]


def _capsule(ptr):
    return ctypes.pythonapi.PyCapsule_New(int(ptr), None, None)


def _attach_scratch(engine, context):
    need = engine.device_memory_size
    if need > _scratch["size"]:
        _scratch["buf"] = cuda.mem_alloc(max(need, 1))
        _scratch["size"] = need
        for c in _scratch["contexts"]:
            c.device_memory = _capsule(_scratch["buf"])
    context.device_memory = _capsule(_scratch["buf"])
    _scratch["contexts"].append(context)


def cuda_context():
    global _ctx
    if _ctx is None:
        cuda.init()
        _ctx = cuda.Device(0).make_context()
        _ctx.pop()
    return _ctx


class TrtModel(object):
    """Deserialized engine with one input and one output, batch 1."""

    def __init__(self, engine_path):
        ctx = cuda_context()
        ctx.push()
        try:
            with open(engine_path, "rb") as fh, trt.Runtime(TRT_LOGGER) as rt:
                self.engine = rt.deserialize_cuda_engine(fh.read())
            if self.engine is None:
                raise RuntimeError("failed to deserialize " + str(engine_path))
            self.context = self.engine.create_execution_context_without_device_memory()
            _attach_scratch(self.engine, self.context)
            self.stream = cuda.Stream()
            self.host, self.dev, self.bindings = [], [], []
            for i in range(self.engine.num_bindings):
                shape = tuple(self.engine.get_binding_shape(i))
                dtype = trt.nptype(self.engine.get_binding_dtype(i))
                h = cuda.pagelocked_empty(shape, dtype)
                d = cuda.mem_alloc(h.nbytes)
                self.host.append(h)
                self.dev.append(d)
                self.bindings.append(int(d))
                if self.engine.binding_is_input(i):
                    self.in_idx, self.in_shape = i, shape
                else:
                    self.out_idx = i
        finally:
            ctx.pop()

    def __call__(self, x):
        x = np.ascontiguousarray(x, dtype=self.host[self.in_idx].dtype).reshape(self.in_shape)
        with _lock:
            ctx = cuda_context()
            ctx.push()
            try:
                np.copyto(self.host[self.in_idx], x)
                cuda.memcpy_htod_async(self.dev[self.in_idx], self.host[self.in_idx], self.stream)
                self.context.execute_async_v2(self.bindings, self.stream.handle)
                cuda.memcpy_dtoh_async(self.host[self.out_idx], self.dev[self.out_idx], self.stream)
                self.stream.synchronize()
                return np.array(self.host[self.out_idx], dtype=np.float32)
            finally:
                ctx.pop()

    def close(self):
        """Free the engine's device memory now (TensorRT objects must be
        destroyed while the CUDA context is current)."""
        if getattr(self, "engine", None) is None:
            return
        ctx = cuda_context()
        ctx.push()
        try:
            if self.context in _scratch["contexts"]:
                _scratch["contexts"].remove(self.context)
            for d in self.dev:
                d.free()
            self.context = self.engine = self.stream = None
            self.host, self.dev, self.bindings = [], [], []
        finally:
            ctx.pop()


class TrtChain(object):
    """A backbone built as consecutive chunk engines (<name>.c0.engine, ...);
    each chunk's output is the next chunk's input. Needed on the 4 GB Nano,
    where a whole ViT-L cannot be built as one engine."""

    def __init__(self, engine_paths):
        self.parts = [TrtModel(p) for p in engine_paths]

    def __call__(self, x):
        for part in self.parts:
            x = part(x)
        return x

    def close(self):
        for part in self.parts:
            part.close()
        self.parts = []


def chunk_engines(trt_dir, name):
    def idx(p):
        return int(p.rsplit(".c", 1)[1].split(".")[0])
    return sorted(glob.glob(os.path.join(trt_dir, name + ".c*.engine")), key=idx)


def run_backbone_batched(trt_dir, name, xs):
    """Run a batch of inputs through a backbone chunk-major: load chunk 1, run
    every input through it, free it, load chunk 2, ... Only ONE engine is
    resident at any time (fp32 DINOv3 chunks are ~0.65 GB each once loaded),
    and intermediate hidden states (~2 MB per image) are kept in host memory.

    Returns (outputs, load_ms_total, compute_ms_per_input)."""
    import gc
    import time
    parts = chunk_engines(trt_dir, name) or [os.path.join(trt_dir, name + ".engine")]
    hs = list(xs)
    load_ms = 0.0
    comp = [0.0] * len(hs)
    for path in parts:
        t0 = time.time()
        model = TrtModel(path)
        load_ms += (time.time() - t0) * 1000
        try:
            for i in range(len(hs)):
                t1 = time.time()
                hs[i] = model(hs[i])
                comp[i] += (time.time() - t1) * 1000
        finally:
            model.close()
            del model
            gc.collect()
    return hs, load_ms, comp


def load_backbone(trt_dir, name):
    """Single engine <name>.engine if present, otherwise the chunk chain."""
    single = os.path.join(trt_dir, name + ".engine")
    if os.path.exists(single):
        return TrtModel(single)
    parts = chunk_engines(trt_dir, name)
    if not parts:
        raise FileNotFoundError("no engine for %s in %s" % (name, trt_dir))
    return TrtChain(parts)
