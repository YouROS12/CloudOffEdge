"""Cloud server for the Jetson Nano (JetPack 4.6, Python 3.6): TensorRT
fp16 backbones + numpy QualityGate head. Same wire protocol as
cloud/server.py for synchronous requests.

Two modes (CLOUD_MODE):

  batch (default on the 4 GB Nano)
    The three ViT-L backbones do not fit in memory together (each is four
    chunk engines; at most 9 of the 12 can be resident). Escalations are
    queued, and the queue is processed backbone-major: load backbone 1, run
    every queued image, unload, backbone 2, ... then the head. The edge
    answers the farmer immediately; the cloud's diagnosis follows at the
    next flush.
      POST /enqueue   ClassifyRequest -> {job_id, queued}
      POST /flush     -> {n, load_ms, results: [ClassifyResponse + job_id]}
      POST /classify  enqueue + flush of a single image (slow; for tests)

  resident
    All engines loaded at start-up, synchronous POST /classify. For cloud
    nodes with enough memory (e.g. 8 GB Jetson modules).

Env: TRT_DIR, SERVER_HOST/PORT (0.0.0.0:8001), NODE_ID, CLOUD_MODE.
"""

import base64
import gc
import io
import itertools
import logging
import os
import socket
import sys
import threading
import time
from typing import Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cloud import trt_common as tc  # noqa: E402
from cloud.trt_runtime import load_backbone, run_backbone_batched  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [cloud-trt] %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("cloud-trt")

TRT_DIR = os.environ.get("TRT_DIR", "/opt/plant-deploy/trt")
MODE = os.environ.get("CLOUD_MODE", "batch").lower()
NODE_ID = os.environ.get("NODE_ID") or socket.gethostname()
# Largest number of queued images passed through a chunk in one pass. Each image holds
# about 4.5 MB in host memory during a flush (decoded payload ~0.8 MB, preprocessed input
# ~1.4 MB, hidden state up to 2.4 MB for DINOv2's 577 tokens); 128 keeps that near 0.6 GB,
# within the headroom left beside one loaded fp32 chunk. Larger queues are processed in sub-batches, each
# reloading the chunk engines. (Added after the measured runs, whose batches were <= 61.)
MAX_BATCH = int(os.environ.get("CLOUD_MAX_BATCH", "128"))


# Mirrors shared/protocol.py (which needs Python >= 3.7 to import).
class ClassifyRequest(BaseModel):
    image_b64: str
    image_kb: float = 0.0
    request_id: Optional[str] = None


class ClassifyResponse(BaseModel):
    label: str
    confidence: float
    top5: List[str] = Field(default_factory=list)
    top5_probs: List[float] = Field(default_factory=list)
    latency_ms: float
    backbones_used: List[str] = Field(default_factory=list)
    timings_ms: Dict[str, float] = Field(default_factory=dict)
    job_id: Optional[str] = None
    request_id: Optional[str] = None


class State(object):
    def __init__(self):
        self.classes = tc.load_class_names(os.path.join(TRT_DIR, "class_names.txt"))
        self.head = tc.QualityGateNP(os.path.join(TRT_DIR, "qualitygate_head.npz"))
        self.models = {}
        self.queue = []
        self.qlock = threading.Lock()
        self.flock = threading.Lock()
        self.ids = itertools.count(1)
        self.n_done = 0
        self.n_flushes = 0
        if MODE == "resident":
            for spec in tc.BACKBONES:
                t0 = time.time()
                self.models[spec["name"]] = load_backbone(TRT_DIR, spec["name"])
                log.info("loaded %s in %.1fs", spec["name"], time.time() - t0)
        log.info("ready: node=%s mode=%s classes=%d", NODE_ID, MODE, len(self.classes))

    # -- shared --------------------------------------------------------------
    def _result(self, feats, timings, job):
        t0 = time.perf_counter()
        probs = tc.softmax(self.head(tc.fuse(feats)))[0]
        timings["head"] = (time.perf_counter() - t0) * 1000
        order = np.argsort(-probs)[:5]
        return ClassifyResponse(
            label=self.classes[int(order[0])], confidence=float(probs[order[0]]),
            top5=[self.classes[int(i)] for i in order], top5_probs=[float(probs[i]) for i in order],
            latency_ms=sum(timings.values()), backbones_used=[s["name"] for s in tc.BACKBONES],
            timings_ms=timings, job_id=job.get("job_id"), request_id=job.get("request_id"))

    # -- resident mode -------------------------------------------------------
    def classify_resident(self, img, request_id=None):
        feats, timings = [], {}
        for spec in tc.BACKBONES:
            t0 = time.perf_counter()
            feats.append(self.models[spec["name"]](tc.preprocess(img, spec["size"], spec["mean"], spec["std"])))
            timings[spec["name"]] = (time.perf_counter() - t0) * 1000
        return self._result(feats, timings, {"request_id": request_id})

    # -- batch mode ----------------------------------------------------------
    def enqueue(self, img, req):
        job = {"job_id": "j%06d" % next(self.ids), "img": img, "request_id": req.request_id,
               "image_kb": req.image_kb, "t": time.time()}
        with self.qlock:
            self.queue.append(job)
            return job["job_id"], len(self.queue)

    def flush(self):
        with self.flock:  # one flush at a time; enqueues keep flowing meanwhile
            with self.qlock:
                jobs, self.queue = self.queue, []
            if not jobs:
                return {"n": 0, "load_ms": {}, "results": []}
            results, load_ms = [], {}
            for s in range(0, len(jobs), MAX_BATCH):
                part = jobs[s:s + MAX_BATCH]
                feats = [[] for _ in part]
                timings = [{} for _ in part]
                for spec in tc.BACKBONES:
                    xs, pre_ms = [], []
                    for job in part:
                        t1 = time.perf_counter()
                        xs.append(tc.preprocess(job["img"], spec["size"], spec["mean"], spec["std"]))
                        pre_ms.append((time.perf_counter() - t1) * 1000)
                    # chunk-major: one engine resident at a time
                    outs, ms, comp = run_backbone_batched(TRT_DIR, spec["name"], xs)
                    load_ms[spec["name"]] = load_ms.get(spec["name"], 0.0) + ms
                    for i, o in enumerate(outs):
                        feats[i].append(o)
                        timings[i][spec["name"]] = pre_ms[i] + comp[i]
                results += [self._result(f, tm, job) for f, tm, job in zip(feats, timings, part)]
            self.n_done += len(jobs)
            self.n_flushes += 1
            log.info("flush: %d images | load %s | compute median %.0f ms/img", len(jobs),
                     " ".join("%s=%.1fs" % (k.split("_")[0], v / 1000) for k, v in load_ms.items()),
                     float(np.median([r.latency_ms for r in results])))
            return {"n": len(jobs), "load_ms": load_ms, "results": [r.dict() for r in results]}


state = None
app = FastAPI(title="plant-cloud-trt", version="2.0")


def _decode(req):
    try:
        return Image.open(io.BytesIO(base64.b64decode(req.image_b64))).convert("RGB")
    except Exception as e:
        raise HTTPException(400, "decode failed: %s" % e)


@app.on_event("startup")
def _startup():
    global state
    state = State()


@app.get("/health")
def health():
    return {
        "status": "ok", "node_id": NODE_ID, "role": "cloud", "mode": MODE,
        "device": "jetson-nano-tensorrt-fp16", "n_classes": len(state.classes),
        "feat_dim": sum(s["dim"] for s in tc.BACKBONES),
        "backbones": [s["name"] for s in tc.BACKBONES],
        "queued": len(state.queue), "n_requests": state.n_done, "n_flushes": state.n_flushes,
    }


@app.post("/enqueue")
def enqueue(req: ClassifyRequest):
    job_id, n = state.enqueue(_decode(req), req)
    log.info("enqueue rid=%s job=%s queued=%d in=%.1fKB", req.request_id or "-", job_id, n, req.image_kb)
    return {"job_id": job_id, "queued": n}


@app.post("/flush")
def flush():
    return state.flush()


@app.post("/classify", response_model=ClassifyResponse)
def classify(req: ClassifyRequest):
    img = _decode(req)
    if MODE == "resident":
        return state.classify_resident(img, req.request_id)
    job_id, _ = state.enqueue(img, req)
    out = state.flush()
    for r in out["results"]:
        if r["job_id"] == job_id:
            return ClassifyResponse(**r)
    raise HTTPException(500, "job lost in flush")


def main():
    import uvicorn
    uvicorn.run("cloud.server_trt:app", host=os.environ.get("SERVER_HOST", "0.0.0.0"),
                port=int(os.environ.get("SERVER_PORT", "8001")), log_level="warning")


if __name__ == "__main__":
    main()
