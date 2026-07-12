"""Wire protocol for edge <-> cloud communication.

All messages are JSON. Images are sent base64-encoded JPEG inside the JSON
body (simple, debuggable with curl). Schema mirrors the predict / classify
calls used by both servers.

Also defines the dashboard heartbeat protocol used by every node to report
its identity + stats to a central dashboard service running on the user's
laptop.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class EdgePrediction(BaseModel):
    """Edge-only prediction returned by the Pi server."""
    label: str
    confidence: float          # max softmax probability
    margin: float              # top1 - top2 softmax probability
    entropy: float             # softmax entropy (nats)
    top5: List[str] = Field(default_factory=list)
    top5_probs: List[float] = Field(default_factory=list)


class ClassifyRequest(BaseModel):
    """Request to cloud server (POST /classify)."""
    image_b64: str             # base64 JPEG
    image_kb: float = 0.0      # observed encoded size, for accounting
    request_id: Optional[str] = None


class ClassifyResponse(BaseModel):
    """Response from cloud server."""
    label: str
    confidence: float
    top5: List[str] = Field(default_factory=list)
    top5_probs: List[float] = Field(default_factory=list)
    latency_ms: float
    backbones_used: List[str] = Field(default_factory=list)


class PredictRequest(BaseModel):
    """Request to edge server (POST /predict).

    The edge server itself decides whether to escalate to the cloud based
    on the routing config. Caller can override with `force_route`.
    """
    image_b64: str
    request_id: Optional[str] = None
    force_route: Optional[str] = None  # 'edge' | 'cloud' | None


class PredictResponse(BaseModel):
    """Response from edge server."""
    label: str
    confidence: float
    source: str                # 'edge' | 'cloud'
    routed: bool               # True iff source == 'cloud'
    edge_prediction: EdgePrediction
    cloud_prediction: Optional[ClassifyResponse] = None
    routing_signal: float      # the margin value (or whatever was used)
    threshold: float           # the threshold the signal was compared against
    latency_ms: float          # total wall time for this request
    bandwidth_kb: float        # bytes uploaded for this sample (0 if no escalation)


# ---------------------------------------------------------------------------
# Dashboard heartbeat protocol. Every node POSTs Heartbeat to the dashboard
# every HEARTBEAT_INTERVAL_S seconds (default 15). Dashboard derives "online"
# from `now - last_seen < HEARTBEAT_TIMEOUT_S`.
# ---------------------------------------------------------------------------

class NodeStats(BaseModel):
    """Role-agnostic counters that the dashboard renders. Both edge and
    cloud populate the subset that makes sense for them."""
    uptime_s: float = 0.0
    requests_total: int = 0
    # Edge-only:
    requests_routed: int = 0      # escalated to cloud
    requests_edge_only: int = 0
    avg_margin: Optional[float] = None
    avg_bandwidth_kb: Optional[float] = None
    bandwidth_total_kb: float = 0.0
    # Both:
    avg_latency_ms: Optional[float] = None
    last_label: Optional[str] = None
    last_at: Optional[float] = None  # epoch seconds
    # Cloud-only:
    avg_classify_ms: Optional[float] = None


class Heartbeat(BaseModel):
    node_id: str
    role: str                  # 'edge' | 'cloud'
    hostname: str
    port: int
    version: str = "1.0"
    started_at: float          # epoch seconds, set once on boot
    stats: NodeStats
    config: Dict[str, str] = Field(default_factory=dict)
    # Set true by /register so the dashboard can log a fresh boot vs ongoing heartbeat:
    is_register: bool = False
