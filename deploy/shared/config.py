"""Config loader. Reads a YAML file + environment overrides.

Single source of truth for both edge and cloud server config.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional  # noqa: F401  (Optional used in DashboardConfig)

import yaml


@dataclass
class EdgeConfig:
    arch: str = "mobilenet_v3_large"
    ckpt_path: str = "checkpoints/edge_v3l.pt"
    n_classes: int = 89
    img_size: int = 224
    device: str = "cpu"        # Pi has no CUDA; use 'cuda' on a desktop test
    margin_threshold: float = 0.20  # route if margin < threshold
    cloud_url: str = "http://localhost:8001"
    cloud_timeout_s: float = 60.0
    jpeg_quality: int = 90
    class_names_path: str = "checkpoints/class_names.txt"


@dataclass
class CloudConfig:
    head_ckpt_path: str = "checkpoints/qualitygate_fusion_3L.pt"
    class_names_path: str = "checkpoints/class_names.txt"
    backbones: List[str] = field(default_factory=lambda: [
        "dinov2_vitl14", "dinov3_vitl16", "clip_vitl14"
    ])
    device: str = "cuda"
    sequential: bool = True    # load one backbone at a time (Nano-friendly)
    n_classes: int = 89


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    log_dir: str = "logs"


@dataclass
class DashboardConfig:
    """Dashboard reporting (optional; unset = disabled)."""
    url: Optional[str] = None       # e.g. http://my-laptop:9000
    node_id: Optional[str] = None   # default = socket.gethostname()
    interval_s: float = 15.0


@dataclass
class DeployConfig:
    role: str = "edge"         # 'edge' | 'cloud'
    edge: EdgeConfig = field(default_factory=EdgeConfig)
    cloud: CloudConfig = field(default_factory=CloudConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)


def _override_from_env(cfg: DeployConfig) -> DeployConfig:
    """Allow a few key knobs to be overridden by env (for systemd / docker)."""
    if v := os.environ.get("DEPLOY_ROLE"):
        cfg.role = v
    if v := os.environ.get("CLOUD_URL"):
        cfg.edge.cloud_url = v
    if v := os.environ.get("MARGIN_THRESHOLD"):
        cfg.edge.margin_threshold = float(v)
    if v := os.environ.get("EDGE_DEVICE"):
        cfg.edge.device = v
    if v := os.environ.get("CLOUD_DEVICE"):
        cfg.cloud.device = v
    if v := os.environ.get("CLOUD_SEQUENTIAL"):
        cfg.cloud.sequential = v.lower() in ("1", "true", "yes")
    if v := os.environ.get("SERVER_HOST"):
        cfg.server.host = v
    if v := os.environ.get("SERVER_PORT"):
        cfg.server.port = int(v)
    if v := os.environ.get("DASHBOARD_URL"):
        cfg.dashboard.url = v
    if v := os.environ.get("NODE_ID"):
        cfg.dashboard.node_id = v
    if v := os.environ.get("HEARTBEAT_INTERVAL_S"):
        cfg.dashboard.interval_s = float(v)
    return cfg


def load(config_path: Optional[str] = None) -> DeployConfig:
    """Load config from YAML + env overrides.

    If config_path is None, looks at DEPLOY_CONFIG env var, falling back to
    deploy/config/config.yaml in the repo root.
    """
    if config_path is None:
        config_path = os.environ.get(
            "DEPLOY_CONFIG",
            str(Path(__file__).resolve().parents[1] / "config" / "config.yaml"),
        )
    cfg = DeployConfig()
    p = Path(config_path)
    if p.exists():
        with p.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        if "role" in raw:
            cfg.role = raw["role"]
        if "edge" in raw:
            for k, v in (raw["edge"] or {}).items():
                if hasattr(cfg.edge, k):
                    setattr(cfg.edge, k, v)
        if "cloud" in raw:
            for k, v in (raw["cloud"] or {}).items():
                if hasattr(cfg.cloud, k):
                    setattr(cfg.cloud, k, v)
        if "server" in raw:
            for k, v in (raw["server"] or {}).items():
                if hasattr(cfg.server, k):
                    setattr(cfg.server, k, v)
        if "dashboard" in raw:
            for k, v in (raw["dashboard"] or {}).items():
                if hasattr(cfg.dashboard, k):
                    setattr(cfg.dashboard, k, v)
    return _override_from_env(cfg)
