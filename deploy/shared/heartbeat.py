"""Background heartbeat sender.

Each node (edge or cloud) starts a HeartbeatThread that POSTs a Heartbeat
to the dashboard every `interval_s` seconds. The thread is fire-and-forget:
network errors are logged at WARNING and never propagate.

Disabled cleanly by passing dashboard_url=None — used when DASHBOARD_URL is
unset (deployment without a dashboard, e.g. running just the two devices).
"""

from __future__ import annotations

import logging
import socket
import threading
import time
from typing import Callable, Optional

import httpx

from shared.protocol import Heartbeat
from shared.stats import StatTracker


log = logging.getLogger("heartbeat")


def default_node_id() -> str:
    """Use hostname; the user picked 'Hostname' as identity scheme."""
    try:
        return socket.gethostname()
    except Exception:
        return "unknown-node"


class HeartbeatThread(threading.Thread):
    """Daemon thread that periodically POSTs heartbeats to the dashboard."""

    def __init__(
        self,
        *,
        dashboard_url: Optional[str],
        node_id: str,
        role: str,
        port: int,
        stats: StatTracker,
        config: dict,
        interval_s: float = 15.0,
        timeout_s: float = 5.0,
    ):
        super().__init__(name=f"heartbeat-{role}", daemon=True)
        self.dashboard_url = dashboard_url
        self.node_id = node_id
        self.role = role
        self.port = port
        self.stats = stats
        self.config = {k: str(v) for k, v in config.items()}
        self.interval_s = interval_s
        self.timeout_s = timeout_s
        self._stop = threading.Event()
        self._first = True

    def stop(self) -> None:
        self._stop.set()

    def _payload(self, is_register: bool) -> dict:
        return Heartbeat(
            node_id=self.node_id,
            role=self.role,
            hostname=socket.gethostname(),
            port=self.port,
            started_at=self.stats.started_at,
            stats=self.stats.snapshot(),
            config=self.config,
            is_register=is_register,
        ).dict()

    def _send_once(self, client: httpx.Client, is_register: bool) -> None:
        try:
            path = "/register" if is_register else "/heartbeat"
            r = client.post(path, json=self._payload(is_register))
            r.raise_for_status()
        except Exception as e:
            log.warning("dashboard %s failed (%s); will retry", path, e)

    def run(self) -> None:
        if not self.dashboard_url:
            log.info("DASHBOARD_URL unset; heartbeat disabled")
            return
        log.info("heartbeat -> %s every %.1fs (node_id=%s role=%s)",
                 self.dashboard_url, self.interval_s, self.node_id, self.role)
        with httpx.Client(base_url=self.dashboard_url, timeout=self.timeout_s) as c:
            while not self._stop.is_set():
                self._send_once(c, is_register=self._first)
                self._first = False
                # Sleep in 1s slices so .stop() is responsive
                end = time.monotonic() + self.interval_s
                while not self._stop.is_set() and time.monotonic() < end:
                    time.sleep(0.5)


def maybe_start(
    dashboard_url: Optional[str],
    *,
    node_id: str,
    role: str,
    port: int,
    stats: StatTracker,
    config: dict,
    interval_s: float = 15.0,
) -> Optional[HeartbeatThread]:
    """Start a heartbeat if dashboard_url is set; otherwise return None."""
    if not dashboard_url:
        return None
    t = HeartbeatThread(
        dashboard_url=dashboard_url, node_id=node_id, role=role, port=port,
        stats=stats, config=config, interval_s=interval_s,
    )
    t.start()
    return t
