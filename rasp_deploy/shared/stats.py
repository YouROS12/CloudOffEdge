"""Lightweight in-memory stat tracking for edge and cloud servers.

Thread-safe, no dependencies. Each server creates one StatTracker on
startup and updates it after every prediction.
"""

from __future__ import annotations

import collections
import threading
import time
from typing import Optional

from shared.protocol import NodeStats


class StatTracker:
    """Rolling-window stats. Window size determines the avg_* horizon."""

    def __init__(self, role: str, window: int = 100):
        self.role = role
        self.window = window
        self.started_at = time.time()
        self._lock = threading.Lock()
        # Edge counters
        self.requests_total = 0
        self.requests_routed = 0
        self.requests_edge_only = 0
        self.bandwidth_total_kb = 0.0
        self._margins = collections.deque(maxlen=window)
        self._bw = collections.deque(maxlen=window)
        # Both
        self._latencies = collections.deque(maxlen=window)
        self.last_label: Optional[str] = None
        self.last_at: Optional[float] = None
        # Cloud-only
        self._classify_latencies = collections.deque(maxlen=window)

    def record_edge(self, *, margin: float, latency_ms: float, label: str,
                    routed: bool, bandwidth_kb: float) -> None:
        with self._lock:
            self.requests_total += 1
            if routed:
                self.requests_routed += 1
            else:
                self.requests_edge_only += 1
            self._margins.append(margin)
            self._latencies.append(latency_ms)
            self._bw.append(bandwidth_kb)
            self.bandwidth_total_kb += bandwidth_kb
            self.last_label = label
            self.last_at = time.time()

    def record_cloud(self, *, latency_ms: float, label: str) -> None:
        with self._lock:
            self.requests_total += 1
            self._latencies.append(latency_ms)
            self._classify_latencies.append(latency_ms)
            self.last_label = label
            self.last_at = time.time()

    @staticmethod
    def _mean(d) -> Optional[float]:
        return float(sum(d) / len(d)) if d else None

    def snapshot(self) -> NodeStats:
        with self._lock:
            return NodeStats(
                uptime_s=time.time() - self.started_at,
                requests_total=self.requests_total,
                requests_routed=self.requests_routed,
                requests_edge_only=self.requests_edge_only,
                avg_margin=self._mean(self._margins),
                avg_bandwidth_kb=self._mean(self._bw),
                bandwidth_total_kb=self.bandwidth_total_kb,
                avg_latency_ms=self._mean(self._latencies),
                last_label=self.last_label,
                last_at=self.last_at,
                avg_classify_ms=self._mean(self._classify_latencies),
            )
