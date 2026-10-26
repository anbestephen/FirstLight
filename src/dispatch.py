"""
Live commander-to-field dispatch hub (`src/dispatch.py`).

Holds the active field assignment (vehicle class, origin, destination) and pushes a fresh
route snapshot to every connected field device over Server-Sent Events whenever the
commander changes the assignment, simulates a debris clearance, or the internet uplink
drops. Everything runs over the local Wi-Fi LAN; the only outbound traffic is the optional
TCP reachability probe against the Meta Model API host used to detect uplink loss.
"""

from __future__ import annotations

import asyncio
import json
import socket
import threading
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Awaitable, Callable

DEFAULT_UPLINK_PROBE_HOST = "api.meta.ai"
DEFAULT_UPLINK_PROBE_PORT = 443
UPLINK_PROBE_INTERVAL_S = 5.0
UPLINK_PROBE_TIMEOUT_S = 2.0
STREAM_POLL_INTERVAL_S = 0.25
STREAM_HEARTBEAT_S = 15.0


def probe_tcp_uplink(
    host: str = DEFAULT_UPLINK_PROBE_HOST,
    port: int = DEFAULT_UPLINK_PROBE_PORT,
    timeout_s: float = UPLINK_PROBE_TIMEOUT_S,
) -> bool:
    """Return True when a TCP connection to `host:port` succeeds within `timeout_s`."""
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return True
    except OSError:
        return False


def detect_lan_ip() -> str | None:
    """
    Return this laptop's LAN IPv4 address without sending any packets (a UDP `connect`
    only selects the outbound interface), so the field URL can be shown offline.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("10.255.255.255", 1))
        ip = sock.getsockname()[0]
        return None if ip.startswith("127.") else ip
    except OSError:
        return None
    finally:
        sock.close()


@dataclass
class DispatchAssignment:
    """Active field assignment pushed from the commander dashboard."""

    vehicle_class: str = "fire_engine"
    origin_lonlat: tuple[float, float] | None = None
    destination_lonlat: tuple[float, float] | None = None
    force_offline: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "vehicle_class": self.vehicle_class,
            "origin": list(self.origin_lonlat) if self.origin_lonlat else None,
            "destination": list(self.destination_lonlat) if self.destination_lonlat else None,
            "force_offline": self.force_offline,
        }


@dataclass
class DispatchHub:
    """
    Versioned dispatch state shared by the commander dashboard and field devices.

    Every mutation bumps `version`; each SSE stream re-sends the snapshot when the version
    (or the connected-device count) changes. Snapshots are built by `snapshot_builder` and
    cached per version so N phones never trigger N route computations.
    """

    uplink_probe: Callable[[], bool] | None = probe_tcp_uplink
    assignment: DispatchAssignment = field(default_factory=DispatchAssignment)
    version: int = 0
    last_change_reason: str = "initial"
    last_change_detail: str = ""
    last_change_at: float = field(default_factory=time.time)
    uplink_online: bool | None = None
    field_clients: int = 0
    command_clients: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _cached_version: int = -1
    _cached_snapshot: dict[str, Any] | None = None
    _last_probe_at: float = 0.0
    _probe_task: asyncio.Task[None] | None = None

    def bump(self, reason: str, detail: str = "") -> int:
        """Record a state change so every connected stream pushes a fresh snapshot."""
        with self._lock:
            self.version += 1
            self.last_change_reason = reason
            self.last_change_detail = detail
            self.last_change_at = time.time()
            return self.version

    def update_assignment(
        self,
        *,
        vehicle_class: str | None = None,
        origin_lonlat: tuple[float, float] | None = None,
        destination_lonlat: tuple[float, float] | None = None,
        force_offline: bool | None = None,
    ) -> int:
        """Apply commander changes; returns the new version (unchanged if nothing changed)."""
        with self._lock:
            before = self.assignment.to_dict()
            if vehicle_class is not None:
                self.assignment.vehicle_class = vehicle_class
            if origin_lonlat is not None:
                self.assignment.origin_lonlat = origin_lonlat
            if destination_lonlat is not None:
                self.assignment.destination_lonlat = destination_lonlat
            if force_offline is not None:
                self.assignment.force_offline = force_offline
            after = self.assignment.to_dict()
        if before == after:
            return self.version
        if before["force_offline"] != after["force_offline"]:
            reason = "comms_blackout_on" if after["force_offline"] else "comms_blackout_off"
        elif before["vehicle_class"] != after["vehicle_class"]:
            reason = "vehicle_changed"
        else:
            reason = "route_reassigned"
        return self.bump(reason)

    def ai_tier(self) -> str:
        """Effective reasoning tier given manual blackout and the measured uplink state."""
        if self.assignment.force_offline or self.uplink_online is False:
            return "tier_2_edge"
        return "tier_1_cloud"

    def connectivity(self) -> dict[str, Any]:
        return {
            "uplink_online": self.uplink_online,
            "force_offline": self.assignment.force_offline,
            "ai_tier": self.ai_tier(),
            "field_clients": self.field_clients,
            "command_clients": self.command_clients,
        }

    def get_snapshot(self, builder: Callable[[DispatchAssignment], dict[str, Any]]) -> dict[str, Any]:
        """Return the cached snapshot for the current version, rebuilding it when stale."""
        with self._lock:
            version = self.version
            if self._cached_snapshot is not None and self._cached_version == version:
                body = self._cached_snapshot
            else:
                body = None
            assignment = DispatchAssignment(**vars(self.assignment))
        if body is None:
            body = builder(assignment)
            with self._lock:
                if self.version == version:
                    self._cached_snapshot = body
                    self._cached_version = version
        return {
            "version": version,
            "reason": self.last_change_reason,
            "reason_detail": self.last_change_detail,
            "changed_at": self.last_change_at,
            "assignment": assignment.to_dict(),
            "connectivity": self.connectivity(),
            **body,
        }

    async def _run_uplink_probe(self) -> None:
        if self.uplink_probe is None:
            return
        online = await asyncio.to_thread(self.uplink_probe)
        if online != self.uplink_online:
            was_known = self.uplink_online is not None
            self.uplink_online = online
            if was_known:
                self.bump("uplink_restored" if online else "uplink_lost")

    def _maybe_schedule_probe(self) -> None:
        if self.uplink_probe is None:
            return
        now = time.monotonic()
        if self._probe_task is not None and not self._probe_task.done():
            return
        if now - self._last_probe_at < UPLINK_PROBE_INTERVAL_S:
            return
        self._last_probe_at = now
        self._probe_task = asyncio.create_task(self._run_uplink_probe())

    async def stream(
        self,
        builder: Callable[[DispatchAssignment], dict[str, Any]],
        *,
        role: str = "field",
        is_disconnected: Callable[[], Awaitable[bool]] | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[str]:
        """
        Yield SSE frames (`event: dispatch`) whenever the dispatch version, uplink state, or
        connected-device count changes, plus a comment heartbeat every 15 s so proxies and
        phones keep the connection open. `limit` stops after N dispatch events (tests).
        """
        counter_attr = "field_clients" if role == "field" else "command_clients"
        # Presence changes are part of the change key below, not a version bump, so a phone
        # joining never invalidates the cached route or shows up as a "route updated" toast.
        with self._lock:
            setattr(self, counter_attr, getattr(self, counter_attr) + 1)

        sent = 0
        last_key: tuple[Any, ...] | None = None
        last_beat = time.monotonic()
        try:
            yield "retry: 2000\n\n"
            while True:
                if is_disconnected is not None and await is_disconnected():
                    break
                self._maybe_schedule_probe()
                key = (self.version, self.uplink_online, self.field_clients, self.command_clients)
                if key != last_key:
                    snapshot = await asyncio.to_thread(self.get_snapshot, builder)
                    last_key = key
                    yield f"event: dispatch\nid: {snapshot['version']}\ndata: {json.dumps(snapshot)}\n\n"
                    sent += 1
                    last_beat = time.monotonic()
                    if limit is not None and sent >= limit:
                        break
                elif time.monotonic() - last_beat >= STREAM_HEARTBEAT_S:
                    yield ": heartbeat\n\n"
                    last_beat = time.monotonic()
                await asyncio.sleep(STREAM_POLL_INTERVAL_S)
        finally:
            with self._lock:
                setattr(self, counter_attr, max(0, getattr(self, counter_attr) - 1))
