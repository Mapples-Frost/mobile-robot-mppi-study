"""Framed TCP transport for split PC/Pi physical deployment.

The PC opens the connection, so Windows needs no inbound firewall rule.  The
Pi streams the existing versioned Livox datagrams and chassis status over the
same full-duplex socket that carries bounded control commands back to the Pi.
"""

from dataclasses import dataclass
import json
import queue
import socket
import struct
import threading
import time
from typing import Optional

import numpy as np

from .livox_scan_adapter import LivoxPointCloudFrame
from .livox_udp import decode_livox_datagram


FRAME_MAGIC = b"RMP1"
FRAME_SCAN = 1
FRAME_STATUS = 2
FRAME_COMMAND = 3
FRAME_HELLO = 4
FRAME_STOP = 5
FRAME_HEADER = struct.Struct("!4sBI")
MAX_FRAME_BYTES = 1_000_000


def encode_json(payload) -> bytes:
    return json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")


def decode_json(payload: bytes):
    value = json.loads(payload.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("remote JSON payload must be an object")
    return value


def _receive_exact(sock: socket.socket, size: int) -> bytes:
    chunks = []
    remaining = int(size)
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("remote deployment socket closed")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def send_frame(sock: socket.socket, frame_type: int, payload: bytes,
               lock: Optional[threading.Lock] = None) -> None:
    payload = bytes(payload)
    if not 0 <= int(frame_type) <= 255:
        raise ValueError("frame type is outside uint8 range")
    if len(payload) > MAX_FRAME_BYTES:
        raise ValueError("remote deployment frame is too large")
    packet = FRAME_HEADER.pack(FRAME_MAGIC, int(frame_type), len(payload)) + payload
    if lock is None:
        sock.sendall(packet)
    else:
        with lock:
            sock.sendall(packet)


def receive_frame(sock: socket.socket):
    header = _receive_exact(sock, FRAME_HEADER.size)
    magic, frame_type, payload_size = FRAME_HEADER.unpack(header)
    if magic != FRAME_MAGIC:
        raise ValueError("invalid remote deployment frame magic")
    if payload_size > MAX_FRAME_BYTES:
        raise ValueError("remote deployment frame exceeds size limit")
    return int(frame_type), _receive_exact(sock, int(payload_size))


@dataclass(frozen=True)
class RemoteStatus:
    received_monotonic: float
    v_mps: float
    omega_radps: float
    battery_v: Optional[float]
    control_mode: Optional[int]
    fault: Optional[int]
    armed: bool
    last_sequence: int
    target_v_mps: float = 0.0
    target_omega_radps: float = 0.0
    applied_v_mps: float = 0.0
    applied_omega_radps: float = 0.0


def _drain_latest(queue_value):
    """Return only the newest currently queued payload and drop backlog."""
    latest = None
    count = 0
    while True:
        try:
            latest = queue_value.get_nowait()
            count += 1
        except queue.Empty:
            break
    return latest, max(0, count - 1)


def _drain_all(queue_value):
    """Return every currently queued payload in arrival order.

    A Livox datagram is a different slice of the non-repetitive scan, not a
    replaceable state update.  Collapsing this queue to its newest item made a
    close obstacle alternate between present and absent in consecutive
    controller cycles.  Freshness is decided from the sensor timestamps after
    decoding, so retaining the queue here does not retain stale geometry.
    """
    values = []
    while True:
        try:
            values.append(queue_value.get_nowait())
        except queue.Empty:
            break
    return values


class RemoteDeploymentClient:
    """PC-side client with bounded scan buffering and latest-only status."""

    def __init__(self, host: str, port: int, token: str,
                 timeout_s: float = 2.0, maximum_scan_packets: int = 4096,
                 scan_udp_port: int = 57701):
        self._scan_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._scan_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4_000_000)
        self._scan_socket.bind(("0.0.0.0", int(scan_udp_port)))
        self._scan_socket.settimeout(0.10)
        self.socket = socket.create_connection((str(host), int(port)), timeout_s)
        # Status and command frames are small and latency-sensitive.  Leaving
        # Nagle enabled can batch several 20 Hz status frames behind delayed
        # ACKs, making a healthy Pi look stale at the PC safety gate.
        self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.socket.settimeout(None)
        self._send_lock = threading.Lock()
        self._scan_packets = queue.Queue(maxsize=int(maximum_scan_packets))
        self._status_lock = threading.Lock()
        self._status = None
        self._running = True
        self._error = None
        self.last_sequence = None
        self.received_packets = 0
        self.sequence_gaps = 0
        self.decode_errors = 0
        self.backlog_packets_dropped = 0
        self.backlog_packets_preserved = 0
        self.stale_points_dropped = 0
        self.status_packets_received = 0
        self._last_frame_timestamp_ns = None
        self._scan_history = None
        send_frame(self.socket, FRAME_HELLO, encode_json({
            "contract": "pc_pi_full_proposed_v1",
            "token": str(token),
            "scan_udp_port": int(scan_udp_port),
        }), self._send_lock)
        self._scan_thread = threading.Thread(
            target=self._scan_receive_loop, name="remote-livox-udp", daemon=True
        )
        self._scan_thread.start()
        self._thread = threading.Thread(
            target=self._receive_loop, name="remote-deployment-client", daemon=True
        )
        self._thread.start()

    def _scan_receive_loop(self):
        while self._running:
            try:
                payload, _ = self._scan_socket.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                self._scan_packets.put_nowait(payload)
            except queue.Full:
                try:
                    self._scan_packets.get_nowait()
                    self.backlog_packets_dropped += 1
                except queue.Empty:
                    pass
                self._scan_packets.put_nowait(payload)

    def _receive_loop(self):
        try:
            while self._running:
                frame_type, payload = receive_frame(self.socket)
                if frame_type == FRAME_SCAN:
                    try:
                        self._scan_packets.put_nowait(payload)
                    except queue.Full:
                        try:
                            self._scan_packets.get_nowait()
                            self.backlog_packets_dropped += 1
                        except queue.Empty:
                            pass
                        self._scan_packets.put_nowait(payload)
                elif frame_type == FRAME_STATUS:
                    data = decode_json(payload)
                    status = RemoteStatus(
                        received_monotonic=time.monotonic(),
                        v_mps=float(data.get("v_mps", 0.0)),
                        omega_radps=float(data.get("omega_radps", 0.0)),
                        battery_v=(None if data.get("battery_v") is None
                                   else float(data["battery_v"])),
                        control_mode=(None if data.get("control_mode") is None
                                      else int(data["control_mode"])),
                        fault=(None if data.get("fault") is None
                               else int(data["fault"])),
                        armed=bool(data.get("armed", False)),
                        last_sequence=int(data.get("last_sequence", -1)),
                        target_v_mps=float(data.get("target_v_mps", 0.0)),
                        target_omega_radps=float(
                            data.get("target_omega_radps", 0.0)
                        ),
                        applied_v_mps=float(data.get("applied_v_mps", 0.0)),
                        applied_omega_radps=float(
                            data.get("applied_omega_radps", 0.0)
                        ),
                    )
                    with self._status_lock:
                        self._status = status
                    self.status_packets_received += 1
        except Exception as exc:
            if self._running:
                self._error = exc
        finally:
            self._running = False

    def status(self) -> Optional[RemoteStatus]:
        with self._status_lock:
            return self._status

    def wait_for_status(self, timeout_s: float = 3.0) -> RemoteStatus:
        deadline = time.monotonic() + float(timeout_s)
        while time.monotonic() < deadline:
            status = self.status()
            if status is not None:
                return status
            if self._error is not None:
                raise RuntimeError("remote deployment receiver failed") from self._error
            time.sleep(0.01)
        raise TimeoutError("no chassis status received from Pi gateway")

    def send_command(self, sequence: int, v_mps: float, omega_radps: float,
                     arm: bool, *, immediate_translation_stop: bool = False,
                     immediate_all_stop: bool = False) -> None:
        send_frame(self.socket, FRAME_COMMAND, encode_json({
            "sequence": int(sequence),
            "v_mps": float(v_mps),
            "omega_radps": float(omega_radps),
            "arm": bool(arm),
            "immediate_translation_stop": bool(immediate_translation_stop),
            "immediate_all_stop": bool(immediate_all_stop),
        }), self._send_lock)

    def receive_livox_frame(self, accumulation_s: float = 0.06,
                            maximum_points: int = 200000,
                            first_packet_timeout_s: float = 0.25
                            ) -> LivoxPointCloudFrame:
        # An SSH tunnel may deliver a short burst after CUDA work.  Give the
        # first packet a bounded wait, then preserve the original scan
        # accumulation window from that packet onward.
        first_packet_deadline = time.monotonic() + float(first_packet_timeout_s)
        deadline = None
        points = []
        reflectivity = []
        tags = []
        point_timestamps_ns = []
        timestamp_ns = None
        oldest_timestamp_ns = None
        point_count = 0
        prefetched = _drain_all(self._scan_packets)
        self.backlog_packets_preserved += len(prefetched)
        prefetched_index = 0
        while point_count < maximum_points:
            active_deadline = first_packet_deadline if deadline is None else deadline
            if time.monotonic() >= active_deadline:
                break
            timeout = max(0.001, active_deadline - time.monotonic())
            if prefetched_index < len(prefetched):
                payload = prefetched[prefetched_index]
                prefetched_index += 1
            else:
                try:
                    payload = self._scan_packets.get(timeout=timeout)
                except queue.Empty:
                    break
            try:
                datagram = decode_livox_datagram(payload)
            except ValueError:
                self.decode_errors += 1
                continue
            self.received_packets += 1
            if self.last_sequence is not None:
                delta = (datagram.sequence - self.last_sequence) & 0xFFFF
                if delta > 1:
                    self.sequence_gaps += delta - 1
            self.last_sequence = datagram.sequence
            frame = datagram.frame
            frame_timestamp_ns = int(frame.timestamp_ns)
            if (self._last_frame_timestamp_ns is not None and
                    frame_timestamp_ns <= self._last_frame_timestamp_ns):
                continue
            if deadline is None:
                deadline = time.monotonic() + float(accumulation_s)
            take = min(frame.points.shape[0], maximum_points - point_count)
            points.append(frame.points[:take])
            reflectivity.append(frame.reflectivity[:take])
            tags.append(frame.tags[:take])
            point_timestamps_ns.append(np.full(
                take, frame_timestamp_ns, dtype=np.int64
            ))
            point_count += int(take)
            timestamp_ns = (frame_timestamp_ns if timestamp_ns is None else
                            max(timestamp_ns, frame_timestamp_ns))
            oldest_timestamp_ns = (
                frame_timestamp_ns
                if oldest_timestamp_ns is None
                else min(oldest_timestamp_ns, frame_timestamp_ns)
            )
            if (
                prefetched_index >= len(prefetched)
                and oldest_timestamp_ns is not None
                and timestamp_ns - oldest_timestamp_ns
                >= int(float(accumulation_s) * 1.0e9)
            ):
                # The backlog already spans the requested acquisition window.
                # Waiting another fixed 40 ms only adds control latency; newly
                # arriving packets remain queued for the next overlapping
                # rolling frame.
                break
        if not points or timestamp_ns is None:
            if self._error is not None:
                raise RuntimeError("remote deployment receiver failed") from self._error
            raise TimeoutError(
                "no Livox packets received from Pi gateway "
                f"(queue={self._scan_packets.qsize()}, "
                f"thread_alive={self._thread.is_alive()}, "
                f"received={self.received_packets}, "
                f"decode_errors={self.decode_errors})"
            )
        point_values = np.concatenate(points, axis=0)
        reflectivity_values = np.concatenate(reflectivity)
        tag_values = np.concatenate(tags)
        timestamp_values = np.concatenate(point_timestamps_ns)
        if self._scan_history is not None:
            history_points, history_reflectivity, history_tags, history_times = (
                self._scan_history
            )
            point_values = np.concatenate(
                (history_points, point_values), axis=0
            )
            reflectivity_values = np.concatenate(
                (history_reflectivity, reflectivity_values)
            )
            tag_values = np.concatenate((history_tags, tag_values))
            timestamp_values = np.concatenate(
                (history_times, timestamp_values)
            )
        # Overlap consecutive controller observations.  The Livox Mid-360 scan
        # is non-repetitive, so disjoint 80--100 ms batches made the same leg or
        # bumper alternate between visible and absent.  Point timestamps are
        # retained and the caller deskews this bounded window before scan
        # projection.
        maximum_span_s = max(0.16, 4.0 * float(accumulation_s))
        cutoff_ns = int(timestamp_ns) - int(maximum_span_s * 1.0e9)
        fresh = timestamp_values >= cutoff_ns
        self.stale_points_dropped += int(np.sum(~fresh))
        point_values = point_values[fresh]
        reflectivity_values = reflectivity_values[fresh]
        tag_values = tag_values[fresh]
        timestamp_values = timestamp_values[fresh]
        if point_values.shape[0] > int(maximum_points):
            keep = np.argsort(timestamp_values)[-int(maximum_points):]
            point_values = point_values[keep]
            reflectivity_values = reflectivity_values[keep]
            tag_values = tag_values[keep]
            timestamp_values = timestamp_values[keep]
        self._scan_history = (
            point_values.copy(),
            reflectivity_values.copy(),
            tag_values.copy(),
            timestamp_values.copy(),
        )
        self._last_frame_timestamp_ns = int(timestamp_ns)
        return LivoxPointCloudFrame(
            timestamp_ns=int(timestamp_ns),
            points=point_values,
            reflectivity=reflectivity_values,
            tags=tag_values,
            point_timestamps_ns=timestamp_values,
        )

    def close(self) -> None:
        if not self._running:
            try:
                self._scan_socket.close()
            except OSError:
                pass
            try:
                self.socket.close()
            except OSError:
                pass
            return
        try:
            send_frame(self.socket, FRAME_STOP, b"", self._send_lock)
        except OSError:
            pass
        self._running = False
        try:
            self._scan_socket.close()
        except OSError:
            pass
        try:
            self.socket.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.socket.close()
        self._thread.join(timeout=1.0)
        self._scan_thread.join(timeout=1.0)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False


__all__ = [
    "FRAME_COMMAND", "FRAME_HELLO", "FRAME_SCAN", "FRAME_STATUS",
    "FRAME_STOP", "RemoteDeploymentClient", "RemoteStatus", "decode_json",
    "encode_json", "receive_frame", "send_frame",
]
