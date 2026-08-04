"""Python receiver for the versioned localhost Livox UDP wire format."""

from dataclasses import dataclass
import socket
import struct
import time
from typing import List, Optional

import numpy as np

from mobile_robot_mppi.real_robot.livox_scan_adapter import (
    LivoxPointCloudFrame,
)


_HEADER = struct.Struct("<4sHHIQ")
_POINT_DTYPE = np.dtype([
    ("x", "<f4"),
    ("y", "<f4"),
    ("z", "<f4"),
    ("reflectivity", "u1"),
    ("tag", "u1"),
])


@dataclass(frozen=True)
class LivoxDatagram:
    sequence: int
    frame: LivoxPointCloudFrame


def decode_livox_datagram(payload: bytes) -> LivoxDatagram:
    if len(payload) < _HEADER.size:
        raise ValueError("truncated Livox UDP header")
    magic, version, count, sequence, timestamp_ns = _HEADER.unpack_from(payload)
    if magic != b"LVX1" or version != 1:
        raise ValueError("unsupported Livox UDP wire contract")
    expected = _HEADER.size + int(count) * _POINT_DTYPE.itemsize
    if len(payload) != expected:
        raise ValueError("Livox UDP payload length does not match point_count")
    packed = np.frombuffer(payload, dtype=_POINT_DTYPE, count=int(count),
                           offset=_HEADER.size)
    points = np.column_stack((packed["x"], packed["y"], packed["z"]))
    return LivoxDatagram(
        sequence=int(sequence),
        frame=LivoxPointCloudFrame(
            timestamp_ns=int(timestamp_ns),
            points=points,
            reflectivity=packed["reflectivity"],
            tags=packed["tag"],
        ),
    )


class LivoxUdpReceiver:
    """Accumulate SDK packets into bounded-duration point cloud frames."""

    def __init__(self, host: str = "127.0.0.1", port: int = 57701,
                 timeout_s: float = 0.25):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind((str(host), int(port)))
        self.socket.settimeout(float(timeout_s))
        self.last_sequence = None  # type: Optional[int]
        self.received_packets = 0
        self.sequence_gaps = 0
        self.decode_errors = 0

    def close(self) -> None:
        self.socket.close()

    def receive_frame(self, accumulation_s: float = 0.10,
                      maximum_points: int = 200000) -> LivoxPointCloudFrame:
        if accumulation_s <= 0.0 or maximum_points <= 0:
            raise ValueError("frame accumulation limits must be positive")
        deadline = time.monotonic() + float(accumulation_s)
        points = []  # type: List[np.ndarray]
        reflectivity = []  # type: List[np.ndarray]
        tags = []  # type: List[np.ndarray]
        timestamp_ns = None
        point_count = 0
        while time.monotonic() < deadline and point_count < maximum_points:
            try:
                payload, _ = self.socket.recvfrom(65535)
            except socket.timeout:
                if points:
                    break
                raise TimeoutError("no Livox UDP packet received")
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
            remaining = maximum_points - point_count
            take = min(int(frame.points.shape[0]), remaining)
            points.append(frame.points[:take])
            reflectivity.append(frame.reflectivity[:take])
            tags.append(frame.tags[:take])
            point_count += take
            timestamp_ns = frame.timestamp_ns
        if not points or timestamp_ns is None:
            raise TimeoutError("no valid Livox UDP packet received")
        return LivoxPointCloudFrame(
            timestamp_ns=timestamp_ns,
            points=np.concatenate(points, axis=0),
            reflectivity=np.concatenate(reflectivity),
            tags=np.concatenate(tags),
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False


__all__ = [
    "LivoxDatagram",
    "LivoxUdpReceiver",
    "decode_livox_datagram",
]
