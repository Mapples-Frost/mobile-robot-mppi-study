#!/usr/bin/env python3
"""Five-second wire-format smoke test; does not access the vehicle CAN bus."""

import socket
import struct
import time


def main():
    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.bind(("127.0.0.1", 57701))
    receiver.settimeout(2.0)
    deadline = time.monotonic() + 5.0
    packets = 0
    points = 0
    bad = 0
    first = None
    while time.monotonic() < deadline:
        try:
            payload, _ = receiver.recvfrom(65535)
        except socket.timeout:
            continue
        if len(payload) < 20:
            bad += 1
            continue
        magic, version, count, _, _ = struct.unpack_from(
            "<4sHHIQ", payload
        )
        if (
            magic != b"LVX1"
            or version != 1
            or len(payload) != 20 + 14 * count
        ):
            bad += 1
            continue
        if first is None and count:
            first = struct.unpack_from("<fffBB", payload, 20)
        packets += 1
        points += count
    result = {
        "packets": packets,
        "points": points,
        "bad": bad,
        "first": first,
    }
    print(result)
    if packets == 0 or bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
