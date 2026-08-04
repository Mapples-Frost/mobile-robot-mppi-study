import struct

import numpy as np
import pytest

from mobile_robot_mppi.real_robot.livox_udp import decode_livox_datagram


def _payload(points):
    body = b"".join(
        struct.pack("<fffBB", x, y, z, reflectivity, tag)
        for x, y, z, reflectivity, tag in points
    )
    return struct.pack("<4sHHIQ", b"LVX1", 1, len(points), 17, 123456) + body


def test_decodes_versioned_wire_contract_without_losing_raw_bytes():
    value = decode_livox_datagram(_payload([
        (1.0, -2.0, 0.5, 91, 0xA5),
        (3.0, 4.0, -1.0, 7, 0x11),
    ]))
    assert value.sequence == 17
    assert value.frame.timestamp_ns == 123456
    np.testing.assert_array_equal(value.frame.reflectivity, (91, 7))
    np.testing.assert_array_equal(value.frame.tags, (0xA5, 0x11))
    np.testing.assert_allclose(
        value.frame.points,
        ((1.0, -2.0, 0.5), (3.0, 4.0, -1.0)),
    )


def test_rejects_truncated_or_unknown_wire_contract():
    with pytest.raises(ValueError, match="truncated"):
        decode_livox_datagram(b"LVX1")
    bad = bytearray(_payload([(1.0, 2.0, 3.0, 4, 5)]))
    bad[:4] = b"NOPE"
    with pytest.raises(ValueError, match="unsupported"):
        decode_livox_datagram(bytes(bad))
    with pytest.raises(ValueError, match="length"):
        decode_livox_datagram(_payload([(1.0, 2.0, 3.0, 4, 5)])[:-1])
