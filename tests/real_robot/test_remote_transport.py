import socket
import threading
import queue

from mobile_robot_mppi.real_robot.remote_transport import (
    FRAME_COMMAND, decode_json, encode_json, receive_frame, send_frame,
    _drain_all, _drain_latest,
)


def test_remote_frame_round_trip_handles_fragmentation():
    left, right = socket.socketpair()
    payload = encode_json({"sequence": 7, "v_mps": 0.05})
    thread = threading.Thread(
        target=lambda: send_frame(left, FRAME_COMMAND, payload), daemon=True
    )
    thread.start()
    frame_type, received = receive_frame(right)
    thread.join(timeout=1.0)
    left.close()
    right.close()
    assert frame_type == FRAME_COMMAND
    assert decode_json(received) == {"sequence": 7, "v_mps": 0.05}


def test_remote_json_rejects_non_object():
    try:
        decode_json(b"[1,2,3]")
    except ValueError as exc:
        assert "object" in str(exc)
    else:
        raise AssertionError("non-object JSON was accepted")


def test_scan_backlog_is_collapsed_to_latest_packet():
    values = queue.Queue()
    for value in (b"oldest", b"middle", b"latest"):
        values.put(value)
    latest, dropped = _drain_latest(values)
    assert latest == b"latest"
    assert dropped == 2
    assert values.empty()


def test_livox_scan_backlog_is_preserved_in_arrival_order():
    values = queue.Queue()
    for value in (b"oldest", b"middle", b"latest"):
        values.put(value)
    assert _drain_all(values) == [b"oldest", b"middle", b"latest"]
    assert values.empty()
