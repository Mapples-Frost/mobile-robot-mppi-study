import struct
import sys
import time
from types import SimpleNamespace

import pytest

from mobile_robot_mppi.real_robot.scout_can import (
    CONTROL_MODE_ID,
    MOTION_COMMAND_ID,
    ScoutGuardedCanGateway,
    pack_scout_motion,
    unpack_scout_motion_feedback,
)


def test_scout_motion_wire_contract_is_big_endian_signed_int16():
    payload = pack_scout_motion(-0.1, 0.35)
    assert payload == struct.pack(">hhHH", -100, 350, 0, 0)
    assert unpack_scout_motion_feedback(payload) == pytest.approx((-0.1, 0.35))


def test_scout_motion_rejects_wire_overflow():
    with pytest.raises(ValueError, match="int16"):
        pack_scout_motion(40.0, 0.0)


class _FakeBus:
    def __init__(self):
        self.sent = []
        self.shutdown_called = False

    def send(self, message):
        self.sent.append(message)

    def recv(self, timeout):
        time.sleep(min(float(timeout), 0.001))
        return None

    def shutdown(self):
        self.shutdown_called = True


class _FakeMessage:
    def __init__(self, arbitration_id, data, is_extended_id=False):
        self.arbitration_id = int(arbitration_id)
        self.data = bytes(data)
        self.is_extended_id = bool(is_extended_id)


@pytest.fixture(autouse=True)
def _fake_python_can(monkeypatch):
    monkeypatch.setitem(sys.modules, "can", SimpleNamespace(Message=_FakeMessage))


def _decoded_motion(message):
    assert message.arbitration_id == MOTION_COMMAND_ID
    return unpack_scout_motion_feedback(message.data)


def test_guarded_gateway_is_disarmed_by_default_and_clips_commands():
    bus = _FakeBus()
    gateway = ScoutGuardedCanGateway(
        bus=bus, period_s=0.01, watchdog_timeout_s=0.10,
        max_v_mps=0.05, max_omega_radps=0.10,
    )
    with pytest.raises(RuntimeError, match="disarmed"):
        gateway.command(1.0, 1.0)
    gateway.start()
    gateway.arm()
    assert gateway.command(1.0, -1.0) == pytest.approx((0.05, -0.10))
    time.sleep(0.07)
    gateway.close()
    motions = [m for m in bus.sent if m.arbitration_id == MOTION_COMMAND_ID]
    assert any(_decoded_motion(m) == pytest.approx((0.05, -0.10)) for m in motions)
    modes = [m for m in bus.sent if m.arbitration_id == CONTROL_MODE_ID]
    assert bytes(modes[0].data) == b"\x01"
    assert bytes(modes[-1].data) == b"\x00"
    assert _decoded_motion(motions[-1]) == pytest.approx((0.0, 0.0))
    assert bus.shutdown_called


def test_guarded_gateway_watchdog_returns_to_zero():
    bus = _FakeBus()
    gateway = ScoutGuardedCanGateway(
        bus=bus, period_s=0.01, watchdog_timeout_s=0.10,
        max_v_mps=0.05, max_omega_radps=0.10,
    )
    gateway.start()
    gateway.arm()
    gateway.command(0.04, 0.08)
    time.sleep(0.16)
    gateway.close()
    motions = [m for m in bus.sent if m.arbitration_id == MOTION_COMMAND_ID]
    assert any(_decoded_motion(m) == pytest.approx((0.04, 0.08)) for m in motions)
    assert gateway.watchdog_zero_events == 1
    assert _decoded_motion(motions[-1]) == pytest.approx((0.0, 0.0))


def test_guarded_gateway_applies_asymmetric_linear_limits():
    bus = _FakeBus()
    gateway = ScoutGuardedCanGateway(
        bus=bus, max_v_mps=0.50, max_reverse_v_mps=0.20,
        max_omega_radps=0.40,
    )
    gateway.start()
    gateway.arm()
    assert gateway.command(0.8, 0.8) == (0.5, 0.4)
    assert gateway.command(-0.8, -0.8) == (-0.2, -0.4)
    gateway.close()


def test_revoke_motion_authority_does_not_override_human_control_mode():
    bus = _FakeBus()
    gateway = ScoutGuardedCanGateway(
        bus=bus, period_s=0.01, watchdog_timeout_s=0.10,
        max_v_mps=0.05, max_omega_radps=0.10,
    )
    gateway.start()
    gateway.arm()
    gateway.command(0.04, 0.08)
    gateway.revoke_motion_authority()
    mode_count = len([
        message for message in bus.sent
        if message.arbitration_id == CONTROL_MODE_ID
    ])
    assert mode_count == 1
    with pytest.raises(RuntimeError, match="disarmed"):
        gateway.command(0.04, 0.08)
    gateway.close(release_control_mode=False)
    modes = [
        message for message in bus.sent
        if message.arbitration_id == CONTROL_MODE_ID
    ]
    assert len(modes) == 1


def test_gateway_interpolates_normal_targets_at_can_rate():
    bus = _FakeBus()
    gateway = ScoutGuardedCanGateway(
        bus=bus,
        period_s=0.01,
        watchdog_timeout_s=0.20,
        max_v_mps=0.50,
        max_reverse_v_mps=0.30,
        max_omega_radps=0.60,
        max_linear_acceleration_mps2=1.0,
        max_linear_deceleration_mps2=2.0,
        max_angular_acceleration_radps2=4.0,
    )
    gateway.start()
    gateway.arm()
    gateway.command(0.50, 0.60)
    time.sleep(0.045)
    snapshot = gateway.command_snapshot()
    gateway.close()

    assert snapshot.target_v_mps == pytest.approx(0.50)
    assert 0.0 < snapshot.applied_v_mps < 0.10
    assert 0.0 < snapshot.applied_omega_radps < 0.30


def test_gateway_hard_stop_bypasses_interpolation_immediately():
    bus = _FakeBus()
    gateway = ScoutGuardedCanGateway(
        bus=bus,
        period_s=0.01,
        watchdog_timeout_s=0.20,
        max_v_mps=0.50,
        max_reverse_v_mps=0.30,
        max_omega_radps=0.60,
    )
    gateway.start()
    gateway.arm()
    gateway.command(0.50, 0.40)
    time.sleep(0.035)
    moving = gateway.command_snapshot()
    target = gateway.command(
        0.50,
        -0.40,
        immediate_translation_stop=True,
    )
    stopped = gateway.command_snapshot()
    gateway.close()

    assert moving.applied_v_mps > 0.0
    assert target[0] == 0.0
    assert stopped.target_v_mps == 0.0
    assert stopped.applied_v_mps == 0.0
    # Translation braking does not erase turn-away authority.
    assert stopped.target_omega_radps == pytest.approx(-0.40)
