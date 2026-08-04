"""Fail-closed SCOUT MINI CAN helpers for the Raspberry Pi deployment."""

from dataclasses import dataclass
import struct
import threading
import time
from typing import Optional


MOTION_COMMAND_ID = 0x111
MOTION_FEEDBACK_ID = 0x221
CHASSIS_STATUS_ID = 0x211
CONTROL_MODE_ID = 0x421


def pack_scout_motion(v_mps: float, omega_radps: float) -> bytes:
    v_raw = int(round(float(v_mps) * 1000.0))
    omega_raw = int(round(float(omega_radps) * 1000.0))
    if not -32768 <= v_raw <= 32767 or not -32768 <= omega_raw <= 32767:
        raise ValueError("SCOUT motion command exceeds int16 wire range")
    return struct.pack(">hhHH", v_raw, omega_raw, 0, 0)


def unpack_scout_motion_feedback(payload: bytes):
    if len(payload) < 4:
        raise ValueError("SCOUT motion feedback requires four bytes")
    v_raw, omega_raw = struct.unpack(">hh", bytes(payload[:4]))
    return v_raw / 1000.0, omega_raw / 1000.0


@dataclass(frozen=True)
class ScoutCanSnapshot:
    timestamp: float
    v_mps: float
    omega_radps: float
    battery_v: Optional[float]
    control_mode: Optional[int]
    fault: Optional[int]


@dataclass(frozen=True)
class ScoutCommandSnapshot:
    """Latest PC target and the command actually emitted by the 20 Hz loop."""

    timestamp: float
    target_v_mps: float
    target_omega_radps: float
    applied_v_mps: float
    applied_omega_radps: float
    armed: bool


class ScoutZeroOnlyCanGuard:
    """Continuously transmit only zero motion and collect chassis feedback.

    There is intentionally no method accepting a non-zero command.  The first
    deployment milestone therefore cannot move the robot even if planner or
    safety code produces a non-zero value.
    """

    def __init__(self, channel="can0", period_s=0.10, bus=None):
        self.channel = str(channel)
        self.period_s = float(period_s)
        if self.period_s <= 0.0 or self.period_s > 0.25:
            raise ValueError("zero guard period must be in (0, 0.25] seconds")
        if bus is None:
            import can
            bus = can.Bus(interface="socketcan", channel=self.channel)
        self.bus = bus
        self._lock = threading.Lock()
        self._snapshot = ScoutCanSnapshot(
            timestamp=0.0,
            v_mps=0.0,
            omega_radps=0.0,
            battery_v=None,
            control_mode=None,
            fault=None,
        )
        self._running = False
        self._thread = None

    def _zero_message(self):
        import can
        return can.Message(
            arbitration_id=MOTION_COMMAND_ID,
            data=pack_scout_motion(0.0, 0.0),
            is_extended_id=False,
        )

    def _update_feedback(self, message):
        with self._lock:
            current = self._snapshot
            if message.arbitration_id == MOTION_FEEDBACK_ID:
                v_mps, omega = unpack_scout_motion_feedback(message.data)
                self._snapshot = ScoutCanSnapshot(
                    time.monotonic(), v_mps, omega, current.battery_v,
                    current.control_mode, current.fault,
                )
            elif message.arbitration_id == CHASSIS_STATUS_ID and len(message.data) >= 6:
                battery = ((message.data[2] << 8) | message.data[3]) / 10.0
                self._snapshot = ScoutCanSnapshot(
                    time.monotonic(), current.v_mps, current.omega_radps,
                    battery, int(message.data[1]), int(message.data[5]),
                )

    def _run(self):
        next_send = time.monotonic()
        message = self._zero_message()
        while self._running:
            now = time.monotonic()
            if now >= next_send:
                self.bus.send(message)
                next_send = now + self.period_s
            received = self.bus.recv(timeout=min(0.02, self.period_s / 2.0))
            if received is not None:
                self._update_feedback(received)

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._run, name="scout-zero-only-can", daemon=True
        )
        self._thread.start()

    def snapshot(self):
        with self._lock:
            return self._snapshot

    def close(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, 4.0 * self.period_s))
        zero = self._zero_message()
        for _ in range(5):
            try:
                self.bus.send(zero)
            except Exception:
                pass
            time.sleep(0.02)
        self.bus.shutdown()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False


class ScoutGuardedCanGateway(ScoutZeroOnlyCanGuard):
    """Low-speed, watchdog-protected SCOUT command gateway.

    The gateway starts disarmed and continues transmitting zero motion.  A
    caller must explicitly call :meth:`arm` before non-zero commands are
    accepted.  Commands are clipped to the frozen smoke-test envelope and
    automatically return to zero if the planner stops refreshing them.
    """

    def __init__(self, channel="can0", period_s=0.05,
                 watchdog_timeout_s=0.60, max_v_mps=0.05,
                 max_omega_radps=0.10, max_reverse_v_mps=None, bus=None,
                 max_linear_acceleration_mps2=1.0,
                 max_linear_deceleration_mps2=2.0,
                 max_angular_acceleration_radps2=4.0):
        super().__init__(channel=channel, period_s=period_s, bus=bus)
        self.watchdog_timeout_s = float(watchdog_timeout_s)
        self.max_v_mps = float(max_v_mps)
        self.max_reverse_v_mps = float(
            max_v_mps if max_reverse_v_mps is None else max_reverse_v_mps
        )
        self.max_omega_radps = float(max_omega_radps)
        self.max_linear_acceleration_mps2 = float(
            max_linear_acceleration_mps2
        )
        self.max_linear_deceleration_mps2 = float(
            max_linear_deceleration_mps2
        )
        self.max_angular_acceleration_radps2 = float(
            max_angular_acceleration_radps2
        )
        if not 0.10 <= self.watchdog_timeout_s <= 1.0:
            raise ValueError("watchdog timeout must be in [0.10, 1.0] seconds")
        if not 0.0 < self.max_v_mps <= 0.50:
            raise ValueError("guarded forward speed limit must be in (0, 0.50]")
        if not 0.0 < self.max_reverse_v_mps <= 0.30:
            raise ValueError("guarded reverse speed limit must be in (0, 0.30]")
        if not 0.0 < self.max_omega_radps <= 0.60:
            raise ValueError("guarded angular speed limit must be in (0, 0.60]")
        if (
            self.max_linear_acceleration_mps2 <= 0.0
            or self.max_linear_deceleration_mps2 <= 0.0
            or self.max_angular_acceleration_radps2 <= 0.0
        ):
            raise ValueError("guarded command interpolation rates must be positive")
        self._armed = False
        self._desired_v = 0.0
        self._desired_omega = 0.0
        self._applied_v = 0.0
        self._applied_omega = 0.0
        self._applied_timestamp = 0.0
        self._last_command_time = 0.0
        self.nonzero_commands_sent = 0
        self.watchdog_zero_events = 0

    @staticmethod
    def _can_message(arbitration_id, data):
        import can
        return can.Message(
            arbitration_id=int(arbitration_id),
            data=bytes(data),
            is_extended_id=False,
        )

    def _mode_message(self, enabled):
        return self._can_message(CONTROL_MODE_ID, [0x01 if enabled else 0x00])

    def arm(self):
        if not self._running:
            raise RuntimeError("CAN gateway must be running before arming")
        with self._lock:
            self._desired_v = 0.0
            self._desired_omega = 0.0
            self._applied_v = 0.0
            self._applied_omega = 0.0
            self._applied_timestamp = time.monotonic()
            self._last_command_time = time.monotonic()
            self._armed = True
        self.bus.send(self._mode_message(True))

    def command(self, v_mps, omega_radps, *,
                immediate_translation_stop=False,
                immediate_all_stop=False):
        """Set a bounded target without tying smoothness to the PC loop rate.

        Normal targets are interpolated at the CAN transmit rate.  Safety
        stops are applied to the interpolation state immediately, so neither
        a low PC planning frequency nor a prior positive target can postpone
        braking.
        """
        with self._lock:
            if not self._armed:
                raise RuntimeError("CAN gateway is disarmed")
            v = max(
                -self.max_reverse_v_mps,
                min(self.max_v_mps, float(v_mps)),
            )
            omega = max(
                -self.max_omega_radps,
                min(self.max_omega_radps, float(omega_radps)),
            )
            if bool(immediate_all_stop):
                v = 0.0
                omega = 0.0
                self._applied_v = 0.0
                self._applied_omega = 0.0
            elif bool(immediate_translation_stop):
                v = 0.0
                self._applied_v = 0.0
            self._desired_v = v
            self._desired_omega = omega
            self._last_command_time = time.monotonic()
            return v, omega

    def command_snapshot(self):
        with self._lock:
            return ScoutCommandSnapshot(
                timestamp=float(self._applied_timestamp),
                target_v_mps=float(self._desired_v),
                target_omega_radps=float(self._desired_omega),
                applied_v_mps=float(self._applied_v),
                applied_omega_radps=float(self._applied_omega),
                armed=bool(self._armed),
            )

    def revoke_motion_authority(self):
        """Latch zero motion without changing a human-selected control mode."""
        with self._lock:
            self._armed = False
            self._desired_v = 0.0
            self._desired_omega = 0.0
            self._applied_v = 0.0
            self._applied_omega = 0.0
            self._applied_timestamp = time.monotonic()
            self._last_command_time = 0.0
        zero = self._zero_message()
        for _ in range(5):
            try:
                self.bus.send(zero)
            except Exception:
                pass
            time.sleep(0.02)

    def disarm(self, release_control_mode=True):
        self.revoke_motion_authority()
        if release_control_mode:
            try:
                self.bus.send(self._mode_message(False))
            except Exception:
                pass

    def _run(self):
        next_send = time.monotonic()
        previous_send = next_send
        watchdog_latched = False
        while self._running:
            now = time.monotonic()
            if now >= next_send:
                with self._lock:
                    armed = self._armed
                    expired = (
                        armed
                        and now - self._last_command_time
                        > self.watchdog_timeout_s
                    )
                    if armed and not expired:
                        dt = max(0.0, now - previous_send)
                        v_target = self._desired_v
                        omega_target = self._desired_omega
                        linear_rate = self.max_linear_acceleration_mps2
                        if (
                            abs(v_target) < abs(self._applied_v)
                            or self._applied_v * v_target < 0.0
                        ):
                            linear_rate = self.max_linear_deceleration_mps2
                        # A sign change is a braking transaction: reach zero
                        # first, then accelerate in the opposite direction on
                        # a later 20 Hz transmit tick.
                        v_step_target = v_target
                        if self._applied_v * v_target < 0.0:
                            v_step_target = 0.0
                        self._applied_v = self._move_toward(
                            self._applied_v,
                            v_step_target,
                            linear_rate * dt,
                        )
                        self._applied_omega = self._move_toward(
                            self._applied_omega,
                            omega_target,
                            self.max_angular_acceleration_radps2 * dt,
                        )
                        v = self._applied_v
                        omega = self._applied_omega
                    else:
                        self._desired_v = 0.0
                        self._desired_omega = 0.0
                        self._applied_v = 0.0
                        self._applied_omega = 0.0
                        v = 0.0
                        omega = 0.0
                    self._applied_timestamp = now
                    if expired and not watchdog_latched:
                        self.watchdog_zero_events += 1
                    watchdog_latched = expired
                self.bus.send(self._can_message(
                    MOTION_COMMAND_ID, pack_scout_motion(v, omega)
                ))
                if abs(v) > 1.0e-12 or abs(omega) > 1.0e-12:
                    self.nonzero_commands_sent += 1
                previous_send = now
                next_send = now + self.period_s
            received = self.bus.recv(timeout=min(0.02, self.period_s / 2.0))
            if received is not None:
                self._update_feedback(received)

    def close(self, release_control_mode=True):
        self.disarm(release_control_mode=release_control_mode)
        super().close()

    @staticmethod
    def _move_toward(current, target, maximum_delta):
        delta = float(target) - float(current)
        limit = max(0.0, float(maximum_delta))
        if abs(delta) <= limit:
            return float(target)
        return float(current) + (limit if delta > 0.0 else -limit)


__all__ = [
    "ScoutCommandSnapshot",
    "ScoutCanSnapshot",
    "ScoutGuardedCanGateway",
    "ScoutZeroOnlyCanGuard",
    "pack_scout_motion",
    "unpack_scout_motion_feedback",
]
