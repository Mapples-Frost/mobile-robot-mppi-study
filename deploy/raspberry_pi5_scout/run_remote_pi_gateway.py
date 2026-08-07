#!/usr/bin/env python3
"""Pi-side Livox stream and watchdog-protected SCOUT command gateway."""

import argparse
import json
import math
from pathlib import Path
import socket
import threading
import time

from mobile_robot_mppi.real_robot import ScoutGuardedCanGateway
from mobile_robot_mppi.real_robot.remote_transport import (
    FRAME_COMMAND, FRAME_HELLO, FRAME_SCAN, FRAME_STATUS, FRAME_STOP,
    decode_json, encode_json, receive_frame, send_frame,
)


def _control_mode_interlock_reason(control_mode, confirmed, arm_age_s):
    """Return a fail-closed reason for loss of CAN motion authority."""
    if control_mode == 3:
        return "remote_takeover"
    if confirmed and control_mode != 1:
        return "can_mode_lost"
    if not confirmed and float(arm_age_s) > 1.0:
        return "can_mode_not_confirmed"
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=57720)
    parser.add_argument("--livox-port", type=int, default=57701)
    parser.add_argument("--duration-s", type=float, default=30.0)
    parser.add_argument("--allow-arm", action="store_true")
    parser.add_argument("--max-v-mps", type=float, default=0.05)
    parser.add_argument("--max-reverse-v-mps", type=float, default=0.05)
    parser.add_argument("--max-omega-radps", type=float, default=0.10)
    parser.add_argument("--watchdog-timeout-s", type=float, default=0.35)
    parser.add_argument("--max-linear-acceleration-mps2", type=float, default=1.0)
    parser.add_argument("--max-linear-deceleration-mps2", type=float, default=2.0)
    parser.add_argument("--max-angular-acceleration-radps2", type=float, default=4.0)
    parser.add_argument("--accept-timeout-s", type=float, default=120.0)
    parser.add_argument("--scan-stride", type=int, default=8)
    args = parser.parse_args()
    if args.scan_stride < 1:
        raise ValueError("scan-stride must be at least 1")
    args.output.mkdir(parents=True, exist_ok=False)

    events = []
    event_lock = threading.Lock()
    running = threading.Event()
    running.set()
    send_lock = threading.Lock()
    armed = threading.Event()
    can_mode_confirmed = threading.Event()
    arm_started = [None]
    mode_interlock_triggered = [False]
    last_sequence = [-1]

    def record(kind, **values):
        row = {"monotonic": time.monotonic(), "kind": kind, **values}
        with event_lock:
            events.append(row)

    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp.bind(("127.0.0.1", args.livox_port))
    udp.settimeout(0.05)
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("0.0.0.0", args.port))
    server.listen(1)
    # The CUDA PC deliberately constructs and validates the frozen full stack
    # before opening the control connection.  Model loading can take longer
    # than a lightweight sensor-only client startup.
    server.settimeout(args.accept_timeout_s)

    scan_packets = 0
    scan_packets_received = 0
    status_packets_sent = 0
    scan_thread = None
    scan_output = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    scan_destination = None
    client = None
    gateway = ScoutGuardedCanGateway(
        "can0", max_v_mps=args.max_v_mps,
        max_reverse_v_mps=args.max_reverse_v_mps,
        max_omega_radps=args.max_omega_radps,
        watchdog_timeout_s=args.watchdog_timeout_s,
        max_linear_acceleration_mps2=args.max_linear_acceleration_mps2,
        max_linear_deceleration_mps2=args.max_linear_deceleration_mps2,
        max_angular_acceleration_radps2=args.max_angular_acceleration_radps2,
    )
    try:
        gateway.start()

        def scan_loop():
            nonlocal scan_packets, scan_packets_received
            while running.is_set():
                try:
                    packet, _ = udp.recvfrom(65535)
                except socket.timeout:
                    continue
                except OSError:
                    break
                packet_index = scan_packets_received
                scan_packets_received += 1
                destination = scan_destination
                if destination is not None and packet_index % args.scan_stride == 0:
                    sent = scan_output.sendto(packet, destination)
                    if sent == len(packet):
                        scan_packets += 1
                time.sleep(0)

        # Keep draining UDP while the PC loads the CUDA models, and forward
        # scans independently from the status/control TCP threads.
        scan_thread = threading.Thread(
            target=scan_loop, name="livox-udp-drain", daemon=True
        )
        scan_thread.start()
        client, address = server.accept()
        # Do not aggregate the small 20 Hz status frames.  The PC independently
        # rejects status older than its frozen safety limit.
        client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        client.settimeout(2.0)
        frame_type, payload = receive_frame(client)
        if frame_type != FRAME_HELLO:
            raise RuntimeError("first remote frame was not HELLO")
        hello = decode_json(payload)
        if hello.get("token") != args.token:
            raise RuntimeError("remote session token mismatch")
        scan_udp_port = int(hello.get("scan_udp_port", 0))
        if not 1 <= scan_udp_port <= 65535:
            raise RuntimeError("remote scan UDP port is invalid")
        scan_destination = (address[0], scan_udp_port)
        client.settimeout(None)
        record("connected", address=address[0], allow_arm=args.allow_arm)
        # The requested session duration begins only after the authenticated PC
        # is attached; package import, CAN startup and operator connection time
        # must not consume the experiment window.
        started = time.monotonic()

        def command_loop():
            try:
                while running.is_set():
                    frame_type, payload = receive_frame(client)
                    if frame_type == FRAME_STOP:
                        record("stop_received")
                        running.clear()
                        break
                    if frame_type != FRAME_COMMAND:
                        continue
                    command = decode_json(payload)
                    sequence = int(command["sequence"])
                    if sequence <= last_sequence[0]:
                        record("command_rejected", reason="non_increasing_sequence",
                               sequence=sequence)
                        continue
                    v_mps = float(command["v_mps"])
                    omega = float(command["omega_radps"])
                    if not math.isfinite(v_mps) or not math.isfinite(omega):
                        record("command_rejected", reason="non_finite",
                               sequence=sequence)
                        continue
                    last_sequence[0] = sequence
                    request_arm = bool(command.get("arm", False))
                    immediate_translation_stop = bool(
                        command.get("immediate_translation_stop", False)
                    )
                    immediate_all_stop = bool(
                        command.get("immediate_all_stop", False)
                    )
                    if request_arm and args.allow_arm and not armed.is_set():
                        snapshot = gateway.snapshot()
                        if snapshot.fault != 0 or snapshot.battery_v is None:
                            record("arm_rejected", fault=snapshot.fault,
                                   battery_v=snapshot.battery_v)
                            continue
                        gateway.arm()
                        armed.set()
                        can_mode_confirmed.clear()
                        arm_started[0] = time.monotonic()
                        record("armed", sequence=sequence)
                    if armed.is_set():
                        applied = gateway.command(
                            v_mps,
                            omega,
                            immediate_translation_stop=(
                                immediate_translation_stop
                            ),
                            immediate_all_stop=immediate_all_stop,
                        )
                        record("command", sequence=sequence,
                               requested=[v_mps, omega], target=list(applied),
                               immediate_translation_stop=(
                                   immediate_translation_stop
                               ),
                               immediate_all_stop=immediate_all_stop)
                    else:
                        record("shadow_command", sequence=sequence,
                               requested=[v_mps, omega])
            except Exception as exc:
                record("command_receiver_stopped", error=repr(exc))
                running.clear()

        command_thread = threading.Thread(
            target=command_loop, name="remote-command-receiver", daemon=True
        )
        command_thread.start()
        next_status = 0.0
        while running.is_set() and time.monotonic() - started < args.duration_s:
            now = time.monotonic()
            if now >= next_status:
                snapshot = gateway.snapshot()
                command_snapshot = gateway.command_snapshot()
                if armed.is_set():
                    if snapshot.control_mode == 1:
                        can_mode_confirmed.set()
                    arm_age_s = (
                        0.0
                        if arm_started[0] is None
                        else now - float(arm_started[0])
                    )
                    mode_reason = _control_mode_interlock_reason(
                        snapshot.control_mode,
                        can_mode_confirmed.is_set(),
                        arm_age_s,
                    )
                    if mode_reason is not None:
                        gateway.revoke_motion_authority()
                        armed.clear()
                        mode_interlock_triggered[0] = True
                        record(
                            "control_mode_interlock",
                            reason=mode_reason,
                            control_mode=snapshot.control_mode,
                        )
                        running.clear()
                send_frame(client, FRAME_STATUS, encode_json({
                    "gateway_monotonic_s": now,
                    "v_mps": snapshot.v_mps,
                    "omega_radps": snapshot.omega_radps,
                    "feedback_timestamp_s": (
                        None
                        if snapshot.feedback_timestamp <= 0.0
                        else snapshot.feedback_timestamp
                    ),
                    "feedback_age_s": (
                        None
                        if snapshot.feedback_timestamp <= 0.0
                        else max(0.0, now - snapshot.feedback_timestamp)
                    ),
                    "battery_v": snapshot.battery_v,
                    "control_mode": snapshot.control_mode,
                    "fault": snapshot.fault,
                    "armed": armed.is_set(),
                    "last_sequence": last_sequence[0],
                    "target_v_mps": command_snapshot.target_v_mps,
                    "target_omega_radps": command_snapshot.target_omega_radps,
                    "applied_v_mps": command_snapshot.applied_v_mps,
                    "applied_omega_radps": command_snapshot.applied_omega_radps,
                    "command_timestamp_s": command_snapshot.timestamp,
                }), send_lock)
                status_packets_sent += 1
                next_status = now + 0.05
            time.sleep(0.01)
        running.clear()
        command_thread.join(timeout=1.0)
    finally:
        gateway.close(
            release_control_mode=not mode_interlock_triggered[0]
        )
        if client is not None:
            try:
                client.close()
            except OSError:
                pass
        server.close()
        udp.close()
        scan_output.close()
        if scan_thread is not None:
            scan_thread.join(timeout=1.0)
        summary = {
            "contract": "pc_pi_full_proposed_v1",
            "allow_arm": bool(args.allow_arm),
            "max_forward_v_mps": args.max_v_mps,
            "max_reverse_v_mps": args.max_reverse_v_mps,
            "max_omega_radps": args.max_omega_radps,
            "scan_packets_forwarded": scan_packets,
            "scan_packets_received": scan_packets_received,
            "scan_stride": args.scan_stride,
            "scan_transport": "direct_udp",
            "status_packets_sent": status_packets_sent,
            "last_sequence": last_sequence[0],
            "nonzero_can_commands_sent": gateway.nonzero_commands_sent,
            "watchdog_zero_events": gateway.watchdog_zero_events,
            "command_interpolation": {
                "period_s": gateway.period_s,
                "max_linear_acceleration_mps2": (
                    gateway.max_linear_acceleration_mps2
                ),
                "max_linear_deceleration_mps2": (
                    gateway.max_linear_deceleration_mps2
                ),
                "max_angular_acceleration_radps2": (
                    gateway.max_angular_acceleration_radps2
                ),
            },
            "control_mode_interlock_triggered": bool(
                mode_interlock_triggered[0]
            ),
            "events": len(events),
        }
        (args.output / "events.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in events),
            encoding="utf-8",
        )
        (args.output / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
