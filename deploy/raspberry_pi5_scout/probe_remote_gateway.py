#!/usr/bin/env python3
"""Read one Scout Pi gateway status without arming or sending a command."""

import argparse
from dataclasses import asdict
import json
from typing import Any, Callable, Dict

from mobile_robot_mppi.real_robot.remote_transport import RemoteDeploymentClient


def probe_gateway(
    host: str,
    port: int,
    token: str,
    timeout_s: float = 3.0,
    client_factory: Callable[..., RemoteDeploymentClient] = RemoteDeploymentClient,
) -> Dict[str, Any]:
    """Return the first decoded status from the production transport.

    The probe deliberately exposes no arm or command argument. Closing the
    client sends the protocol STOP frame, preserving the gateway's fail-closed
    behavior.
    """

    with client_factory(
        host=str(host),
        port=int(port),
        token=str(token),
        timeout_s=float(timeout_s),
        scan_udp_port=0,
    ) as client:
        status = client.wait_for_status(timeout_s=float(timeout_s))
        payload = asdict(status)
        payload["transport_probe"] = "status_only_non_arming_v1"
        return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Probe Scout Pi status without issuing a motion command"
    )
    parser.add_argument("--host", required=True, help="Raspberry Pi address")
    parser.add_argument("--port", type=int, default=57720)
    parser.add_argument("--token", required=True)
    parser.add_argument("--timeout-s", type=float, default=3.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    payload = probe_gateway(
        host=args.host,
        port=args.port,
        token=args.token,
        timeout_s=args.timeout_s,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
