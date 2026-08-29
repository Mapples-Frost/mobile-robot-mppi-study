from dataclasses import dataclass

from deploy.raspberry_pi5_scout.probe_remote_gateway import (
    build_parser,
    probe_gateway,
)


@dataclass(frozen=True)
class _Status:
    received_monotonic: float = 1.25
    v_mps: float = 0.0
    omega_radps: float = 0.0
    battery_v: float = 27.1
    control_mode: int = 0
    fault: int = 0
    armed: bool = False
    last_sequence: int = 12


class _FakeClient:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.wait_timeout = None
        self.closed = False
        self.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.closed = True

    def wait_for_status(self, timeout_s):
        self.wait_timeout = timeout_s
        return _Status()


def test_probe_is_status_only_and_uses_ephemeral_scan_port():
    _FakeClient.instances.clear()
    payload = probe_gateway(
        "10.65.117.219",
        57720,
        "test-token",
        timeout_s=2.5,
        client_factory=_FakeClient,
    )

    client = _FakeClient.instances[0]
    assert client.kwargs == {
        "host": "10.65.117.219",
        "port": 57720,
        "token": "test-token",
        "timeout_s": 2.5,
        "scan_udp_port": 0,
    }
    assert client.wait_timeout == 2.5
    assert client.closed is True
    assert payload["armed"] is False
    assert payload["transport_probe"] == "status_only_non_arming_v1"


def test_probe_cli_has_no_arm_or_motion_options():
    parser = build_parser()
    option_strings = {
        option
        for action in parser._actions
        for option in action.option_strings
    }
    assert "--arm" not in option_strings
    assert "--v" not in option_strings
    assert "--omega" not in option_strings
