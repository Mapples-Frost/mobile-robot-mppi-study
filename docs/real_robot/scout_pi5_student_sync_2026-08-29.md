# Scout Mini / Raspberry Pi 5 student-work synchronization

Date: 2026-08-29

## Scope and source

This note synchronizes the useful real-robot progress found in
`D:/Projects/mobile-robot-mppi-study-main/mobile-robot-mppi-study-main` with
the existing Scout Mini / Raspberry Pi 5 deployment stack rooted at commit
`3c8ad38` (`real-robot-control-baseline-3c8ad38`). The source directory is not
a Git checkout, so statements below distinguish source-code inspection from
newly reproduced evidence.

The active physical platform is the new vehicle:

- AgileX Scout Mini differential-drive chassis;
- Raspberry Pi 5 running 64-bit Debian / Raspberry Pi OS;
- SocketCAN interface `can0`, configured for 500000 bit/s;
- PC-to-Pi control transport on TCP port 57720;
- optional Livox scan transport on UDP port 57721/57701 depending on launch
  configuration.

This is not the earlier ROS 1 vehicle. No ROS 1 deployment assumptions are
used by this synchronization.

## Progress recovered from the student directory

The student record dated 2026-08-29 reports:

1. `can0` was brought up and a separate `test_move2.py` chassis motion test
   succeeded;
2. a PC/Pi `RMP1` framed TCP handshake was exercised, including HELLO,
   STATUS, COMMAND and STOP frames;
3. a status-only prototype gateway listened on TCP 57720 and printed incoming
   commands;
4. a PC-side proportional-controller prototype connected to that gateway;
5. an experimental SLAM-map/MPPI v9 simulation and a separate BC policy
   deployment package were produced.

The first two items are reports in the supplied notes, not evidence reproduced
in this commit. The original `test_move2.py` and raw terminal logs were not in
the supplied directory.

## Audit result

The supplied prototypes are not a safe replacement for the existing physical
deployment stack:

- `_gw4.py` and `_simple_gateway.py` acknowledge commands but do not send
  them to the guarded CAN gateway;
- `simple_pc_controller.py` arms automatically and dead-reckons commanded
  motion instead of consuming trusted chassis state;
- `pc_driver.py` reads six bytes for a nine-byte `!4sBI` frame header and has
  inconsistent SSH result unpacking/launch commands;
- the BC controller evaluates a simulated kinematic state and static scene;
  it is not connected to localization, Livox perception or the CAN gateway;
- `dynamic_mppi_sim_v9.py` is an isolated prototype with hard-coded local
  paths and tuple-shape inconsistencies around map obstacles.

Those files are therefore not wired into actuation. The existing guarded
stack remains authoritative:

- `src/mobile_robot_mppi/real_robot/remote_transport.py` for framed transport,
  bounded buffering, status freshness and execution odometry;
- `src/mobile_robot_mppi/real_robot/scout_can.py` for zero-only and guarded CAN
  operation;
- `deploy/raspberry_pi5_scout/run_remote_pi_gateway.py` for the Pi gateway;
- `deploy/raspberry_pi5_scout/run_remote_cuda_full.py` for the PC planner;
- `tests/real_robot/` for physical-deployment safety contracts.

## Integrated change

`deploy/raspberry_pi5_scout/probe_remote_gateway.py` is a non-arming status
probe derived from the useful part of the student TCP experiment. It reuses
the production `RemoteDeploymentClient`, so the frame size, token contract,
status decoding and shutdown behavior have one implementation. The probe has
no command-sending or arm option.

The student algorithm artifacts are recorded under
`contrib/student_scout_mini/` as non-production provenance. They do not alter
the paper method, planning costs, control authority or physical actuation.

## Current real-car gate

The synchronized state is:

| Gate | State | Evidence |
| --- | --- | --- |
| Pi identity/runtime | observed | Pi 5, Debian aarch64, Python 3.13.5 |
| SocketCAN bring-up | student-reported | `can0`, 500000 bit/s |
| Direct chassis motion | student-reported | missing `test_move2.py` log |
| PC/Pi framed status | student-reported + code-audited | RMP1/TCP 57720 |
| Production guarded gateway | implemented in baseline | offline tests only in this sync |
| Livox end-to-end freshness | not requalified | requires lab evidence |
| Planner shadow run | not requalified | requires non-arming run |
| Armed MPPI run | not authorized by this commit | requires all preceding gates |

## Next lab sequence

1. Run the status probe with the current Pi address and a freshly generated
   token. This does not arm or send a motion command.
2. Run the existing zero-CAN and watchdog tests with wheels lifted or chassis
   mechanically restrained.
3. Capture CAN state, Pi gateway log, PC transport diagnostics and Livox
   freshness into one timestamped run directory.
4. Run the full planner in non-arming/shadow mode and verify that proposed,
   guarded and applied controls are all logged.
5. Only after the above gates pass, conduct a bounded low-speed armed test with
   an independent emergency stop operator.

No claim in this document upgrades the student prototype to production or
asserts that an armed end-to-end MPPI experiment has passed.
