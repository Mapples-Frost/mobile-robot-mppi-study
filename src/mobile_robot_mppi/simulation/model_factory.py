"""Deterministic MJCF generation for the differential-drive research plant."""

import hashlib
import json
from typing import Mapping, Sequence, Tuple


def _f(value):
    return "%.9g" % float(value)


def _rgba(obstacle):
    values = tuple(
        float(value)
        for value in obstacle.get("rgba", (0.65, 0.32, 0.28, 1.0))
    )
    if len(values) != 4 or any(value < 0.0 or value > 1.0 for value in values):
        raise ValueError("obstacle rgba must contain four values in [0, 1]")
    return " ".join(_f(value) for value in values)


def _geom_for_obstacle(index, obstacle, local=False):
    kind = str(obstacle.get("type", "cylinder"))
    position = (0.0, 0.0) if local else obstacle.get("position", (0.0, 0.0))
    height = float(obstacle.get("height", 0.5))
    if kind == "box":
        size = obstacle.get("size", (0.25, 0.25))
        yaw = float(obstacle.get("yaw", 0.0))
        return (
            '<geom name="obstacle_%d" type="box" pos="%s %s %s" '
            'size="%s %s %s" euler="0 0 %s" rgba="%s" group="1"/>'
            % (index, _f(position[0]), _f(position[1]), _f(height / 2.0),
               _f(size[0]), _f(size[1]), _f(height / 2.0), _f(yaw),
               _rgba(obstacle))
        )
    radius = float(obstacle.get("radius", 0.25))
    return (
        '<geom name="obstacle_%d" type="cylinder" pos="%s %s %s" '
        'size="%s %s" rgba="%s" group="1"/>'
        % (index, _f(position[0]), _f(position[1]), _f(height / 2.0),
           _f(radius), _f(height / 2.0), _rgba(obstacle))
    )


def _obstacle_element(index, obstacle):
    """Render a static world geom or a prescribed mocap obstacle body.

    Dynamic obstacles remain ordinary MuJoCo collision/raycast geometry.  A
    mocap body only supplies their deterministic world pose; the planner still
    receives them exclusively through the simulated LaserScan.
    """

    motion = obstacle.get("motion")
    if motion is None:
        return _geom_for_obstacle(index, obstacle)
    if not isinstance(motion, Mapping):
        raise TypeError("obstacle motion must be a mapping")
    motion_type = str(motion.get("type", "linear_ping_pong"))
    if motion_type not in (
        "linear_ping_pong",
        "recurrent_semimarkov_v3",
    ):
        raise ValueError("unknown obstacle motion type: %s" % motion_type)
    position = (
        motion.get("start", obstacle.get("position", (0.0, 0.0)))
        if motion_type == "linear_ping_pong"
        else obstacle.get("position", (0.0, 0.0))
    )
    if len(position) != 2:
        raise ValueError("dynamic obstacle start must contain x and y")
    return (
        '<body name="dynamic_obstacle_%d" mocap="true" pos="%s %s 0">\n'
        '      %s\n'
        '    </body>'
        % (
            index,
            _f(position[0]),
            _f(position[1]),
            _geom_for_obstacle(index, obstacle, local=True),
        )
    )


def build_diff_drive_mjcf(config: Mapping[str, object], scene: Mapping[str, object]):
    robot = config.get("robot", {})
    physics = config.get("physics", {})
    contact = config.get("contact", {})
    actuator = config.get("actuator", {})
    wheel_radius = float(robot.get("wheel_radius", 0.08))
    wheel_width = float(robot.get("wheel_width", 0.035))
    track_width = float(robot.get("track_width", 0.32))
    chassis_size = robot.get("chassis_half_size", (0.19, 0.14, 0.045))
    chassis_mass = float(robot.get("chassis_mass", 9.0))
    inertia = robot.get("chassis_inertia", (0.08, 0.12, 0.16))
    wheel_mass = float(robot.get("wheel_mass", 0.32))
    base_height = wheel_radius + float(chassis_size[2]) * 0.85
    timestep = float(physics.get("timestep", 0.002))
    integrator = str(physics.get("integrator", "implicitfast"))
    solver = str(physics.get("solver", "Newton"))
    iterations = int(physics.get("iterations", 60))
    tolerance = float(physics.get("tolerance", 1e-10))
    wheel_friction = contact.get("wheel_friction", (1.2, 0.01, 0.002))
    floor_friction = contact.get("floor_friction", (1.0, 0.005, 0.0005))
    caster_friction = contact.get("caster_friction", (0.05, 0.001, 0.0001))
    solref = str(contact.get("solref", "0.01 1"))
    solimp = str(contact.get("solimp", "0.95 0.99 0.001"))
    profile = str(actuator.get("profile", "torque_pi"))
    torque_limit = float(actuator.get("torque_limit", 2.2))
    velocity_gain = float(actuator.get("velocity_gain", 8.0))
    obstacles = "\n".join(
        "    " + _obstacle_element(index, value)
        for index, value in enumerate(scene.get("obstacles", ()))
    )
    if profile == "ideal_velocity":
        actuators = (
            '<velocity name="left_drive" joint="left_wheel_joint" kv="%s" '
            'ctrllimited="true" ctrlrange="-40 40" forcelimited="true" forcerange="-%s %s"/>\n'
            '<velocity name="right_drive" joint="right_wheel_joint" kv="%s" '
            'ctrllimited="true" ctrlrange="-40 40" forcelimited="true" forcerange="-%s %s"/>'
            % (_f(velocity_gain), _f(torque_limit), _f(torque_limit),
               _f(velocity_gain), _f(torque_limit), _f(torque_limit))
        )
    elif profile == "torque_pi":
        actuators = (
            '<motor name="left_drive" joint="left_wheel_joint" gear="1" '
            'ctrllimited="true" ctrlrange="-%s %s"/>\n'
            '<motor name="right_drive" joint="right_wheel_joint" gear="1" '
            'ctrllimited="true" ctrlrange="-%s %s"/>'
            % (_f(torque_limit), _f(torque_limit), _f(torque_limit), _f(torque_limit))
        )
    else:
        raise ValueError("unknown actuator profile: %s" % profile)
    xml = '''<mujoco model="mppi_diff_drive">
  <compiler angle="radian" inertiafromgeom="false"/>
  <option timestep="{timestep}" integrator="{integrator}" solver="{solver}"
          iterations="{iterations}" tolerance="{tolerance}" gravity="0 0 -9.81"/>
  <visual><headlight ambient="0.45 0.45 0.45" diffuse="0.7 0.7 0.7"/></visual>
  <default>
    <geom condim="4" solref="{solref}" solimp="{solimp}" contype="1" conaffinity="1"/>
    <joint damping="0.01" armature="0.001"/>
  </default>
  <worldbody>
    <light name="sun" pos="0 0 6" dir="0 0 -1" directional="true"/>
    <geom name="floor" type="plane" size="10 10 0.1" friction="{floor_friction}" rgba="0.66 0.67 0.65 1" group="1"/>
{obstacles}
    <body name="base" pos="0 0 {base_height}">
      <freejoint name="base_free"/>
      <inertial pos="0 0 0" mass="{chassis_mass}" diaginertia="{inertia}"/>
      <geom name="chassis" type="box" size="{chassis_size}" rgba="0.18 0.40 0.65 1" group="0"/>
      <site name="imu_site" pos="0 0 0.06" size="0.008"/>
      <site name="lidar_site" pos="0.10 0 0.10" size="0.012"/>
      <body name="left_wheel" pos="0 {half_track} {wheel_z}">
        <joint name="left_wheel_joint" type="hinge" axis="0 1 0"/>
        <inertial pos="0 0 0" mass="{wheel_mass}" diaginertia="0.001 0.001 0.001"/>
        <geom name="left_wheel_geom" type="cylinder" size="{wheel_radius} {half_width}"
              quat="0.707106781 0.707106781 0 0" friction="{wheel_friction}" rgba="0.12 0.12 0.12 1" group="0"/>
      </body>
      <body name="right_wheel" pos="0 -{half_track} {wheel_z}">
        <joint name="right_wheel_joint" type="hinge" axis="0 1 0"/>
        <inertial pos="0 0 0" mass="{wheel_mass}" diaginertia="0.001 0.001 0.001"/>
        <geom name="right_wheel_geom" type="cylinder" size="{wheel_radius} {half_width}"
              quat="0.707106781 0.707106781 0 0" friction="{wheel_friction}" rgba="0.12 0.12 0.12 1" group="0"/>
      </body>
      <geom name="front_caster" type="sphere" pos="0.14 0 {caster_z}" size="0.022"
            friction="{caster_friction}" rgba="0.2 0.2 0.2 1" group="0"/>
      <geom name="rear_caster" type="sphere" pos="-0.14 0 {caster_z}" size="0.022"
            friction="{caster_friction}" rgba="0.2 0.2 0.2 1" group="0"/>
    </body>
  </worldbody>
  <actuator>{actuators}</actuator>
  <sensor>
    <jointpos name="left_encoder" joint="left_wheel_joint"/>
    <jointpos name="right_encoder" joint="right_wheel_joint"/>
    <jointvel name="left_wheel_velocity" joint="left_wheel_joint"/>
    <jointvel name="right_wheel_velocity" joint="right_wheel_joint"/>
    <actuatorfrc name="left_actuator_force" actuator="left_drive"/>
    <actuatorfrc name="right_actuator_force" actuator="right_drive"/>
    <gyro name="imu_gyro" site="imu_site"/>
    <accelerometer name="imu_accelerometer" site="imu_site"/>
  </sensor>
</mujoco>'''.format(
        timestep=_f(timestep), integrator=integrator, solver=solver,
        iterations=iterations, tolerance=_f(tolerance), solref=solref, solimp=solimp,
        floor_friction=" ".join(_f(v) for v in floor_friction), obstacles=obstacles,
        base_height=_f(base_height), chassis_mass=_f(chassis_mass),
        inertia=" ".join(_f(v) for v in inertia),
        chassis_size=" ".join(_f(v) for v in chassis_size),
        # A small preload avoids intermittent zero-contact frames at the exact
        # analytical touching height without making the chassis scrape.
        half_track=_f(track_width / 2.0), wheel_z=_f(-base_height + 0.97 * wheel_radius),
        wheel_mass=_f(wheel_mass), wheel_radius=_f(wheel_radius),
        half_width=_f(wheel_width / 2.0),
        wheel_friction=" ".join(_f(v) for v in wheel_friction),
        caster_z=_f(-base_height + 0.020),
        caster_friction=" ".join(_f(v) for v in caster_friction),
        actuators=actuators,
    )
    return xml


def model_hash(xml: str) -> str:
    return hashlib.sha256(xml.encode("utf-8")).hexdigest()
