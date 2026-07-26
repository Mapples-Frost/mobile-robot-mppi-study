"""Deterministic MJCF generation for the differential-drive research plant."""

import hashlib
import json
import math
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


def _rgb(values, label):
    values = tuple(float(value) for value in values)
    if len(values) != 3 or any(value < 0.0 or value > 1.0 for value in values):
        raise ValueError("%s must contain three values in [0, 1]" % label)
    return " ".join(_f(value) for value in values)


def _geom_for_obstacle(index, obstacle, local=False, geom_name=None):
    kind = str(obstacle.get("type", "cylinder"))
    if kind == "segment":
        start = tuple(float(value) for value in obstacle["start"])
        end = tuple(float(value) for value in obstacle["end"])
        if len(start) != 2 or len(end) != 2:
            raise ValueError("segment start and end must contain x and y")
        delta_x = end[0] - start[0]
        delta_y = end[1] - start[1]
        length = math.hypot(delta_x, delta_y)
        thickness = float(obstacle.get("thickness", 0.20))
        if length <= 0.0 or thickness <= 0.0:
            raise ValueError("segment length and thickness must be positive")
        position = (
            0.5 * (start[0] + end[0]),
            0.5 * (start[1] + end[1]),
        )
        size = (0.5 * length, 0.5 * thickness)
        yaw = math.atan2(delta_y, delta_x)
        kind = "box"
    else:
        position = (
            obstacle.get("offset", (0.0, 0.0))
            if local
            else obstacle.get("position", (0.0, 0.0))
        )
        size = obstacle.get("size", (0.25, 0.25))
        yaw = float(obstacle.get("yaw", 0.0))
    name = geom_name or "obstacle_%d" % index
    height = float(obstacle.get("height", 0.5))
    if kind == "box":
        return (
            '<geom name="%s" type="box" pos="%s %s %s" '
            'size="%s %s %s" euler="0 0 %s" rgba="%s" group="1"/>'
            % (name, _f(position[0]), _f(position[1]), _f(height / 2.0),
               _f(size[0]), _f(size[1]), _f(height / 2.0), _f(yaw),
               _rgba(obstacle))
        )
    radius = float(obstacle.get("radius", 0.25))
    return (
        '<geom name="%s" type="cylinder" pos="%s %s %s" '
        'size="%s %s" rgba="%s" group="1"/>'
        % (name, _f(position[0]), _f(position[1]), _f(height / 2.0),
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
            _dynamic_obstacle_geoms(index, obstacle),
        )
    )


def _dynamic_obstacle_geoms(index, obstacle):
    """Render one irregular moving obstacle as a rigid multi-geom body."""

    primary = dict(obstacle)
    # World yaw belongs to the mocap body.  Geom yaws are relative to it.
    primary["yaw"] = 0.0
    output = [_geom_for_obstacle(index, primary, local=True)]
    parts = obstacle.get("parts", ())
    if not isinstance(parts, Sequence) or isinstance(parts, (str, bytes)):
        raise TypeError("dynamic obstacle parts must be a sequence")
    for part_index, part in enumerate(parts):
        if not isinstance(part, Mapping):
            raise TypeError("dynamic obstacle part must be a mapping")
        shape = dict(part)
        shape.setdefault("type", obstacle.get("type", "box"))
        shape.setdefault("height", obstacle.get("height", 0.5))
        shape.setdefault("rgba", obstacle.get("rgba", (0.65, 0.32, 0.28, 1.0)))
        output.append(
            _geom_for_obstacle(
                index,
                shape,
                local=True,
                geom_name="obstacle_%d_part_%d" % (index, part_index),
            )
        )
    return "\n      ".join(output)


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
    visual = scene.get("visual", {})
    if not isinstance(visual, Mapping):
        raise TypeError("scene visual must be a mapping")
    floor_rgba = _rgba(
        {"rgba": visual.get("floor_rgba", (0.66, 0.67, 0.65, 1.0))}
    )
    headlight_ambient = _rgb(
        visual.get("headlight_ambient", (0.45, 0.45, 0.45)),
        "headlight ambient",
    )
    headlight_diffuse = _rgb(
        visual.get("headlight_diffuse", (0.70, 0.70, 0.70)),
        "headlight diffuse",
    )
    sun_diffuse = _rgb(
        visual.get("sun_diffuse", (0.70, 0.70, 0.70)),
        "sun diffuse",
    )
    if visual:
        sun_style = (
            ' diffuse="%s" specular="0.10 0.10 0.12"' % sun_diffuse
        )
        fill_light = (
            '    <light name="fill" pos="-4 -3 4" dir="0.6 0.4 -1"\n'
            '           directional="true" diffuse="0.22 0.28 0.35" '
            'specular="0.08 0.08 0.10"/>\n'
        )
    else:
        sun_style = ""
        fill_light = ""
    checker = visual.get("floor_checker")
    if checker is None:
        visual_assets = ""
        floor_material = ""
    else:
        if not isinstance(checker, Mapping):
            raise TypeError("floor_checker must be a mapping")
        rgb1 = _rgb(checker.get("rgb1", (0.16, 0.19, 0.23)), "checker rgb1")
        rgb2 = _rgb(checker.get("rgb2", (0.22, 0.26, 0.31)), "checker rgb2")
        repeat = checker.get("repeat", (10.0, 6.0))
        if len(repeat) != 2 or any(float(value) <= 0.0 for value in repeat):
            raise ValueError("checker repeat must contain two positive values")
        visual_assets = (
            '  <asset>\n'
            '    <texture name="research_floor_grid" type="2d" builtin="checker" '
            'rgb1="%s" rgb2="%s" width="512" height="512"/>\n'
            '    <material name="research_floor_material" '
            'texture="research_floor_grid" texuniform="true" '
            'texrepeat="%s %s" reflectance="%s"/>\n'
            '  </asset>\n'
            % (
                rgb1,
                rgb2,
                _f(repeat[0]),
                _f(repeat[1]),
                _f(checker.get("reflectance", 0.08)),
            )
        )
        floor_material = ' material="research_floor_material"'
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
  <visual><headlight ambient="{headlight_ambient}" diffuse="{headlight_diffuse}"/></visual>
  <default>
    <geom condim="4" solref="{solref}" solimp="{solimp}" contype="1" conaffinity="1"/>
    <joint damping="0.01" armature="0.001"/>
  </default>
{visual_assets}  <worldbody>
    <light name="sun" pos="0 0 6" dir="0 0 -1" directional="true"{sun_style}/>
{fill_light}    <geom name="floor" type="plane" size="10 10 0.1" friction="{floor_friction}"
          rgba="{floor_rgba}"{floor_material} group="1"/>
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
        visual_assets=visual_assets, floor_rgba=floor_rgba,
        floor_material=floor_material,
        headlight_ambient=headlight_ambient,
        headlight_diffuse=headlight_diffuse,
        sun_diffuse=sun_diffuse,
        sun_style=sun_style,
        fill_light=fill_light,
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
