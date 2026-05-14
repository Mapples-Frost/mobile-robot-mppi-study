# -*- coding: utf-8 -*-
from __future__ import print_function

import math
import time


STUCK_LOCAL_MIN = "STUCK_LOCAL_MIN"
LOW_PROGRESS_CORRIDOR = "LOW_PROGRESS_CORRIDOR"
SPIN_TRAP = "SPIN_TRAP"
NEAR_OBSTACLE_TRAP = "NEAR_OBSTACLE_TRAP"
HARD_STOP_RECOVERY_TRAP = "HARD_STOP_RECOVERY_TRAP"


def _cfg_get(cfg, section, name, default):
    try:
        section_obj = getattr(cfg, section)
        return getattr(section_obj, name, default)
    except AttributeError:
        return default


def _clip(value, lower, upper):
    return max(lower, min(upper, value))


def _normalize(vec):
    x_value, y_value = float(vec[0]), float(vec[1])
    norm = math.hypot(x_value, y_value)
    if norm <= 1e-9:
        return None
    return (x_value / norm, y_value / norm)


class MemoryFeature(object):
    def __init__(
        self,
        feature_type,
        position,
        radius,
        strength,
        now,
        escape_direction=None,
        heading=None,
        decay=0.995,
    ):
        self.type = str(feature_type)
        self.position = (float(position[0]), float(position[1]))
        self.heading = heading
        self.escape_direction = _normalize(escape_direction or (0.0, 0.0))
        self.radius = float(radius)
        self.strength = float(strength)
        self.last_seen_time = float(now)
        self.hit_count = 1
        self.successful_escape_direction = self.escape_direction
        self.decay = float(decay)

    def to_debug_dict(self):
        return {
            "type": self.type,
            "position": self.position,
            "heading": self.heading,
            "escape_direction": self.escape_direction,
            "radius": self.radius,
            "strength": self.strength,
            "last_seen_time": self.last_seen_time,
            "hit_count": self.hit_count,
            "successful_escape_direction": self.successful_escape_direction,
            "decay": self.decay,
        }


class MppiMemoryField(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.enabled = bool(_cfg_get(cfg, "memory", "enable", True))
        self.max_features = int(_cfg_get(cfg, "memory", "max_features", 30))
        self.stuck_window_sec = float(_cfg_get(cfg, "memory", "stuck_window_sec", 3.0))
        self.stuck_pos_radius = float(_cfg_get(cfg, "memory", "stuck_pos_radius", 0.18))
        self.stuck_goal_progress_threshold = float(
            _cfg_get(cfg, "memory", "stuck_goal_progress_threshold", 0.04)
        )
        self.spin_omega_threshold = float(
            _cfg_get(cfg, "memory", "spin_omega_threshold", 0.16)
        )
        self.spin_v_threshold = float(_cfg_get(cfg, "memory", "spin_v_threshold", 0.025))
        self.feature_merge_distance = float(
            _cfg_get(cfg, "memory", "feature_merge_distance", 0.30)
        )
        self.feature_radius = float(_cfg_get(cfg, "memory", "feature_radius", 0.45))
        self.feature_strength_initial = float(
            _cfg_get(cfg, "memory", "feature_strength_initial", 1.0)
        )
        self.feature_strength_max = float(
            _cfg_get(cfg, "memory", "feature_strength_max", 5.0)
        )
        self.feature_decay = float(_cfg_get(cfg, "memory", "feature_decay", 0.995))
        self.local_min_cost_weight = float(
            _cfg_get(cfg, "memory", "local_min_cost_weight", 1.0)
        )
        self.spin_trap_cost_weight = float(
            _cfg_get(cfg, "memory", "spin_trap_cost_weight", 0.8)
        )
        self.low_progress_cost_weight = float(
            _cfg_get(cfg, "memory", "low_progress_cost_weight", 0.6)
        )
        self.escape_direction_cost_weight = float(
            _cfg_get(cfg, "memory", "escape_direction_cost_weight", 0.8)
        )
        self.temperature_boost_max = float(
            _cfg_get(cfg, "memory", "temperature_boost_max", 1.8)
        )
        self.features = []
        self.history = []
        self.last_debug = self.debug_snapshot()

    def reset(self):
        self.features = []
        self.history = []
        self.last_debug = self.debug_snapshot()

    def _weight_for_type(self, feature_type):
        if feature_type == SPIN_TRAP:
            return self.spin_trap_cost_weight
        if feature_type == LOW_PROGRESS_CORRIDOR:
            return self.low_progress_cost_weight
        if feature_type == HARD_STOP_RECOVERY_TRAP:
            return max(self.local_min_cost_weight, 1.4)
        return self.local_min_cost_weight

    def decay_features(self):
        if not self.enabled:
            return False
        kept = []
        decay_applied = False
        for feature in self.features:
            old_strength = feature.strength
            feature.strength *= feature.decay
            if feature.strength != old_strength:
                decay_applied = True
            if feature.strength >= 0.05:
                kept.append(feature)
        self.features = kept
        return decay_applied

    def nearest_feature(self, state):
        if not self.enabled or not self.features or state is None:
            return None, None
        x_value, y_value = float(state[0]), float(state[1])
        best_feature = None
        best_distance = None
        for feature in self.features:
            distance = math.hypot(
                x_value - feature.position[0],
                y_value - feature.position[1],
            )
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_feature = feature
        return best_feature, best_distance

    def add_or_update_feature(self, feature_type, position, escape_direction, now):
        if not self.enabled:
            return None, False
        position = (float(position[0]), float(position[1]))
        best_feature = None
        best_distance = None
        for feature in self.features:
            if feature.type != feature_type:
                continue
            distance = math.hypot(
                position[0] - feature.position[0],
                position[1] - feature.position[1],
            )
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_feature = feature

        added = False
        if best_feature is not None and best_distance <= self.feature_merge_distance:
            alpha = 1.0 / float(best_feature.hit_count + 1)
            best_feature.position = (
                (1.0 - alpha) * best_feature.position[0] + alpha * position[0],
                (1.0 - alpha) * best_feature.position[1] + alpha * position[1],
            )
            new_escape = _normalize(escape_direction or (0.0, 0.0))
            if new_escape is not None:
                best_feature.escape_direction = new_escape
                best_feature.successful_escape_direction = new_escape
            best_feature.hit_count += 1
            best_feature.last_seen_time = float(now)
            best_feature.strength = min(
                self.feature_strength_max,
                best_feature.strength + 0.35 * self.feature_strength_initial,
            )
            feature = best_feature
        else:
            feature = MemoryFeature(
                feature_type=feature_type,
                position=position,
                radius=self.feature_radius,
                strength=self.feature_strength_initial,
                now=now,
                escape_direction=escape_direction,
                decay=self.feature_decay,
            )
            self.features.append(feature)
            added = True

        self._trim_features()
        return feature, added

    def _trim_features(self):
        max_features = max(1, int(self.max_features))
        if len(self.features) <= max_features:
            return
        self.features.sort(
            key=lambda feature: (float(feature.strength), int(feature.hit_count))
        )
        self.features = self.features[-max_features:]

    def update(
        self,
        state,
        goal_distance,
        control,
        min_front_range=None,
        avoidance_state="CLEAR",
        now=None,
    ):
        if now is None:
            now = time.time()
        debug = self.debug_snapshot(state)
        if not self.enabled or state is None:
            self.last_debug = debug
            return debug

        x_value, y_value, theta = float(state[0]), float(state[1]), float(state[2])
        v_cmd = float(control[0]) if control is not None else 0.0
        omega_cmd = float(control[1]) if control is not None else 0.0
        entry = {
            "time": float(now),
            "x": x_value,
            "y": y_value,
            "theta": theta,
            "goal_distance": float(goal_distance),
            "v_cmd": v_cmd,
            "omega_cmd": omega_cmd,
            "min_front_range": min_front_range,
            "avoidance_state": str(avoidance_state),
        }
        self.history.append(entry)
        cutoff = float(now) - max(0.5, self.stuck_window_sec)
        self.history = [item for item in self.history if item["time"] >= cutoff]
        decay_applied = self.decay_features()

        if len(self.history) < 2:
            debug = self.debug_snapshot(state)
            debug["memory_decay_applied"] = decay_applied
            self.last_debug = debug
            return debug

        first = self.history[0]
        last = self.history[-1]
        position_span = 0.0
        for item in self.history:
            position_span = max(
                position_span,
                math.hypot(item["x"] - first["x"], item["y"] - first["y"]),
            )
        displacement = math.hypot(last["x"] - first["x"], last["y"] - first["y"])
        goal_progress = float(first["goal_distance"]) - float(last["goal_distance"])
        mean_abs_omega = sum(abs(item["omega_cmd"]) for item in self.history) / float(
            len(self.history)
        )
        mean_abs_v = sum(abs(item["v_cmd"]) for item in self.history) / float(
            len(self.history)
        )
        elapsed = max(0.0, last["time"] - first["time"])

        escape_direction = _normalize(
            (
                math.cos(theta),
                math.sin(theta),
            )
        )
        if goal_progress > 0.01 and displacement > 1e-6:
            escape_direction = _normalize((last["x"] - first["x"], last["y"] - first["y"]))

        added_types = []
        if (
            elapsed >= 0.8 * self.stuck_window_sec
            and position_span <= self.stuck_pos_radius
            and goal_progress <= self.stuck_goal_progress_threshold
        ):
            _, added = self.add_or_update_feature(
                STUCK_LOCAL_MIN,
                (x_value, y_value),
                escape_direction,
                now,
            )
            if added:
                added_types.append(STUCK_LOCAL_MIN)

        if (
            elapsed >= 0.8 * self.stuck_window_sec
            and mean_abs_v > self.spin_v_threshold
            and displacement <= self.stuck_pos_radius
            and goal_progress <= self.stuck_goal_progress_threshold
        ):
            _, added = self.add_or_update_feature(
                LOW_PROGRESS_CORRIDOR,
                (x_value, y_value),
                escape_direction,
                now,
            )
            if added:
                added_types.append(LOW_PROGRESS_CORRIDOR)

        if (
            elapsed >= 0.8 * self.stuck_window_sec
            and mean_abs_omega >= self.spin_omega_threshold
            and mean_abs_v <= self.spin_v_threshold
            and goal_progress <= self.stuck_goal_progress_threshold
        ):
            _, added = self.add_or_update_feature(
                SPIN_TRAP,
                (x_value, y_value),
                escape_direction,
                now,
            )
            if added:
                added_types.append(SPIN_TRAP)

        if min_front_range is not None:
            try:
                front_value = float(min_front_range)
            except (TypeError, ValueError):
                front_value = None
            if (
                front_value is not None
                and front_value < 0.40
                and elapsed >= 0.8 * self.stuck_window_sec
                and goal_progress <= self.stuck_goal_progress_threshold
            ):
                _, added = self.add_or_update_feature(
                    NEAR_OBSTACLE_TRAP,
                    (x_value, y_value),
                    escape_direction,
                    now,
                )
                if added:
                    added_types.append(NEAR_OBSTACLE_TRAP)

        if (
            str(avoidance_state) == "HARD_STOP_RECOVERY"
            and elapsed >= 0.8 * self.stuck_window_sec
        ):
            _, added = self.add_or_update_feature(
                HARD_STOP_RECOVERY_TRAP,
                (x_value, y_value),
                escape_direction,
                now,
            )
            if added:
                added_types.append(HARD_STOP_RECOVERY_TRAP)

        debug = self.debug_snapshot(state)
        debug["stuck_feature_added"] = bool(added_types)
        debug["memory_added_types"] = ",".join(added_types) if added_types else "none"
        debug["memory_decay_applied"] = decay_applied
        self.last_debug = debug
        return debug

    def cost_for_state(self, state):
        if not self.enabled or not self.features or state is None:
            return 0.0
        x_value, y_value = float(state[0]), float(state[1])
        total = 0.0
        for feature in self.features:
            dx = x_value - feature.position[0]
            dy = y_value - feature.position[1]
            distance = math.hypot(dx, dy)
            if distance >= feature.radius:
                continue
            influence = (1.0 - distance / max(feature.radius, 1e-6)) ** 2
            total += self._weight_for_type(feature.type) * feature.strength * influence
            if feature.escape_direction is not None:
                projected_progress = (
                    dx * feature.escape_direction[0] + dy * feature.escape_direction[1]
                )
                if projected_progress < 0.0:
                    total += (
                        self.escape_direction_cost_weight
                        * feature.strength
                        * influence
                    )
        return total

    def cost_for_trajectory(self, trajectory, step_stride=3):
        if not self.enabled or not self.features or not trajectory:
            return 0.0
        stride = max(1, int(step_stride))
        total = 0.0
        count = 0
        for index, state in enumerate(trajectory):
            if index != len(trajectory) - 1 and index % stride != 0:
                continue
            total += self.cost_for_state(state)
            count += 1
        if count <= 0:
            return 0.0
        return total / float(count)

    def temperature_scale(self, state, stuck_trap_active=False):
        if not self.enabled:
            return 1.0
        feature, distance = self.nearest_feature(state)
        scale = 1.0
        if feature is not None and distance is not None and distance < feature.radius:
            influence = 1.0 - distance / max(feature.radius, 1e-6)
            scale = 1.0 + (self.temperature_boost_max - 1.0) * _clip(influence, 0.0, 1.0)
        if stuck_trap_active:
            scale = max(scale, min(self.temperature_boost_max, 1.25))
        return _clip(scale, 1.0, max(1.0, self.temperature_boost_max))

    def debug_snapshot(self, state=None):
        nearest, distance = self.nearest_feature(state)
        nearest_type = "none"
        nearest_strength = 0.0
        nearest_hit_count = 0
        nearest_escape = None
        if nearest is not None:
            nearest_type = nearest.type
            nearest_strength = nearest.strength
            nearest_hit_count = nearest.hit_count
            nearest_escape = nearest.escape_direction
        return {
            "memory_enabled": bool(self.enabled),
            "memory_feature_count": len(self.features),
            "memory_nearest_type": nearest_type,
            "memory_nearest_distance": distance,
            "memory_nearest_strength": nearest_strength,
            "memory_escape_direction": nearest_escape,
            "memory_temperature_scale": self.temperature_scale(state),
            "memory_feature_hit_count": nearest_hit_count,
            "stuck_feature_added": False,
            "memory_decay_applied": False,
        }


def _run_basic_tests():
    class Section(object):
        pass

    class Cfg(object):
        pass

    cfg = Cfg()
    cfg.memory = Section()
    cfg.memory.enable = True
    field = MppiMemoryField(cfg)
    now = time.time()
    for idx in range(8):
        field.update(
            state=(0.0 + 0.005 * idx, 0.0, 0.0),
            goal_distance=3.0 - 0.001 * idx,
            control=(0.02, 0.20),
            min_front_range=0.5,
            avoidance_state="CLEAR",
            now=now + 0.5 * idx,
        )
    if len(field.features) <= 0:
        raise AssertionError("stuck history should add memory features")
    if field.cost_for_state((0.0, 0.0, 0.0)) <= 0.0:
        raise AssertionError("near feature memory cost should be positive")
    feature = field.features[0]
    feature.escape_direction = (1.0, 0.0)
    forward_cost = field.cost_for_state((0.20, 0.0, 0.0))
    backward_cost = field.cost_for_state((-0.20, 0.0, 0.0))
    if backward_cost <= forward_cost:
        raise AssertionError("opposite escape direction should cost more")
    field.max_features = 2
    for idx in range(5):
        field.add_or_update_feature(
            STUCK_LOCAL_MIN,
            (float(idx), 0.0),
            (1.0, 0.0),
            now + idx,
        )
    if len(field.features) > 2:
        raise AssertionError("feature cap should trim weak features")
    cfg.memory.enable = False
    disabled = MppiMemoryField(cfg)
    disabled.add_or_update_feature(STUCK_LOCAL_MIN, (0, 0), (1, 0), now)
    if disabled.cost_for_state((0.0, 0.0, 0.0)) != 0.0:
        raise AssertionError("disabled memory should have zero cost")
    print("mppi_memory_field basic tests passed")


if __name__ == "__main__":
    _run_basic_tests()
