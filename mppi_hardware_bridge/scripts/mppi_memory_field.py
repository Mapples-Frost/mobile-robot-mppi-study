# -*- coding: utf-8 -*-
from __future__ import print_function

import math
import time


STUCK_LOCAL_MIN = "STUCK_LOCAL_MIN"
LOW_PROGRESS_CORRIDOR = "LOW_PROGRESS_CORRIDOR"
SPIN_TRAP = "SPIN_TRAP"
NEAR_OBSTACLE_TRAP = "NEAR_OBSTACLE_TRAP"
HARD_STOP_RECOVERY_TRAP = "HARD_STOP_RECOVERY_TRAP"
HIGH_CURVATURE_REGION = "HIGH_CURVATURE_REGION"

FEATURE_TYPES = (
    STUCK_LOCAL_MIN,
    LOW_PROGRESS_CORRIDOR,
    SPIN_TRAP,
    NEAR_OBSTACLE_TRAP,
    HARD_STOP_RECOVERY_TRAP,
    HIGH_CURVATURE_REGION,
)


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
        self.feature_type = self.type
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
        self.near_obstacle_cost_weight = float(
            _cfg_get(cfg, "memory", "near_obstacle_cost_weight", self.local_min_cost_weight * 1.15)
        )
        self.hard_stop_cost_weight = float(
            _cfg_get(cfg, "memory", "hard_stop_cost_weight", self.local_min_cost_weight * 1.35)
        )
        self.high_curvature_cost_weight = float(
            _cfg_get(cfg, "memory", "high_curvature_cost_weight", 0.45)
        )
        self.escape_direction_cost_weight = float(
            _cfg_get(cfg, "memory", "escape_direction_cost_weight", 0.8)
        )
        self.temperature_boost_max = float(
            _cfg_get(cfg, "memory", "temperature_boost_max", 1.8)
        )
        self.near_obstacle_range_threshold = float(
            _cfg_get(cfg, "memory", "near_obstacle_range_threshold", 0.40)
        )
        self.high_curvature_omega_threshold = float(
            _cfg_get(cfg, "memory", "high_curvature_omega_threshold", 0.35)
        )
        self.high_curvature_sign_flip_threshold = int(
            _cfg_get(cfg, "memory", "high_curvature_sign_flip_threshold", 3)
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
        if feature_type == NEAR_OBSTACLE_TRAP:
            return self.near_obstacle_cost_weight
        if feature_type == HARD_STOP_RECOVERY_TRAP:
            return self.hard_stop_cost_weight
        if feature_type == HIGH_CURVATURE_REGION:
            return self.high_curvature_cost_weight
        return self.local_min_cost_weight

    def _feature_priority(self, feature_type):
        priorities = {
            HARD_STOP_RECOVERY_TRAP: 6,
            NEAR_OBSTACLE_TRAP: 5,
            SPIN_TRAP: 4,
            STUCK_LOCAL_MIN: 3,
            LOW_PROGRESS_CORRIDOR: 2,
            HIGH_CURVATURE_REGION: 1,
        }
        return priorities.get(feature_type, 0)

    def _combine_feature_type(self, old_type, new_type):
        if self._feature_priority(new_type) > self._feature_priority(old_type):
            return new_type
        return old_type

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
        best_same_type = None
        best_same_type_distance = None
        best_any_feature = None
        best_any_distance = None
        for feature in self.features:
            distance = math.hypot(
                position[0] - feature.position[0],
                position[1] - feature.position[1],
            )
            if best_any_distance is None or distance < best_any_distance:
                best_any_distance = distance
                best_any_feature = feature
            if feature.type == feature_type and (
                best_same_type_distance is None or distance < best_same_type_distance
            ):
                best_same_type_distance = distance
                best_same_type = feature

        added = False
        best_feature = best_same_type
        best_distance = best_same_type_distance
        if best_feature is None:
            best_feature = best_any_feature
            best_distance = best_any_distance

        if best_feature is not None and best_distance <= self.feature_merge_distance:
            alpha = 1.0 / float(best_feature.hit_count + 1)
            best_feature.position = (
                (1.0 - alpha) * best_feature.position[0] + alpha * position[0],
                (1.0 - alpha) * best_feature.position[1] + alpha * position[1],
            )
            best_feature.type = self._combine_feature_type(best_feature.type, feature_type)
            best_feature.feature_type = best_feature.type
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
            best_feature.radius = max(best_feature.radius, self.feature_radius)
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
            key=lambda feature: (
                float(feature.strength),
                float(feature.last_seen_time),
                int(feature.hit_count),
            )
        )
        self.features = self.features[-max_features:]

    def _omega_sign_flips(self, history):
        signs = []
        for item in history:
            omega_value = float(item.get("omega_cmd", 0.0))
            if abs(omega_value) < self.high_curvature_omega_threshold:
                continue
            signs.append(1 if omega_value > 0.0 else -1)
        flips = 0
        previous = None
        for sign in signs:
            if previous is not None and sign != previous:
                flips += 1
            previous = sign
        return flips

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
        avoidance_upper = str(avoidance_state or "CLEAR").upper()
        omega_sign_flips = self._omega_sign_flips(self.history)

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
            and displacement > 0.55 * self.stuck_pos_radius
            and position_span <= 2.20 * self.stuck_pos_radius
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
                and front_value < self.near_obstacle_range_threshold
                and elapsed >= 0.8 * self.stuck_window_sec
                and goal_progress <= self.stuck_goal_progress_threshold
                and avoidance_upper in (
                    "CREEP_ESCAPE",
                    "FRONT_SOFT_BLOCK",
                    "FRONT_OBSTACLE_SLOW",
                    "SIDE_OBSTACLE_SOFT",
                    "APPROACH_SLOW",
                )
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
            avoidance_upper == "HARD_STOP_RECOVERY"
            and elapsed >= 0.5 * self.stuck_window_sec
        ):
            _, added = self.add_or_update_feature(
                HARD_STOP_RECOVERY_TRAP,
                (x_value, y_value),
                escape_direction,
                now,
            )
            if added:
                added_types.append(HARD_STOP_RECOVERY_TRAP)

        if (
            elapsed >= 0.8 * self.stuck_window_sec
            and omega_sign_flips >= self.high_curvature_sign_flip_threshold
            and mean_abs_omega >= self.high_curvature_omega_threshold
            and goal_progress <= self.stuck_goal_progress_threshold
        ):
            _, added = self.add_or_update_feature(
                HIGH_CURVATURE_REGION,
                (x_value, y_value),
                escape_direction,
                now,
            )
            if added:
                added_types.append(HIGH_CURVATURE_REGION)

        debug = self.debug_snapshot(state)
        debug["stuck_feature_added"] = bool(added_types)
        debug["memory_added_types"] = ",".join(added_types) if added_types else "none"
        debug["memory_decay_applied"] = decay_applied
        self.last_debug = debug
        return debug

    def _feature_radius_for_cost(self, feature):
        if feature.type == HARD_STOP_RECOVERY_TRAP:
            return 0.82 * feature.radius
        if feature.type == NEAR_OBSTACLE_TRAP:
            return 0.88 * feature.radius
        if feature.type == HIGH_CURVATURE_REGION:
            return 0.72 * feature.radius
        return feature.radius

    def _cost_contribution_for_state(self, feature, state):
        x_value, y_value = float(state[0]), float(state[1])
        dx = x_value - feature.position[0]
        dy = y_value - feature.position[1]
        distance = math.hypot(dx, dy)
        effective_radius = max(self._feature_radius_for_cost(feature), 1e-6)
        if distance >= effective_radius:
            return 0.0

        influence = (1.0 - distance / effective_radius) ** 2
        weight = self._weight_for_type(feature.type)

        if feature.type == HIGH_CURVATURE_REGION:
            base = 0.55 * weight * feature.strength * influence
        elif feature.type == LOW_PROGRESS_CORRIDOR:
            base = 0.70 * weight * feature.strength * influence
        else:
            base = weight * feature.strength * influence

        directional = 0.0
        if feature.escape_direction is not None:
            projected_progress = (
                dx * feature.escape_direction[0] + dy * feature.escape_direction[1]
            )
            if projected_progress < 0.0:
                direction_scale = 1.0
                if feature.type == LOW_PROGRESS_CORRIDOR:
                    direction_scale = 1.20
                elif feature.type == HIGH_CURVATURE_REGION:
                    direction_scale = 0.35
                directional = (
                    direction_scale
                    * self.escape_direction_cost_weight
                    * feature.strength
                    * influence
                )
        return base + directional

    def memory_cost_breakdown_for_state(self, state):
        by_type = {}
        for feature_type in FEATURE_TYPES:
            by_type[feature_type] = 0.0

        if not self.enabled or not self.features or state is None:
            return {
                "total": 0.0,
                "by_type": by_type,
                "nearest_type": "none",
                "nearest_distance": None,
                "nearest_strength": 0.0,
                "active_feature_count": 0,
                "temperature_scale": 1.0,
            }

        active_count = 0
        total = 0.0
        for feature in self.features:
            contribution = self._cost_contribution_for_state(feature, state)
            if contribution > 0.0:
                active_count += 1
                by_type[feature.type] = by_type.get(feature.type, 0.0) + contribution
                total += contribution

        nearest, distance = self.nearest_feature(state)
        return {
            "total": total,
            "by_type": by_type,
            "nearest_type": nearest.type if nearest is not None else "none",
            "nearest_distance": distance,
            "nearest_strength": nearest.strength if nearest is not None else 0.0,
            "active_feature_count": active_count,
            "temperature_scale": self.temperature_scale(state),
        }

    def cost_for_state(self, state):
        if not self.enabled or not self.features or state is None:
            return 0.0
        return float(self.memory_cost_breakdown_for_state(state).get("total", 0.0))

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

    def memory_cost_breakdown_for_trajectory(self, trajectory, controls=None, stride=3):
        by_type = {}
        for feature_type in FEATURE_TYPES:
            by_type[feature_type] = 0.0

        if not self.enabled or not self.features or not trajectory:
            return {
                "total": 0.0,
                "by_type": by_type,
                "nearest_type": "none",
                "nearest_distance": None,
                "nearest_strength": 0.0,
                "active_feature_count": 0,
                "temperature_scale": 1.0,
            }

        stride = max(1, int(stride))
        total = 0.0
        count = 0
        active_feature_count = 0
        nearest_state = trajectory[0]
        nearest_distance = None
        nearest_type = "none"
        nearest_strength = 0.0

        for index, state in enumerate(trajectory):
            if index != len(trajectory) - 1 and index % stride != 0:
                continue
            breakdown = self.memory_cost_breakdown_for_state(state)
            total += float(breakdown.get("total", 0.0))
            count += 1
            active_feature_count = max(
                active_feature_count,
                int(breakdown.get("active_feature_count", 0)),
            )
            for key, value in breakdown.get("by_type", {}).items():
                by_type[key] = by_type.get(key, 0.0) + float(value)
            distance = breakdown.get("nearest_distance")
            if distance is not None and (nearest_distance is None or distance < nearest_distance):
                nearest_distance = distance
                nearest_type = breakdown.get("nearest_type", "none")
                nearest_strength = breakdown.get("nearest_strength", 0.0)
                nearest_state = state

        if count <= 0:
            count = 1
        for key in list(by_type.keys()):
            by_type[key] = by_type[key] / float(count)
        total = total / float(count)

        if controls:
            sign_flips = 0
            previous_sign = None
            for control in controls:
                omega_value = float(control[1])
                if abs(omega_value) < self.high_curvature_omega_threshold:
                    continue
                sign = 1 if omega_value > 0.0 else -1
                if previous_sign is not None and sign != previous_sign:
                    sign_flips += 1
                previous_sign = sign
            if sign_flips > 0:
                high_curvature_nearby = False
                high_strength = 0.0
                for feature in self.features:
                    if feature.type != HIGH_CURVATURE_REGION:
                        continue
                    for state in trajectory[::stride]:
                        distance = math.hypot(
                            float(state[0]) - feature.position[0],
                            float(state[1]) - feature.position[1],
                        )
                        if distance <= max(feature.radius, 1e-6):
                            high_curvature_nearby = True
                            high_strength = max(high_strength, feature.strength)
                            break
                if high_curvature_nearby:
                    oscillation_cost = (
                        self.high_curvature_cost_weight
                        * max(1.0, high_strength)
                        * float(sign_flips)
                        / float(max(1, len(controls)))
                    )
                    by_type[HIGH_CURVATURE_REGION] = (
                        by_type.get(HIGH_CURVATURE_REGION, 0.0) + oscillation_cost
                    )
                    total += oscillation_cost

        return {
            "total": total,
            "by_type": by_type,
            "nearest_type": nearest_type,
            "nearest_distance": nearest_distance,
            "nearest_strength": nearest_strength,
            "active_feature_count": active_feature_count,
            "temperature_scale": self.temperature_scale(nearest_state),
        }

    def memory_cost_for_state(self, state):
        return self.cost_for_state(state)

    def memory_cost_for_trajectory(self, trajectory, stride=3):
        return self.cost_for_trajectory(trajectory, step_stride=stride)

    def temperature_scale_for_state(self, state):
        return self.temperature_scale(state)

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

    def suggest_escape_direction(self, state, goal=None):
        debug = {
            "has_escape_direction": False,
            "nearest_type": "none",
            "nearest_distance": None,
            "nearest_strength": 0.0,
            "source": "none",
        }
        feature, distance = self.nearest_feature(state)
        if not self.enabled or feature is None or state is None:
            return None, debug

        current_xy = (float(state[0]), float(state[1]))
        escape_direction = feature.escape_direction
        source = "feature_escape_direction"
        if escape_direction is None:
            escape_direction = _normalize(
                (
                    current_xy[0] - feature.position[0],
                    current_xy[1] - feature.position[1],
                )
            )
            source = "radial_from_feature"

        goal_direction = None
        if goal is not None:
            goal_direction = _normalize(
                (
                    float(goal[0]) - current_xy[0],
                    float(goal[1]) - current_xy[1],
                )
            )

        if escape_direction is None:
            escape_direction = goal_direction
            source = "goal_direction"
        if escape_direction is None:
            debug.update(
                {
                    "nearest_type": feature.type,
                    "nearest_distance": distance,
                    "nearest_strength": feature.strength,
                }
            )
            return None, debug

        if goal_direction is not None:
            escape_direction = _normalize(
                (
                    0.65 * escape_direction[0] + 0.35 * goal_direction[0],
                    0.65 * escape_direction[1] + 0.35 * goal_direction[1],
                )
            )
            source = source + "+goal"

        debug.update(
            {
                "has_escape_direction": escape_direction is not None,
                "nearest_type": feature.type,
                "nearest_distance": distance,
                "nearest_strength": feature.strength,
                "source": source,
            }
        )
        return escape_direction, debug

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
        cost_breakdown = self.memory_cost_breakdown_for_state(state)
        feature_types = {}
        for feature in self.features:
            feature_types[feature.type] = feature_types.get(feature.type, 0) + 1
        return {
            "memory_enabled": bool(self.enabled),
            "memory_feature_count": len(self.features),
            "memory_active_feature_count": int(cost_breakdown.get("active_feature_count", 0)),
            "memory_nearest_type": nearest_type,
            "memory_nearest_distance": distance,
            "memory_nearest_strength": nearest_strength,
            "memory_cost": float(cost_breakdown.get("total", 0.0)),
            "memory_cost_total": float(cost_breakdown.get("total", 0.0)),
            "memory_cost_by_type": cost_breakdown.get("by_type", {}),
            "memory_feature_types": feature_types,
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
