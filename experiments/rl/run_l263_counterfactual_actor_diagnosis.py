#!/usr/bin/env python3
"""L263 fixed-state counterfactual diagnosis for the direct SAC Actor."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from experiments.rl.train_rl_sampling_prior import _scene_configs
from mobile_robot_mppi.core.config import git_sha, load_yaml
from mobile_robot_mppi.rl.checkpointing import load_sac_checkpoint
from mobile_robot_mppi.rl.environment import DirectControlEnv, _path_tracking_state
from mobile_robot_mppi.rl.observation import RunningNormalizer
from mobile_robot_mppi.rl.sac import SACAgent, SACConfig


PROTOCOL = "L263"
EXPECTED_CHECKPOINT_SHA256 = (
    "48e662fd728c0d1d62b1ff79681e5cdd924b120d9411c8df6e062a814b4d42a7"
)
DEFAULT_CONFIG = ROOT / "configs/rl/l262_coverage_gated_value_6k.yaml"
DEFAULT_CHECKPOINT = ROOT / (
    "results/research_platform/rl/"
    "l262_coverage_gated_value_seed20262611_6k/"
    "checkpoints/step_000006000.pt"
)
SCENES = tuple(
    ROOT / "configs/research/l261_value_stability" / role / name
    for role, names in (
        ("train", (
            "l261_train_s_bend.yaml",
            "l261_train_offset_hairpin.yaml",
            "l261_train_near_double_loop.yaml",
            "l261_train_curvature_ramp.yaml",
            "l261_train_corridor_switchback.yaml",
            "l261_train_compound_turns.yaml",
        )),
        ("validation", (
            "l261_validation_wave.yaml",
            "l261_validation_switchback.yaml",
            "l261_validation_loop_exit.yaml",
        )),
    )
    for name in names
)
HORIZONS = (1, 10, 20, 40)
GRID_LEVELS = (-1.0, -0.5, 0.0, 0.5, 1.0)
OFFSET_BLOCK = (0.70, 1.10, 1.80)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    seen = set()
    for row in rows:
        for name in row:
            if name not in seen:
                seen.add(name)
                fields.append(name)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def observation_feature_names(encoder_config):
    names = [
        "goal_body_x", "goal_body_y", "goal_distance", "goal_bearing",
        "measured_v", "measured_omega", "sin_theta", "cos_theta",
    ]
    if encoder_config.get("include_absolute_pose", False):
        names.extend(("absolute_x", "absolute_y"))
    if encoder_config.get("include_previous_action", True):
        names.extend(("previous_v", "previous_omega"))
    if encoder_config.get("include_safety_state", True):
        names.append("safety_override")
    if encoder_config.get("include_path_context", False):
        names.extend((
            "signed_cross_track", "sin_heading_error", "cos_heading_error",
            "path_curvature", "path_remaining", "path_valid",
        ))
    names.extend(
        "lidar_sector_%02d" % index
        for index in range(int(encoder_config["lidar_sectors"]))
    )
    names.append("scan_valid")
    if encoder_config.get("include_residual_context", False):
        names.extend(
            "residual_context_%02d" % index
            for index in range(int(encoder_config["residual_context_dimension"]))
        )
    if encoder_config.get("include_path_preview", False):
        for distance in encoder_config["path_preview_distances"]:
            names.extend((
                "preview_%g_body_x" % float(distance),
                "preview_%g_body_y" % float(distance),
            ))
    return tuple(names)


def _load_agent(checkpoint: Path, device: str):
    payload = load_sac_checkpoint(checkpoint, map_location=device)
    state = payload["agent"]
    agent = SACAgent(
        int(state["observation_dim"]),
        int(state["action_dim"]),
        SACConfig.from_mapping(state["config"]),
        device=device,
        seed=0,
    )
    agent.load_state_dict(state, load_optimizers=False)
    agent.eval()
    normalizer = RunningNormalizer.from_state_dict(payload["normalizer"])
    return payload, agent, normalizer


def _prepare_reset(environment, initial_state, seed):
    observation, reset_info = environment.reset(
        seed=int(seed),
        initial_state=np.asarray(initial_state, dtype=np.float64).copy(),
    )
    # The state vector contains measured v/omega.  Make the previous-command
    # feature and rate limiter consistent with that dynamic state.
    environment.previous_control = np.asarray((
        environment.truth.twist.v, environment.truth.twist.omega
    ), dtype=np.float64)
    environment.encoder.reset()
    observation = environment._encoded_observation()
    path_state = _path_tracking_state(
        environment.components["reference"], environment.truth.pose
    )
    environment.previous_path_progress = float(path_state["progress"])
    return observation, reset_info, path_state


def _candidate_progress(reference):
    cumulative = np.asarray(reference.cumulative, dtype=np.float64)
    lengths = np.asarray(reference.segment_lengths, dtype=np.float64)
    progress = cumulative[:-1] + 0.5 * lengths
    valid = (progress >= 0.08 * reference.total_length) & (
        progress <= 0.92 * reference.total_length
    )
    if not np.any(valid):
        valid[:] = True
    values = []
    for value in progress:
        pose = reference.poses_at_progress(np.asarray([value]))[0]
        projection = reference.project(pose[:2], minimum_progress=None)
        values.append(float(projection.curvature))
    return progress, np.asarray(values), valid


def _state_specs(reference, scene_index):
    progress, curvature, valid = _candidate_progress(reference)
    indices = np.flatnonzero(valid)
    low = int(indices[np.argmin(np.abs(curvature[indices]))])
    high = int(indices[np.argmax(np.abs(curvature[indices]))])
    reversal = None
    for index in indices[:-1]:
        if curvature[index] * curvature[index + 1] < 0.0:
            reversal = int(index)
            break
    event = high if reversal is None else reversal
    return (
        {
            "kind": "center_low_curvature",
            "progress": float(progress[low]),
            "offset": 0.0,
            "curvature_event": False,
        },
        {
            "kind": "center_curvature_reversal" if reversal is not None
            else "center_high_curvature",
            "progress": float(progress[event]),
            "offset": 0.0,
            "curvature_event": True,
        },
        {
            "kind": (
                "near_boundary" if scene_index % 3 == 0 else
                "moderate_offtrack" if scene_index % 3 == 1 else
                "far_offtrack"
            ),
            "progress": float(0.58 * reference.total_length),
            "offset": float(OFFSET_BLOCK[scene_index % 3]),
            "curvature_event": False,
        },
    )


def _initial_state(reference, progress, signed_offset):
    pose = reference.poses_at_progress(np.asarray([progress]))[0]
    tangent = float(pose[2])
    normal = np.asarray((-math.sin(tangent), math.cos(tangent)))
    position = pose[:2] + float(signed_offset) * normal
    # MujocoPlant reset initializes dynamic velocity to zero.  Keep the
    # declared five-state vector honest instead of pretending a non-zero twist
    # that the simulator does not apply.
    return np.asarray((position[0], position[1], tangent, 0.0, 0.0))


def _geometric_recovery_action(environment, initial_state):
    reference = environment.components["reference"]
    projection = reference.project(
        np.asarray(initial_state[:2]), minimum_progress=getattr(reference, "progress", None)
    )
    target = reference.poses_at_progress(np.asarray([
        min(reference.total_length, projection.progress + 0.80)
    ]))[0]
    delta = target[:2] - np.asarray(initial_state[:2])
    cosine, sine = math.cos(initial_state[2]), math.sin(initial_state[2])
    body_x = cosine * delta[0] + sine * delta[1]
    body_y = -sine * delta[0] + cosine * delta[1]
    bearing = math.atan2(body_y, body_x)
    v_physical = 0.18 * max(0.25, math.cos(bearing))
    omega_physical = float(np.clip(
        2.0 * bearing - 0.8 * projection.signed_cross_track_error,
        environment.action_spec.lower[1], environment.action_spec.upper[1],
    ))
    center = 0.5 * (environment.action_spec.lower + environment.action_spec.upper)
    half = 0.5 * (environment.action_spec.upper - environment.action_spec.lower)
    return np.clip(
        (np.asarray((v_physical, omega_physical)) - center) / half,
        -1.0, 1.0,
    )


def _unique_actions(actor_action, recovery_action, levels):
    semantic = []
    semantic.append(("actor", np.asarray(actor_action, dtype=np.float64)))
    semantic.append(("recovery", np.asarray(recovery_action, dtype=np.float64)))
    for v_value in levels:
        for omega_value in levels:
            semantic.append((
                "grid_v%+.1f_w%+.1f" % (v_value, omega_value),
                np.asarray((v_value, omega_value), dtype=np.float64),
            ))
    unique = {}
    for label, action in semantic:
        key = tuple(np.round(action, decimals=8))
        if key not in unique:
            unique[key] = {"action": action, "labels": []}
        unique[key]["labels"].append(label)
    result = []
    for index, value in enumerate(unique.values()):
        result.append({
            "action_id": "action_%03d" % index,
            "action": value["action"],
            "labels": tuple(value["labels"]),
        })
    return result


def _feature_audit(payload, agent, normalizer):
    replay = payload.get("replay_buffer")
    if not replay or int(replay.get("size", 0)) <= 0:
        raise ValueError("L263 requires a checkpoint with replay data")
    size = int(replay["size"])
    raw = np.asarray(replay["observations"][:size], dtype=np.float32)
    normalized = normalizer.normalize(raw)
    actions = agent.select_action_batch(normalized, deterministic=True)
    names = observation_feature_names(payload["encoder_config"])
    if len(names) != raw.shape[1]:
        raise ValueError("feature schema does not match checkpoint observation size")
    rows = []
    for index, name in enumerate(names):
        values = raw[:, index]
        z_values = normalized[:, index]
        rows.append({
            "index": index,
            "name": name,
            "normalizer_mean": float(normalizer.mean[index]),
            "normalizer_std": float(normalizer.std[index]),
            "raw_mean": float(np.mean(values)),
            "raw_std": float(np.std(values)),
            "raw_min": float(np.min(values)),
            "raw_max": float(np.max(values)),
            "normalized_mean": float(np.mean(z_values)),
            "normalized_std": float(np.std(z_values)),
            "normalized_abs_max": float(np.max(np.abs(z_values))),
            "normalizer_clip_fraction": float(np.mean(
                np.abs(z_values) >= normalizer.clip - 1e-6
            )),
        })
    groups = np.asarray(replay["groups"][:size], dtype=np.int64)
    group_counts = {
        str(int(group)): int(np.sum(groups == group)) for group in np.unique(groups)
    }
    physical = np.column_stack((
        0.5 * (payload["action_spec"]["lower"][0] + payload["action_spec"]["upper"][0])
        + 0.5 * (payload["action_spec"]["upper"][0] - payload["action_spec"]["lower"][0]) * actions[:, 0],
        0.5 * (payload["action_spec"]["lower"][1] + payload["action_spec"]["upper"][1])
        + 0.5 * (payload["action_spec"]["upper"][1] - payload["action_spec"]["lower"][1]) * actions[:, 1],
    ))
    summary = {
        "normalizer_count": int(normalizer.count),
        "replay_size": size,
        "replay_group_counts": group_counts,
        "features_at_minimum_std": [
            names[index] for index in np.flatnonzero(
                normalizer.std <= normalizer.min_std + 1e-12
            )
        ],
        "features_with_clip_fraction_above_1pct": [
            row["name"] for row in rows
            if row["normalizer_clip_fraction"] > 0.01
        ],
        "actor_normalized_mean": np.mean(actions, axis=0).tolist(),
        "actor_normalized_std": np.std(actions, axis=0).tolist(),
        "actor_normalized_saturation_fraction": np.mean(
            np.abs(actions) >= 0.95, axis=0
        ).tolist(),
        "actor_physical_mean": np.mean(physical, axis=0).tolist(),
        "actor_physical_std": np.std(physical, axis=0).tolist(),
        "alpha": float(agent.alpha.detach().cpu()),
        "target_entropy": float(agent.target_entropy),
        "critic_distribution": agent.config.critic_distribution,
        "critic_num_quantiles": int(agent.config.critic_num_quantiles),
        "actor_cvar_fraction": float(agent.config.actor_cvar_fraction),
        "signed_cte_feature_saturation_m": float(
            payload["encoder_config"]["path_cross_track_scale"]
        ),
        "transition_level_far_offtrack_coverage_identifiable": False,
        "transition_level_far_offtrack_note": (
            "the replay stores the encoded signed CTE after clipping, not the "
            "unclipped simulator CTE"
        ),
    }
    return rows, summary


def _rankdata(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def _spearman(left, right):
    left_rank, right_rank = _rankdata(left), _rankdata(right)
    if np.std(left_rank) <= 1e-12 or np.std(right_rank) <= 1e-12:
        return None
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def _rollout(
    environment, initial_state, seed, expected_observation, candidate_action,
    continuation, agent, normalizer, horizons,
):
    observation, _, initial_path = _prepare_reset(environment, initial_state, seed)
    replay_error = float(np.max(np.abs(observation - expected_observation)))
    if replay_error > 1e-6:
        raise RuntimeError("fixed-state reset observation drifted")
    initial_distance = float(environment._distance_to_final(environment.truth))
    initial_cte = float(initial_path["cross_track_error"])
    initial_signed_cte = float(initial_path["signed_cross_track_error"])
    initial_progress = float(initial_path["progress"])
    gamma = float(environment.gamma)
    cumulative = 0.0
    term_sums = defaultdict(float)
    corridor_reentry = initial_cte <= 0.75
    terminated = truncated = False
    last_info = None
    records = {}
    maximum = max(horizons)
    for step in range(1, maximum + 1):
        if not (terminated or truncated):
            if step == 1 or continuation == "constant":
                action = candidate_action
            else:
                action, _ = agent.select_action(
                    normalizer.normalize(observation), deterministic=True
                )
            observation, reward, terminated, truncated, last_info = environment.step(action)
            cumulative += gamma ** (step - 1) * float(reward)
            for name, value in last_info["reward_terms"].items():
                term_sums[name] += gamma ** (step - 1) * float(value)
            corridor_reentry = corridor_reentry or float(
                last_info["cross_track_error"]
            ) <= 0.75
        if step in horizons:
            if last_info is None:
                raise RuntimeError("counterfactual rollout produced no transition")
            records[step] = {
                "horizon": int(step),
                "steps_executed": int(environment.steps),
                "discounted_return": float(cumulative),
                "cross_track_delta": float(last_info["cross_track_error"] - initial_cte),
                "signed_cross_track_delta": float(
                    last_info["signed_cross_track_error"] - initial_signed_cte
                ),
                "path_progress_delta": float(last_info["path_progress"] - initial_progress),
                "goal_distance_delta": float(last_info["goal_distance"] - initial_distance),
                "corridor_reentry": bool(corridor_reentry),
                "collision": bool(last_info["collision"]),
                "safety_override": bool(last_info["safety_override"]),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "proposed_v": float(last_info["proposed_control"][0]),
                "proposed_omega": float(last_info["proposed_control"][1]),
                "executed_v": float(last_info["executed_control"][0]),
                "executed_omega": float(last_info["executed_control"][1]),
                "applied_v": float(last_info["applied_control"][0]),
                "applied_omega": float(last_info["applied_control"][1]),
                "max_reset_observation_error": replay_error,
                "reward_terms": json.dumps(dict(term_sums), sort_keys=True),
            }
    return records


def _ranking_summary(action_rows, rollout_rows, horizons):
    actions = {(row["state_id"], row["action_id"]): row for row in action_rows}
    grouped = defaultdict(list)
    for row in rollout_rows:
        if "grid_" in row["semantic_labels"]:
            grouped[(row["state_id"], row["continuation"], row["horizon"])].append(row)
    summaries = []
    for (state_id, continuation, horizon), rows in sorted(grouped.items()):
        q_values = [float(actions[(state_id, row["action_id"])]["target_minimum_q"]) for row in rows]
        returns = [float(row["discounted_return"]) for row in rows]
        best_q = max(q_values)
        best_return = max(returns)
        q_top = {rows[i]["action_id"] for i, value in enumerate(q_values) if best_q - value <= 1e-6}
        return_top = {rows[i]["action_id"] for i, value in enumerate(returns) if best_return - value <= 1e-6}
        actor_rows = [row for row in rollout_rows if row["state_id"] == state_id and row["continuation"] == continuation and row["horizon"] == horizon and "actor" in row["semantic_labels"].split("|")]
        actor_return = None if not actor_rows else float(actor_rows[0]["discounted_return"])
        summaries.append({
            "state_id": state_id,
            "continuation": continuation,
            "horizon": int(horizon),
            "grid_actions": len(rows),
            "spearman_q_return": _spearman(q_values, returns),
            "top1_agreement": bool(q_top & return_top),
            "critic_q_spread": float(max(q_values) - min(q_values)),
            "realized_return_spread": float(max(returns) - min(returns)),
            "actor_realized_regret": None if actor_return is None else float(best_return - actor_return),
        })
    aggregate = {}
    for continuation in ("constant", "actor_follow"):
        for horizon in horizons:
            rows = [row for row in summaries if row["continuation"] == continuation and row["horizon"] == horizon]
            correlations = [row["spearman_q_return"] for row in rows if row["spearman_q_return"] is not None]
            aggregate["%s_h%d" % (continuation, horizon)] = {
                "states": len(rows),
                "mean_spearman_q_return": None if not correlations else float(np.mean(correlations)),
                "top1_agreement_fraction": None if not rows else float(np.mean([row["top1_agreement"] for row in rows])),
                "mean_actor_realized_regret": None if not rows else float(np.mean([
                    row["actor_realized_regret"] for row in rows
                    if row["actor_realized_regret"] is not None
                ])),
            }
    return summaries, aggregate


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "results/research_platform/rl/l263_counterfactual_actor_diagnosis",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)

    checkpoint = args.checkpoint.resolve()
    checkpoint_hash = _sha256(checkpoint)
    if checkpoint_hash != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("L263 checkpoint SHA256 mismatch: %s" % checkpoint_hash)
    payload, agent, normalizer = _load_agent(checkpoint, args.device)
    if int(payload["agent"]["observation_dim"]) != 69:
        raise ValueError("L263 requires the frozen 69D Actor")
    feature_rows, feature_summary = _feature_audit(payload, agent, normalizer)

    base = load_yaml(args.config.resolve())
    scene_paths = list(SCENES[:1] if args.smoke else SCENES)
    configs = _scene_configs(base, [str(path) for path in scene_paths])
    output_dir = args.output_dir.resolve()
    if args.smoke:
        output_dir = output_dir.with_name(output_dir.name + "_smoke")
    output_dir.mkdir(parents=True, exist_ok=True)
    levels = (-1.0, 0.0, 1.0) if args.smoke else GRID_LEVELS
    horizons = (1, 3) if args.smoke else HORIZONS

    state_rows = []
    action_rows = []
    rollout_rows = []
    rejected_rows = []
    for scene_index, (scene_path, config) in enumerate(zip(scene_paths, configs)):
        config = copy.deepcopy(config)
        configured_state = np.asarray(config["experiment"]["initial_state"]).reshape(-1)
        config["rl"]["training"]["initial_state_noise"] = [0.0] * configured_state.size
        config["experiment"]["max_steps"] = max(max(horizons) + 1, 50)
        environment = DirectControlEnv(config, ROOT, seed=20262731 + scene_index)
        try:
            reference = environment.components["reference"]
            specs = _state_specs(reference, scene_index)
            if args.smoke:
                specs = specs[:1]
            for local_index, spec in enumerate(specs):
                state_id = "scene_%02d_state_%02d" % (scene_index, local_index)
                signs = (0.0,) if spec["offset"] == 0.0 else (1.0, -1.0)
                accepted = None
                rejection_reasons = []
                for sign in signs:
                    initial = _initial_state(
                        reference, spec["progress"], sign * spec["offset"]
                    )
                    observation, reset_info, path_state = _prepare_reset(
                        environment, initial, 20262731 + scene_index
                    )
                    if bool(environment.truth.collision):
                        rejection_reasons.append("collision_at_reset_sign_%+.0f" % sign)
                        continue
                    if not np.isfinite(observation).all():
                        rejection_reasons.append("nonfinite_observation_sign_%+.0f" % sign)
                        continue
                    accepted = (initial, observation, reset_info, path_state, sign)
                    break
                if accepted is None:
                    rejected_rows.append({
                        "state_id": state_id,
                        "scene": str(config["scene"]["name"]),
                        "kind": spec["kind"],
                        "reasons": "|".join(rejection_reasons),
                    })
                    continue
                initial, observation, reset_info, path_state, sign = accepted
                normalized = normalizer.normalize(observation)
                actor_action, actor_diagnostics = agent.select_action(
                    normalized, deterministic=True
                )
                recovery_action = _geometric_recovery_action(environment, initial)
                actions = _unique_actions(actor_action, recovery_action, levels)
                action_matrix = np.stack([item["action"] for item in actions]).astype(np.float32)
                observations = np.repeat(normalized[None, :], len(actions), axis=0)
                online_q = agent.expected_twin_q(observations, action_matrix, "online")
                target_q = agent.expected_twin_q(observations, action_matrix, "target")
                distributions = agent.twin_q_distribution(observations, action_matrix, "target")
                state_rows.append({
                    "state_id": state_id,
                    "scene": str(config["scene"]["name"]),
                    "scene_role": "train" if scene_index < 6 else "validation",
                    "scene_config": str(scene_path.resolve()),
                    "seed": int(20262731 + scene_index),
                    "kind": spec["kind"],
                    "requested_offset_m": float(spec["offset"]),
                    "accepted_offset_sign": float(sign),
                    "initial_state": initial.tolist(),
                    "truth_pose": environment.truth.pose.as_array().tolist(),
                    "truth_twist": [float(environment.truth.twist.v), float(environment.truth.twist.omega)],
                    "truth_collision": bool(environment.truth.collision),
                    "truth_minimum_clearance": float(environment.truth.minimum_clearance),
                    "path_state": path_state,
                    "raw_observation": observation.tolist(),
                    "normalized_observation": normalized.tolist(),
                    "actor_action": actor_action.tolist(),
                    "actor_diagnostics": actor_diagnostics,
                    "recovery_action": recovery_action.tolist(),
                })
                for action_index, item in enumerate(actions):
                    action_rows.append({
                        "state_id": state_id,
                        "action_id": item["action_id"],
                        "semantic_labels": "|".join(item["labels"]),
                        "normalized_v": float(item["action"][0]),
                        "normalized_omega": float(item["action"][1]),
                        "online_minimum_q": float(online_q["minimum"][action_index]),
                        "target_q1_mean": float(target_q["q1"][action_index]),
                        "target_q2_mean": float(target_q["q2"][action_index]),
                        "target_minimum_q": float(target_q["minimum"][action_index]),
                        "target_twin_disagreement": float(target_q["disagreement"][action_index]),
                        "target_q1_quantiles": json.dumps(distributions["q1"][action_index].tolist()),
                        "target_q2_quantiles": json.dumps(distributions["q2"][action_index].tolist()),
                    })
                    for continuation in ("constant", "actor_follow"):
                        prefixes = _rollout(
                            environment, initial, 20262731 + scene_index,
                            observation, item["action"], continuation,
                            agent, normalizer, horizons,
                        )
                        for horizon, values in prefixes.items():
                            rollout_rows.append({
                                "state_id": state_id,
                                "action_id": item["action_id"],
                                "semantic_labels": "|".join(item["labels"]),
                                "continuation": continuation,
                                **values,
                            })
                print(json.dumps({
                    "protocol": PROTOCOL,
                    "completed_state": state_id,
                    "completed_states": len(state_rows),
                    "rollout_rows": len(rollout_rows),
                }, sort_keys=True), flush=True)
        finally:
            environment.close()

    ranking_rows, ranking_aggregate = _ranking_summary(
        action_rows, rollout_rows, horizons
    )
    implementation = {
        **feature_summary,
        "action_lower": np.asarray(payload["action_spec"]["lower"]).tolist(),
        "action_upper": np.asarray(payload["action_spec"]["upper"]).tolist(),
        "action_rate_limits": configs[0]["action_space"].get("rate_limits"),
        "control_dt": float(configs[0]["experiment"]["control_dt"]),
        "reward_cross_track_cap_m": float(
            configs[0]["rl"]["reward"].get("cross_track_error_cap", 0.0)
        ),
    }
    summary = {
        "protocol": PROTOCOL,
        "status": "engineering_smoke" if args.smoke else "diagnostic_complete",
        "git_sha": git_sha(ROOT),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "device": args.device,
        "scenes": len(configs),
        "accepted_states": len(state_rows),
        "rejected_states": len(rejected_rows),
        "unique_state_actions": len(action_rows),
        "rollout_rows": len(rollout_rows),
        "horizons": list(horizons),
        "continuations": ["constant", "actor_follow"],
        "implementation_audit": implementation,
        "ranking_aggregate": ranking_aggregate,
        "interpretation": (
            "descriptive only; apply the frozen L263 decision tree after "
            "integrity checks"
        ),
    }
    _write_csv(output_dir / "feature_audit.csv", feature_rows)
    _write_csv(output_dir / "critic_actions.csv", action_rows)
    _write_csv(output_dir / "counterfactual_rollouts.csv", rollout_rows)
    _write_csv(output_dir / "ranking_by_state.csv", ranking_rows)
    _write_csv(output_dir / "rejected_states.csv", rejected_rows)
    (output_dir / "state_manifest.json").write_text(
        json.dumps(state_rows, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
