#!/usr/bin/env python3
"""Read-only statistical core bundled with the standalone Redesign E test.

The input tree is never written. Statistics are produced only after the package
integrity Gate passes, all 200 episode directories are complete, configuration
identity is verified, and every arm contains the same 50 matched cases.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import scipy
from scipy import stats
import yaml


GROUPS = (10,)
ARMS = ("B00", "B01", "B10", "B11")
BASELINES = ("B00", "B01", "B10")
EXPECTED_EPISODES_PER_ARM = 50
RAW_DATASET_DIR = "redesignE_four_arm_exact"
REQUIRED_EPISODE_FILES = (
    "case_identity.json",
    "config_resolved.yaml",
    "config_sha256.txt",
    "metrics.json",
    "preflight.json",
    "provenance.json",
    "trajectory.csv.gz",
)
REVISION_KEYS_EXPECTED_DISABLED = {
    "planner.probabilistic_obstacle_traversal_window_uncommitted_hold_retreat_steps",
    "planner.probabilistic_obstacle_traversal_window_uncommitted_temporal_staging_hold_current_hazard_only_enabled",
}
SUPPLEMENTAL_ROOT_FILES = {
    "episode_index.csv",
    "frozen_protocol_and_seed_lists.yaml",
    "master_orchestrator.log",
    "statistical_summary.json",
}

CONTINUOUS_METRICS: dict[str, tuple[str, str]] = {
    "steps": ("完成步数", "lower"),
    "final_goal_distance": ("最终目标距离 (m)", "lower"),
    "minimum_clearance": ("最小净空 (m)", "higher"),
    "planner_compute_ms_mean": ("规划均值时延 (ms)", "lower"),
    "planner_compute_ms_p95": ("规划 P95 时延 (ms)", "lower"),
}

ADDITIONAL_METRICS: dict[str, tuple[str, str]] = {
    "path_progress_ratio_max": ("最大路线进度比例", "higher"),
    "stuck_steps": ("stuck 步数", "lower"),
    "safety_interventions": ("安全干预次数", "lower"),
    "known_static_map_candidate_feasible_fraction_mean": (
        "静态地图候选可行比例",
        "higher",
    ),
    "planner_deadline_miss_rate": ("规划 deadline miss 比例", "lower"),
    "control_jerk": ("控制 jerk", "lower"),
    "cross_track_p95": ("横向误差 P95 (m)", "lower"),
    "minimum_dynamic_obstacle_center_distance": (
        "最小动态障碍物中心距 (m)",
        "higher",
    ),
    "obstacle_pass_events": ("障碍物通过事件", "higher"),
}

BOOTSTRAP_SEED = 20260730
BOOTSTRAP_SAMPLES = 20_000
EPS = 1e-12


@dataclass(frozen=True)
class Episode:
    group: int
    arm: str
    seed: int
    path: Path
    case_identity: Mapping[str, Any]
    metrics: Mapping[str, Any]
    preflight: Mapping[str, Any]
    provenance: Mapping[str, Any]
    config: Mapping[str, Any]
    config_sha256: str

    @property
    def key(self) -> tuple[int, int]:
        return self.group, self.seed

    @property
    def safe_success(self) -> bool:
        # Formal benchmark semantics from derive_episode_metrics.py and the
        # transferred server episode_index.csv/statistical_summary.json.
        return bool(self.metrics["success"]) and not self.collision

    @property
    def boundary_safe_success(self) -> bool | None:
        value = self.metrics.get("boundary_safe_success")
        return None if value is None else bool(value)

    @property
    def collision(self) -> bool:
        return bool(self.metrics["collision"])

    @property
    def termination_reason(self) -> str:
        value = self.metrics.get("termination_reason")
        return "missing" if value is None else str(value)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def canonical_config_hash(config: Mapping[str, Any]) -> str:
    """Match mobile_robot_mppi.core.config.config_hash semantics.

    config_sha256.txt is a canonical JSON hash of the parsed configuration
    mapping, not a byte hash of config_resolved.yaml.  The raw YAML byte hash is
    independently covered by the transferred server SHA-256 manifest.
    """
    canonical = json.dumps(
        config, sort_keys=True, separators=(",", ":"), default=str
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=json_default)
        + "\n",
        encoding="utf-8",
    )


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value)!r}")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_sha256_list(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    pattern = re.compile(r"^([0-9a-fA-F]{64})[ \t]+(?:\*)?(.+?)\s*$")
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        match = pattern.match(line)
        if not match:
            raise ValueError(f"{path.name}:{line_no}: malformed SHA-256 row")
        digest, rel = match.groups()
        rel = rel.replace("\\", "/")
        if rel in result:
            raise ValueError(f"{path.name}:{line_no}: duplicate path {rel}")
        validate_relative_path(rel)
        result[rel] = digest.lower()
    return result


def parse_size_list(path: Path) -> dict[str, int]:
    result: dict[str, int] = {}
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split("\t", 1)
        if len(parts) != 2:
            raise ValueError(f"{path.name}:{line_no}: malformed size row")
        raw_size, rel = parts
        rel = rel.replace("\\", "/")
        validate_relative_path(rel)
        if rel in result:
            raise ValueError(f"{path.name}:{line_no}: duplicate path {rel}")
        result[rel] = int(raw_size)
    return result


def validate_relative_path(rel: str) -> None:
    pure = PurePosixPath(rel)
    if pure.is_absolute() or ".." in pure.parts or not pure.parts:
        raise ValueError(f"Unsafe relative path in manifest: {rel!r}")


def find_root_control_file(raw_root: Path, exact: str, fallback_regex: str) -> Path | None:
    direct = raw_root / exact
    if direct.is_file():
        return direct
    pattern = re.compile(fallback_regex, re.IGNORECASE)
    matches = sorted(p for p in raw_root.iterdir() if p.is_file() and pattern.search(p.name))
    return matches[0] if len(matches) == 1 else None


def audit_transfer_and_hashes(raw_root: Path, output_root: Path) -> dict[str, Any]:
    transfer_manifest = find_root_control_file(
        raw_root, "TRANSFER_MANIFEST.md", r"transfer.*manifest.*\.md$"
    )
    sha_list = find_root_control_file(raw_root, "norev.sha256", r"(sha256|checksum)")
    size_list = find_root_control_file(raw_root, "norev.sizes", r"(sizes|size_list)")

    gate: dict[str, Any] = {
        "audit_started_utc": utc_now(),
        "raw_root": str(raw_root),
        "input_tree_written": False,
        "transfer_manifest": str(transfer_manifest) if transfer_manifest else None,
        "server_sha256_list": str(sha_list) if sha_list else None,
        "server_size_list": str(size_list) if size_list else None,
        "transfer_manifest_present": bool(transfer_manifest),
        "server_sha256_list_present": bool(sha_list),
        "server_size_list_present": bool(size_list),
        "errors": [],
        "warnings": [],
    }
    if not transfer_manifest:
        gate["errors"].append("TRANSFER_MANIFEST.md 尚未到达。")
    if not sha_list:
        gate["errors"].append("服务器 SHA-256 清单未找到。")
    if not size_list:
        gate["errors"].append("服务器 size 清单未找到。")
    if gate["errors"]:
        gate["passed"] = False
        gate["audit_finished_utc"] = utc_now()
        write_json(output_root / "integrity_gate.json", gate)
        return gate

    assert transfer_manifest and sha_list and size_list
    gate["transfer_manifest_sha256"] = sha256_file(transfer_manifest)
    gate["server_sha256_list_sha256"] = sha256_file(sha_list)
    gate["server_size_list_sha256"] = sha256_file(size_list)
    gate["transfer_manifest_text"] = transfer_manifest.read_text(
        encoding="utf-8", errors="replace"
    )
    manifest_text = str(gate["transfer_manifest_text"])
    gate["transfer_manifest_declares_passed"] = bool(
        re.search(
            r"\|\s*\*\*verification\*\*\s*\|\s*\*\*PASSED\*\*\s*\|",
            manifest_text,
            flags=re.IGNORECASE,
        )
    )
    if not gate["transfer_manifest_declares_passed"]:
        gate["errors"].append("TRANSFER_MANIFEST.md 未明确声明 verification=PASSED。")

    try:
        expected_hashes = parse_sha256_list(sha_list)
        expected_sizes = parse_size_list(size_list)
    except Exception as exc:
        gate["errors"].append(f"清单解析失败：{exc}")
        gate["passed"] = False
        gate["audit_finished_utc"] = utc_now()
        write_json(output_root / "integrity_gate.json", gate)
        return gate

    gate["expected_hash_entries"] = len(expected_hashes)
    gate["expected_size_entries"] = len(expected_sizes)
    gate["expected_total_bytes"] = int(sum(expected_sizes.values()))
    if set(expected_hashes) != set(expected_sizes):
        gate["errors"].append("SHA-256 清单与 size 清单的路径集合不一致。")
    manifest_count_match = re.search(r"\|\s*files\s*\|\s*(\d+)\s*\|", manifest_text)
    manifest_bytes_match = re.search(
        r"\|\s*total size\s*\|\s*(\d+)\s+bytes", manifest_text
    )
    gate["transfer_manifest_file_count"] = (
        int(manifest_count_match.group(1)) if manifest_count_match else None
    )
    gate["transfer_manifest_total_bytes"] = (
        int(manifest_bytes_match.group(1)) if manifest_bytes_match else None
    )
    if gate["transfer_manifest_file_count"] != len(expected_hashes):
        gate["errors"].append(
            "TRANSFER_MANIFEST.md 文件数与 SHA-256 清单条目数不一致。"
        )
    if gate["transfer_manifest_total_bytes"] != sum(expected_sizes.values()):
        gate["errors"].append(
            "TRANSFER_MANIFEST.md 总字节数与 size 清单不一致。"
        )

    missing: list[str] = []
    size_mismatch: list[dict[str, Any]] = []
    hash_mismatch: list[dict[str, Any]] = []
    verified = 0
    verified_bytes = 0
    for index, rel in enumerate(sorted(expected_hashes), 1):
        local = raw_root.joinpath(*PurePosixPath(rel).parts)
        if not local.is_file():
            missing.append(rel)
            continue
        actual_size = local.stat().st_size
        wanted_size = expected_sizes.get(rel)
        if wanted_size is None or actual_size != wanted_size:
            size_mismatch.append(
                {"path": rel, "expected": wanted_size, "actual": actual_size}
            )
            continue
        actual_hash = sha256_file(local)
        if actual_hash != expected_hashes[rel]:
            hash_mismatch.append(
                {
                    "path": rel,
                    "expected": expected_hashes[rel],
                    "actual": actual_hash,
                }
            )
            continue
        verified += 1
        verified_bytes += actual_size
        if index % 250 == 0:
            print(
                f"[checksum] {index}/{len(expected_hashes)} listed paths processed",
                flush=True,
            )

    gate["verified_files"] = verified
    gate["verified_bytes"] = verified_bytes
    gate["missing_files"] = missing
    gate["size_mismatches"] = size_mismatch
    gate["hash_mismatches"] = hash_mismatch
    if missing:
        gate["errors"].append(f"缺失 {len(missing)} 个清单文件。")
    if size_mismatch:
        gate["errors"].append(f"{len(size_mismatch)} 个文件大小不匹配。")
    if hash_mismatch:
        gate["errors"].append(f"{len(hash_mismatch)} 个文件 SHA-256 不匹配。")

    allowed_root_controls = {
        transfer_manifest.resolve(),
        sha_list.resolve(),
        size_list.resolve(),
    }
    supplemental_files: list[dict[str, Any]] = []
    for name in sorted(SUPPLEMENTAL_ROOT_FILES):
        path = raw_root / name
        if path.is_file():
            allowed_root_controls.add(path.resolve())
            supplemental_files.append(
                {
                    "path": name,
                    "size": path.stat().st_size,
                    "local_sha256": sha256_file(path),
                    "covered_by_server_sha256_list": False,
                }
            )
    gate["supplemental_root_files"] = supplemental_files
    expected_abs = {
        raw_root.joinpath(*PurePosixPath(rel).parts).resolve() for rel in expected_hashes
    }
    extras = sorted(
        str(p.relative_to(raw_root)).replace("\\", "/")
        for p in raw_root.rglob("*")
        if p.is_file() and p.resolve() not in expected_abs and p.resolve() not in allowed_root_controls
    )
    gate["unexpected_files"] = extras
    if extras:
        gate["warnings"].append(
            f"原始目录含 {len(extras)} 个未列入服务器 SHA-256 清单的额外文件。"
        )

    gate["passed"] = not gate["errors"]
    gate["audit_finished_utc"] = utc_now()
    write_json(output_root / "integrity_gate.json", gate)
    return gate


def read_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def read_yaml(path: Path) -> Mapping[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return value


def nested_get(data: Mapping[str, Any], dotted: str, default: Any = None) -> Any:
    current: Any = data
    for part in dotted.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


def physical_scenario_bundle(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the expanded fields that determine the matched physical task.

    Path-valued provenance such as ``_config_path`` and
    ``experiment.complex_map_source`` is intentionally excluded. The runtime
    builder copies the map's expanded ``scene`` and ``task`` into the resolved
    config; those values, together with the initial state and episode budget,
    determine the physical scenario used by the runner.
    """
    return {
        "scene": config.get("scene"),
        "task": config.get("task"),
        "experiment": {
            "complex_map": nested_get(config, "experiment.complex_map"),
            "initial_state": nested_get(config, "experiment.initial_state"),
            "max_steps": nested_get(config, "experiment.max_steps"),
            "terminate_on_collision": nested_get(
                config, "experiment.terminate_on_collision"
            ),
        },
    }


def physical_scenario_hash(config: Mapping[str, Any]) -> str:
    return canonical_config_hash(physical_scenario_bundle(config))


def recursive_values_for_key(data: Any, wanted: str) -> list[Any]:
    found: list[Any] = []
    if isinstance(data, Mapping):
        for key, value in data.items():
            if str(key) == wanted:
                found.append(value)
            found.extend(recursive_values_for_key(value, wanted))
    elif isinstance(data, list):
        for value in data:
            found.extend(recursive_values_for_key(value, wanted))
    return found


def parse_seed_dir(path: Path) -> int:
    match = re.fullmatch(r"seed(\d+)", path.name)
    if not match:
        raise ValueError(f"Malformed seed directory: {path}")
    return int(match.group(1))


def load_and_audit_episodes(
    raw_root: Path, output_root: Path
) -> tuple[list[Episode], dict[str, Any]]:
    errors: list[str] = []
    warnings: list[str] = []
    episodes: list[Episode] = []
    seed_audit_rows: list[dict[str, Any]] = []
    environment_rows: list[dict[str, Any]] = []
    field_counts: Counter[str] = Counter()

    observed_group_dirs = {
        path.name
        for path in raw_root.iterdir()
        if path.is_dir() and path.name == RAW_DATASET_DIR
    }
    expected_group_dirs = {RAW_DATASET_DIR}
    extra_group_dirs = sorted(observed_group_dirs - expected_group_dirs)
    if extra_group_dirs:
        errors.append(f"发现协议外 group 目录：{extra_group_dirs}")

    for group in GROUPS:
        group_dir = raw_root / RAW_DATASET_DIR
        if not group_dir.is_dir():
            errors.append(f"Group {group:02d} 目录缺失。")
            continue
        observed_arm_dirs = {
            path.name for path in group_dir.iterdir() if path.is_dir()
        }
        extra_arm_dirs = sorted(observed_arm_dirs - set(ARMS))
        if extra_arm_dirs:
            errors.append(
                f"Group {group:02d} 发现协议外 arm 目录：{extra_arm_dirs}"
            )
        group_seed_sets: dict[str, set[int]] = {}
        for arm in ARMS:
            arm_dir = group_dir / arm
            if not arm_dir.is_dir():
                errors.append(f"Group {group:02d}/{arm} 目录缺失。")
                group_seed_sets[arm] = set()
                continue
            seed_dirs = sorted(p for p in arm_dir.iterdir() if p.is_dir())
            malformed_dirs = [p.name for p in seed_dirs if not re.fullmatch(r"seed\d+", p.name)]
            if malformed_dirs:
                errors.append(
                    f"Group {group:02d}/{arm} 含非法 seed 目录：{malformed_dirs}"
                )
            valid_seed_dirs = [
                p for p in seed_dirs if re.fullmatch(r"seed\d+", p.name)
            ]
            seeds = [parse_seed_dir(p) for p in valid_seed_dirs]
            group_seed_sets[arm] = set(seeds)
            duplicate_seed_names = sorted(seed for seed, n in Counter(seeds).items() if n > 1)
            if duplicate_seed_names:
                errors.append(
                    f"Group {group:02d}/{arm} 重复 seed：{duplicate_seed_names}"
                )
            if len(seeds) != EXPECTED_EPISODES_PER_ARM:
                errors.append(
                    f"Group {group:02d}/{arm} 有 {len(seeds)} 个 seed，"
                    f"预期 {EXPECTED_EPISODES_PER_ARM}。"
                )

            for seed_dir in valid_seed_dirs:
                seed = parse_seed_dir(seed_dir)
                present = {p.name for p in seed_dir.iterdir() if p.is_file()}
                missing = sorted(set(REQUIRED_EPISODE_FILES) - present)
                extra = sorted(present - set(REQUIRED_EPISODE_FILES))
                seed_audit_rows.append(
                    {
                        "test_set": RAW_DATASET_DIR,
                        "arm": arm,
                        "display_seed_alias": seed,
                        "required_file_count": len(REQUIRED_EPISODE_FILES),
                        "missing_files": ";".join(missing),
                        "extra_files": ";".join(extra),
                        "complete": not missing,
                    }
                )
                if missing:
                    errors.append(
                        f"Group {group:02d}/{arm}/seed{seed} 缺失：{missing}"
                    )
                    continue
                if extra:
                    warnings.append(
                        f"Group {group:02d}/{arm}/seed{seed} 含额外文件：{extra}"
                    )
                try:
                    case_identity = read_json(seed_dir / "case_identity.json")
                    metrics = read_json(seed_dir / "metrics.json")
                    preflight = read_json(seed_dir / "preflight.json")
                    provenance = read_json(seed_dir / "provenance.json")
                    config = read_yaml(seed_dir / "config_resolved.yaml")
                    config_file_sha256 = sha256_file(
                        seed_dir / "config_resolved.yaml"
                    )
                    actual_config_hash = canonical_config_hash(config)
                    recorded_config_hash = (
                        (seed_dir / "config_sha256.txt")
                        .read_text(encoding="utf-8")
                        .strip()
                        .lower()
                    )
                except Exception as exc:
                    errors.append(
                        f"Group {group:02d}/{arm}/seed{seed} 解析失败：{exc}"
                    )
                    continue

                for required_metric in (
                    "success",
                    "collision",
                    "termination_reason",
                    "steps",
                    "final_goal_distance",
                    "minimum_clearance",
                    "planner_compute_ms_mean",
                    "planner_compute_ms_p95",
                ):
                    if required_metric not in metrics:
                        errors.append(
                            f"Group {group:02d}/{arm}/seed{seed} 缺少指标 "
                            f"{required_metric}。"
                        )
                field_counts.update(metrics.keys())

                hash_candidates = {
                    "config_sha256.txt": recorded_config_hash,
                    "actual": actual_config_hash,
                    "preflight.resolved_config_hash": str(
                        preflight.get("resolved_config_hash", "")
                    ).lower(),
                    "provenance.config_hash": str(
                        provenance.get("config_hash", "")
                    ).lower(),
                }
                nonempty_hashes = {value for value in hash_candidates.values() if value}
                config_hash_ok = (
                    len(nonempty_hashes) == 1 and actual_config_hash in nonempty_hashes
                )
                if not config_hash_ok:
                    errors.append(
                        f"Group {group:02d}/{arm}/seed{seed} config hash 不一致："
                        f"{hash_candidates}"
                    )

                rng_seed_original = int(
                    case_identity.get("rng_seed_original", -1)
                )
                dir_seed_matches = (
                    int(case_identity.get("display_seed_alias", -1)) == seed
                    and str(case_identity.get("arm_short", "")) == arm
                    and int(preflight.get("seed", -1)) == rng_seed_original
                    and int(nested_get(metrics, "metadata.seed", -1))
                    == rng_seed_original
                )
                if not dir_seed_matches:
                    errors.append(
                        f"Group {group:02d}/{arm}/seed{seed} 的内部 seed 不一致。"
                    )
                arm_name = str(preflight.get("arm", ""))
                if not arm_name.startswith(arm):
                    errors.append(
                        f"Group {group:02d}/{arm}/seed{seed} preflight arm={arm_name!r}。"
                    )

                method_revisions = recursive_values_for_key(
                    config, "method_revisions_applied"
                )
                map_method_revisions = recursive_values_for_key(
                    config, "map_method_revisions_applied"
                )
                method_revision_basis = recursive_values_for_key(
                    config, "method_revision_basis"
                )
                revisions_empty = all(
                    value in (None, [], {}) for value in
                    method_revisions + map_method_revisions + method_revision_basis
                )
                if not revisions_empty:
                    errors.append(
                        f"Group {group:02d}/{arm}/seed{seed} 不是 revision-disabled："
                        f"method={method_revisions}, map={map_method_revisions}, "
                        f"basis={method_revision_basis}"
                    )
                disabled_keys = set(preflight.get("disabled_overlay_keys") or [])
                missing_disabled_guards = sorted(
                    REVISION_KEYS_EXPECTED_DISABLED - disabled_keys
                )
                if missing_disabled_guards:
                    errors.append(
                        f"Group {group:02d}/{arm}/seed{seed} 未记录禁用 revision keys："
                        f"{missing_disabled_guards}"
                    )

                env_row = {
                    "test_set": RAW_DATASET_DIR,
                    "arm": arm,
                    "display_seed_alias": seed,
                    "config_hash": actual_config_hash,
                    "config_file_sha256": config_file_sha256,
                    "config_hash_verified": config_hash_ok,
                    "physical_scenario_hash": physical_scenario_hash(config),
                    "config_path_recorded": config.get("_config_path"),
                    "complex_map_source_recorded": nested_get(
                        config, "experiment.complex_map_source"
                    ),
                    "git_sha": provenance.get("git_sha"),
                    "checkpoint_preflight": preflight.get("checkpoint"),
                    "checkpoint_rl": nested_get(config, "rl.checkpoint"),
                    "rl_policy_id": nested_get(metrics, "metadata.rl_policy_id"),
                    "model_hash": nested_get(metrics, "metadata.model_hash"),
                    "torch_version_recorded": first_or_none(
                        recursive_values_for_key(metrics, "torch_version")
                        + recursive_values_for_key(preflight, "torch_version")
                    ),
                    "mujoco_version_recorded": nested_get(
                        metrics, "metadata.mujoco_version"
                    ),
                    "planner_device": nested_get(config, "planner.device"),
                    "method_revision_disabled": revisions_empty,
                    "revision_guard_keys_disabled": not missing_disabled_guards,
                    "frozen_source": preflight.get("frozen_source"),
                    "frozen_source_sha256": preflight.get("frozen_source_sha256"),
                }
                environment_rows.append(env_row)
                episodes.append(
                    Episode(
                        group=group,
                        arm=arm,
                        seed=seed,
                        path=seed_dir,
                        case_identity=case_identity,
                        metrics=metrics,
                        preflight=preflight,
                        provenance=provenance,
                        config=config,
                        config_sha256=actual_config_hash,
                    )
                )

        reference = group_seed_sets.get("B11", set())
        for arm in ARMS:
            observed = group_seed_sets.get(arm, set())
            missing_vs_b11 = sorted(reference - observed)
            extra_vs_b11 = sorted(observed - reference)
            if missing_vs_b11 or extra_vs_b11:
                errors.append(
                    f"Group {group:02d} {arm} 与 B11 seed 不匹配："
                    f"missing={missing_vs_b11}, extra={extra_vs_b11}"
                )

    if len(episodes) != len(GROUPS) * len(ARMS) * EXPECTED_EPISODES_PER_ARM:
        errors.append(
            f"可加载 episode={len(episodes)}，预期 "
            f"{len(GROUPS) * len(ARMS) * EXPECTED_EPISODES_PER_ARM}。"
        )

    episode_keys = [(e.group, e.arm, e.seed) for e in episodes]
    duplicate_keys = [key for key, n in Counter(episode_keys).items() if n > 1]
    if duplicate_keys:
        errors.append(f"重复 episode key：{duplicate_keys}")

    scenario_pair_count = 0
    scenario_mismatches: list[dict[str, Any]] = []
    episodes_by_group_seed: dict[tuple[int, int], list[Episode]] = defaultdict(list)
    for episode in episodes:
        episodes_by_group_seed[(episode.group, episode.seed)].append(episode)
    for (group, seed), matched in sorted(episodes_by_group_seed.items()):
        if len(matched) != len(ARMS):
            continue
        scenario_pair_count += 1
        hashes = {
            episode.arm: physical_scenario_hash(episode.config)
            for episode in matched
        }
        if len(set(hashes.values())) != 1:
            scenario_mismatches.append(
                {"group": group, "seed": seed, "hashes_by_arm": hashes}
            )
    if scenario_mismatches:
        errors.append(
            "同 seed 四臂的展开物理场景不一致："
            f"{scenario_mismatches[:10]}"
        )

    write_csv(output_root / "seed_integrity_audit.csv", seed_audit_rows)
    write_csv(output_root / "config_environment_audit.csv", environment_rows)
    field_rows = [
        {
            "field": field,
            "episode_count": count,
            "coverage_fraction": count / max(len(episodes), 1),
        }
        for field, count in sorted(field_counts.items())
    ]
    write_csv(output_root / "metrics_field_availability.csv", field_rows)

    audit = {
        "passed": not errors,
        "episodes_loaded": len(episodes),
        "expected_episodes": len(GROUPS) * len(ARMS) * EXPECTED_EPISODES_PER_ARM,
        "errors": errors,
        "warnings": warnings,
        "metric_field_count_union": len(field_counts),
        "required_revision_guard_keys": sorted(REVISION_KEYS_EXPECTED_DISABLED),
        "physical_scenario_matched_sets_checked": scenario_pair_count,
        "physical_scenario_mismatch_count": len(scenario_mismatches),
    }
    write_json(output_root / "dataset_structure_audit.json", audit)
    return episodes, audit


def first_or_none(values: Sequence[Any]) -> Any:
    return values[0] if values else None


def finite_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def values_for(episodes: Sequence[Episode], metric: str) -> np.ndarray:
    values = [finite_float(e.metrics.get(metric)) for e in episodes]
    return np.asarray([value for value in values if value is not None], dtype=float)


def describe(values: np.ndarray) -> dict[str, Any]:
    if values.size == 0:
        return {
            "n": 0,
            "mean": None,
            "sd": None,
            "min": None,
            "q25": None,
            "median": None,
            "q75": None,
            "max": None,
        }
    return {
        "n": int(values.size),
        "mean": float(np.mean(values)),
        "sd": float(np.std(values, ddof=1)) if values.size > 1 else None,
        "min": float(np.min(values)),
        "q25": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "q75": float(np.quantile(values, 0.75)),
        "max": float(np.max(values)),
    }


def wilson_interval(successes: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n <= 0:
        return math.nan, math.nan
    z = float(stats.norm.ppf(1 - alpha / 2))
    p = successes / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def termination_category(episode: Episode) -> str:
    reason = episode.termination_reason.lower()
    if episode.safe_success:
        return "safe_success"
    if episode.collision or "collision" in reason:
        return "collision"
    if "timeout" in reason or "max_step" in reason:
        return "timeout"
    if "livelock" in reason:
        return "livelock"
    return f"other:{episode.termination_reason}"


def group_arm_summaries(episodes: Sequence[Episode]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    all_metrics = {**CONTINUOUS_METRICS, **ADDITIONAL_METRICS}
    scopes: list[tuple[str, int | str, str, list[Episode]]] = []
    for group in GROUPS:
        for arm in ARMS:
            subset = [e for e in episodes if e.group == group and e.arm == arm]
            scopes.append(("group", group, arm, subset))
    for arm in ARMS:
        subset = [e for e in episodes if e.arm == arm]
        scopes.append(("combined", "all", arm, subset))

    for scope, group, arm, subset in scopes:
        n = len(subset)
        success = sum(e.safe_success for e in subset)
        boundary_values = [
            e.boundary_safe_success
            for e in subset
            if e.boundary_safe_success is not None
        ]
        boundary_success = sum(bool(value) for value in boundary_values)
        collision = sum(e.collision for e in subset)
        success_ci = wilson_interval(success, n)
        collision_ci = wilson_interval(collision, n)
        term_counts = Counter(termination_category(e) for e in subset)
        row: dict[str, Any] = {
            "scope": scope,
            "group": group,
            "arm": arm,
            "n": n,
            "safe_success_n": success,
            "safe_success_rate": success / n if n else None,
            "safe_success_ci95_low_wilson": success_ci[0],
            "safe_success_ci95_high_wilson": success_ci[1],
            "collision_n": collision,
            "collision_rate": collision / n if n else None,
            "collision_ci95_low_wilson": collision_ci[0],
            "collision_ci95_high_wilson": collision_ci[1],
            "boundary_safe_success_available_n": len(boundary_values),
            "boundary_safe_success_n": boundary_success,
            "boundary_safe_success_rate": (
                boundary_success / len(boundary_values)
                if boundary_values
                else None
            ),
            "timeout_n": term_counts.get("timeout", 0),
            "livelock_n": term_counts.get("livelock", 0),
            "other_termination_n": sum(
                count for key, count in term_counts.items() if key.startswith("other:")
            ),
            "termination_breakdown": json.dumps(
                dict(sorted(term_counts.items())), ensure_ascii=False, sort_keys=True
            ),
        }
        for metric in all_metrics:
            desc = describe(values_for(subset, metric))
            row[f"{metric}_available_n"] = desc["n"]
            row[f"{metric}_mean"] = desc["mean"]
            row[f"{metric}_sd"] = desc["sd"]
            row[f"{metric}_q25"] = desc["q25"]
            row[f"{metric}_median"] = desc["median"]
            row[f"{metric}_q75"] = desc["q75"]
            row[f"{metric}_min"] = desc["min"]
            row[f"{metric}_max"] = desc["max"]
        rows.append(row)
    return rows


def paired_episode_maps(
    episodes: Sequence[Episode], group: int | None = None
) -> dict[str, dict[tuple[int, int], Episode]]:
    by_arm: dict[str, dict[tuple[int, int], Episode]] = {arm: {} for arm in ARMS}
    for episode in episodes:
        if group is None or episode.group == group:
            by_arm[episode.arm][episode.key] = episode
    return by_arm


def bootstrap_paired_ci(
    values: np.ndarray,
    estimator: str,
    rng: np.random.Generator,
    samples: int = BOOTSTRAP_SAMPLES,
) -> tuple[float, float]:
    if values.size == 0:
        return math.nan, math.nan
    if np.allclose(values, values[0], rtol=0, atol=0):
        return float(values[0]), float(values[0])
    estimates: list[np.ndarray] = []
    remaining = samples
    while remaining > 0:
        batch = min(1000, remaining)
        indices = rng.integers(0, values.size, size=(batch, values.size))
        draws = values[indices]
        if estimator == "mean":
            estimates.append(np.mean(draws, axis=1))
        elif estimator == "median":
            estimates.append(np.median(draws, axis=1))
        else:
            raise ValueError(estimator)
        remaining -= batch
    merged = np.concatenate(estimates)
    return float(np.quantile(merged, 0.025)), float(np.quantile(merged, 0.975))


def holm_adjust(p_values: Sequence[float | None]) -> list[float | None]:
    valid = [(i, float(p)) for i, p in enumerate(p_values) if p is not None and math.isfinite(p)]
    adjusted: list[float | None] = [None] * len(p_values)
    running = 0.0
    m = len(valid)
    for rank, (index, p) in enumerate(sorted(valid, key=lambda item: item[1])):
        candidate = min(1.0, (m - rank) * p)
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def binary_paired_comparisons(
    episodes: Sequence[Episode],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for scope, group in [
        *[("group", value) for value in GROUPS],
        ("combined", None),
    ]:
        maps = paired_episode_maps(episodes, group)
        for baseline in BASELINES:
            keys = sorted(set(maps[baseline]) & set(maps["B11"]))
            base = np.asarray(
                [int(maps[baseline][key].safe_success) for key in keys], dtype=int
            )
            b11 = np.asarray(
                [int(maps["B11"][key].safe_success) for key in keys], dtype=int
            )
            both_fail = int(np.sum((base == 0) & (b11 == 0)))
            improved = int(np.sum((base == 0) & (b11 == 1)))
            regressed = int(np.sum((base == 1) & (b11 == 0)))
            both_success = int(np.sum((base == 1) & (b11 == 1)))
            discordant = improved + regressed
            p_exact = (
                float(stats.binomtest(improved, discordant, 0.5).pvalue)
                if discordant
                else 1.0
            )
            diffs = b11.astype(float) - base.astype(float)
            ci = bootstrap_paired_ci(diffs, "mean", rng)
            base_rate = float(np.mean(base)) if base.size else math.nan
            b11_rate = float(np.mean(b11)) if b11.size else math.nan
            matched_or = (
                improved / regressed
                if regressed > 0
                else (improved + 0.5) / (regressed + 0.5)
            )
            rows.append(
                {
                    "scope": scope,
                    "group": "all" if group is None else group,
                    "comparison": f"B11_vs_{baseline}",
                    "baseline": baseline,
                    "n_pairs": len(keys),
                    "both_fail_n00": both_fail,
                    "baseline_fail_b11_success_n01": improved,
                    "baseline_success_b11_fail_n10": regressed,
                    "both_success_n11": both_success,
                    "baseline_success_rate": base_rate,
                    "b11_success_rate": b11_rate,
                    "absolute_difference": b11_rate - base_rate,
                    "absolute_percentage_points": 100 * (b11_rate - base_rate),
                    "relative_improvement": (
                        (b11_rate - base_rate) / base_rate
                        if base_rate > 0
                        else None
                    ),
                    "paired_risk_difference_ci95_low_bootstrap": ci[0],
                    "paired_risk_difference_ci95_high_bootstrap": ci[1],
                    "matched_odds_ratio": matched_or,
                    "matched_odds_ratio_half_cell_corrected": regressed == 0,
                    "exact_mcnemar_p": p_exact,
                }
            )
    for scope in ("group", "combined"):
        indices = [i for i, row in enumerate(rows) if row["scope"] == scope]
        adjusted = holm_adjust([rows[i]["exact_mcnemar_p"] for i in indices])
        for index, value in zip(indices, adjusted):
            rows[index]["exact_mcnemar_p_holm_within_scope"] = value
    return rows


def rank_biserial(differences: np.ndarray) -> float:
    nonzero = differences[np.abs(differences) > EPS]
    if nonzero.size == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(nonzero), method="average")
    positive = float(np.sum(ranks[nonzero > 0]))
    negative = float(np.sum(ranks[nonzero < 0]))
    return (positive - negative) / (positive + negative)


def continuous_paired_comparisons(
    episodes: Sequence[Episode],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(BOOTSTRAP_SEED + 1)
    for scope, group in [
        *[("group", value) for value in GROUPS],
        ("combined", None),
    ]:
        maps = paired_episode_maps(episodes, group)
        for baseline in BASELINES:
            keys = sorted(set(maps[baseline]) & set(maps["B11"]))
            for metric, (label, direction) in CONTINUOUS_METRICS.items():
                pairs: list[tuple[float, float]] = []
                for key in keys:
                    base_value = finite_float(maps[baseline][key].metrics.get(metric))
                    b11_value = finite_float(maps["B11"][key].metrics.get(metric))
                    if base_value is not None and b11_value is not None:
                        pairs.append((base_value, b11_value))
                base = np.asarray([pair[0] for pair in pairs], dtype=float)
                b11 = np.asarray([pair[1] for pair in pairs], dtype=float)
                differences = b11 - base
                row: dict[str, Any] = {
                    "scope": scope,
                    "group": "all" if group is None else group,
                    "comparison": f"B11_vs_{baseline}",
                    "baseline": baseline,
                    "metric": metric,
                    "metric_cn": label,
                    "preferred_direction": direction,
                    "n_pairs_available": int(differences.size),
                    "missing_pair_count": len(keys) - int(differences.size),
                    "baseline_mean": (
                        float(np.mean(base)) if base.size else None
                    ),
                    "b11_mean": float(np.mean(b11)) if b11.size else None,
                    "mean_difference_b11_minus_baseline": (
                        float(np.mean(differences)) if differences.size else None
                    ),
                    "median_difference_b11_minus_baseline": (
                        float(np.median(differences)) if differences.size else None
                    ),
                }
                if differences.size < 3:
                    row.update(
                        {
                            "normality_shapiro_p": None,
                            "test": "数据不足",
                            "statistic": None,
                            "p_value": None,
                            "effect_size_name": None,
                            "effect_size": None,
                            "ci_estimator": None,
                            "ci95_low": None,
                            "ci95_high": None,
                        }
                    )
                elif np.allclose(differences, differences[0], rtol=0, atol=0):
                    constant = float(differences[0])
                    sign_p = (
                        1.0
                        if abs(constant) <= EPS
                        else min(1.0, 2.0 * (0.5 ** int(differences.size)))
                    )
                    row.update(
                        {
                            "normality_shapiro_p": 1.0,
                            "test": "exact_sign_test_constant_difference",
                            "statistic": 0.0,
                            "p_value": sign_p,
                            "effect_size_name": "constant_difference",
                            "effect_size": constant,
                            "ci_estimator": "constant_difference",
                            "ci95_low": constant,
                            "ci95_high": constant,
                        }
                    )
                else:
                    shapiro_p = float(stats.shapiro(differences).pvalue)
                    if shapiro_p >= 0.05:
                        result = stats.ttest_rel(b11, base)
                        mean_diff = float(np.mean(differences))
                        sem = float(stats.sem(differences))
                        critical = float(
                            stats.t.ppf(0.975, df=differences.size - 1)
                        )
                        sd_diff = float(np.std(differences, ddof=1))
                        row.update(
                            {
                                "normality_shapiro_p": shapiro_p,
                                "test": "paired_t",
                                "statistic": float(result.statistic),
                                "p_value": float(result.pvalue),
                                "effect_size_name": "cohen_dz",
                                "effect_size": (
                                    mean_diff / sd_diff if sd_diff > EPS else None
                                ),
                                "ci_estimator": "mean_difference_t",
                                "ci95_low": mean_diff - critical * sem,
                                "ci95_high": mean_diff + critical * sem,
                            }
                        )
                    else:
                        result = stats.wilcoxon(
                            differences,
                            zero_method="wilcox",
                            correction=False,
                            alternative="two-sided",
                            method="auto",
                        )
                        ci = bootstrap_paired_ci(differences, "median", rng)
                        row.update(
                            {
                                "normality_shapiro_p": shapiro_p,
                                "test": "wilcoxon_signed_rank",
                                "statistic": float(result.statistic),
                                "p_value": float(result.pvalue),
                                "effect_size_name": "matched_rank_biserial",
                                "effect_size": rank_biserial(differences),
                                "ci_estimator": "median_difference_bootstrap",
                                "ci95_low": ci[0],
                                "ci95_high": ci[1],
                            }
                        )
                rows.append(row)
    for scope in ("group", "combined"):
        indices = [
            i
            for i, row in enumerate(rows)
            if row["scope"] == scope and row["p_value"] is not None
        ]
        adjusted = holm_adjust([rows[i]["p_value"] for i in indices])
        for index, value in zip(indices, adjusted):
            rows[index]["p_value_holm_within_scope"] = value
    return rows


def between_group_variability(
    summaries: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    group_rows = [row for row in summaries if row["scope"] == "group"]
    for arm in ARMS:
        arm_rows = [row for row in group_rows if row["arm"] == arm]
        for metric in (
            "safe_success_rate",
            "collision_rate",
            "steps_mean",
            "final_goal_distance_mean",
            "minimum_clearance_mean",
            "planner_compute_ms_mean_mean",
            "planner_compute_ms_p95_mean",
        ):
            values = np.asarray(
                [
                    float(row[metric])
                    for row in arm_rows
                    if row.get(metric) is not None
                ],
                dtype=float,
            )
            desc = describe(values)
            rows.append(
                {
                    "arm": arm,
                    "metric": metric,
                    "group_count": desc["n"],
                    "between_group_mean": desc["mean"],
                    "between_group_sd": desc["sd"],
                    "between_group_min": desc["min"],
                    "between_group_q25": desc["q25"],
                    "between_group_median": desc["median"],
                    "between_group_q75": desc["q75"],
                    "between_group_max": desc["max"],
                }
            )
    return rows


def b11_sampling_variability(
    summaries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    rows = sorted(
        (
            row
            for row in summaries
            if row["scope"] == "group" and row["arm"] == "B11"
        ),
        key=lambda row: int(row["group"]),
    )
    successes = np.asarray([int(row["safe_success_n"]) for row in rows], dtype=int)
    rates = successes / EXPECTED_EPISODES_PER_ARM
    pooled_success = int(np.sum(successes))
    pooled_n = len(rows) * EXPECTED_EPISODES_PER_ARM
    pooled_rate = pooled_success / pooled_n
    observed_sd = float(np.std(rates, ddof=1))
    expected_binomial_sd = math.sqrt(
        pooled_rate * (1 - pooled_rate) / EXPECTED_EPISODES_PER_ARM
    )
    max_success = int(np.max(successes))
    single_tail = float(
        stats.binom.sf(max_success - 1, EXPECTED_EPISODES_PER_ARM, pooled_rate)
    )
    any_tail = 1 - (1 - single_tail) ** len(rows)
    contingency = np.column_stack(
        [successes, EXPECTED_EPISODES_PER_ARM - successes]
    )
    chi2, p_value, dof, _ = stats.chi2_contingency(contingency)
    leave_one_out_rates = [
        (pooled_success - int(value)) / (pooled_n - EXPECTED_EPISODES_PER_ARM)
        for value in successes
    ]
    return {
        "group_success_counts": {
            f"Group {int(row['group']):02d}": int(value)
            for row, value in zip(rows, successes)
        },
        "group_rates": {
            f"Group {int(row['group']):02d}": float(value)
            for row, value in zip(rows, rates)
        },
        "pooled_success_n": pooled_success,
        "pooled_n": pooled_n,
        "pooled_rate": pooled_rate,
        "observed_between_group_rate_sd": observed_sd,
        "binomial_expected_single_group_rate_sd": expected_binomial_sd,
        "observed_sd_over_binomial_expected": (
            observed_sd / expected_binomial_sd
            if expected_binomial_sd > 0
            else None
        ),
        "min_group_rate": float(np.min(rates)),
        "median_group_rate": float(np.median(rates)),
        "max_group_rate": float(np.max(rates)),
        "max_group_success_n": max_success,
        "groups_at_max": [
            f"Group {int(row['group']):02d}"
            for row, value in zip(rows, successes)
            if int(value) == max_success
        ],
        "binomial_tail_single_group_at_least_observed_max": single_tail,
        "binomial_tail_any_of_ten_at_least_observed_max": any_tail,
        "homogeneity_chi_square": float(chi2),
        "homogeneity_df": int(dof),
        "homogeneity_p": float(p_value),
        "leave_one_group_out_rate_min": float(np.min(leave_one_out_rates)),
        "leave_one_group_out_rate_max": float(np.max(leave_one_out_rates)),
    }


def consistency_against_baselines(
    summaries: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    lookup = {
        (int(row["group"]), str(row["arm"])): float(row["safe_success_rate"])
        for row in summaries
        if row["scope"] == "group"
    }
    for baseline in BASELINES:
        differences = np.asarray(
            [
                lookup[(group, "B11")] - lookup[(group, baseline)]
                for group in GROUPS
            ],
            dtype=float,
        )
        rows.append(
            {
                "comparison": f"B11_vs_{baseline}",
                "groups_b11_higher": int(np.sum(differences > EPS)),
                "groups_equal": int(np.sum(np.abs(differences) <= EPS)),
                "groups_b11_lower": int(np.sum(differences < -EPS)),
                "mean_group_rate_difference": float(np.mean(differences)),
                "sd_group_rate_difference": float(np.std(differences, ddof=1)),
                "median_group_rate_difference": float(np.median(differences)),
                "min_group_rate_difference": float(np.min(differences)),
                "max_group_rate_difference": float(np.max(differences)),
            }
        )
    return rows


def hard_seed_rows(episodes: Sequence[Episode]) -> list[dict[str, Any]]:
    maps = paired_episode_maps(episodes)
    keys = sorted(set.intersection(*(set(maps[arm]) for arm in ARMS)))
    rows: list[dict[str, Any]] = []
    for group, seed in keys:
        arm_success = {
            arm: int(maps[arm][(group, seed)].safe_success) for arm in ARMS
        }
        arm_collision = {
            arm: int(maps[arm][(group, seed)].collision) for arm in ARMS
        }
        rows.append(
            {
                "group": group,
                "seed": seed,
                **{f"{arm}_safe_success": arm_success[arm] for arm in ARMS},
                **{f"{arm}_collision": arm_collision[arm] for arm in ARMS},
                "success_arm_count": sum(arm_success.values()),
                "failure_arm_count": len(ARMS) - sum(arm_success.values()),
                "collision_arm_count": sum(arm_collision.values()),
                "all_four_failed": sum(arm_success.values()) == 0,
                "all_four_collided": sum(arm_collision.values()) == len(ARMS),
            }
        )
    return rows


def anomalous_episode_rows(episodes: Sequence[Episode]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    metrics_to_check = tuple(CONTINUOUS_METRICS) + tuple(ADDITIONAL_METRICS)
    for episode in episodes:
        issues: list[str] = []
        success_flag = bool(episode.metrics.get("success"))
        if episode.safe_success and episode.collision:
            issues.append("safe_success_and_collision")
        if episode.safe_success and not success_flag:
            issues.append("safe_success_without_raw_success")
        steps = finite_float(episode.metrics.get("steps"))
        if steps is None or steps < 0:
            issues.append("invalid_steps")
        for metric in metrics_to_check:
            raw = episode.metrics.get(metric)
            if raw is not None and finite_float(raw) is None:
                issues.append(f"non_finite_or_non_numeric:{metric}")
        for metric in (
            "planner_compute_ms_mean",
            "planner_compute_ms_p95",
            "planner_compute_ms_max",
        ):
            value = finite_float(episode.metrics.get(metric))
            if value is not None and value < 0:
                issues.append(f"negative:{metric}")
        if issues:
            rows.append(
                {
                    "group": episode.group,
                    "arm": episode.arm,
                    "seed": episode.seed,
                    "issues": ";".join(issues),
                    "path": str(episode.path),
                }
            )
    return rows


def environment_summary(episodes: Sequence[Episode]) -> dict[str, Any]:
    def unique(values: Iterable[Any]) -> list[Any]:
        rendered = {
            json.dumps(value, ensure_ascii=False, sort_keys=True, default=json_default)
            for value in values
        }
        return [json.loads(value) for value in sorted(rendered)]

    arm_contracts = {
        arm: unique(
            nested_get(e.config, "complex_method_contract.arm_contract")
            for e in episodes
            if e.arm == arm
        )
        for arm in ARMS
    }
    map_sources_by_arm = {
        arm: unique(
            nested_get(e.config, "experiment.complex_map_source")
            for e in episodes
            if e.arm == arm
        )
        for arm in ARMS
    }
    config_paths_by_arm = {
        arm: unique(
            e.config.get("_config_path")
            for e in episodes
            if e.arm == arm
        )
        for arm in ARMS
    }
    return {
        "episode_count": len(episodes),
        "arm_contracts": arm_contracts,
        "physical_scenario_hash_values": unique(
            physical_scenario_hash(e.config) for e in episodes
        ),
        "complex_map_source_values_by_arm": map_sources_by_arm,
        "config_path_values_by_arm": config_paths_by_arm,
        "git_sha_values": unique(e.provenance.get("git_sha") for e in episodes),
        "mujoco_version_values": unique(
            nested_get(e.metrics, "metadata.mujoco_version") for e in episodes
        ),
        "torch_version_values_recorded": unique(
            first_or_none(
                recursive_values_for_key(e.metrics, "torch_version")
                + recursive_values_for_key(e.preflight, "torch_version")
            )
            for e in episodes
        ),
        "checkpoint_preflight_values": unique(
            e.preflight.get("checkpoint") for e in episodes
        ),
        "checkpoint_rl_values": unique(
            nested_get(e.config, "rl.checkpoint") for e in episodes
        ),
        "model_hash_values": unique(
            nested_get(e.metrics, "metadata.model_hash") for e in episodes
        ),
        "planner_device_values": unique(
            nested_get(e.config, "planner.device") for e in episodes
        ),
        "frozen_source_values": unique(
            e.preflight.get("frozen_source") for e in episodes
        ),
        "frozen_source_sha256_values": unique(
            e.preflight.get("frozen_source_sha256") for e in episodes
        ),
        "config_hash_count": len({e.config_sha256 for e in episodes}),
        "revision_disabled_episode_count": sum(
            all(
                value in (None, [], {})
                for value in (
                    recursive_values_for_key(e.config, "method_revisions_applied")
                    + recursive_values_for_key(e.config, "map_method_revisions_applied")
                    + recursive_values_for_key(e.config, "method_revision_basis")
                )
            )
            for e in episodes
        ),
        "local_analysis_environment": {
            "python": sys.version.replace("\n", " "),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pyyaml": yaml.__version__,
        },
    }


def episode_index_rows(episodes: Sequence[Episode]) -> list[dict[str, Any]]:
    fields = tuple(CONTINUOUS_METRICS) + tuple(ADDITIONAL_METRICS)
    rows: list[dict[str, Any]] = []
    for episode in sorted(episodes, key=lambda e: (e.group, e.arm, e.seed)):
        row: dict[str, Any] = {
            "test_set": RAW_DATASET_DIR,
            "arm": episode.arm,
            "case_id": episode.case_identity.get("case_id"),
            "display_seed_alias": episode.seed,
            "rng_seed_original": episode.case_identity.get("rng_seed_original"),
            "safe_success": episode.safe_success,
            "success_raw": episode.metrics.get("success"),
            "boundary_safe_success": episode.boundary_safe_success,
            "collision": episode.collision,
            "termination_reason": episode.termination_reason,
            "termination_category": termination_category(episode),
            "config_sha256": episode.config_sha256,
            "episode_path": str(episode.path),
        }
        for field in fields:
            row[field] = episode.metrics.get(field)
        rows.append(row)
    return rows


def crosscheck_server_summaries(
    raw_root: Path,
    episodes: Sequence[Episode],
    summaries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Cross-check independent local endpoint derivation against server indices."""
    errors: list[str] = []
    server_index_path = raw_root / "episode_index.csv"
    server_summary_path = raw_root / "statistical_summary.json"
    local_episode_lookup = {
        (episode.group, episode.arm, episode.seed): episode for episode in episodes
    }
    checked_index_rows = 0
    seen: set[tuple[int, str, int]] = set()
    if not server_index_path.is_file():
        errors.append("服务器 episode_index.csv 缺失。")
    else:
        with server_index_path.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                match = re.fullmatch(
                    re.escape(RAW_DATASET_DIR), str(row["test_set"])
                )
                if not match:
                    errors.append(
                        f"服务器 index test_set 非法：{row['test_set']!r}"
                    )
                    continue
                key = (
                    10,
                    str(row["arm"]),
                    int(row["display_seed_alias"]),
                )
                if key in seen:
                    errors.append(f"服务器 index 重复 key：{key}")
                    continue
                seen.add(key)
                episode = local_episode_lookup.get(key)
                if episode is None:
                    errors.append(f"服务器 index 多余 key：{key}")
                    continue
                server_safe = bool(int(row["safe_success"]))
                server_collision = bool(int(row["collision"]))
                if server_safe != episode.safe_success:
                    errors.append(
                        f"服务器 index safe_success 与本地 producer 语义不一致：{key}"
                    )
                if server_collision != episode.collision:
                    errors.append(
                        f"服务器 index collision 与 metrics 不一致：{key}"
                    )
                checked_index_rows += 1
    missing_index_keys = sorted(set(local_episode_lookup) - seen)
    if missing_index_keys:
        errors.append(
            f"服务器 index 缺失 {len(missing_index_keys)} 个 episode key。"
        )

    checked_summary_cells = 0
    summary_lookup = {
        (row["scope"], str(row["group"]), row["arm"]): row for row in summaries
    }
    if not server_summary_path.is_file():
        errors.append("服务器 statistical_summary.json 缺失。")
    else:
        server_summary = read_json(server_summary_path)
        for group in GROUPS:
            group_name = RAW_DATASET_DIR
            server_group = nested_get(server_summary, f"groups.{group_name}", {})
            for arm in ARMS:
                local = summary_lookup[("group", str(group), arm)]
                server_cell = (
                    server_group.get(arm, {})
                    if isinstance(server_group, Mapping)
                    else {}
                )
                if int(server_cell.get("n", -1)) != int(local["n"]):
                    errors.append(
                        f"服务器 summary n 与本地不一致：Group {group:02d}/{arm}"
                    )
                if int(server_cell.get("safe_success", -1)) != int(
                    local["safe_success_n"]
                ):
                    errors.append(
                        "服务器 summary safe_success 与本地不一致："
                        f"Group {group:02d}/{arm}"
                    )
                checked_summary_cells += 1
        server_pooled = server_summary.get("pooled", {})
        for arm in ARMS:
            local = summary_lookup[("combined", "all", arm)]
            server_cell = (
                server_pooled.get(arm, {})
                if isinstance(server_pooled, Mapping)
                else {}
            )
            if int(server_cell.get("n", -1)) != int(local["n"]):
                errors.append(f"服务器 pooled n 与本地不一致：{arm}")
            if int(server_cell.get("safe_success", -1)) != int(
                local["safe_success_n"]
            ):
                errors.append(
                    f"服务器 pooled safe_success 与本地不一致：{arm}"
                )
            checked_summary_cells += 1

    return {
        "passed": not errors,
        "server_episode_index": str(server_index_path),
        "server_episode_index_sha256": (
            sha256_file(server_index_path) if server_index_path.is_file() else None
        ),
        "server_index_rows_checked": checked_index_rows,
        "server_statistical_summary": str(server_summary_path),
        "server_statistical_summary_sha256": (
            sha256_file(server_summary_path)
            if server_summary_path.is_file()
            else None
        ),
        "server_summary_cells_checked": checked_summary_cells,
        "safe_success_definition": "metrics.success AND NOT metrics.collision",
        "errors": errors,
    }


def fmt_number(value: Any, digits: int = 3) -> str:
    if value is None:
        return "数据未提供"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "数据未提供"
    return f"{number:.{digits}f}"


def fmt_percent(value: Any, digits: int = 1) -> str:
    if value is None:
        return "数据未提供"
    return f"{100 * float(value):.{digits}f}%"


def fmt_p_value(value: Any) -> str:
    if value is None:
        return "数据未提供"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "数据未提供"
    if number < 1e-4:
        return f"{number:.3e}"
    return f"{number:.5f}"


def fmt_recorded_values(values: Sequence[Any]) -> str:
    meaningful = [value for value in values if value not in (None, "", "unknown")]
    if not meaningful:
        if any(value == "unknown" for value in values):
            return "unknown（已记录，但无法定位代码提交）"
        return "数据未提供"
    return "`" + "`, `".join(str(value) for value in meaningful) + "`"


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    def safe(value: Any) -> str:
        return str(value).replace("|", r"\|").replace("\n", " ")

    output = [
        "| " + " | ".join(safe(value) for value in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    output.extend(
        "| " + " | ".join(safe(value) for value in row) + " |" for row in rows
    )
    return "\n".join(output)


def report_markdown(
    raw_root: Path,
    output_root: Path,
    transfer_gate: Mapping[str, Any],
    structure_audit: Mapping[str, Any],
    server_crosscheck: Mapping[str, Any],
    summaries: Sequence[Mapping[str, Any]],
    binary_rows: Sequence[Mapping[str, Any]],
    continuous_rows: Sequence[Mapping[str, Any]],
    variability_rows: Sequence[Mapping[str, Any]],
    b11_variability: Mapping[str, Any],
    consistency_rows: Sequence[Mapping[str, Any]],
    hard_rows: Sequence[Mapping[str, Any]],
    anomaly_rows: Sequence[Mapping[str, Any]],
    env_summary: Mapping[str, Any],
) -> str:
    lines: list[str] = [
        "# Redesign E 四臂独立测试统计分析",
        "",
        "> 本报告仅在传输 manifest、服务器 SHA-256/size 清单、本地逐文件",
        "> SHA-256 校验和 2,000 episode 结构审计全部通过后生成。",
        "> “事实”指原始 metrics/config/provenance 的直接汇总；“推断”会显式标注。",
        "",
        "## 1. 数据与完整性审计",
        "",
        f"- 原始数据目录（只读）：`{raw_root}`",
        f"- 独立分析目录：`{output_root}`",
        f"- 服务器清单条目：{transfer_gate['expected_hash_entries']:,}",
        f"- 本地 SHA-256 通过：{transfer_gate['verified_files']:,}/"
        f"{transfer_gate['expected_hash_entries']:,}",
        f"- 校验总字节数：{transfer_gate['verified_bytes']:,}",
        f"- 结构审计：{structure_audit['episodes_loaded']:,}/"
        f"{structure_audit['expected_episodes']:,} episode",
        f"- 同 seed 四臂展开物理场景核对："
        f"{structure_audit['physical_scenario_matched_sets_checked']:,}/500，"
        f"不一致 {structure_audit['physical_scenario_mismatch_count']:,}",
        f"- 服务器 episode index 逐项交叉核对："
        f"{server_crosscheck['server_index_rows_checked']:,}/2,000",
        f"- 服务器组/arm 与 pooled 摘要单元格交叉核对："
        f"{server_crosscheck['server_summary_cells_checked']:,}/44",
        f"- 原始树写入：否；分析器只读输入。",
        "",
        "每组四个 arm 均要求 50 个 seed，且组内 seed 集合完全匹配；"
        "重复、缺失或额外 seed 均会使 Gate 失败并阻止本报告生成。",
        "",
        "## 2. 实验配置和四个 Arms",
        "",
        "四个 arm 名称以各 episode 的 `preflight.json` 为准：B00、B01、B10、B11。"
        "正式主结果 `safe_success` 按项目 producer 定义为 "
        "`metrics.success AND NOT metrics.collision`；"
        "`boundary_safe_success` 是更严格的次级边界安全指标，单独报告。"
        "`collision` 和 `termination_reason` 分别用于安全与终止模式统计。",
        "",
        f"- METHOD_REVISION disabled：{env_summary['revision_disabled_episode_count']:,}/"
        f"{env_summary['episode_count']:,} episode",
        f"- 记录的 Git SHA："
        f"{fmt_recorded_values(env_summary['git_sha_values'])}",
        f"- 记录的 MuJoCo 版本："
        f"{fmt_recorded_values(env_summary['mujoco_version_values'])}",
        f"- 记录的 Torch 版本："
        f"{fmt_recorded_values(env_summary['torch_version_values_recorded'])}",
        f"- planner device："
        f"{fmt_recorded_values(env_summary['planner_device_values'])}",
        f"- config SHA-256 唯一值数：{env_summary['config_hash_count']:,}（均逐项核验）",
        f"- checkpoint（preflight）："
        f"{fmt_recorded_values(env_summary['checkpoint_preflight_values'])}",
        f"- RL checkpoint："
        f"{fmt_recorded_values(env_summary['checkpoint_rl_values'])}",
        f"- model hash："
        f"{fmt_recorded_values(env_summary['model_hash_values'])}",
        f"- frozen source："
        f"{fmt_recorded_values(env_summary['frozen_source_values'])}",
        f"- frozen source SHA-256："
        f"{fmt_recorded_values(env_summary['frozen_source_sha256_values'])}",
        "",
    ]

    arm_contract_rows: list[list[Any]] = []
    for arm in ARMS:
        contracts = env_summary["arm_contracts"].get(arm, [])
        contract = contracts[0] if len(contracts) == 1 else {}
        arm_contract_rows.append(
            [
                arm,
                contract.get("learning", "数据未提供"),
                contract.get("probability", "数据未提供"),
                contract.get("predictor", "数据未提供"),
                contract.get("icode", "数据未提供"),
                contract.get("hss", "数据未提供"),
            ]
        )
    lines.extend(
        [
            "### 四臂冻结契约（事实）",
            "",
            markdown_table(
                ["Arm", "Learning", "Probability", "Predictor", "ICODE", "HSS"],
                arm_contract_rows,
            ),
            "",
            f"- 展开物理场景 bundle（`scene`、`task`、初始状态、max_steps、"
            f"碰撞停止设置）SHA-256 唯一值数："
            f"{len(env_summary['physical_scenario_hash_values'])}；"
            f"500 个 matched seed 四臂全部一致。",
            "- `experiment.complex_map_source` 的记录字符串按 arm 不一致："
            f"B00={fmt_recorded_values(env_summary['complex_map_source_values_by_arm']['B00'])}；"
            f"B01={fmt_recorded_values(env_summary['complex_map_source_values_by_arm']['B01'])}；"
            f"B10={fmt_recorded_values(env_summary['complex_map_source_values_by_arm']['B10'])}；"
            f"B11={fmt_recorded_values(env_summary['complex_map_source_values_by_arm']['B11'])}。",
            "- `_config_path` 也存在 Linux/Windows 来源字符串差异。"
            "由于展开后的物理场景 bundle 逐 seed 完全相同，这没有形成已观测的"
            "四臂物理地图混杂；但路径 provenance 不一致且 Git SHA 为 unknown，"
            "仍作为审计风险保留。完整值见 `config_environment_audit.csv`。",
            "",
        ]
    )

    summary_lookup = {
        (row["scope"], str(row["group"]), row["arm"]): row for row in summaries
    }
    binary_lookup = {
        (row["scope"], str(row["group"]), row["baseline"]): row
        for row in binary_rows
    }

    for group in GROUPS:
        lines.extend([f"## {group + 2}. Group {group:02d} 独立分析", ""])
        outcome_rows: list[list[Any]] = []
        performance_rows: list[list[Any]] = []
        additional_rows: list[list[Any]] = []
        for arm in ARMS:
            row = summary_lookup[("group", str(group), arm)]
            outcome_rows.append(
                [
                    arm,
                    f"{row['safe_success_n']}/50 ({fmt_percent(row['safe_success_rate'])})",
                    f"{row['boundary_safe_success_n']}/"
                    f"{row['boundary_safe_success_available_n']} "
                    f"({fmt_percent(row['boundary_safe_success_rate'])})",
                    f"{row['collision_n']}/50 ({fmt_percent(row['collision_rate'])})",
                    row["timeout_n"],
                    row["livelock_n"],
                    row["other_termination_n"],
                    row["termination_breakdown"],
                ]
            )
            performance_rows.append(
                [
                    arm,
                    f"{fmt_number(row['steps_mean'], 1)} / "
                    f"{fmt_number(row['steps_q25'], 1)} / "
                    f"{fmt_number(row['steps_median'], 1)} / "
                    f"{fmt_number(row['steps_q75'], 1)}",
                    f"{fmt_number(row['final_goal_distance_mean'])} / "
                    f"{fmt_number(row['final_goal_distance_median'])}",
                    f"{fmt_number(row['minimum_clearance_mean'])} / "
                    f"{fmt_number(row['minimum_clearance_median'])}",
                    f"{fmt_number(row['planner_compute_ms_mean_mean'], 1)} / "
                    f"{fmt_number(row['planner_compute_ms_p95_mean'], 1)}",
                ]
            )
            additional_rows.append(
                [
                    arm,
                    fmt_number(row["path_progress_ratio_max_mean"]),
                    fmt_number(row["stuck_steps_mean"], 1),
                    fmt_number(row["safety_interventions_mean"], 1),
                    fmt_number(
                        row[
                            "known_static_map_candidate_feasible_fraction_mean_mean"
                        ]
                    ),
                    fmt_percent(row["planner_deadline_miss_rate_mean"]),
                    fmt_number(row["control_jerk_mean"]),
                ]
            )
        lines.extend(
            [
                "### 结果与终止模式（事实）",
                "",
                markdown_table(
                    [
                        "Arm",
                        "Safe success",
                        "Boundary-safe success",
                        "Collision",
                        "Timeout",
                        "Livelock",
                        "其他",
                        "完整终止分布",
                    ],
                    outcome_rows,
                ),
                "",
                "### 连续指标（事实）",
                "",
                "步数列依次为均值 / Q1 / 中位数 / Q3；其余成对数值为均值 / 中位数。",
                "",
                markdown_table(
                    [
                        "Arm",
                        "Steps",
                        "Final distance (m)",
                        "Minimum clearance (m)",
                        "Planner mean / episode P95 (ms)",
                    ],
                    performance_rows,
                ),
                "",
                "### 数据中存在的其他关键指标（事实）",
                "",
                markdown_table(
                    [
                        "Arm",
                        "路线进度",
                        "Stuck steps",
                        "Safety interventions",
                        "候选可行比例",
                        "Deadline miss",
                        "Control jerk",
                    ],
                    additional_rows,
                ),
                "",
                "### B11 组内配对成功比较",
                "",
                markdown_table(
                    [
                        "比较",
                        "两者失败",
                        "B11 / baseline",
                        "改善 / 退化",
                        "两者成功",
                        "绝对差",
                        "相对改善",
                        "RD 95% CI",
                        "Exact McNemar p",
                    ],
                    [
                        [
                            f"B11 vs {baseline}",
                            binary_lookup[("group", str(group), baseline)][
                                "both_fail_n00"
                            ],
                            f"{fmt_percent(binary_lookup[('group', str(group), baseline)]['b11_success_rate'])} / "
                            f"{fmt_percent(binary_lookup[('group', str(group), baseline)]['baseline_success_rate'])}",
                            f"{binary_lookup[('group', str(group), baseline)]['baseline_fail_b11_success_n01']} / "
                            f"{binary_lookup[('group', str(group), baseline)]['baseline_success_b11_fail_n10']}",
                            binary_lookup[("group", str(group), baseline)][
                                "both_success_n11"
                            ],
                            f"{binary_lookup[('group', str(group), baseline)]['absolute_percentage_points']:.1f} pp",
                            fmt_percent(
                                binary_lookup[('group', str(group), baseline)][
                                    "relative_improvement"
                                ]
                            ),
                            f"[{fmt_percent(binary_lookup[('group', str(group), baseline)]['paired_risk_difference_ci95_low_bootstrap'])}, "
                            f"{fmt_percent(binary_lookup[('group', str(group), baseline)]['paired_risk_difference_ci95_high_bootstrap'])}]",
                            fmt_p_value(
                                binary_lookup[('group', str(group), baseline)][
                                    "exact_mcnemar_p"
                                ]
                            ),
                        ]
                        for baseline in BASELINES
                    ],
                ),
                "",
                "### B11 组内配对连续指标",
                "",
                "差值为 B11−baseline；完整检验选择、效应量和 Holm 校正值见 "
                "`paired_continuous_comparisons.csv`。",
                "",
                markdown_table(
                    [
                        "比较",
                        "指标",
                        "N",
                        "均值差",
                        "中位差",
                        "方法",
                        "95% CI",
                        "效应量",
                        "p",
                    ],
                    [
                        [
                            row["comparison"],
                            row["metric_cn"],
                            row["n_pairs_available"],
                            fmt_number(
                                row["mean_difference_b11_minus_baseline"]
                            ),
                            fmt_number(
                                row["median_difference_b11_minus_baseline"]
                            ),
                            row["test"],
                            f"[{fmt_number(row['ci95_low'])}, "
                            f"{fmt_number(row['ci95_high'])}]",
                            f"{row['effect_size_name']}="
                            f"{fmt_number(row['effect_size'])}",
                            fmt_p_value(row["p_value"]),
                        ]
                        for row in continuous_rows
                        if row["scope"] == "group"
                        and int(row["group"]) == group
                    ],
                ),
                "",
            ]
        )

    lines.extend(
        [
            "## 13. 实验内比较",
            "",
            "下表统计 B11 相对各基线的方向一致性。",
            "",
            markdown_table(
                [
                    "比较",
                    "B11 更高组数",
                    "相同组数",
                    "B11 更低组数",
                    "平均组差",
                    "组差 SD",
                    "范围",
                ],
                [
                    [
                        row["comparison"],
                        row["groups_b11_higher"],
                        row["groups_equal"],
                        row["groups_b11_lower"],
                        fmt_percent(row["mean_group_rate_difference"]),
                        fmt_percent(row["sd_group_rate_difference"]),
                        f"{fmt_percent(row['min_group_rate_difference'])}–"
                        f"{fmt_percent(row['max_group_rate_difference'])}",
                    ]
                    for row in consistency_rows
                ],
            ),
            "",
            "各 arm 的组间均值、标准差、中位数、最小值和最大值保存在 "
            "`between_group_variability.csv`；成功率的核心结果如下。",
            "",
            markdown_table(
                ["Arm", "组均值", "组 SD", "组中位数", "最小", "最大"],
                [
                    [
                        row["arm"],
                        fmt_percent(row["between_group_mean"]),
                        fmt_percent(row["between_group_sd"]),
                        fmt_percent(row["between_group_median"]),
                        fmt_percent(row["between_group_min"]),
                        fmt_percent(row["between_group_max"]),
                    ]
                    for row in variability_rows
                    if row["metric"] == "safe_success_rate"
                ],
            ),
            "",
            "## 14. 500 Seeds 合并统计",
            "",
            "每个 arm 合并 10×50=500 个 episode。比例区间为 Wilson 95% CI。",
            "",
            markdown_table(
                [
                    "Arm",
                    "Safe success",
                    "95% CI",
                    "Boundary-safe",
                    "Collision",
                    "95% CI",
                    "Timeout",
                    "Livelock",
                    "其他",
                ],
                [
                    [
                        arm,
                        f"{summary_lookup[('combined', 'all', arm)]['safe_success_n']}/500 "
                        f"({fmt_percent(summary_lookup[('combined', 'all', arm)]['safe_success_rate'])})",
                        f"[{fmt_percent(summary_lookup[('combined', 'all', arm)]['safe_success_ci95_low_wilson'])}, "
                        f"{fmt_percent(summary_lookup[('combined', 'all', arm)]['safe_success_ci95_high_wilson'])}]",
                        f"{summary_lookup[('combined', 'all', arm)]['boundary_safe_success_n']}/"
                        f"{summary_lookup[('combined', 'all', arm)]['boundary_safe_success_available_n']} "
                        f"({fmt_percent(summary_lookup[('combined', 'all', arm)]['boundary_safe_success_rate'])})",
                        f"{summary_lookup[('combined', 'all', arm)]['collision_n']}/500 "
                        f"({fmt_percent(summary_lookup[('combined', 'all', arm)]['collision_rate'])})",
                        f"[{fmt_percent(summary_lookup[('combined', 'all', arm)]['collision_ci95_low_wilson'])}, "
                        f"{fmt_percent(summary_lookup[('combined', 'all', arm)]['collision_ci95_high_wilson'])}]",
                        summary_lookup[("combined", "all", arm)]["timeout_n"],
                        summary_lookup[("combined", "all", arm)]["livelock_n"],
                        summary_lookup[("combined", "all", arm)][
                            "other_termination_n"
                        ],
                    ]
                    for arm in ARMS
                ],
            ),
            "",
            "## 15. B11 与各基线的配对检验",
            "",
            "二元结局使用双侧 exact McNemar 检验；效应量报告配对风险差（B11−baseline）"
            "及固定随机种子的配对 bootstrap 95% CI。Holm 校正值见 CSV。",
            "",
            markdown_table(
                [
                    "比较",
                    "N",
                    "两者失败",
                    "改善",
                    "退化",
                    "两者成功",
                    "绝对差",
                    "相对改善",
                    "RD 95% CI",
                    "McNemar p",
                ],
                [
                    [
                        f"B11 vs {baseline}",
                        binary_lookup[("combined", "all", baseline)]["n_pairs"],
                        binary_lookup[("combined", "all", baseline)]["both_fail_n00"],
                        binary_lookup[("combined", "all", baseline)][
                            "baseline_fail_b11_success_n01"
                        ],
                        binary_lookup[("combined", "all", baseline)][
                            "baseline_success_b11_fail_n10"
                        ],
                        binary_lookup[("combined", "all", baseline)][
                            "both_success_n11"
                        ],
                        f"{binary_lookup[('combined', 'all', baseline)]['absolute_percentage_points']:.1f} pp",
                        fmt_percent(
                            binary_lookup[("combined", "all", baseline)][
                                "relative_improvement"
                            ]
                        ),
                        f"[{fmt_percent(binary_lookup[('combined', 'all', baseline)]['paired_risk_difference_ci95_low_bootstrap'])}, "
                        f"{fmt_percent(binary_lookup[('combined', 'all', baseline)]['paired_risk_difference_ci95_high_bootstrap'])}]",
                        fmt_p_value(
                            binary_lookup[("combined", "all", baseline)][
                                "exact_mcnemar_p"
                            ]
                        ),
                    ]
                    for baseline in BASELINES
                ],
            ),
            "",
            "连续指标先检查逐对差值的 Shapiro–Wilk 正态性。差值未拒绝正态时使用"
            "配对 t 检验、均值差 95% t 区间和 Cohen dz；否则使用 Wilcoxon "
            "signed-rank、配对差中位数 bootstrap 95% CI 和 matched rank-biserial。"
            "差值统一定义为 B11−baseline；lower 指标的负值、higher 指标的正值代表 B11 更优。",
            "",
            markdown_table(
                [
                    "比较",
                    "指标",
                    "N",
                    "均值差",
                    "中位差",
                    "方法",
                    "95% CI",
                    "效应量",
                    "p",
                ],
                [
                    [
                        row["comparison"],
                        row["metric_cn"],
                        row["n_pairs_available"],
                        fmt_number(row["mean_difference_b11_minus_baseline"]),
                        fmt_number(row["median_difference_b11_minus_baseline"]),
                        row["test"],
                        f"[{fmt_number(row['ci95_low'])}, {fmt_number(row['ci95_high'])}]",
                        f"{row['effect_size_name']}={fmt_number(row['effect_size'])}",
                        fmt_p_value(row["p_value"]),
                    ]
                    for row in continuous_rows
                    if row["scope"] == "combined"
                ],
            ),
            "",
            "## 16. 组间波动与 80% 成功率问题",
            "",
            f"- B11 分组范围：{fmt_percent(b11_variability['min_group_rate'])}–"
            f"{fmt_percent(b11_variability['max_group_rate'])}；中位数 "
            f"{fmt_percent(b11_variability['median_group_rate'])}。",
            f"- B11 组间成功率 SD："
            f"{fmt_percent(b11_variability['observed_between_group_rate_sd'])}；"
            f"按汇总成功率和 n=50 的二项抽样预期 SD："
            f"{fmt_percent(b11_variability['binomial_expected_single_group_rate_sd'])}。",
            f"- 最高组：{b11_variability['groups_at_max']}，"
            f"{b11_variability['max_group_success_n']}/50。将汇总率作为生成率时，"
            f"单组达到至少该值的二项尾概率为 "
            f"{fmt_number(b11_variability['binomial_tail_single_group_at_least_observed_max'], 4)}；"
            f"重复分组中至少出现一次的机会为 "
            f"{fmt_number(b11_variability['binomial_tail_any_of_ten_at_least_observed_max'], 4)}。",
            f"- 分组比例同质性 χ²({b11_variability['homogeneity_df']})="
            f"{fmt_number(b11_variability['homogeneity_chi_square'], 3)}，"
            f"p={fmt_number(b11_variability['homogeneity_p'], 4)}。",
            f"- B11 汇总率：{fmt_percent(b11_variability['pooled_rate'])}；"
            f"leave-one-group-out 范围："
            f"{fmt_percent(b11_variability['leave_one_group_out_rate_min'])}–"
            f"{fmt_percent(b11_variability['leave_one_group_out_rate_max'])}。",
            "",
            "**推断：** 结果解释应保持在预先确定的实验设计、地图和 seed 范围内。",
            "",
            "## 17. 异常 Seed 与失败模式",
            "",
        ]
    )
    failure_distribution = Counter(int(row["failure_arm_count"]) for row in hard_rows)
    all_failed = [row for row in hard_rows if row["all_four_failed"]]
    all_collided = [row for row in hard_rows if row["all_four_collided"]]
    lines.extend(
        [
            "按同组同 seed 跨四种方法统计，失败 arm 数分布为："
            f"`{dict(sorted(failure_distribution.items()))}`。",
            "",
            f"- 四个 arm 均失败：{len(all_failed)} 个 seed。",
            f"- 四个 arm 均碰撞：{len(all_collided)} 个 seed。",
            f"- 自动异常审计记录：{len(anomaly_rows)} 个 episode（详见 "
            "`anomalous_episodes.csv`）。",
            "- 正式 safe success 与更严格 boundary-safe success 的差异为："
            + "；".join(
                f"{arm} "
                f"{summary_lookup[('combined', 'all', arm)]['safe_success_n']}"
                f" vs "
                f"{summary_lookup[('combined', 'all', arm)]['boundary_safe_success_n']}"
                for arm in ARMS
            )
            + "。该差异是 endpoint 定义差异，不被隐藏或改写为碰撞。",
            "",
            "一致困难 seed 的完整清单保存在 `hard_seed_consistency.csv`，避免在正文中"
            "选择性只展示最显眼案例。",
            "",
            "## 18. 可用于论文的主要结论",
            "",
            "以下表述只能以本报告实际汇总表为依据：",
            "",
            "1. 应报告每个 arm 的预定义实验结果与 Wilson 95% CI，"
            "不得在结果产生后改写 endpoint。",
            "2. B11 相对 B00/B01/B10 的主比较应同时给出配对翻转数、绝对百分点差、"
            "配对风险差区间和 exact McNemar 结果。",
            "3. “稳定优于”只在方向一致性、汇总效应和组间波动共同支持时使用；"
            "单组显著或单组 84% 不足以证明总体稳定。",
            "4. 连续指标结果属于补充证据，需同时考虑缺失率、检验选择、效应量及多重比较。",
            "",
            "## 19. 局限与审计说明",
            "",
            "- 本报告不把当前实验结果外推为其他地图。",
            "- exact McNemar 针对二元 safe success；连续指标的检验由逐对差值分布选择。",
            "- 配对 bootstrap 使用固定 seed "
            f"{BOOTSTRAP_SEED}、{BOOTSTRAP_SAMPLES:,} 次重采样。",
            "- 多重比较的 Holm 校正值已写入中间 CSV；正文保留原始 p 值并强调效应量。",
            "- 若服务器未记录 Torch 版本或 Git SHA 为 unknown，则只能标记为“数据未提供”"
            "或审计风险，不能用本机分析环境替代。",
            "- `experiment.complex_map_source` 与 `_config_path` 的字符串 provenance "
            "在 B11 和三个 baseline 间不一致；展开物理场景 bundle 的 500 组"
            "逐 seed 审计全部相同，因此未观察到地图几何混杂，但来源字符串问题"
            "仍降低了独立 provenance 可追溯性。",
            "- `planner_compute_ms_mean` 的汇总是 episode-level mean 的组间平均，"
            "不是按全部 planner step 加权的全局均值。",
            "",
            "## 20. 数据文件和分析脚本位置",
            "",
            f"- 分析器：`{output_root / 'analyze_independent_experiment.py'}`",
            f"- 完整性 Gate：`{output_root / 'integrity_gate.json'}`",
            f"- episode 索引：`{output_root / 'episode_index.csv'}`",
            f"- 组/arm 汇总：`{output_root / 'group_arm_summary.csv'}`",
            f"- 二元配对：`{output_root / 'paired_binary_comparisons.csv'}`",
            f"- 连续配对：`{output_root / 'paired_continuous_comparisons.csv'}`",
            f"- 组间波动：`{output_root / 'between_group_variability.csv'}`",
            f"- hard seed：`{output_root / 'hard_seed_consistency.csv'}`",
            f"- 环境审计：`{output_root / 'config_environment_audit.csv'}`",
            f"- 服务器摘要交叉核对："
            f"`{output_root / 'server_summary_crosscheck.json'}`",
            f"- 运行命令：`{output_root / 'RUN_COMMAND.txt'}`",
            "",
        ]
    )
    return "\n".join(lines)


def run_analysis(raw_root: Path, output_root: Path) -> int:
    raw_root = raw_root.resolve()
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if raw_root == output_root or raw_root in output_root.parents:
        raise ValueError("Output directory must be outside the raw input tree.")

    gate = audit_transfer_and_hashes(raw_root, output_root)
    if not gate.get("passed"):
        print(
            "[BLOCKED] Transfer/checksum Gate did not pass; no formal statistics "
            f"were generated. See {output_root / 'integrity_gate.json'}",
            file=sys.stderr,
        )
        return 2

    episodes, structure_audit = load_and_audit_episodes(raw_root, output_root)
    if not structure_audit.get("passed"):
        print(
            "[BLOCKED] Dataset structure Gate did not pass; no formal statistics "
            f"were generated. See {output_root / 'dataset_structure_audit.json'}",
            file=sys.stderr,
        )
        return 3

    summaries = group_arm_summaries(episodes)
    server_crosscheck = crosscheck_server_summaries(
        raw_root, episodes, summaries
    )
    write_json(output_root / "server_summary_crosscheck.json", server_crosscheck)
    if not server_crosscheck["passed"]:
        print(
            "[BLOCKED] Server summary cross-check failed; no formal statistics "
            f"were generated. See {output_root / 'server_summary_crosscheck.json'}",
            file=sys.stderr,
        )
        return 4
    binary_rows = binary_paired_comparisons(episodes)
    continuous_rows = continuous_paired_comparisons(episodes)
    variability_rows = between_group_variability(summaries)
    b11_variability = b11_sampling_variability(summaries)
    consistency_rows = consistency_against_baselines(summaries)
    hard_rows = hard_seed_rows(episodes)
    anomaly_rows = anomalous_episode_rows(episodes)
    env_summary = environment_summary(episodes)

    write_csv(output_root / "episode_index.csv", episode_index_rows(episodes))
    write_csv(output_root / "group_arm_summary.csv", summaries)
    write_csv(output_root / "paired_binary_comparisons.csv", binary_rows)
    write_csv(output_root / "paired_continuous_comparisons.csv", continuous_rows)
    write_csv(output_root / "between_group_variability.csv", variability_rows)
    write_csv(output_root / "b11_consistency_across_groups.csv", consistency_rows)
    write_csv(output_root / "hard_seed_consistency.csv", hard_rows)
    write_csv(output_root / "anomalous_episodes.csv", anomaly_rows)
    write_json(output_root / "b11_sampling_variability.json", b11_variability)
    write_json(output_root / "environment_summary.json", env_summary)
    analysis_metadata = {
        "analysis_completed_utc": utc_now(),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "safe_success_definition": "metrics.success AND NOT metrics.collision",
        "secondary_safety_endpoint": "metrics.boundary_safe_success",
        "rate_ci": "Wilson score 95%",
        "binary_test": "two-sided exact McNemar via conditional binomial test",
        "continuous_selection": (
            "paired t when Shapiro-Wilk p>=0.05; otherwise Wilcoxon signed-rank"
        ),
        "raw_root": str(raw_root),
        "output_root": str(output_root),
        "raw_tree_written": False,
    }
    write_json(output_root / "analysis_metadata.json", analysis_metadata)

    report = report_markdown(
        raw_root,
        output_root,
        gate,
        structure_audit,
        server_crosscheck,
        summaries,
        binary_rows,
        continuous_rows,
        variability_rows,
        b11_variability,
        consistency_rows,
        hard_rows,
        anomaly_rows,
        env_summary,
    )
    (output_root / "REDESIGNE_ANALYSIS_CN.md").write_text(
        report + "\n", encoding="utf-8"
    )
    print(f"[OK] Formal analysis written to {output_root}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="read-only raw root")
    parser.add_argument(
        "--output", required=True, type=Path, help="separate analysis output directory"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run_analysis(args.root, args.output)


if __name__ == "__main__":
    raise SystemExit(
        "Run analyze_redesignE_standalone.py; this file is its support module."
    )
