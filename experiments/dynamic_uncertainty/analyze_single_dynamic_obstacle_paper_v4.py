"""Outcome-locked analysis for the single-dynamic-obstacle paper-v4 study.

The ``audit`` command never reads an episode metric or trajectory.  The
``analyze`` command first requires a complete 360-block/1,860-episode formal
matrix and a matching execution manifest; only then may it open outcomes.
All inferential directions, multiplicity families and bootstrap seeds are
loaded from the pre-formal protocol.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import NormalDist
import sys

import numpy as np
import yaml

try:
    from scipy import stats
except ImportError:  # Primitive-only test environments need not install SciPy.
    stats = None


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = ROOT / "configs/research/single_dynamic_obstacle_paper_v4.yaml"
TANGO_TOLERANCE = 1.0e-12
REQUIRED_FORMAL_ARTIFACTS = (
    "paper_v4_job.json",
    "config_resolved.yaml",
    "provenance.json",
    "metrics.json",
    "trajectory.csv",
)


def _load_json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("JSON root must be a mapping: %s" % path)
    return value


def _load_yaml(path):
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("YAML root must be a mapping: %s" % path)
    return value


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_csv(path, rows):
    path = Path(path)
    rows = list(rows)
    keys = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value):
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _resolve(root, value):
    text = str(value).replace("\\", os.sep).replace("/", os.sep)
    path = Path(text)
    return (path if path.is_absolute() else Path(root) / path).resolve()


def _finite_vector(values, name="values"):
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or result.size == 0:
        raise ValueError("%s must be a nonempty vector" % name)
    if not np.isfinite(result).all():
        raise ValueError("%s contains nonfinite values" % name)
    return result


def paired_table(treatment, control):
    treatment = np.asarray(treatment, dtype=bool)
    control = np.asarray(control, dtype=bool)
    if treatment.shape != control.shape or treatment.ndim != 1:
        raise ValueError("paired binary arrays must be equal-length vectors")
    return {
        "n11": int(np.sum(treatment & control)),
        "n10": int(np.sum(treatment & ~control)),
        "n01": int(np.sum(~treatment & control)),
        "n00": int(np.sum(~treatment & ~control)),
    }


def exact_mcnemar_p(n10, n01):
    discordant = int(n10) + int(n01)
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, k)
        for k in range(min(int(n10), int(n01)) + 1)
    ) / (2.0 ** discordant)
    return float(min(1.0, 2.0 * tail))


def _constrained_discordance(n10, n01, n00_plus_n11, delta):
    """Constrained MLE of q=P(+1)+P(-1), given E[X]=delta."""

    delta = float(delta)
    lower = abs(delta) + 1.0e-14
    upper = 1.0 - 1.0e-14
    if lower >= upper:
        return min(1.0, max(abs(delta), 1.0))

    def log_likelihood(q):
        p10 = 0.5 * (q + delta)
        p01 = 0.5 * (q - delta)
        p0 = 1.0 - q
        value = 0.0
        for count, probability in (
            (int(n10), p10),
            (int(n01), p01),
            (int(n00_plus_n11), p0),
        ):
            if count:
                if probability <= 0.0:
                    return -float("inf")
                value += count * math.log(probability)
        return value

    lo, hi = lower, upper
    for _ in range(200):
        left = lo + (hi - lo) / 3.0
        right = hi - (hi - lo) / 3.0
        if log_likelihood(left) < log_likelihood(right):
            lo = left
        else:
            hi = right
        if hi - lo <= TANGO_TOLERANCE:
            break
    return float(0.5 * (lo + hi))


def tango_score_statistic(n10, n01, n_concordant, null_difference):
    n = int(n10) + int(n01) + int(n_concordant)
    if n <= 0:
        raise ValueError("paired sample size must be positive")
    delta = float(null_difference)
    q = _constrained_discordance(n10, n01, n_concordant, delta)
    variance = max(q - delta * delta, np.finfo(float).tiny) / n
    estimate = (int(n10) - int(n01)) / n
    return float((estimate - delta) / math.sqrt(variance))


def tango_matched_score_interval(
    n10, n01, n_concordant, confidence=0.95, side="two-sided"
):
    """Invert Tango's matched-pair score statistic by deterministic bisection."""

    if side not in ("two-sided", "lower", "upper"):
        raise ValueError("side must be two-sided, lower, or upper")
    alpha = 1.0 - float(confidence)
    critical = float(
        NormalDist().inv_cdf(1.0 - alpha / 2.0)
        if side == "two-sided"
        else NormalDist().inv_cdf(1.0 - alpha)
    )

    def root(target_sign):
        lo, hi = -1.0 + 1e-13, 1.0 - 1e-13
        target = target_sign * critical
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            value = tango_score_statistic(n10, n01, n_concordant, mid) - target
            if value > 0.0:
                lo = mid
            else:
                hi = mid
            if hi - lo <= TANGO_TOLERANCE:
                break
        return 0.5 * (lo + hi)

    lower = root(+1.0) if side in ("two-sided", "lower") else -1.0
    upper = root(-1.0) if side in ("two-sided", "upper") else 1.0
    return float(lower), float(upper)


def holm_adjust(p_values):
    values = [float(value) for value in p_values]
    if any(not 0.0 <= value <= 1.0 for value in values):
        raise ValueError("p-values must be in [0,1]")
    order = sorted(range(len(values)), key=lambda index: values[index])
    adjusted = [0.0] * len(values)
    running = 0.0
    count = len(values)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (count - rank) * values[index]))
        adjusted[index] = running
    return adjusted


def _wilcoxon_p(values):
    if stats is None:
        raise RuntimeError("formal paper-v4 analysis requires scipy")
    values = _finite_vector(values, "paired differences")
    if np.all(np.abs(values) <= 1.0e-15):
        return 1.0
    result = stats.wilcoxon(
        values,
        zero_method="pratt",
        correction=False,
        alternative="two-sided",
        method="approx",
    )
    return float(result.pvalue)


def _paired_t_interval(values, confidence=0.95):
    if stats is None:
        raise RuntimeError("formal paper-v4 analysis requires scipy")
    values = _finite_vector(values, "paired differences")
    mean = float(np.mean(values))
    if values.size < 2:
        return [mean, mean]
    standard_error = float(stats.sem(values))
    if standard_error <= 0.0:
        return [mean, mean]
    critical = float(stats.t.ppf(0.5 + confidence / 2.0, values.size - 1))
    return [mean - critical * standard_error, mean + critical * standard_error]


def _bootstrap_mean_intervals(
    matrix, replicates, seed, confidence=0.95, chunk_size=1024
):
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise ValueError("bootstrap matrix must be nonempty and two-dimensional")
    if not np.isfinite(matrix).all():
        raise ValueError("bootstrap matrix contains nonfinite values")
    replicates = int(replicates)
    if replicates <= 0:
        raise ValueError("bootstrap replicates must be positive")
    rng = np.random.default_rng(int(seed))
    samples = np.empty((replicates, matrix.shape[1]), dtype=np.float64)
    written = 0
    while written < replicates:
        size = min(int(chunk_size), replicates - written)
        indices = rng.integers(0, matrix.shape[0], size=(size, matrix.shape[0]))
        samples[written:written + size] = np.mean(matrix[indices], axis=1)
        written += size
    alpha = 0.5 * (1.0 - float(confidence))
    quantiles = np.quantile(samples, [alpha, 1.0 - alpha], axis=0)
    return [
        [float(quantiles[0, index]), float(quantiles[1, index])]
        for index in range(matrix.shape[1])
    ]


def binary_contrast(treatment, control):
    treatment = np.asarray(treatment, dtype=bool)
    control = np.asarray(control, dtype=bool)
    table = paired_table(treatment, control)
    n = int(treatment.size)
    if n == 0:
        raise ValueError("binary contrast has no pairs")
    two_sided = tango_matched_score_interval(
        table["n10"], table["n01"], table["n11"] + table["n00"]
    )
    lower = tango_matched_score_interval(
        table["n10"], table["n01"], table["n11"] + table["n00"], side="lower"
    )
    upper = tango_matched_score_interval(
        table["n10"], table["n01"], table["n11"] + table["n00"], side="upper"
    )
    return {
        "n": n,
        "treatment_events": int(np.sum(treatment)),
        "control_events": int(np.sum(control)),
        "treatment_rate": float(np.mean(treatment)),
        "control_rate": float(np.mean(control)),
        "treatment_minus_control": float(np.mean(treatment) - np.mean(control)),
        "paired_table": table,
        "exact_mcnemar_two_sided_p": exact_mcnemar_p(table["n10"], table["n01"]),
        "tango_two_sided_95ci": list(two_sided),
        "tango_one_sided_95ci_lower": float(lower[0]),
        "tango_one_sided_95ci_upper": float(upper[1]),
    }


def continuous_contrast(
    treatment,
    control,
    higher_is_better,
    bootstrap_replicates,
    bootstrap_seed,
):
    treatment = _finite_vector(treatment, "treatment")
    control = _finite_vector(control, "control")
    if treatment.shape != control.shape:
        raise ValueError("continuous contrast arrays must be paired")
    raw = treatment - control
    favorable = raw if bool(higher_is_better) else -raw
    bootstrap = _bootstrap_mean_intervals(
        raw, bootstrap_replicates, bootstrap_seed
    )[0]
    return {
        "n": int(raw.size),
        "higher_is_better": bool(higher_is_better),
        "treatment_mean": float(np.mean(treatment)),
        "control_mean": float(np.mean(control)),
        "treatment_median": float(np.median(treatment)),
        "control_median": float(np.median(control)),
        "mean_treatment_minus_control": float(np.mean(raw)),
        "median_treatment_minus_control": float(np.median(raw)),
        "mean_favorable_effect": float(np.mean(favorable)),
        "paired_t_two_sided_95ci": _paired_t_interval(raw),
        "whole_seed_bootstrap_mean_95ci": bootstrap,
        "wilcoxon_two_sided_p": _wilcoxon_p(raw),
    }


def factorial_effect_vectors(values_by_arm):
    required = (
        "B00_strong_nominal_mppi",
        "B10_learning_only",
        "B01_probability_only",
        "B11_full_proposed",
    )
    values = {name: _finite_vector(values_by_arm[name], name) for name in required}
    shapes = {value.shape for value in values.values()}
    if len(shapes) != 1:
        raise ValueError("factorial arms must contain the same paired seeds")
    b00 = values[required[0]]
    b10 = values[required[1]]
    b01 = values[required[2]]
    b11 = values[required[3]]
    return {
        "learning_main": 0.5 * ((b10 - b00) + (b11 - b01)),
        "probability_main": 0.5 * ((b01 - b00) + (b11 - b10)),
        "learning_by_probability_interaction": b11 - b10 - b01 + b00,
    }


def factorial_summary(values_by_arm, bootstrap_replicates, bootstrap_seed):
    effects = factorial_effect_vectors(values_by_arm)
    names = list(effects)
    matrix = np.column_stack([effects[name] for name in names])
    intervals = _bootstrap_mean_intervals(
        matrix, bootstrap_replicates, bootstrap_seed
    )
    raw_p = [_wilcoxon_p(effects[name]) for name in names]
    adjusted = holm_adjust(raw_p)
    return [
        {
            "effect": name,
            "n": int(matrix.shape[0]),
            "mean_favorable_effect": float(np.mean(effects[name])),
            "median_favorable_effect": float(np.median(effects[name])),
            "whole_seed_bootstrap_mean_95ci": intervals[index],
            "wilcoxon_two_sided_p": raw_p[index],
            "holm_adjusted_p_three_contrasts": adjusted[index],
        }
        for index, name in enumerate(names)
    ]


def _expected_jobs(registry):
    jobs = []
    seen = set()
    for block in registry["schedule"]:
        for arm in block["arm_sequence"]:
            key = (str(block["split"]), int(block["seed"]), str(arm))
            if key in seen:
                raise ValueError("duplicate formal job: %r" % (key,))
            seen.add(key)
            jobs.append({"block": block, "arm": str(arm), "key": key})
    return jobs


def _manifest_path(root, name):
    text = str(name).replace("\\", os.sep).replace("/", os.sep)
    path = Path(text)
    return path if path.is_absolute() else Path(root) / path


def audit_formal_completion(protocol_path=DEFAULT_PROTOCOL, formal_output=None):
    """Audit completeness and hashes without reading any episode outcome."""

    protocol_path = Path(protocol_path).resolve()
    root = protocol_path.parents[2]
    protocol = _load_yaml(protocol_path)
    registry_path = _resolve(root, protocol["formal_registry"])
    if not registry_path.is_file():
        raise FileNotFoundError("sealed registry missing: %s" % registry_path)
    registry = _load_yaml(registry_path)
    if registry.get("status") != "sealed_before_formal_execution":
        raise ValueError("formal registry is not sealed")
    if _canonical_sha256(registry["schedule"]) != registry["schedule_sha256"]:
        raise ValueError("registry schedule SHA-256 mismatch")
    if _canonical_sha256(registry["ablation_subset"]) != registry["ablation_subset_sha256"]:
        raise ValueError("registry ablation-subset SHA-256 mismatch")
    sidecar = registry_path.with_suffix(registry_path.suffix + ".sha256")
    if not sidecar.is_file():
        raise FileNotFoundError("registry SHA-256 sidecar missing")
    expected_registry_sha = sidecar.read_text(encoding="ascii").split()[0].lower()
    actual_registry_sha = _sha256(registry_path)
    if actual_registry_sha != expected_registry_sha:
        raise ValueError("registry file SHA-256 mismatch")

    jobs = _expected_jobs(registry)
    design = protocol["design"]
    if len(registry["splits"]["id"]) != int(design["core_seed_count_per_split"]):
        raise ValueError("ID seed count changed")
    if len(registry["splits"]["ood"]) != int(design["core_seed_count_per_split"]):
        raise ValueError("OOD seed count changed")
    if len(jobs) != int(design["total_episode_jobs"]):
        raise ValueError("formal job count changed")

    output = (
        Path(formal_output).resolve()
        if formal_output is not None
        else _resolve(root, protocol["formal_output"])
    )
    progress_path = output / "progress.json"
    schedule_path = output / "schedule.json"
    manifest_path = output / "execution_manifest.json"
    for path in (progress_path, schedule_path, manifest_path):
        if not path.is_file():
            raise RuntimeError("formal matrix is not complete; missing %s" % path)
    progress = _load_json(progress_path)
    required_progress = {
        "status": "complete",
        "completed_blocks": len(registry["schedule"]),
        "total_blocks": len(registry["schedule"]),
        "completed_episode_jobs": len(jobs),
        "total_episode_jobs": len(jobs),
        "outcomes_opened": False,
    }
    for key, expected in required_progress.items():
        if progress.get(key) != expected:
            raise RuntimeError(
                "formal outcomes remain locked: progress.%s=%r, expected %r"
                % (key, progress.get(key), expected)
            )
    if progress.get("failures"):
        raise RuntimeError("formal progress contains failures")
    schedule = _load_json(schedule_path)
    if schedule.get("scope") != "paper_v4_formal":
        raise ValueError("formal schedule scope mismatch")
    if schedule.get("blocks") != registry["schedule"]:
        raise ValueError("formal schedule differs from sealed registry")
    if schedule.get("schedule_sha256") != registry["schedule_sha256"]:
        raise ValueError("formal schedule hash differs from sealed registry")

    manifest = _load_json(manifest_path)
    if manifest.get("formal_experiment_started") is not True:
        raise ValueError("formal execution manifest was not activated")
    if _canonical_sha256(manifest["files"]) != manifest.get("manifest_sha256"):
        raise ValueError("execution manifest file-map hash mismatch")
    if _canonical_sha256(manifest["runtime"]) != manifest.get("runtime_sha256"):
        raise ValueError("execution manifest runtime hash mismatch")
    changed = []
    for name, expected in manifest["files"].items():
        path = _manifest_path(root, name).resolve()
        actual = _sha256(path) if path.is_file() else None
        if actual != str(expected).lower():
            changed.append({"path": str(path), "expected": expected, "actual": actual})
    if changed:
        raise ValueError("formal execution files changed: %s" % changed[:3])

    missing = []
    for job in jobs:
        split, seed, arm = job["key"]
        run_dir = output / "runs" / split / ("seed_%d" % seed) / arm
        job["run_dir"] = run_dir
        for name in REQUIRED_FORMAL_ARTIFACTS:
            if not (run_dir / name).is_file():
                missing.append(str(run_dir / name))
    if missing:
        raise RuntimeError(
            "formal outcomes remain locked: %d required artifacts missing; first=%s"
            % (len(missing), missing[0])
        )
    return {
        "status": "complete_outcomes_may_be_opened",
        "outcomes_read": False,
        "formal_output": str(output),
        "registry": str(registry_path),
        "registry_sha256": actual_registry_sha,
        "schedule_sha256": registry["schedule_sha256"],
        "ablation_subset_sha256": registry["ablation_subset_sha256"],
        "seed_blocks": len(registry["schedule"]),
        "episode_jobs": len(jobs),
        "execution_manifest_sha256": manifest["manifest_sha256"],
        "runtime_sha256": manifest["runtime_sha256"],
        "jobs": jobs,
        "protocol": protocol,
        "registry_data": registry,
        "root": root,
    }


def _read_trajectory(path):
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("trajectory is empty: %s" % path)
    return rows


def _number(row, key, default=0.0):
    value = row.get(key)
    if value in (None, ""):
        return float(default)
    return float(value)


def _qualified_direction_runs(trajectory, threshold=0.05, minimum_cycles=2):
    velocities = np.asarray(
        [_number(row, "applied_v") for row in trajectory], dtype=np.float64
    )
    signs = np.where(velocities > threshold, 1, np.where(velocities < -threshold, -1, 0))
    raw = []
    index = 0
    while index < signs.size:
        sign = int(signs[index])
        end = index + 1
        while end < signs.size and int(signs[end]) == sign:
            end += 1
        if sign and end - index >= int(minimum_cycles):
            raw.append({"sign": sign, "start": index, "end": end})
        index = end
    collapsed = []
    for run in raw:
        if collapsed and collapsed[-1]["sign"] == run["sign"]:
            collapsed[-1]["end"] = run["end"]
        else:
            collapsed.append(dict(run))
    return collapsed


def _release_delay(trajectory):
    times = np.asarray([_number(row, "time") for row in trajectory])
    closing = np.asarray([
        _number(row, "temporal_scan_closing_rate_mps") > 0.0
        for row in trajectory
    ], dtype=bool)
    forward = np.asarray([
        _number(row, "applied_v") > 0.05 for row in trajectory
    ], dtype=bool)
    delays = []
    censored = 0
    for index in range(1, max(1, len(trajectory) - 1)):
        if closing[index - 1] and not closing[index] and not closing[index + 1]:
            release = None
            for candidate in range(index + 2, len(trajectory) - 1):
                if forward[candidate] and forward[candidate + 1]:
                    release = candidate
                    break
            if release is None:
                delays.append(float(times[-1] - times[index]))
                censored += 1
            else:
                delays.append(float(times[release] - times[index]))
    return {
        "release_delay_event_count": len(delays),
        "release_delay_max_s": float(max(delays)) if delays else 0.0,
        "release_delay_mean_s": float(np.mean(delays)) if delays else 0.0,
        "release_delay_censored_events": int(censored),
    }


def _behavior_metrics(trajectory, block, protocol, episode_minimum_clearance):
    runs = _qualified_direction_runs(trajectory)
    times = np.asarray([_number(row, "time") for row in trajectory])
    direction_switches = max(0, len(runs) - 1)
    three_phase = 0
    for first, _, third in zip(runs, runs[1:], runs[2:]):
        if times[third["start"]] - times[first["start"]] <= 2.0 + 1.0e-12:
            three_phase += 1
    premature_reverse = 0
    for run in runs:
        if run["sign"] >= 0:
            continue
        row = trajectory[run["start"]]
        if _number(row, "clearance", -float("inf")) >= 0.80 and _number(
            row, "temporal_scan_ttc_s", float("inf")
        ) > 1.50:
            premature_reverse += 1

    windows = list(block["certificate"]["conflict_windows_s"])
    window_clearance = []
    for begin, end in windows:
        window_clearance.extend(
            _number(row, "clearance")
            for row in trajectory
            if float(begin) - 1.0e-12 <= _number(row, "time") <= float(end) + 1.0e-12
        )
    if window_clearance:
        conflict_q05 = float(np.quantile(window_clearance, 0.05))
        clearance_fallback = False
    else:
        conflict_q05 = float(episode_minimum_clearance)
        clearance_fallback = True

    contract = protocol["conflict_certificate"]
    start = np.asarray(contract["robot_start_xy"], dtype=np.float64)
    goal = np.asarray(contract["robot_goal_xy"], dtype=np.float64)
    displacement = goal - start
    route_length = float(np.linalg.norm(displacement))
    direction = displacement / route_length
    xy = np.asarray([
        [_number(row, "x"), _number(row, "y")] for row in trajectory
    ])
    progress = np.clip((xy - start) @ direction, 0.0, route_length)
    traversal = 0.0
    traversal_censored = False
    horizon = float(protocol["design"]["maximum_episode_steps"]) * float(
        protocol["design"]["control_period_s"]
    )
    speed = float(contract["ghost_speed_mps"])
    for begin, end in windows:
        entry = min(speed * float(begin), route_length)
        exit_value = min(speed * float(end), route_length)
        entered = np.flatnonzero(progress >= entry - 1.0e-12)
        if entered.size == 0:
            traversal_censored = True
            break
        exited = np.flatnonzero(
            (np.arange(progress.size) >= int(entered[0]))
            & (progress >= exit_value - 1.0e-12)
        )
        if exited.size == 0:
            traversal_censored = True
            break
        traversal += float(times[int(exited[0])] - times[int(entered[0])])
    if traversal_censored:
        traversal = horizon

    hard_or_closing = np.asarray([
        _number(row, "temporal_scan_risk_alpha") >= 1.0 - 1.0e-12
        or _number(row, "temporal_scan_closing_rate_mps") > 0.0
        for row in trajectory
    ], dtype=bool)
    stopped = np.asarray([
        abs(_number(row, "applied_v")) <= 0.05 for row in trajectory
    ], dtype=bool)
    risk_steps = int(np.sum(hard_or_closing))
    release = _release_delay(trajectory)
    return {
        "direction_switch_count": int(direction_switches),
        "three_phase_oscillation_count": int(three_phase),
        "premature_reverse_count": int(premature_reverse),
        "conflict_q05_clearance": conflict_q05,
        "conflict_q05_used_episode_fallback": bool(clearance_fallback),
        "conflict_window_sample_count": int(len(window_clearance)),
        "conflict_traversal_time_failure_penalized_s": float(traversal),
        "conflict_traversal_censored": bool(traversal_censored),
        "zero_speed_risk_steps": int(np.sum(hard_or_closing & stopped)),
        "risk_or_raw_closing_steps": risk_steps,
        "zero_speed_risk_fraction": (
            float(np.sum(hard_or_closing & stopped) / risk_steps)
            if risk_steps else 0.0
        ),
        **release,
    }


def _load_episode_rows(audit):
    protocol = audit["protocol"]
    learned = {
        name for name, contract in protocol["arm_contracts"].items()
        if bool(contract["learning"])
    }
    maximum_steps = int(protocol["design"]["maximum_episode_steps"])
    contract = protocol["conflict_certificate"]
    start = np.asarray(contract["robot_start_xy"], dtype=np.float64)
    goal = np.asarray(contract["robot_goal_xy"], dtype=np.float64)
    direction = (goal - start) / np.linalg.norm(goal - start)
    rows = []
    file_hashes = []
    for job in audit["jobs"]:
        run_dir = Path(job["run_dir"])
        for name in REQUIRED_FORMAL_ARTIFACTS:
            path = run_dir / name
            file_hashes.append({
                "path": str(path.relative_to(Path(audit["formal_output"]))),
                "sha256": _sha256(path),
            })
        metrics = _load_json(run_dir / "metrics.json")
        provenance = _load_json(run_dir / "provenance.json")
        metadata = _load_json(run_dir / "paper_v4_job.json")
        config = _load_yaml(run_dir / "config_resolved.yaml")
        trajectory = _read_trajectory(run_dir / "trajectory.csv")
        split, seed, arm = job["key"]
        block = job["block"]
        if metadata.get("arm") != arm or metadata.get("block") != block:
            raise ValueError("job metadata differs from sealed schedule: %s" % run_dir)
        if metrics.get("provenance") != provenance:
            raise ValueError("metrics/provenance mismatch: %s" % run_dir)
        experiment = config.get("experiment", {})
        if (
            int(experiment.get("seed", -1)) != seed
            or experiment.get("paper_v4_arm") != arm
            or experiment.get("paper_v4_split") != split
            or int(experiment.get("paper_v4_model_block", -1)) != int(block["model_block"])
        ):
            raise ValueError("resolved config identity mismatch: %s" % run_dir)
        if _canonical_sha256(config) != metadata.get("resolved_config_sha256"):
            raise ValueError("resolved config SHA-256 mismatch: %s" % run_dir)
        iterations = int(config["planner"].get("paper_rl_driven", {}).get("iterations", 1))
        if int(config["planner"]["num_samples"]) * iterations != int(
            protocol["design"]["total_rollouts_per_decision"]
        ):
            raise ValueError("rollout budget mismatch: %s" % run_dir)
        if arm in learned and abs(
            float(metrics.get("paper_total_rollouts_mean", 0.0))
            - float(protocol["design"]["total_rollouts_per_decision"])
        ) > 1.0e-9:
            raise ValueError("learned-arm rollout telemetry mismatch: %s" % run_dir)
        required_metrics = (
            "steps", "success", "collision", "final_goal_distance",
            "trajectory_length", "minimum_clearance", "applied_control_jerk",
            "stuck_steps", "spin_steps", "planner_compute_ms_p95",
            "planner_deadline_miss_rate",
        )
        missing = [name for name in required_metrics if metrics.get(name) is None]
        if missing:
            raise ValueError("missing metrics %s: %s" % (missing, run_dir))
        collision = bool(metrics["collision"])
        boundary_steps = int(metrics.get("boundary_violation_steps", 0))
        safe_success = bool(metrics["success"] and not collision and boundary_steps == 0)
        xy = np.asarray([
            [_number(record, "x"), _number(record, "y")] for record in trajectory
        ])
        path_progress = float(np.max(np.clip((xy - start) @ direction, 0.0, None)))
        minimum_clearance = float(metrics["minimum_clearance"])
        behavior = _behavior_metrics(
            trajectory, block, protocol, minimum_clearance
        )
        rows.append({
            "split": split,
            "seed": seed,
            "arm": arm,
            "model_block": int(block["model_block"]),
            "ablation_block": bool(block["ablation_block"]),
            "block_order": int(block["block_order"]),
            "safe_success": safe_success,
            "collision": collision,
            "collision_free": not collision,
            "boundary_violation": boundary_steps > 0,
            "boundary_violation_steps": boundary_steps,
            "steps": int(metrics["steps"]),
            "failure_penalized_steps": int(metrics["steps"]) if safe_success else maximum_steps,
            "time_to_goal_s": (
                float(metrics["time_to_goal_s"])
                if safe_success and metrics.get("time_to_goal_s") is not None else None
            ),
            "final_goal_distance": float(metrics["final_goal_distance"]),
            "path_progress_m_max": path_progress,
            "minimum_clearance": minimum_clearance,
            "trajectory_length": float(metrics["trajectory_length"]),
            "applied_control_jerk": float(metrics["applied_control_jerk"]),
            "stuck_steps": int(metrics["stuck_steps"]),
            "spin_steps": int(metrics["spin_steps"]),
            "planner_compute_ms_p95_health_only": float(metrics["planner_compute_ms_p95"]),
            "planner_deadline_miss_rate_health_only": float(metrics["planner_deadline_miss_rate"]),
            **behavior,
        })
    return rows, file_hashes


BINARY_ENDPOINTS = {
    "safe_success": True,
    "collision": False,
    "boundary_violation": False,
}

CONTINUOUS_ENDPOINTS = {
    "failure_penalized_steps": False,
    "final_goal_distance": False,
    "path_progress_m_max": True,
    "minimum_clearance": True,
    "conflict_q05_clearance": True,
    "trajectory_length": False,
    "applied_control_jerk": False,
    "stuck_steps": False,
    "spin_steps": False,
    "direction_switch_count": False,
    "three_phase_oscillation_count": False,
    "premature_reverse_count": False,
    "release_delay_max_s": False,
    "conflict_traversal_time_failure_penalized_s": False,
    "zero_speed_risk_steps": False,
    "planner_compute_ms_p95_health_only": False,
    "planner_deadline_miss_rate_health_only": False,
}


def _selected(rows, split="pooled", arms=None):
    arms = set(arms) if arms is not None else None
    return [
        row for row in rows
        if (split == "pooled" or row["split"] == split)
        and (arms is None or row["arm"] in arms)
    ]


def _pairs(rows, treatment, control, split="pooled"):
    selected = _selected(rows, split, (treatment, control))
    indexed = {(row["split"], row["seed"], row["arm"]): row for row in selected}
    keys = sorted({(row["split"], row["seed"]) for row in selected})
    pairs = []
    for key in keys:
        left = indexed.get((key[0], key[1], treatment))
        right = indexed.get((key[0], key[1], control))
        if left is None or right is None:
            raise ValueError("incomplete pair %s: %s vs %s" % (key, treatment, control))
        pairs.append((left, right))
    if not pairs:
        raise ValueError("no pairs for %s vs %s" % (treatment, control))
    return pairs


def _descriptives(rows, core_arms):
    output = []
    endpoints = {**BINARY_ENDPOINTS, **CONTINUOUS_ENDPOINTS}
    for split in ("pooled", "id", "ood"):
        for arm in core_arms:
            selected = _selected(rows, split, (arm,))
            for endpoint in endpoints:
                values = np.asarray([row[endpoint] for row in selected], dtype=np.float64)
                record = {
                    "split": split,
                    "arm": arm,
                    "endpoint": endpoint,
                    "n": int(values.size),
                    "mean_or_rate": float(np.mean(values)),
                }
                if endpoint not in BINARY_ENDPOINTS:
                    record.update({
                        "sd": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
                        "median": float(np.median(values)),
                        "q25": float(np.quantile(values, 0.25)),
                        "q75": float(np.quantile(values, 0.75)),
                    })
                output.append(record)
    return output


def _primary_analysis(rows, protocol):
    treatment = "B11_full_proposed"
    control = "B00_strong_nominal_mppi"
    by_split = {}
    for split in ("pooled", "id", "ood"):
        pairs = _pairs(rows, treatment, control, split)
        success = binary_contrast(
            [left["safe_success"] for left, _ in pairs],
            [right["safe_success"] for _, right in pairs],
        )
        collision = binary_contrast(
            [left["collision"] for left, _ in pairs],
            [right["collision"] for _, right in pairs],
        )
        by_split[split] = {"success_rate": success, "collision_rate": collision}
    alpha = float(protocol["analysis"]["alpha"])
    pooled_success = by_split["pooled"]["success_rate"]
    pooled_collision = by_split["pooled"]["collision_rate"]
    success_confirmed = bool(
        pooled_success["treatment_minus_control"] > 0.0
        and pooled_success["exact_mcnemar_two_sided_p"] < alpha
        and pooled_success["tango_two_sided_95ci"][0] > 0.0
    )
    practical = bool(
        success_confirmed
        and pooled_success["treatment_minus_control"] >= 0.10
    )
    collision_ni = bool(
        pooled_collision["tango_one_sided_95ci_upper"]
        < float(protocol["analysis"]["collision_noninferiority_margin"])
    )
    split_gates = {}
    for split in ("id", "ood"):
        split_gates[split] = {
            "success_no_harm": bool(
                by_split[split]["success_rate"]["tango_one_sided_95ci_lower"]
                > float(protocol["analysis"]["split_success_harm_margin"])
            ),
            "collision_no_harm": bool(
                by_split[split]["collision_rate"]["tango_one_sided_95ci_upper"]
                < float(protocol["analysis"]["split_collision_harm_margin"])
            ),
        }
    behavior_gate = bool(
        success_confirmed
        and collision_ni
        and all(all(value.values()) for value in split_gates.values())
    )
    return {
        "contrast": "%s_minus_%s" % (treatment, control),
        "by_split": by_split,
        "gates": {
            "statistically_confirmed_success_improvement": success_confirmed,
            "practically_meaningful_success_improvement_ge_10pp": practical,
            "pooled_collision_rate_noninferiority_margin_2pp": collision_ni,
            "split_harm_gates": split_gates,
            "behavior_matrix_confirmatory_gate": behavior_gate,
            "publication_claim_pending_independent_timing_cohort": True,
        },
    }


def _continuous_primary(rows, protocol):
    output = []
    base_seed = int(protocol["analysis"]["bootstrap_seed"])
    replicates = int(protocol["analysis"]["bootstrap_replicates"])
    for split_index, split in enumerate(("pooled", "id", "ood")):
        pairs = _pairs(rows, "B11_full_proposed", "B00_strong_nominal_mppi", split)
        for endpoint_index, (endpoint, higher) in enumerate(CONTINUOUS_ENDPOINTS.items()):
            result = continuous_contrast(
                [left[endpoint] for left, _ in pairs],
                [right[endpoint] for _, right in pairs],
                higher,
                replicates,
                base_seed + 1000 * split_index + endpoint_index,
            )
            result.update({"split": split, "endpoint": endpoint})
            output.append(result)
        both_success = [
            (left, right) for left, right in pairs
            if left["safe_success"] and right["safe_success"]
        ]
        if both_success:
            result = continuous_contrast(
                [left["time_to_goal_s"] for left, _ in both_success],
                [right["time_to_goal_s"] for _, right in both_success],
                False,
                replicates,
                base_seed + 1000 * split_index + 999,
            )
            result.update({
                "split": split,
                "endpoint": "time_to_goal_s_both_success_only",
                "selected_pair_fraction": float(len(both_success) / len(pairs)),
            })
            output.append(result)
    return output


def _factorial_analysis(rows, protocol):
    output = []
    endpoints = {
        "safe_success": True,
        "collision_free": True,
        "failure_penalized_steps": False,
        "final_goal_distance": False,
        "path_progress_m_max": True,
        "minimum_clearance": True,
        "conflict_q05_clearance": True,
    }
    core = list(protocol["design"]["core_arms"])
    replicates = int(protocol["analysis"]["bootstrap_replicates"])
    base_seed = int(protocol["analysis"]["bootstrap_seed"]) + 10000
    for split_index, split in enumerate(("pooled", "id", "ood")):
        selected = _selected(rows, split, core)
        keys = sorted({(row["split"], row["seed"]) for row in selected})
        indexed = {(row["split"], row["seed"], row["arm"]): row for row in selected}
        for endpoint_index, (endpoint, higher) in enumerate(endpoints.items()):
            by_arm = {}
            for arm in core:
                raw = np.asarray([
                    indexed[(key[0], key[1], arm)][endpoint] for key in keys
                ], dtype=np.float64)
                by_arm[arm] = raw if higher else -raw
            summaries = factorial_summary(
                by_arm,
                replicates,
                base_seed + 1000 * split_index + endpoint_index,
            )
            for summary in summaries:
                summary.update({
                    "split": split,
                    "endpoint": endpoint,
                    "effect_scale": "positive_is_favorable",
                    "confirmatory_holm_family": bool(
                        split == "pooled" and endpoint == "safe_success"
                    ),
                })
                output.append(summary)
    return output


def _ablation_analysis(rows, protocol):
    full = "B11_full_proposed"
    ablations = list(protocol["design"]["ablation_arms"])
    # Core arms run on all 360 seeds, whereas the three ablations run only on
    # the controller-independent 140-seed subset frozen in the registry.  The
    # block flag comes directly from that sealed schedule and is checked while
    # loading every formal job.  Restrict both sides to those blocks before
    # constructing pairs; otherwise the full arm's 220 core-only seeds appear
    # as incomplete ablation pairs.
    ablation_rows = [row for row in rows if bool(row["ablation_block"])]
    output = []
    replicates = int(protocol["analysis"]["bootstrap_replicates"])
    base_seed = int(protocol["analysis"]["bootstrap_seed"]) + 20000
    endpoints = {
        "safe_success": True,
        "collision": False,
        "failure_penalized_steps": False,
        "final_goal_distance": False,
        "minimum_clearance": True,
        "conflict_q05_clearance": True,
    }
    for split_index, split in enumerate(("pooled", "id", "ood")):
        for endpoint_index, (endpoint, higher) in enumerate(endpoints.items()):
            summaries = []
            raw_p = []
            for ablation in ablations:
                pairs = _pairs(ablation_rows, full, ablation, split)
                if endpoint in ("safe_success", "collision"):
                    result = binary_contrast(
                        [left[endpoint] for left, _ in pairs],
                        [right[endpoint] for _, right in pairs],
                    )
                    p_value = result["exact_mcnemar_two_sided_p"]
                    result["mean_favorable_effect"] = (
                        result["treatment_minus_control"]
                        if higher else -result["treatment_minus_control"]
                    )
                else:
                    result = continuous_contrast(
                        [left[endpoint] for left, _ in pairs],
                        [right[endpoint] for _, right in pairs],
                        higher,
                        replicates,
                        base_seed + 1000 * split_index + 10 * endpoint_index + len(summaries),
                    )
                    p_value = result["wilcoxon_two_sided_p"]
                result.update({
                    "split": split,
                    "endpoint": endpoint,
                    "contrast": "%s_minus_%s" % (full, ablation),
                    "effect_scale": "positive_is_favorable",
                })
                summaries.append(result)
                raw_p.append(float(p_value))
            adjusted = holm_adjust(raw_p)
            for index, result in enumerate(summaries):
                result["holm_adjusted_p_three_contrasts"] = adjusted[index]
                result["confirmatory_holm_family"] = bool(
                    split == "pooled" and endpoint == "safe_success"
                )
                output.append(result)
    return output


def _result_file_manifest(output):
    files = {}
    for path in sorted(Path(output).glob("*")):
        if path.is_file() and path.name != "analysis_bundle_manifest.json":
            files[path.name] = _sha256(path)
    return {
        "files": files,
        "bundle_sha256": _canonical_sha256(files),
    }


def analyze(protocol_path=DEFAULT_PROTOCOL, formal_output=None, analysis_output=None):
    audit = audit_formal_completion(protocol_path, formal_output)
    protocol = audit["protocol"]
    rows, input_hashes = _load_episode_rows(audit)
    if len(rows) != int(protocol["design"]["total_episode_jobs"]):
        raise RuntimeError("loaded formal row count changed")
    output = (
        Path(analysis_output).resolve()
        if analysis_output is not None
        else _resolve(audit["root"], protocol["analysis_output"])
    )
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("analysis output already exists: %s" % output)
    output.mkdir(parents=True, exist_ok=True)
    public_audit = {key: value for key, value in audit.items() if key not in (
        "jobs", "protocol", "registry_data", "root"
    )}
    public_audit["outcomes_read"] = True
    public_audit["outcome_unlock_condition"] = "all_1860_jobs_complete_before_first_metric_read"
    _write_json(output / "analysis_integrity_audit.json", public_audit)
    _write_csv(output / "input_file_hashes.csv", input_hashes)
    _write_csv(output / "episode_table.csv", rows)
    _write_csv(
        output / "core_descriptive_table.csv",
        _descriptives(rows, protocol["design"]["core_arms"]),
    )
    primary = _primary_analysis(rows, protocol)
    _write_json(output / "primary_confirmatory_analysis.json", primary)
    _write_csv(output / "continuous_full_vs_baseline.csv", _continuous_primary(rows, protocol))
    _write_csv(output / "factorial_mechanism_effects.csv", _factorial_analysis(rows, protocol))
    _write_csv(output / "full_system_ablation_effects.csv", _ablation_analysis(rows, protocol))
    conclusion = {
        "schema_version": 1,
        "study_id": protocol["study_id"],
        "formal_outcomes_opened_after_complete_matrix": True,
        "primary_gates": primary["gates"],
        "timing_conclusion": "pending_independent_exclusive_rtx5060_timing_cohort",
        "analysis_code_sha256": _sha256(Path(__file__)),
        "protocol_sha256": _sha256(Path(protocol_path)),
        "registry_sha256": audit["registry_sha256"],
        "input_file_count": len(input_hashes),
        "input_file_manifest_sha256": _canonical_sha256(input_hashes),
    }
    _write_json(output / "paper_v4_result_summary.json", conclusion)
    bundle = _result_file_manifest(output)
    _write_json(output / "analysis_bundle_manifest.json", bundle)
    print(json.dumps({
        "status": "analysis_complete",
        "output": str(output),
        "episode_jobs": len(rows),
        "bundle_sha256": bundle["bundle_sha256"],
        "primary_gates": primary["gates"],
    }, indent=2, sort_keys=True))
    return conclusion


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("audit", "analyze"))
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--formal-output", type=Path)
    parser.add_argument("--analysis-output", type=Path)
    args = parser.parse_args(argv)
    if args.command == "audit":
        audit = audit_formal_completion(args.protocol, args.formal_output)
        public = {key: value for key, value in audit.items() if key not in (
            "jobs", "protocol", "registry_data", "root"
        )}
        print(json.dumps(public, indent=2, sort_keys=True))
    else:
        analyze(args.protocol, args.formal_output, args.analysis_output)
    return 0


__all__ = [
    "TANGO_TOLERANCE",
    "audit_formal_completion",
    "binary_contrast",
    "continuous_contrast",
    "exact_mcnemar_p",
    "factorial_effect_vectors",
    "factorial_summary",
    "holm_adjust",
    "paired_table",
    "tango_matched_score_interval",
    "tango_score_statistic",
]


if __name__ == "__main__":
    raise SystemExit(main())
