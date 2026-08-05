#!/usr/bin/env python3
"""Summarize the shadow encounter state from one physical-robot run."""

import argparse
from collections import Counter
import json
import math
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


ACTIVE_MODES = frozenset({
    "straight_crossing", "oblique_crossing", "frontal_approach"
})
CROSSING_MODES = frozenset({"straight_crossing", "oblique_crossing"})
EXPECTED_SCENARIOS = (
    "auto", "left_to_right", "right_to_left", "cross_to_frontal", "frontal"
)


def _finite(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _percentile(values: Sequence[float], fraction: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = fraction * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return (1.0 - weight) * ordered[lower] + weight * ordered[upper]


def load_cycles(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, raw in enumerate(stream, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    "invalid JSON at %s:%d: %s" % (path, line_number, error)
                ) from error
            if not isinstance(value, dict):
                raise ValueError(
                    "cycle row at %s:%d is not an object" % (path, line_number)
                )
            rows.append(value)
    return rows


def latest_run(root: Path) -> Path:
    candidates = []
    for cycles in Path(root).glob("*/cycles.jsonl"):
        if cycles.is_file() and cycles.stat().st_size > 0:
            candidates.append(cycles)
    if not candidates:
        raise FileNotFoundError("no non-empty cycles.jsonl below %s" % root)
    return max(candidates, key=lambda value: value.stat().st_mtime).parent


def _event(
    row: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    field: str,
    previous: Optional[str],
    first_timestamp: float,
) -> Dict[str, Any]:
    timestamp = _finite(row.get("timestamp"))
    return {
        "cycle": int(row.get("cycle", 0) or 0),
        "time_from_start_s": (
            None if timestamp is None else timestamp - first_timestamp
        ),
        "field": field,
        "from": previous,
        "to": diagnostics.get(field),
        "track_index": diagnostics.get("encounter_track_index"),
        "confidence": _finite(diagnostics.get("encounter_confidence")),
        "approach_angle_deg": _finite(
            diagnostics.get("encounter_approach_angle_deg")
        ),
        "human_longitudinal_velocity_mps": _finite(diagnostics.get(
            "encounter_human_longitudinal_velocity_mps"
        )),
        "human_lateral_velocity_mps": _finite(diagnostics.get(
            "encounter_human_lateral_velocity_mps"
        )),
        "strategy": diagnostics.get("encounter_strategy"),
        "locked_steering_side": diagnostics.get(
            "encounter_locked_steering_side"
        ),
        "change_detected": bool(diagnostics.get(
            "encounter_change_detected", False
        )),
        "change_score": _finite(diagnostics.get("encounter_change_score")),
        "change_reason": diagnostics.get("encounter_change_reason"),
    }


def _first_active(
    records: Iterable[Mapping[str, Any]], field: str
) -> Optional[Mapping[str, Any]]:
    for record in records:
        if record["diagnostics"].get(field) in ACTIVE_MODES:
            return record
    return None


def _record_fact(record: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    if record is None:
        return None
    row = record["row"]
    diagnostics = record["diagnostics"]
    return {
        "cycle": int(row.get("cycle", 0) or 0),
        "timestamp": _finite(row.get("timestamp")),
        "mode": diagnostics.get(record["field"]),
        "track_index": diagnostics.get("encounter_track_index"),
        "confidence": _finite(diagnostics.get("encounter_confidence")),
    }


def analyze_rows(
    rows: Sequence[Mapping[str, Any]], expected: str = "auto"
) -> Dict[str, Any]:
    if expected not in EXPECTED_SCENARIOS:
        raise ValueError("unknown expected scenario: %s" % expected)
    records = []
    for row in rows:
        diagnostics = row.get("encounter_mode_shadow")
        if isinstance(diagnostics, dict):
            records.append({"row": row, "diagnostics": diagnostics})
    if not records:
        return {
            "schema": "encounter_shadow_audit_v1",
            "usable": False,
            "reason": "no_encounter_mode_shadow_rows",
            "total_cycle_rows": len(rows),
            "encounter_rows": 0,
            "expected_scenario": expected,
        }

    first_timestamp = _finite(records[0]["row"].get("timestamp")) or 0.0
    counts = {
        "candidate": Counter(), "confirmed": Counter(), "phase": Counter()
    }
    field_map = {
        "candidate": "encounter_candidate_mode",
        "confirmed": "encounter_confirmed_mode",
        "phase": "encounter_phase",
    }
    previous = {field: None for field in field_map.values()}
    events = []
    direction_samples = []
    discontinuities = []
    changes = []
    line_crossings = []
    encounter_timings = []

    for record in records:
        row = record["row"]
        diagnostics = record["diagnostics"]
        for name, field in field_map.items():
            value = str(diagnostics.get(field, "missing"))
            counts[name][value] += 1
            if value != previous[field]:
                events.append(_event(
                    row, diagnostics, field, previous[field], first_timestamp
                ))
                previous[field] = value

        phase = str(diagnostics.get("encounter_phase", "idle"))
        strategy = str(diagnostics.get("encounter_strategy", "none"))
        lateral_velocity = _finite(diagnostics.get(
            "encounter_human_lateral_velocity_mps"
        ))
        steering_side = diagnostics.get("encounter_locked_steering_side")
        if (
            phase in CROSSING_MODES
            and strategy in ("behind_pass", "front_pass")
            and lateral_velocity is not None
            and abs(lateral_velocity) > 1.0e-6
            and steering_side in (-1, 1)
        ):
            # Keep the audit independent of the manager implementation:
            # behind-pass steers opposite human motion, while front-pass
            # steers into the human's forward side.
            if strategy == "behind_pass":
                expected_side = -1 if lateral_velocity > 0.0 else 1
            else:
                expected_side = 1 if lateral_velocity > 0.0 else -1
            direction_samples.append({
                "cycle": int(row.get("cycle", 0) or 0),
                "strategy": strategy,
                "human_lateral_velocity_mps": lateral_velocity,
                "locked_steering_side": int(steering_side),
                "expected_steering_side": expected_side,
                "consistent": int(steering_side) == expected_side,
            })
        reason = str(diagnostics.get("encounter_change_reason", ""))
        if (
            diagnostics.get("encounter_track_index") is not None
            and not bool(diagnostics.get("encounter_track_continuous", False))
            and reason != "no_track_history"
        ):
            discontinuities.append({
                "cycle": int(row.get("cycle", 0) or 0),
                "track_index": diagnostics.get("encounter_track_index"),
                "reason": reason,
                "residual_m": _finite(diagnostics.get(
                    "encounter_track_residual_m"
                )),
            })
        if bool(diagnostics.get("encounter_change_detected", False)):
            changes.append({
                "cycle": int(row.get("cycle", 0) or 0),
                "score": _finite(diagnostics.get("encounter_change_score")),
                "reason": reason,
                "candidate_mode": diagnostics.get("encounter_candidate_mode"),
                "confirmed_mode": diagnostics.get("encounter_confirmed_mode"),
            })
        if bool(diagnostics.get("encounter_line_crossed", False)):
            line_crossings.append(int(row.get("cycle", 0) or 0))
        timing = _finite((row.get("timing_ms") or {}).get("encounter"))
        if timing is not None:
            encounter_timings.append(timing)

    candidate_records = [dict(record, field=field_map["candidate"])
                         for record in records]
    confirmed_records = [dict(record, field=field_map["confirmed"])
                         for record in records]
    first_candidate = _first_active(candidate_records, field_map["candidate"])
    first_confirmed = _first_active(confirmed_records, field_map["confirmed"])
    confirmation_delay_s = None
    confirmation_delay_cycles = None
    if first_candidate is not None and first_confirmed is not None:
        candidate_time = _finite(first_candidate["row"].get("timestamp"))
        confirmed_time = _finite(first_confirmed["row"].get("timestamp"))
        if candidate_time is not None and confirmed_time is not None:
            confirmation_delay_s = confirmed_time - candidate_time
        confirmation_delay_cycles = int(
            first_confirmed["row"].get("cycle", 0) or 0
        ) - int(first_candidate["row"].get("cycle", 0) or 0)

    confirmed_modes = counts["confirmed"]
    lateral_values = [
        sample["human_lateral_velocity_mps"] for sample in direction_samples
    ]
    direction_consistent = sum(
        sample["consistent"] for sample in direction_samples
    )
    checks = {
        "has_encounter_candidate": first_candidate is not None,
        "has_confirmed_encounter": first_confirmed is not None,
        "confirmation_within_0_5_s": bool(
            confirmation_delay_s is not None
            and 0.0 <= confirmation_delay_s <= 0.5
        ),
        "locked_direction_consistent": bool(
            direction_samples
            and direction_consistent == len(direction_samples)
        ),
        "no_track_identity_jump": not discontinuities,
    }
    if expected == "left_to_right":
        checks.update({
            "confirmed_crossing": any(
                confirmed_modes[mode] > 0 for mode in CROSSING_MODES
            ),
            "observed_left_to_right_velocity": bool(
                lateral_values and median(lateral_values) < 0.0
            ),
        })
    elif expected == "right_to_left":
        checks.update({
            "confirmed_crossing": any(
                confirmed_modes[mode] > 0 for mode in CROSSING_MODES
            ),
            "observed_right_to_left_velocity": bool(
                lateral_values and median(lateral_values) > 0.0
            ),
        })
    elif expected == "cross_to_frontal":
        checks.update({
            "confirmed_crossing": any(
                confirmed_modes[mode] > 0 for mode in CROSSING_MODES
            ),
            "confirmed_frontal": confirmed_modes["frontal_approach"] > 0,
            "detected_behaviour_change": bool(changes),
        })
    elif expected == "frontal":
        checks["confirmed_frontal"] = (
            confirmed_modes["frontal_approach"] > 0
        )

    return {
        "schema": "encounter_shadow_audit_v1",
        "usable": True,
        "expected_scenario": expected,
        "total_cycle_rows": len(rows),
        "encounter_rows": len(records),
        "mode_counts": {
            name: dict(sorted(value.items())) for name, value in counts.items()
        },
        "first_active_candidate": _record_fact(first_candidate),
        "first_active_confirmation": _record_fact(first_confirmed),
        "confirmation_delay_cycles": confirmation_delay_cycles,
        "confirmation_delay_s": confirmation_delay_s,
        "mode_transition_events": events,
        "change_events": changes,
        "track_discontinuities": discontinuities,
        "line_crossing_cycles": line_crossings,
        "direction_consistency": {
            "sample_count": len(direction_samples),
            "consistent_count": direction_consistent,
            "samples": direction_samples,
        },
        "encounter_timing_ms": {
            "count": len(encounter_timings),
            "p50": median(encounter_timings) if encounter_timings else None,
            "p95": _percentile(encounter_timings, 0.95),
            "maximum": max(encounter_timings) if encounter_timings else None,
        },
        "checks": checks,
        "all_checks_passed": bool(checks and all(checks.values())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path, nargs="?")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("codex_tmp/real_robot_cuda_runs"),
        help="root searched when no run directory is given",
    )
    parser.add_argument("--expected", choices=EXPECTED_SCENARIOS, default="auto")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run = latest_run(args.root) if args.run is None else args.run
    cycles = run if run.name == "cycles.jsonl" else run / "cycles.jsonl"
    result = analyze_rows(load_cycles(cycles), expected=args.expected)
    result["run"] = str(cycles.parent)
    encoded = json.dumps(result, indent=2, sort_keys=True, allow_nan=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
