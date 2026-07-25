"""Offline evaluation of causal Change-Aware IMM trigger records."""

from typing import Mapping, Sequence

import numpy as np


def evaluate_change_detections(
    detection_events: Sequence[Mapping[str, object]],
    times,
    change_flags,
    warmup_s=1.0,
    maximum_match_delay_s=1.5,
):
    """Match online NIS triggers to hidden V3 changes after prediction."""
    times = np.asarray(times, dtype=np.float64)
    change_flags = np.asarray(change_flags, dtype=bool)
    if (
        times.ndim != 1
        or change_flags.shape != times.shape
        or times.size < 2
        or not np.isfinite(times).all()
        or np.any(np.diff(times) <= 0.0)
    ):
        raise ValueError("detection evaluation inputs are invalid")
    warmup_s = float(warmup_s)
    maximum_match_delay_s = float(maximum_match_delay_s)
    if warmup_s < 0.0 or maximum_match_delay_s <= 0.0:
        raise ValueError("detection evaluation windows are invalid")
    truth_times = [
        float(time)
        for time, changed in zip(times, change_flags)
        if changed and float(time) >= warmup_s
    ]
    nis_events = sorted(
        (
            {
                "timestamp": float(event["timestamp"]),
                "nis": float(event["nis"]),
            }
            for event in detection_events
            if str(event["kind"]) == "nis_change"
            and float(event["timestamp"]) >= warmup_s
        ),
        key=lambda event: event["timestamp"],
    )
    dropout_events = [
        event
        for event in detection_events
        if str(event["kind"]) == "dropout_guard"
        and float(event["timestamp"]) >= warmup_s
    ]
    unmatched_truth = set(range(len(truth_times)))
    delays = []
    false_triggers = 0
    matches = []
    for trigger in nis_events:
        eligible = [
            index
            for index in unmatched_truth
            if 0.0
            <= trigger["timestamp"] - truth_times[index]
            <= maximum_match_delay_s
        ]
        if not eligible:
            false_triggers += 1
            continue
        truth_index = max(eligible, key=lambda index: truth_times[index])
        unmatched_truth.remove(truth_index)
        delay = trigger["timestamp"] - truth_times[truth_index]
        delays.append(delay)
        matches.append(
            {
                "truth_timestamp": truth_times[truth_index],
                "trigger_timestamp": trigger["timestamp"],
                "delay_s": delay,
                "nis": trigger["nis"],
            }
        )
    duration_minutes = max(
        (float(times[-1]) - warmup_s) / 60.0,
        np.finfo(np.float64).eps,
    )
    return {
        "true_change_count": len(truth_times),
        "nis_trigger_count": len(nis_events),
        "dropout_guard_count": len(dropout_events),
        "matched_trigger_count": len(delays),
        "false_trigger_count": int(false_triggers),
        "false_triggers_per_minute": float(
            false_triggers / duration_minutes
        ),
        "true_change_recall": (
            None
            if not truth_times
            else float(len(delays) / len(truth_times))
        ),
        "median_detection_delay_s": (
            None if not delays else float(np.median(delays))
        ),
        "p90_detection_delay_s": (
            None if not delays else float(np.quantile(delays, 0.90))
        ),
        "matches": matches,
    }


__all__ = ["evaluate_change_detections"]
