#!/usr/bin/env python3
"""Post-gate EDA for paired trajectory-utility datasets."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from mobile_robot_mppi.core.config import load_yaml
from mobile_robot_mppi.rl.utility_dataset import (
    load_counterfactual_utility_dataset,
)
from mobile_robot_mppi.rl.utility_model import (
    CounterfactualUtilityConfig,
    Standardizer,
)


def _correlation(x, y):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.size != y.size or x.size < 2:
        raise ValueError("correlation arrays must align")
    if np.std(x) <= 1.0e-12 or np.std(y) <= 1.0e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _feature_correlations(features, targets):
    return np.asarray(
        [_correlation(features[:, index], targets) for index in range(features.shape[1])],
        dtype=np.float64,
    )


def _group_means(features, targets, groups):
    keys = sorted(set(tuple(int(v) for v in row) for row in groups))
    feature_rows = []
    target_rows = []
    for key in keys:
        mask = np.all(groups == np.asarray(key)[None, :], axis=1)
        feature_rows.append(np.mean(features[mask], axis=0))
        target_rows.append(float(np.mean(targets[mask])))
    return np.asarray(feature_rows), np.asarray(target_rows), keys


def _top(correlations, names, count=10):
    order = np.argsort(-np.abs(correlations))[:count]
    return [
        {"name": names[index], "correlation": float(correlations[index])}
        for index in order
    ]


def _branch_summary(split):
    rows = []
    for step in sorted(set(int(value) for value in split["branch_step"])):
        values = split["targets"][split["branch_step"] == step]
        rows.append({
            "branch_step": step,
            "rows": int(values.size),
            "mean_m": float(np.mean(values)),
            "std_m": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
            "min_m": float(np.min(values)),
            "max_m": float(np.max(values)),
        })
    return rows


def _nearest_train_rmse(train, query, dimension):
    standardizer = Standardizer.fit(train["features"][:, :dimension])
    train_x = standardizer.transform(train["features"][:, :dimension])
    query_x = standardizer.transform(query["features"][:, :dimension])
    predictions = []
    for row in query_x:
        index = int(np.argmin(np.sum((train_x - row[None, :]) ** 2, axis=1)))
        predictions.append(train["targets"][index])
    predictions = np.asarray(predictions)
    return float(np.sqrt(np.mean((query["targets"] - predictions) ** 2)))


def _markdown(result):
    lines = [
        "# L24 trajectory-utility dataset EDA",
        "",
        "This report is explanatory analysis performed only after the frozen development gate was evaluated. It does not alter the preregistered model, features, thresholds, or sealed-test decision.",
        "",
        "## Data quality",
        "",
        "- Rows: `%d`" % result["rows"],
        "- Feature dimension: `%d` (`%d` state + `3 x %d` trajectory metrics)" % (
            result["feature_dim"], result["state_feature_dim"], result["trajectory_metric_dim"]
        ),
        "- Finite arrays: `%s`" % str(result["finite"]).lower(),
        "- Exact candidate-minus-base block: `%s`" % str(result["delta_contract_exact"]).lower(),
        "",
        "## Split diagnostics",
        "",
        "| split | rows | groups | state NN RMSE | full NN RMSE | predicted-progress delta r |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in ("train", "selection", "calibration"):
        item = result["splits"][name]
        lines.append(
            "| %s | %d | %d | %s | %s | %.3f |" % (
                name,
                item["rows"],
                item["groups"],
                "--" if item["state_nearest_train_rmse_m"] is None else "%.3f cm" % (100 * item["state_nearest_train_rmse_m"]),
                "--" if item["full_nearest_train_rmse_m"] is None else "%.3f cm" % (100 * item["full_nearest_train_rmse_m"]),
                item["predicted_final_progress_delta_correlation"],
            )
        )
    lines.extend([
        "",
        "## Stable associations",
        "",
        "Features with absolute group-mean correlation >= 0.2 and the same sign in train, selection, and calibration: `%d`." % len(result["stable_group_features"]),
        "",
    ])
    if result["stable_group_features"]:
        lines.append("| feature | train r | selection r | calibration r |")
        lines.append("|---|---:|---:|---:|")
        for item in result["stable_group_features"]:
            lines.append("| %s | %.3f | %.3f | %.3f |" % (
                item["name"], item["train"], item["selection"], item["calibration"]
            ))
    else:
        lines.append("No trajectory or state feature met this stability criterion.")
    lines.extend(["", "## Strongest group-level correlations by split", ""])
    for name in ("train", "selection", "calibration"):
        lines.append("### %s" % name)
        lines.append("")
        lines.append("| feature | r |")
        lines.append("|---|---:|")
        for item in result["splits"][name]["top_group_correlations"][:8]:
            lines.append("| %s | %.3f |" % (item["name"], item["correlation"]))
        lines.append("")
    lines.extend([
        "## Interpretation guard",
        "",
        "Correlations and nearest-neighbour diagnostics are descriptive. Branch rows remain nested within checkpoint/episode groups, and this report must not be used to tune L24 after its development gate failed.",
    ])
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    config_path = Path(args.config).resolve()
    resolved = load_yaml(config_path)
    config = CounterfactualUtilityConfig.from_mapping(
        resolved["rl"]["counterfactual_utility"]
    )
    dataset = load_counterfactual_utility_dataset(
        Path(args.dataset_dir).resolve(),
        config.model_selection_episode_seeds,
        config.calibration_episode_seeds,
    )
    schema = dataset["feature_schema"]
    state_dim = int(schema["state_feature_dim"])
    metric_names = list(schema["trajectory_metric_names"])
    metric_dim = len(metric_names)
    if dataset["feature_dim"] != state_dim + 3 * metric_dim:
        raise ValueError("trajectory dataset layout differs from schema")
    names = (
        ["state_%03d" % index for index in range(state_dim)]
        + ["base_%s" % name for name in metric_names]
        + ["candidate_%s" % name for name in metric_names]
        + ["delta_%s" % name for name in metric_names]
    )
    all_features = np.concatenate(
        [split["features"] for split in dataset["splits"].values()], axis=0
    )
    base = all_features[:, state_dim:state_dim + metric_dim]
    candidate = all_features[:, state_dim + metric_dim:state_dim + 2 * metric_dim]
    delta = all_features[:, state_dim + 2 * metric_dim:]
    result = {
        "rows": int(all_features.shape[0]),
        "feature_dim": int(all_features.shape[1]),
        "state_feature_dim": state_dim,
        "trajectory_metric_dim": metric_dim,
        "finite": bool(np.isfinite(all_features).all()),
        "delta_contract_exact": bool(np.array_equal(delta, candidate - base)),
        "splits": {},
    }
    group_correlations = {}
    progress_delta_index = names.index("delta_predicted_final_progress_m")
    train = dataset["splits"]["train"]
    for split_name, split in dataset["splits"].items():
        row_correlations = _feature_correlations(
            split["features"], split["targets"]
        )
        group_x, group_y, keys = _group_means(
            split["features"], split["targets"], split["groups"]
        )
        group_corr = _feature_correlations(group_x, group_y)
        group_correlations[split_name] = group_corr
        result["splits"][split_name] = {
            "rows": int(split["targets"].size),
            "groups": len(keys),
            "target_mean_m": float(np.mean(split["targets"])),
            "target_std_m": float(np.std(split["targets"], ddof=1)),
            "predicted_final_progress_delta_correlation": float(
                row_correlations[progress_delta_index]
            ),
            "top_row_correlations": _top(row_correlations, names),
            "top_group_correlations": _top(group_corr, names),
            "branch_steps": _branch_summary(split),
            "state_nearest_train_rmse_m": (
                None if split_name == "train" else _nearest_train_rmse(
                    train, split, state_dim
                )
            ),
            "full_nearest_train_rmse_m": (
                None if split_name == "train" else _nearest_train_rmse(
                    train, split, dataset["feature_dim"]
                )
            ),
        }
    stable = []
    for index, name in enumerate(names):
        values = [group_correlations[split][index] for split in (
            "train", "selection", "calibration"
        )]
        if min(abs(value) for value in values) >= 0.2 and (
            all(value > 0.0 for value in values)
            or all(value < 0.0 for value in values)
        ):
            stable.append({
                "name": name,
                "train": float(values[0]),
                "selection": float(values[1]),
                "calibration": float(values[2]),
            })
    result["stable_group_features"] = stable
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "trajectory_eda.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    (output / "trajectory_eda.md").write_text(
        _markdown(result), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
