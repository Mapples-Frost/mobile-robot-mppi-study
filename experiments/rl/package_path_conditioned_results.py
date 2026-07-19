#!/usr/bin/env python3
"""Build a versionable, lossless L185--L188 research-result package."""

import argparse
import csv
import gzip
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

EXPERIMENT_DIRS = (
    "results/research_platform/rl/path_conditioned_l185_dev",
    "results/research_platform/rl/path_conditioned_l185_seed20261901_60k_v2",
    "results/research_platform/rl/path_conditioned_l185_seed20261902_60k",
    "results/research_platform/rl/path_conditioned_l185_seed20261903_60k",
    "results/research_platform/rl/path_conditioned_l185_mppi_dev_screen_reverse_s_nominal",
    "results/research_platform/rl/path_conditioned_l187_terminal_diagnostic_t0_seed553",
    "results/research_platform/rl/path_conditioned_l187_terminal_diagnostic_t1_seed553",
    "results/research_platform/rl/path_conditioned_l187_terminal_confirmation_t1_seeds554_555",
    "results/research_platform/rl/path_conditioned_l188_terminal_parity_dev_seeds556_557",
)

SELECTED_CHECKPOINTS = (
    "results/research_platform/rl/gate1_direct_control_multidomain_screen_l175/checkpoints/best.pt",
    "results/research_platform/rl/path_conditioned_l185_seed20261901_60k_v2/checkpoints/step_000050000.pt",
    "results/research_platform/rl/path_conditioned_l185_seed20261902_60k/checkpoints/step_000040000.pt",
    "results/research_platform/rl/path_conditioned_l185_seed20261903_60k/checkpoints/step_000020000.pt",
    "results/research_platform/l57_icode_high_dynamic_h36_seed20261201_v1/best.pt",
    "results/research_platform/l57_icode_high_dynamic_h36_seed20261202_v1/best.pt",
    "results/research_platform/l57_icode_high_dynamic_h36_seed20261203_v1/best.pt",
    "results/research_platform/gate3_value_aligned_ensemble_member1_l192/best.pt",
    "results/research_platform/gate3_value_aligned_ensemble_member2_l192/best.pt",
    "results/research_platform/gate3_value_aligned_ensemble_member3_l192/best.pt",
)

CALIBRATION_ARTIFACTS = (
    "results/research_platform/rl/gate4_ordinary_reliability_calibration_l203/calibration_summary.json",
    "results/research_platform/rl/gate3_reliability_calibration_l193/calibration_summary.json",
)

AUXILIARY_ANALYSES = (
    "results/research_platform/rl/path_conditioned_l187_terminal_amendment_analysis.json",
    "results/research_platform/rl/path_conditioned_l188_terminal_parity_analysis.json",
)

TEXT_SUFFIXES = {
    ".csv",
    ".json",
    ".yaml",
    ".yml",
    ".log",
    ".sh",
    ".txt",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy(source, destination, compress_threshold):
    destination.parent.mkdir(parents=True, exist_ok=True)
    compressed = bool(
        source.suffix == ".csv"
        and source.stat().st_size >= int(compress_threshold)
    )
    if compressed:
        destination = destination.with_suffix(destination.suffix + ".gz")
        with source.open("rb") as input_handle:
            with destination.open("wb") as raw_output:
                with gzip.GzipFile(
                    filename="",
                    mode="wb",
                    fileobj=raw_output,
                    mtime=0,
                ) as output_handle:
                    shutil.copyfileobj(input_handle, output_handle)
    else:
        shutil.copy2(source, destination)
    return destination, compressed


def _record(source, destination, compressed, role, package_root):
    return {
        "role": str(role),
        "source": source.relative_to(ROOT).as_posix(),
        "source_bytes": source.stat().st_size,
        "source_sha256": sha256(source),
        "packaged": destination.relative_to(package_root).as_posix(),
        "packaged_bytes": destination.stat().st_size,
        "packaged_sha256": sha256(destination),
        "lossless_gzip": bool(compressed),
    }


def _bool(value):
    return str(value).strip().lower() == "true"


def _direct_actor_index(directory):
    summary = json.loads(
        (directory / "summary.json").read_text(encoding="utf-8")
    )
    return {
        "experiment": directory.name,
        "stage": "direct_actor_development",
        "variant": Path(summary["checkpoint"]).stem,
        "seed": "",
        "episodes": int(summary["episodes"]),
        "successes": int(round(
            float(summary["success_rate"]) * int(summary["episodes"])
        )),
        "success_rate": float(summary["success_rate"]),
        "collision_rate": float(summary["collision_rate"]),
        "cross_track_rmse": float(summary["mean_cross_track_rmse"]),
        "control_jerk": float(summary["mean_control_jerk"]),
        "path_completion_ratio": float(
            summary["mean_path_completion_ratio"]
        ),
        "source": directory.relative_to(ROOT).as_posix(),
    }


def _selection_index(directory):
    selection = json.loads(
        (directory / "path_checkpoint_selection.json").read_text(
            encoding="utf-8"
        )
    )
    selected_step = int(selection["selected_global_step"])
    selected = next(
        item
        for item in selection["aggregates"]
        if int(item["global_step"]) == selected_step
    )
    return {
        "experiment": directory.name,
        "stage": "sac_checkpoint_selection",
        "variant": "step_%09d" % selected_step,
        "seed": "",
        "episodes": int(selected["episodes"]),
        "successes": int(selected["successes"]),
        "success_rate": float(selected["success_rate"]),
        "collision_rate": float(selected["collision_rate"]),
        "cross_track_rmse": float(selected["mean_cross_track_rmse"]),
        "control_jerk": "",
        "path_completion_ratio": float(
            selected["mean_path_completion_ratio"]
        ),
        "source": directory.relative_to(ROOT).as_posix(),
    }


def _factorial_index(directory):
    progress = directory / "progress.csv"
    if not progress.is_file():
        return []
    with progress.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [
        {
            "experiment": directory.name,
            "stage": "integrated_mppi_development",
            "variant": row["factorial_arm"],
            "seed": int(row["seed"]),
            "episodes": 1,
            "successes": int(_bool(row["success"])),
            "success_rate": float(_bool(row["success"])),
            "collision_rate": float(_bool(row["collision"])),
            "cross_track_rmse": float(row["cross_track_rmse"]),
            "control_jerk": float(row["control_jerk"]),
            "path_completion_ratio": float(
                row["path_completion_ratio"]
            ),
            "source": progress.relative_to(ROOT).as_posix(),
        }
        for row in rows
    ]


def experiment_index():
    rows = []
    development = ROOT / EXPERIMENT_DIRS[0]
    for directory in sorted(
        path for path in development.iterdir()
        if path.is_dir() and (path / "summary.json").is_file()
    ):
        rows.append(_direct_actor_index(directory))
    for relative in EXPERIMENT_DIRS[1:4]:
        rows.append(_selection_index(ROOT / relative))
    for relative in EXPERIMENT_DIRS[4:]:
        rows.extend(_factorial_index(ROOT / relative))
    return rows


def build(output, compress_threshold=5 * 1024 * 1024):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(
            "output already exists; use a new package path: %s" % output
        )
    output.mkdir(parents=True)
    records = []
    for relative in EXPERIMENT_DIRS:
        source_root = ROOT / relative
        if not source_root.is_dir():
            raise FileNotFoundError(source_root)
        for source in sorted(source_root.rglob("*")):
            if not source.is_file() or source.suffix.lower() not in TEXT_SUFFIXES:
                continue
            destination = (
                output / "experiments" / source.relative_to(
                    ROOT / "results/research_platform/rl"
                )
            )
            packaged, compressed = _copy(
                source, destination, compress_threshold
            )
            records.append(_record(
                source, packaged, compressed, "experiment_data", output
            ))
    for relative in SELECTED_CHECKPOINTS:
        source = ROOT / relative
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = output / "checkpoints" / Path(relative)
        packaged, compressed = _copy(
            source, destination, compress_threshold
        )
        records.append(_record(
            source, packaged, compressed, "selected_checkpoint", output
        ))
    for relative in CALIBRATION_ARTIFACTS:
        source = ROOT / relative
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = output / "calibration" / Path(relative)
        packaged, compressed = _copy(
            source, destination, compress_threshold
        )
        records.append(_record(
            source, packaged, compressed, "calibration_audit", output
        ))
    for relative in AUXILIARY_ANALYSES:
        source = ROOT / relative
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = output / "analyses" / Path(relative).name
        packaged, compressed = _copy(
            source, destination, compress_threshold
        )
        records.append(_record(
            source, packaged, compressed, "analysis_summary", output
        ))

    index_rows = experiment_index()
    index_path = output / "experiment_index.csv"
    with index_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(index_rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(index_rows)
    manifest = {
        "schema_version": 1,
        "package": output.name,
        "git_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True
        ).strip(),
        "evidence_class": "development_only",
        "sealed_data_included": False,
        "file_count": len(records),
        "source_bytes": sum(item["source_bytes"] for item in records),
        "packaged_bytes": sum(
            item["packaged_bytes"] for item in records
        ),
        "records": records,
    }
    (output / "package_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "README.md").write_text(
        "# Path-conditioned ICODE--RL--MPPI development data\n\n"
        "This package contains the versioned L185--L188 development evidence.\n"
        "It includes per-step/per-episode CSV data, trajectories, configuration\n"
        "snapshots, logs, selected SAC and ICODE checkpoints, calibration\n"
        "summaries, and SHA256 provenance for every copied artifact.\n\n"
        "The data are development-only. L186 sealed geometries and seeds\n"
        "561--565 are deliberately absent. `experiment_index.csv` is the\n"
        "compact success/metric index. Files ending in `.csv.gz` are lossless\n"
        "gzip copies; their uncompressed hashes are in `package_manifest.json`.\n",
        encoding="utf-8",
    )
    return manifest, index_rows


def validate_package(output):
    output = Path(output).resolve()
    manifest_path = output / "package_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    total_packaged = 0
    for record in manifest["records"]:
        packaged = output / record["packaged"]
        if not packaged.is_file():
            raise FileNotFoundError(packaged)
        if sha256(packaged) != record["packaged_sha256"]:
            raise ValueError("packaged SHA256 mismatch: %s" % packaged)
        if packaged.stat().st_size != int(record["packaged_bytes"]):
            raise ValueError("packaged size mismatch: %s" % packaged)
        total_packaged += packaged.stat().st_size
        source = ROOT / record["source"]
        if not source.is_file():
            raise FileNotFoundError(source)
        if sha256(source) != record["source_sha256"]:
            raise ValueError("source SHA256 mismatch: %s" % source)
        if record["lossless_gzip"]:
            digest = hashlib.sha256()
            uncompressed_bytes = 0
            with gzip.open(packaged, "rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
                    uncompressed_bytes += len(chunk)
            if digest.hexdigest() != record["source_sha256"]:
                raise ValueError(
                    "gzip content SHA256 mismatch: %s" % packaged
                )
            if uncompressed_bytes != int(record["source_bytes"]):
                raise ValueError(
                    "gzip content size mismatch: %s" % packaged
                )
    if total_packaged != int(manifest["packaged_bytes"]):
        raise ValueError("manifest packaged-byte total is inconsistent")
    with (output / "experiment_index.csv").open(
        "r", newline="", encoding="utf-8"
    ) as handle:
        index_rows = list(csv.DictReader(handle))
    if not index_rows:
        raise ValueError("experiment index is empty")
    return {
        "files": len(manifest["records"]),
        "index_rows": len(index_rows),
        "packaged_bytes": total_packaged,
        "valid": True,
    }


def augment_auxiliary_analyses(output, compress_threshold):
    """Add newly declared analysis summaries to an existing package."""

    output = Path(output).resolve()
    manifest_path = output / "package_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    packaged_names = {
        str(record["packaged"]) for record in manifest["records"]
    }
    added = []
    for relative in AUXILIARY_ANALYSES:
        source = ROOT / relative
        destination = output / "analyses" / Path(relative).name
        packaged_name = destination.relative_to(output).as_posix()
        if packaged_name in packaged_names:
            continue
        packaged, compressed = _copy(
            source, destination, compress_threshold
        )
        record = _record(
            source, packaged, compressed, "analysis_summary", output
        )
        manifest["records"].append(record)
        packaged_names.add(packaged_name)
        added.append(record)
    manifest["file_count"] = len(manifest["records"])
    manifest["source_bytes"] = sum(
        int(item["source_bytes"]) for item in manifest["records"]
    )
    manifest["packaged_bytes"] = sum(
        int(item["packaged_bytes"]) for item in manifest["records"]
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {"added": len(added), "file_count": manifest["file_count"]}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default=str(
            ROOT
            / "research_artifacts"
            / "path_conditioned_l185_l188_2026-07-19"
        ),
    )
    parser.add_argument(
        "--compress-threshold-mb", type=float, default=5.0
    )
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--augment-auxiliary", action="store_true")
    args = parser.parse_args(argv)
    if args.validate_only:
        print(json.dumps(
            validate_package(args.output), indent=2, sort_keys=True
        ))
        return 0
    if args.augment_auxiliary:
        print(json.dumps(
            augment_auxiliary_analyses(
                args.output,
                int(args.compress_threshold_mb * 1024 * 1024),
            ),
            indent=2,
            sort_keys=True,
        ))
        return 0
    if args.compress_threshold_mb <= 0.0:
        raise ValueError("compression threshold must be positive")
    manifest, rows = build(
        args.output,
        int(args.compress_threshold_mb * 1024 * 1024),
    )
    print(json.dumps({
        "output": str(Path(args.output).resolve()),
        "files": manifest["file_count"],
        "index_rows": len(rows),
        "source_mb": manifest["source_bytes"] / (1024 ** 2),
        "packaged_mb": manifest["packaged_bytes"] / (1024 ** 2),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
