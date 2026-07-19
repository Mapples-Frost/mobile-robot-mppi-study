#!/usr/bin/env python3
"""Build a versionable, lossless L190--L194 evidence package."""

import argparse
import csv
import gzip
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIRS = (
    "results/research_platform/datasets/path_aware_reliability_l190",
    "results/research_platform/datasets/path_aware_reliability_innovation_l191",
    "results/research_platform/datasets/path_aware_reliability_continuous_l192",
    "results/research_platform/rl/path_aware_ordinary_reliability_l190",
    "results/research_platform/rl/path_aware_value_reliability_l190",
    "results/research_platform/rl/path_aware_ordinary_reliability_innovation_l191",
    "results/research_platform/rl/path_aware_value_reliability_innovation_l191",
    "results/research_platform/rl/path_aware_ordinary_reliability_continuous_l192",
    "results/research_platform/rl/path_aware_value_reliability_continuous_l192",
    "results/research_platform/rl/path_aware_innovation_closed_loop_dev_l193",
    "results/research_platform/rl/path_aware_source_competence_closed_loop_dev_l194",
)
SOURCE_FILES = (
    "results/research_platform/rl/path_aware_reliability_l190_component_analysis.json",
)
CHECKPOINT_REFERENCES = (
    "results/research_platform/rl/path_conditioned_l185_seed20261901_60k_v2/checkpoints/step_000050000.pt",
    "results/research_platform/l57_icode_high_dynamic_h36_seed20261201_v1/best.pt",
    "results/research_platform/l57_icode_high_dynamic_h36_seed20261202_v1/best.pt",
    "results/research_platform/l57_icode_high_dynamic_h36_seed20261203_v1/best.pt",
    "results/research_platform/gate3_value_aligned_ensemble_member1_l192/best.pt",
    "results/research_platform/gate3_value_aligned_ensemble_member2_l192/best.pt",
    "results/research_platform/gate3_value_aligned_ensemble_member3_l192/best.pt",
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy(source, destination, compress_threshold):
    destination.parent.mkdir(parents=True, exist_ok=True)
    compressed = bool(
        source.suffix.lower() == ".csv"
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


def _record(source, destination, compressed, package):
    return {
        "source": source.relative_to(ROOT).as_posix(),
        "source_bytes": source.stat().st_size,
        "source_sha256": sha256(source),
        "packaged": destination.relative_to(package).as_posix(),
        "packaged_bytes": destination.stat().st_size,
        "packaged_sha256": sha256(destination),
        "lossless_gzip": bool(compressed),
    }


def _calibration_index():
    rows = []
    for relative in SOURCE_DIRS:
        directory = ROOT / relative
        summary_path = directory / "calibration_summary.json"
        if not summary_path.is_file():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        selected = summary.get("selected_candidate", {})
        splits = {
            "validation": selected.get("validation_gate", {}),
        }
        splits.update(summary.get("evaluation", {}))
        for split, result in sorted(splits.items()):
            rows.append({
                "experiment": directory.name,
                "evidence_stage": "offline_reliability_calibration",
                "split": split,
                "gate_passed": bool(result.get("passed", False)),
                "episode_count": result.get(
                    "episode_count", result.get("episode_count", "")
                ),
                "rank_correlation": result.get(
                    "authority_error_rank_correlation", ""
                ),
                "rank_ci_low": result.get(
                    "authority_error_rank_correlation_ci95", ["", ""]
                )[0],
                "rank_ci_high": result.get(
                    "authority_error_rank_correlation_ci95", ["", ""]
                )[1],
                "tail_error_separation": result.get(
                    "relative_tail_error_separation", ""
                ),
                "source": summary_path.relative_to(ROOT).as_posix(),
            })
    return rows


def _write_index(package):
    rows = _calibration_index()
    for experiment in (
        "path_aware_innovation_closed_loop_dev_l193",
        "path_aware_source_competence_closed_loop_dev_l194",
    ):
        progress = (
            ROOT
            / "results/research_platform/rl"
            / experiment
            / "progress.csv"
        )
        if progress.is_file():
            with progress.open(
                "r", newline="", encoding="utf-8"
            ) as handle:
                for row in csv.DictReader(handle):
                    rows.append({
                        "experiment": progress.parent.name,
                        "evidence_stage": "closed_loop_development",
                        "split": "%s_seed%s" % (
                            row["factorial_arm"], row["seed"]
                        ),
                        "gate_passed": row["success"],
                        "episode_count": 1,
                        "rank_correlation": "",
                        "rank_ci_low": "",
                        "rank_ci_high": "",
                        "tail_error_separation": "",
                        "source": progress.relative_to(ROOT).as_posix(),
                    })
    if not rows:
        raise ValueError("evidence index would be empty")
    path = package / "evidence_index.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def build(package, compress_threshold):
    package = Path(package).resolve()
    if package.exists():
        raise FileExistsError(
            "package already exists; use a new immutable path: %s"
            % package
        )
    package.mkdir(parents=True)
    records = []
    for relative in SOURCE_DIRS:
        source_root = ROOT / relative
        if not source_root.is_dir():
            raise FileNotFoundError(source_root)
        for source in sorted(source_root.rglob("*")):
            if not source.is_file() or "__pycache__" in source.parts:
                continue
            destination = (
                package / "evidence" / source.relative_to(ROOT)
            )
            packaged, compressed = _copy(
                source, destination, compress_threshold
            )
            records.append(_record(
                source, packaged, compressed, package
            ))
    for relative in SOURCE_FILES:
        source = ROOT / relative
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = package / "evidence" / source.relative_to(ROOT)
        packaged, compressed = _copy(
            source, destination, compress_threshold
        )
        records.append(_record(source, packaged, compressed, package))

    checkpoint_references = []
    for relative in CHECKPOINT_REFERENCES:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        checkpoint_references.append({
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "copied": False,
        })
    index_rows = _write_index(package)
    manifest = {
        "schema_version": 1,
        "package": package.name,
        "git_sha_before_package_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True
        ).strip(),
        "evidence_class": "development_only",
        "sealed_geometry_included": False,
        "sealed_seeds_included": False,
        "source_file_count": len(records),
        "source_bytes": sum(item["source_bytes"] for item in records),
        "packaged_bytes": sum(
            item["packaged_bytes"] for item in records
        ),
        "records": records,
        "checkpoint_references": checkpoint_references,
    }
    (package / "package_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (package / "README.md").write_text(
        "# Path-aware reliability L190--L194 evidence\n\n"
        "This immutable package contains the complete development evidence "
        "for the path-aware reliability redesign: independently generated "
        "datasets, failed historical calibrations, causal innovation-anchor "
        "calibrations, continuous held-out confirmation, the failed L193 "
        "integration block, and the passed L194 source-competence "
        "remediation block.\n\n"
        "Failures are retained as first-class evidence. The package excludes "
        "the sealed L186 geometry and seeds 561--565. Large CSV files are "
        "stored as deterministic lossless gzip streams. Checkpoints are "
        "referenced by SHA256 because their bytes are already versioned in "
        "the preceding L185--L188 artifact package.\n",
        encoding="utf-8",
    )
    return manifest, index_rows


def validate(package):
    package = Path(package).resolve()
    manifest = json.loads(
        (package / "package_manifest.json").read_text(encoding="utf-8")
    )
    for record in manifest["records"]:
        source = ROOT / record["source"]
        packaged = package / record["packaged"]
        if sha256(source) != record["source_sha256"]:
            raise ValueError("source hash mismatch: %s" % source)
        if sha256(packaged) != record["packaged_sha256"]:
            raise ValueError("package hash mismatch: %s" % packaged)
        if record["lossless_gzip"]:
            digest = hashlib.sha256()
            size = 0
            with gzip.open(packaged, "rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
                    size += len(chunk)
            if digest.hexdigest() != record["source_sha256"]:
                raise ValueError(
                    "lossless content mismatch: %s" % packaged
                )
            if size != int(record["source_bytes"]):
                raise ValueError(
                    "lossless content size mismatch: %s" % packaged
                )
    with (package / "evidence_index.csv").open(
        "r", newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("evidence index is empty")
    return {
        "valid": True,
        "source_files": len(manifest["records"]),
        "index_rows": len(rows),
        "packaged_bytes": manifest["packaged_bytes"],
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default=str(
            ROOT
            / "research_artifacts/"
            "path_reliability_l190_l194_2026-07-19"
        ),
    )
    parser.add_argument(
        "--compress-threshold-mb", type=float, default=1.0
    )
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    if args.validate_only:
        print(json.dumps(
            validate(args.output), indent=2, sort_keys=True
        ))
        return 0
    if args.compress_threshold_mb <= 0.0:
        raise ValueError("compression threshold must be positive")
    manifest, index_rows = build(
        args.output,
        int(args.compress_threshold_mb * 1024 * 1024),
    )
    print(json.dumps({
        "output": str(Path(args.output).resolve()),
        "files": manifest["source_file_count"],
        "index_rows": len(index_rows),
        "source_mb": manifest["source_bytes"] / (1024 ** 2),
        "packaged_mb": manifest["packaged_bytes"] / (1024 ** 2),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
