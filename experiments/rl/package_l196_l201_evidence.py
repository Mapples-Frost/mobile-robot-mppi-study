#!/usr/bin/env python3
"""Package compact, checksummed L196--L201 evidence for version control."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_tree_files(source, destination, names):
    copied = []
    source = ROOT / source
    if not source.exists():
        return copied
    for path in sorted(source.rglob("*")):
        if not path.is_file() or path.name not in names:
            continue
        relative = path.relative_to(source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        copied.append(target)
    return copied


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="research_artifacts/l196_l201_2026-07-19",
    )
    args = parser.parse_args(argv)
    output = (ROOT / args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    selected = {
        "progress.csv",
        "ordinary_fixed_episodes.csv",
        "value_fixed_episodes.csv",
        "ordinary_adaptive_episodes.csv",
        "full_proposed_episodes.csv",
        "paired_comparisons.json",
        "factorial_contrasts.json",
        "provenance.json",
        "schedule.json",
        "multidomain_gate.json",
        "sealed_gate.json",
        "training_summary.json",
        "dataset_manifest.json",
        "episodes.csv",
    }
    sources = (
        "results/research_platform/rl/path_aware_multiphysics_dev_l196",
        "results/research_platform/rl/path_aware_multiphysics_dev_l197",
        "results/research_platform/rl/sealed_path_physics_confirmation_l198",
        "results/research_platform/path_policy_value_aligned_member1_l199",
        "results/research_platform/path_policy_value_aligned_member2_l199",
        "results/research_platform/path_policy_value_aligned_member3_l199",
        "results/research_platform/route_balanced_value_aligned_member1_l201",
        "results/research_platform/route_balanced_value_aligned_member2_l201",
        "results/research_platform/route_balanced_value_aligned_member3_l201",
        "results/research_platform/datasets/path_policy_value_alignment_l199",
        "results/research_platform/datasets/route_balanced_value_alignment_l201",
    )
    copied = []
    for source in sources:
        destination = output / Path(source).name
        copied.extend(_copy_tree_files(source, destination, selected))
    manifest = {
        "schema_version": 1,
        "artifact_scope": "L196--L201 compact research evidence",
        "excludes": [
            "raw NPZ datasets",
            "model checkpoints",
            "per-step trajectories",
        ],
        "files": [
            {
                "path": str(path.relative_to(output)),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in copied
        ],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "README.md").write_text(
        "# L196--L201 compact evidence\n\n"
        "This package preserves episode-level summaries, randomized schedules, "
        "checkpoint/config provenance, paired analyses, Gate decisions, and "
        "training summaries. Large reproducible arrays, checkpoints, and "
        "per-step trajectories remain excluded from Git. See "
        "`docs/rl/202_l196_l201_reliability_and_value_alignment_results_2026-07-19.md`.\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output_dir": str(output),
        "files": len(copied),
        "bytes": sum(path.stat().st_size for path in copied),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
