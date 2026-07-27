import csv
import hashlib
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
AMENDMENT = (
    ROOT
    / "configs"
    / "research"
    / "complex_maneuver_proposal_execution_equivalence_amendment_v1.yaml"
)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _normalized_config(path):
    text = path.read_text(encoding="utf-8")
    for prefix in (
        r"D:\Projects\mobile-robot-mppi-study-gate-d-reference",
        r"D:\Projects\mobile-robot-mppi-study-single-v6",
    ):
        text = text.replace(prefix, "${REPO}")
    return text


def test_amendment_replaces_only_the_unreproducible_reference_oracle():
    protocol = yaml.safe_load(AMENDMENT.read_text(encoding="utf-8"))

    assert protocol["status"] == "frozen_before_actor_on_equivalence"
    assert protocol["scope"] == "phase_zero_equivalence_reference_only"
    assert protocol["comparison"]["numeric_tolerance"] is None
    assert protocol["authorization"]["nonzero_margin_rounds"].startswith(
        "forbidden"
    )
    assert protocol["authorization"]["training_or_data_expansion"] == (
        "forbidden"
    )
    assert protocol["authorization"]["paid_server_or_formal_execution"] == (
        "forbidden"
    )


def test_concurrent_reference_and_diagnostic_are_exact_except_timing():
    protocol = yaml.safe_load(AMENDMENT.read_text(encoding="utf-8"))
    evidence = protocol["evidence"]
    reference_path = ROOT / evidence["concurrent_reference_trajectory"]["path"]
    diagnostic_path = (
        ROOT / evidence["concurrent_diagnostic_trajectory"]["path"]
    )

    assert _sha256(reference_path) == (
        evidence["concurrent_reference_trajectory"]["sha256"]
    )
    assert _sha256(diagnostic_path) == (
        evidence["concurrent_diagnostic_trajectory"]["sha256"]
    )

    reference = _rows(reference_path)
    diagnostic = _rows(diagnostic_path)
    excluded = set(protocol["comparison"]["excluded_timing_columns"])
    common = [
        column
        for column in reference[0]
        if column in diagnostic[0] and column not in excluded
    ]

    assert len(reference) == len(diagnostic) == 211
    assert len(common) == 593
    for reference_row, diagnostic_row in zip(reference, diagnostic):
        assert {
            column: reference_row[column] for column in common
        } == {
            column: diagnostic_row[column] for column in common
        }
    assert {
        float(row["paper_total_rollouts"]) for row in reference
    } == {0.0, 600.0}
    assert {
        float(row["paper_total_rollouts"]) for row in diagnostic
    } == {0.0, 600.0}


def test_resolved_configs_match_after_worktree_path_normalization():
    protocol = yaml.safe_load(AMENDMENT.read_text(encoding="utf-8"))
    reference = (
        ROOT
        / "research_artifacts"
        / "complex_maneuver_proposal_execution_consistency_v1"
        / "reference_runtime_reproduction"
        / "chapter1_seed791101019_frozen_full_af691f1_current_runtime"
        / "config_resolved.yaml"
    )
    diagnostic = (
        ROOT
        / "research_artifacts"
        / "complex_maneuver_proposal_execution_consistency_v1"
        / "equivalence_audit_isolated"
        / "chapter1_seed791101019_frozen_full_post_isolation"
        / "config_resolved.yaml"
    )
    normalized_reference = _normalized_config(reference)
    normalized_diagnostic = _normalized_config(diagnostic)

    assert normalized_reference == normalized_diagnostic
    assert hashlib.sha256(normalized_reference.encode()).hexdigest() == (
        protocol["frozen_identity"]["normalized_resolved_config_sha256"]
    )
