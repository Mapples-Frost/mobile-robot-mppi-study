"""Render the audited min-Q extension without selecting seeds or scenes."""
import json

from min_q_protocol import OUT
from runtime import ROOT


REPORT = ROOT / "docs/reports/bohn2021_min_q_extension_2026-09-24.md"


def read(path):
    return json.loads(path.read_text())


def table(lines, audit):
    lines += ["", "## %s" % audit["split"].capitalize(), "",
              "| Task | Method | Seed | Total | Physical | H proxy | Constraint | Goals | Constraint stops | Solver failures | Mean H |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in sorted(audit["rows"], key=lambda r: (r["task"], r["method"], r["seed"])):
        lines.append("| %s | %s | %d | %.3f | %.3f | %.3f | %.3f | %d | %d | %d/%d | %.2f |" % (
            row["task"], row["method"], row["seed"], row["mean_total_cost"],
            row["mean_physical_cost"], row["mean_H_cost"], row["mean_constraint_cost"],
            row["goal_episodes"], row["constraint_episodes"], row["solver_failure_steps"],
            row["evaluated_steps"], row["mean_H"]))
    lines += ["", "| Task | Fixed H | Mean author RL | Mean min-Q | Mean fixed H | min-Q minus author by seed | min-Q minus fixed by seed | Registered robust advantage |",
              "|---|---:|---:|---:|---:|---|---|---|"]
    for result in audit["comparisons"]:
        delta_author = ", ".join("%+.3f" % row["min_q_minus_author"] for row in result["paired"])
        delta_fixed = ", ".join("%+.3f" % row["min_q_minus_fixed"] for row in result["paired"])
        averages = result["mean_cost_across_seeds"]
        lines.append("| %s | %d | %.3f | %.3f | %.3f | %s | %s | %s |" % (
            result["task"], result["fixed_H"], averages["author_rl"],
            averages["min_q"], averages["fixed"], delta_author, delta_fixed,
            "yes" if result["registered_robust_advantage"] else "no"))


def diagnostics(lines, audit):
    lines += ["", "## Failure and interpretation checks", "",
              "The vehicle test mean is sensitive to single long-running scenes. The following is a post-test diagnostic, not a replacement for the registered mean-cost endpoint. Scene indices are zero-based.",
              "", "| Vehicle method | Seed | Highest-cost scene | Scene cost | Share of seed's 20-scene total |",
              "|---|---:|---:|---:|---:|"]
    for row in sorted((r for r in audit["rows"] if r["task"] == "vehicle"),
                      key=lambda r: (r["method"], r["seed"])):
        index, cost = max(enumerate(row["scene_costs"]), key=lambda item: item[1])
        share = cost / sum(row["scene_costs"])
        lines.append("| %s | %d | %d | %.3f | %.1f%% |" % (
            row["method"], row["seed"], index, cost, 100 * share))
    scene = read(OUT / "evaluations/test/vehicle/min_q_s0/summary.json")["episodes"][14]
    assert scene["termination"] == "steps" and scene["steps"] == 150
    assert scene["solver_failure_steps"] == 0
    trace = read(OUT / "evaluations/test/vehicle/min_q_s0/trace_14.json")
    assert max(step["horizon"] for step in trace) == 3
    lines += ["", "Vehicle min-Q seed0, scene 14 used H1-3 throughout the 150-step timeout; its %.3f cost is almost entirely physical tracking error, with zero reported solver-failed steps. The fixed-H25 seed0 controller reached the goal on this scene at cost %.3f. This associates short H with the observed failure; different trained terminal functions and the reset warmup prevent attributing the full difference to H alone." % (
        scene["total_cost"], next(r for r in audit["rows"] if r["task"] == "vehicle" and r["method"] == "fixed" and r["seed"] == 0)["scene_costs"][14]),
              "", "On the pendulum test, all nine evaluated models stop on constraint in 12 of 20 scenes and none has a goal termination. These are matched scene conditions, not 108 independent failures. Solver-failed steps also occur and are reported separately above; a passed cost audit does not establish controller reliability or the paper's adaptive-H claim."]


def main():
    validation = read(OUT / "validation_audit.json")
    test = read(OUT / "test_audit.json")
    assert validation["passed"] and test["passed"]
    assert read(OUT / "validation_audit_passed.json")["passed"]
    assert read(OUT / "test_audit_passed.json")["passed"]
    robust = all(row["registered_robust_advantage"] for row in test["comparisons"])
    lines = ["# Bohn 2021 min-Q actor training extension", "",
             "This is a one-factor method extension, not a strict reproduction of the author's SAC. The actor uses min(Q1,Q2) where the pinned author implementation uses Q1. The terminal value, task configuration and training budget match the paper-grid reconstruction. The original experiment configuration and test files remain unavailable.",
             "", "## Verdict", "",
             "Data audit: **passed** for %d validation and %d independent-test model-episode conditions. The registered robust adaptive-H advantage: **%s** across both tasks. A negative or mixed result does not refute the original paper because the task configuration was reconstructed." % (
                 validation["episode_conditions"], test["episode_conditions"],
                 "observed" if robust else "not observed"),
             "", "All three training seeds are reported. Fixed H was chosen on the older validation set from the complete ten-H grid at seed0; selected H was then trained at seeds1/2 with the same 15,000-step settings. Neither new bank was used to choose H, seed or checkpoint. The historical holdout was already exposed and is not used for this claim."]
    table(lines, validation)
    table(lines, test)
    diagnostics(lines, test)
    lines += ["", "## Audit and budget", "",
              "The independent audit recomputes physical, H and constraint costs from every saved trace step, checks episode and summary totals, solver failures, goal/constraint counts, model hashes, bank hashes and frozen inference. Input-bound excess and any pendulum step-100 state violation labeled as time limit are recorded separately, not erased by a passing cost audit.",
              "", "The inherited paper grid contains 26 models and 390,000 retained training steps, including 300,000 fixed-H search steps and 90,000 original RL steps. This extension adds six min-Q and four fixed-H seed-matched models, 150,000 steps total (90,000 adaptive training and 60,000 extra fixed-H training); the retained total is 540,000 steps across 36 models. New training updates: %d. Each split evaluates all 18 models, with %d validation and %d test episodes per model. These counts exclude interrupted historical work and the reset warmup; H cost is the paper's computation proxy, not measured latency. The fixed-H search was more expensive than the adaptive training and must be included when comparing development budgets." % (
                  test["new_training_updates"], 10, 20),
              "", "The frozen protocol and bank hashes are under `research_artifacts/bohn2021_reproduction_2026-09-17/results/min_q_training_2026-09-24/`. The per-seed and paired-scene values, failure cases and audit details are in `validation_audit.json` and `test_audit.json` there. Even a local advantage would not establish exact numerical reproduction of the original paper."]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(REPORT), "audit_passed": True,
                      "robust_advantage_both_tasks": robust}))


if __name__ == "__main__":
    main()
