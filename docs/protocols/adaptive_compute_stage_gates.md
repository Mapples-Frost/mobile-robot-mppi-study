# Adaptive-compute sequential protocol — development freeze, 2026-09-07

## Material Passport

Scope: algorithm verification, source fidelity, development gates and conditional
training escalation. User-defined question and sequence; AI-authored code and
threshold operationalization. Formal outcomes are not yet available. No patient
or participant data. Raw logs are retained; failed runs are not discarded.

## Stage order and immutable decisions

1. Verify continuous algorithm; commit implementation.
2. Profile CPU latency and freeze development rectangle.
3. Verify published sources and implement explicitly labelled ports.
4. Baseline kernel/smoke sanity.
5. Gate R below.
6. Only on Gate R pass: Gate S.
7. Only on R/S pass: fixed landscape, paired seed0.
8. Only on useful landscape: PPO smoke then T1.
9. Only on positive T1: T2; only on positive T2: separately frozen formal T3.

An explicit failed gate stops subsequent experiments. Debugging may correct an
implementation error with a new version and retained failure record, but a
negative mechanism result is not a bug and does not trigger arbitrary retuning.

## Gate R (frozen before measurement)

Code: `experiments/compute_allocation/reliability_gate.py`. All candidate configs,
input probes and protocol are saved/hashes recorded before physical simulation.
Reference plant: strong baseline. Ordered development candidates:

1. Existing L70 combined payload/friction/actuator condition.
2. Same L70 condition plus existing seen-domain delay .08s.
3. Existing seen .08s delay alone.

The .10s combined-unseen configuration remains reserved, not examined for selection.
All use the same configured .35m/s/.9rad/s action bounds, rate limits, nominal model
and frozen ICODE checkpoint; none is adjusted from the gate outcome.

Eighteen probes: six command shapes (straight, left, right, sine steering,
stop/start, spin) at three headings. Same resting initial observable states, empty
actuator history and slew-limited commands; four seconds per probe. Distinct models
necessarily generate distinct future states. Forcing identical futures would
erase the manipulation. MuJoCo reset's initial v/omega limitations are respected:
all probes start at rest and exact initial observation equality is asserted.

Model predictions start from the exact same states and receive the exact same
commands, including fixed .04s assumed-delay correction. No use of actual future
applied-control truth. R3 reuses R2 physical outcomes, changing only prediction to
nominal+ICODE. Offline truth is an error target, never a policy input.

Report H=1,10,20,40 position/heading/v/omega RMSE. Primary H40 gate requires ALL:

* R2 position RMSE >=1.20*R1 and R2-R1 >=.02m.
* R3 position RMSE <=.85*R2 and R2-R3 >=.01m.
* R2 normalized state RMSE >=1.10*R1.
* R3 normalized state RMSE <=R2.

State error scales: [.25,.25,.35,.25,.60], wrapped heading; aggregate across fixed
probe endpoints. First passing candidate in the frozen order is selected; all
candidate outcomes remain visible. No pass means stop before Gate S/landscape/RL.
This is a manipulation check, not inferential evidence from 18 independent seeds.
Maximum2880 physical control steps and 300s hard runtime. No p-value gate.

## Gate S and landscape (conditional, not yet executed)

Development-only scene candidates must come from existing navigation benchmarks.
Easy is the existing open goal; initial Medium candidate is the strong-baseline
single obstacle, followed by existing path/slalom alternatives if needed. Hard is
the existing corridor/clutter task. Before execution, exact candidate files, order,
seed0, original success thresholds/max steps and representative budgets must be
written into a separate immutable Gate-S manifest.

Medium must have both success and failure among fixed-budget arms. Do not count
near-goal failure as success or extend a failed episode. No selected Medium means
stop. Conditional fixed landscape uses all selected scenes, reference/mismatch/
corrected conditions, frozen representative K points within128..256 and H10/20/30/40.
At least one scene-condition must show >=.30m final-distance spread or a mixed
success outcome; otherwise allocation sensitivity is inadequate and no RL starts.

## Training escalation (conditional design, not a formal freeze)

T0: at most8+4 transitions for numerical plumbing if changes require it.
T1: single training seed0, initial2048-step development screen; Easy/selected Medium,
reference/selected mismatch only. Shared reward/action ranges/PPO budgets across
scene-only, full and H-only comparator. Freeze episode sampling and evaluation
manifest before training. Inspect task failure, mean/P95 latency, raw/quantized
allocation and deterministic policy variation; boundary collapse >90% or no
conditional variation is not a positive mechanism result. A screen does not prove
convergence or superiority. No automatic reward search.

Only if T1 provides positive task/latency and noncollapse signals should a precise
T2 protocol be frozen (2–4 training seeds). Formal T3 (8–12 paired scenario seeds)
requires stable T2 and resolved baseline fidelity, then a separate immutable
commit with train/validation/test membership, model hashes and primary contrasts.
No T3 conditions are designated frozen now.

Primary formal contrasts planned: proposed vs strongest verified published
comparator, best development-fixed ICODE, H-only adaptation, scene-only adaptation.
Paired continuous effect estimates/CIs plus distribution-appropriate paired tests;
paired success/collision proportions; Holm adjustment within these four comparisons.
Control cycles/overlapping windows are not independent replicates. Completion
times among successes alone cannot stand in for unconditional performance.
Pareto claims require comparable full task episodes and actual timing; no frontier
will be inferred from four-step baseline sanity records.
