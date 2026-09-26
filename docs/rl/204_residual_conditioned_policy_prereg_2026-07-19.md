# L204 Residual-Conditioned Policy Preregistration

## Frozen scientific question

Does exposing causal ordinary-ICODE residual context to the offline SAC Actor
improve the policy prior used by RL-Driven MPPI under seen and unseen MuJoCo
physics, relative to the same Actor without this context?

This is the first frozen coupling mechanism in the project plan:

\[
\text{ICODE residual / innovation / reliability} \rightarrow
\text{RL policy prior} \rightarrow \text{MPPI}.
\]

It does not change the paper direction and does not introduce terminal value,
value-aligned ICODE, adaptive rollout budgets, memory cost, or RL control of the
safety layer.

## Causal input contract

The seven appended features are:

1. normalized predicted residual acceleration \(\hat r_v\);
2. normalized predicted residual yaw acceleration \(\hat r_\omega\);
3. signed EMA of the completed-transition velocity innovation;
4. signed EMA of the completed-transition yaw-rate innovation;
5. normalized ensemble disagreement;
6. training-support confidence;
7. a completed-innovation validity flag.

The innovation at decision \(t\) may use transition \((t-1,t)\), but never the
future transition \((t,t+1)\). MuJoCo physics-domain labels, simulator contact
state, ground-truth obstacle coordinates, and future outcomes are forbidden
policy inputs.

## Initialization control

The 54 legacy L185 features remain in their original order. The seven context
features are appended. The first Actor layer copies all L185 weights exactly;
the seven new columns are exactly zero. The legacy normalizer statistics are
copied, while new features start at zero mean and unit standard deviation. Thus
the deterministic step-zero policy must equal L185 before learning.

Source checkpoint:

`results/research_platform/rl/path_conditioned_l185_seed20261901_60k_v2/checkpoints/step_000050000.pt`

Ordinary ICODE ensemble:

- `l57_icode_high_dynamic_h36_seed20261201_v1/best.pt`
- `l57_icode_high_dynamic_h36_seed20261202_v1/best.pt`
- `l57_icode_high_dynamic_h36_seed20261203_v1/best.pt`

## Development protocol

- Train on the four L185 path geometries and only `seen` physics domains.
- Validate on reverse-S and hairpin paths under both seen and unseen domains.
- Freeze ICODE checkpoints, MPPI costs, action bounds, LaserScan, scan_guard,
  local obstacle layer, and safety arbitration.
- Select a checkpoint without inspecting the later sealed confirmation seeds.
- Compare at identical MPPI rollout budget \(K\).

## Required ablation

1. ordinary ICODE + ordinary L185 RL prior (simple combination);
2. ordinary ICODE + residual-conditioned RL prior (proposed coupling);
3. ordinary ICODE-MPPI without RL prior, as an interpretability reference.

## Advancement gate

Relative to the simple combination, the selected policy must satisfy:

- no success-rate loss and no collision-rate increase;
- lower pooled cross-track RMSE on fresh development episodes;
- the improvement must not be confined to one seed or one physics domain;
- unseen-domain cross-track RMSE must not regress materially;
- mean absolute control jerk may increase by at most 5%;
- planner rollout count and safety configuration must be identical.

Only if the development gate passes will a disjoint sealed seed set be run.
The sealed result is accepted only when the direction remains favorable and its
paired uncertainty interval is reported. Smoke runs verify plumbing only and
are not scientific evidence.

## Second frozen coupling mechanism

After this gate, the project proceeds to RL-to-ICODE coupling using a
value-*ranking* consistency objective rather than the rejected raw value-MSE
objective. This second mechanism remains preregistered conceptually here but
will receive its own implementation-level preregistration before training.
