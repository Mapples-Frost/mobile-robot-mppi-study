# Redesigning the experiment from Bøhn 2021

Status: DESIGN, 2026-09-23. Written after the Bøhn reproduction completed
**negative**, and after reading our own tune-split table. It proposes a new,
separately pre-registered study. It does **not** edit the frozen ladder study,
whose test sweep is in flight.

---

## 0. The two things we now know

**The reproduction is done and it is negative.**
`research_artifacts/bohn2021_reproduction_2026-09-17/report/README.md`: across 54
training runs and 810,000 experiences, RL-tuned H did not beat the best fixed
horizon on either of Bøhn's systems.

| task | RL (3-seed mean) | best fixed H | gap |
|---|---:|---|---:|
| inverted pendulum | 468.674 | H30: 410.761 | +14.1% |
| vehicle | 93.537 | H15: **10.641** | +779% |

Refinements (Riccati init, smaller perturbations, low entropy, SAC-teacher,
teacher-value supervision) all stayed negative. The vehicle number is not a
tuning failure; it is a different regime.

**Our own study is null at α=1 for a reason we can now name.** From
`analysis_tune.json`, per control cycle, at `dt = 0.10 s`:

| arm | ms/cycle | ρ = latency/dt |
|---|---:|---:|
| B1 (K64,H24) | 11.5 – 14.6 | 0.115 – 0.146 |
| B2 (H-only) | 10.6 – 13.1 | 0.106 – 0.131 |
| OURS (joint) | 18.4 – 20.3 | 0.184 – 0.203 |

**Compute is nearly free in this environment.** The deadline is never
threatened, so drawing more samples never hurts, so there is nothing to
allocate. A fixed `(64, 24)` is close to optimal because no tension exists for
an adaptive policy to resolve. Our arm spends 1.65× the compute and buys
nothing measurable — and that is the correct outcome *given this design*.

This is precisely the axis Bøhn built his experiment around and we did not.
His cost function (9) charges the horizon directly,

```
R(s,a) = R_P(s') + λ_C (t_max − t) R_C(s') + λ_N R_N(a),   R_N(a) = a
```

with `λ_N = 1e-3` (vehicle) / `3e-3` (pendulum). The agent pays for `H` in the
reward, every step. We charge `−0.05 · Σ measured_e2e_s`, which at ~12 ms/cycle
is roughly 17% of the tracking term — present, but not binding. **Bøhn priced
compute until it bit. We did not.**

So the redesign is not "find a scenario where we win". It is: *restore the
scarcity that makes budget allocation a real decision*, which is the thing
Bøhn's design has and ours lacks.

---

## 1. Change 1 — compute scarcity becomes a declared axis (axis D)

### What changes

The applied actuation latency stops being this laptop's wall-clock time and
becomes

```
latency(K, H, arm) = α · ĉ(K, H, arm)
```

where `ĉ` is a cost model of the deployed pipeline (context build + actor
inference + budget mapping + MPPI rollout + safety) and **α is a
pre-registered scarcity level**. α = 1 reproduces today's compute by
construction.

### The mechanism already exists — no shared-source edit is required

`environment.py:75-84` already has the branch:

```python
elif delay_mode == 'frozen':
    latency = self.latency_table.predict(mode, k, h)
```

It resolves *after* `k, h` are bound, so the latency genuinely depends on the
allocation the policy just chose. `FrozenLatencyTable` is an exact-cell lookup
that raises on an unprofiled cell — so axis D is implemented by **passing a
table and a delay mode from a new roller**, not by touching any file the live
sweep imports. `RouteObservableEnv.__init__` already accepts `latency_table`;
the current roller simply never passes one.

Two traps this exposes, both handled in the new roller rather than discovered
later:

1. **The table must cover every reachable cell.** The action grid is
   `K ∈ {16,32,…,256}` (16 values) × `H ∈ {8,…,40}` (33 values) = **528 cells**.
   The existing frozen profile has 24. A learned policy will address cells the
   old profile never measured and the lookup will raise mid-roll. The new
   profile must be dense over the full grid, or `ĉ` must be the fitted
   surface evaluated per cell (preferred — it is exact by construction and
   cheap).
2. **`mode` is part of the key.** The existing profile was taken at
   `mode='nominal'`; every ladder roll runs `'residual'`. Reusing it would
   silently price the wrong pipeline.

### ĉ is already measurable from data on disk

The B1 grid was rolled at 16 `(K,H)` combinations over 42 tune scenes under
measured latency. Fitting per-cycle cost to a bilinear form gives

```
ĉ(K,H) = 4.6533 + 0.003860·K + 0.24049·H + 0.00120441·K·H     [ms]
```

with **max residual 0.159 ms (0.91% of the mean) and RMS residual 0.082 ms**
across all 16 cells. The `K·H` term is the MPPI rollout; the 4.65 ms intercept
is fixed per-cycle overhead — 37% of the cost at `(64,24)`, which is itself
worth reporting: a third of the budget is not allocation-sensitive at all.

This fit is a *feasibility check*, not the pre-registered model: it is fitted
to rolls taken under contention and covers 16 of 528 cells. The real `ĉ` comes
from a dedicated profile on an idle machine. But it establishes the shape and
lets the α grid be chosen from arithmetic rather than guesswork.

### Choosing the α grid

Using the fit, the fraction of the 528-cell action grid that still meets the
100 ms deadline:

| α | feasible cells | ρ at B1 (64,24) | ρ at OURS (170,31) | ρ at (256,40) |
|---:|---:|---:|---:|---:|
| 1 | 528 (100%) | 0.13 | 0.19 | 0.28 |
| 2 | 528 (100%) | 0.25 | 0.38 | 0.55 |
| 3 | 528 (100%) | 0.38 | 0.57 | 0.83 |
| 4 | 516 (97.7%) | 0.50 | 0.76 | 1.10 |
| 5 | 445 (84.3%) | 0.63 | 0.96 | 1.38 |
| 6 | 353 (66.9%) | 0.75 | 1.15 | 1.66 |
| 8 | 186 (35.2%) | 1.00 | 1.53 | 2.21 |

The structure is clear. Below α = 4 **nothing is infeasible** — the allocation
problem stays vacuous no matter how the scene varies, which is exactly today's
situation and explains the null result directly. The interesting region is
α ∈ [4, 8], where the expensive third of the action grid starts missing the
deadline while the cheap part does not.

So the pre-registered grid is **α ∈ {1, 3, 5, 6, 8}**: one free-compute anchor
that reproduces the current study, one still-slack point, and three inside the
regime where allocation binds. Chosen from the feasibility arithmetic above,
before any outcome at any α is known, and frozen.

What this does to the deadline ratio measured today:

| arm | α=1 | α=3 | α=5 | α=6 | α=8 |
|---|---:|---:|---:|---:|---:|
| B1 | 0.12 | 0.37 | 0.61 | 0.73 | 0.98 |
| B2 | 0.12 | 0.36 | 0.60 | 0.72 | 0.96 |
| OURS | 0.19 | 0.58 | 0.97 | 1.17 | 1.55 |

At α ≥ 5 a policy that allocates like ours does today **misses the deadline**,
and must learn to stop. That is the point: the question becomes *where to spend
a budget that is genuinely too small*, which is the question our method claims
to answer and the one Bøhn's λ_N poses.

### Why this is a legitimate change and not a thumb on the scale

- **Bøhn's own design is the precedent.** `λ_N R_N(a)` exists for exactly this
  reason: on a real platform the horizon costs time, and if the experiment does
  not charge for it the adaptive-budget question is vacuous. α is the same
  device, made physical instead of hand-priced.
- **α is applied identically to every arm**, from the same fitted model, with
  the arm-dependent term (actor inference) charged to the arms that actually
  pay it.
- **B1 is re-tuned at every α by the same frozen rule.** At high α the best
  fixed `(K,H)` will move down; it must be allowed to. Our claim is then
  "adaptive beats the best fixed point *at that scarcity*" — which is Bøhn's
  claim too. Reopening the baseline grid per α is *required* for fairness; it
  is not the manipulation pre-registration forbids, because the rule is
  unchanged and frozen, and it can only make the baseline stronger.
- **α = 1 is reported with equal prominence**, and we expect to lose there.
  A sweep whose first point is our own null result is not a cherry-pick.

### A large side benefit: C116 disappears

`delay_mode='measured'` is what makes every competing process a confound, what
forced 25 rolls into quarantine, and what makes machine exclusivity a
reproducibility condition. A fitted, deterministic cost model removes this
entirely: latency no longer depends on what else the machine is doing.
Analysis, plotting and rolls can run concurrently again.

The trade must be disclosed: we move from "wall-clock on one contended laptop"
to "a profiled cost model of that same pipeline". That is strictly *more*
faithful to Bøhn (whose cost is a pure model) and more faithful to a real
platform than a contended laptop is. Both framings go in the paper; the
measured-latency ladder remains as the α = 1 anchor that shows the model and
the wall-clock agree.

---

## 2. Change 2 — horizon-dependent preview uncertainty (Bøhn's cone)

### The gap

Bøhn's collision-avoidance task grows the obstacle's position uncertainty with
prediction distance, as a 2-D cone. That is the mechanism that makes a *longer*
horizon genuinely worse sometimes: far-future predictions are not merely
expensive, they are wrong, and planning against them makes the robot swerve at
phantoms.

Searching `physical_tradeoff_v2/*.py` and `adapter.py` for
`uncertain|preview_noise|horizon_noise|cone|obstacle_noise|sensor` returns
essentially nothing. **Our environment has no such mechanism.** Consequences:

- `H` is monotonically helpful up to its cost, so the H-decision is partly
  degenerate;
- B2's only available failure mode is the collapse we already observe;
- a reviewer can say our H-only baseline fails only because nothing in the
  environment punishes a *long* horizon, so the comparison is one-sided.

### What changes

Obstacle inflation grows along the prediction horizon:

```
r_eff(j) = r_robot + r_obs + β · (j · dt) · v_ref        j = 1 … H
```

applied inside the MPPI rollout cost, **identically for every arm**, as a
module constant (like `PREDICTION_MODE`), not a per-arm flag.

`β` is anchored, not tuned: it is fitted to the **measured growth of this
planner's own multi-step prediction error with rollout depth** `j`. The cone
then represents a real epistemic fact about the planner rather than a knob
chosen for outcome.

That growth curve does not yet exist. C46 establishes aggregate error at
`H = 36`, not error as a function of `j`, and no artifact on disk carries a
per-step profile. So β requires one new measurement: replay recorded
transitions through the residual dynamics, record position error at every
rollout depth `1 … 40`, fit the slope. This is an **offline** job — no plant
stepping, no measured latency — so it is cheap and unaffected by the α design.
It is computed once, written into the pre-registration with its residuals, and
not revisited. If the curve turns out to be flat, the cone is not justified and
this change is dropped; that outcome is recorded rather than worked around.

Effect: `H` acquires an **interior optimum that moves with the scene** —
short in open corridors, longer before a burst of corners, shorter again where
clearance is tight and phantom inflation would close the gap. That is a
well-posed adaptive-H problem, which is what Bøhn's vehicle task has and ours
does not.

---

## 3. Change 3 — a Bøhn-faithful H-only baseline

Our B2 is **not literally Bøhn's method**. It is Bøhn's idea trained on our
reward:

| Bøhn (9) | ours |
|---|---|
| `R_N(a) = a`, `λ_N = 1e-3` | `resource = −0.05 · Σ measured_e2e_s` |
| `λ_C (t_max − t) R_C` — penalty ∝ how early the episode ended | `collision = −10`, constant |

Deliberate and defensible, but a reviewer will call it a modified baseline, and
they will be right.

So we add `B2_bohn`: H-only SAC whose trainer reward is Bøhn's (9) with our
stage cost in `R_P`, a horizon term `λ_N · H`, and a time-to-failure-weighted
collision term `λ_C (t_max − t)`. `λ_N`, `λ_C` are carried over from Bøhn and
rescaled to our cost magnitudes by a rule frozen before its inputs are read —
the same discipline used for the compute-price calibration.

The comparison set becomes:

| arm | K | H | reward |
|---|---|---|---|
| B1 | fixed, tuned per α | fixed, tuned per α | — |
| B2 | fixed | learned | ours |
| **B2_bohn** | fixed | learned | **Bøhn (9)** |
| OURS | learned | learned | ours |

If H-only collapses under **both** rewards, the finding is about the *action
space* — adapting the horizon alone is unsafe — and not about our reward
function. That is a substantially stronger and more defensible claim than the
one we can make today, and it is the claim the B4 cell already hints at.

---

## 4. Change 4 — retrain at hold = 1

The deployed checkpoints were trained with a 5-cycle decision hold and are
evaluated at 1-cycle hold (C117). Both identified mechanisms run *against* the
learned arms. `sac_route_independent_realtime_2026-09-16` has never been
trained — the flag exists, the stage directory does not.

Retraining at hold = 1 removes a known adverse confound. It cannot be called
cherry-picking: the confound's direction is already argued and recorded as
running against us, so fixing it is a correction, not a selection. It also makes
the compute figures a measurement of the method rather than of a mismatch.

---

## 5. What the headline result becomes

The figure is **success rate and compute cost against α**, one line per arm, at
each ladder cell — with the B4 collapse as the mechanism inset:

| | B3→B4 (tune, α=1, existing) |
|---|---|
| B2 mean H | 17.0 → **13.2** |
| B2 success | 1.00 → **0.00** |
| OURS mean H | 31.5 → 31.7 |
| OURS success | 1.00 → **0.889** |

A complete causal chain already exists there: the horizon-only arm *shortens* H
exactly where the scene demands a longer one, and times out. Under α > 1 and
the uncertainty cone, this should sharpen and appear across cells rather than at
one level, because both changes raise the cost of getting the horizon wrong.

**The honest prediction, stated before the runs:** at α = 1 we expect to keep
losing — B1 remains near-optimal where compute is free. The result we expect is
a *crossing*, and the crossing point is the contribution. A sweep that begins
with our own null result and reports it at equal prominence is the opposite of
a cherry-pick; it is what lets the win mean something.

---

## 6. Scope discipline

- This is a **new study**, `kh_compute_scarcity_2026-09-24`. The frozen ladder
  study is untouched; its test sweep finishes and is adjudicated exactly once,
  as pre-registered.
- **No file on the live sweep's import path may be edited.** `kh_ladder_roll.py`,
  `adapter.py`, `physical_tradeoff_v2/environment.py`, `core.py`,
  `kh_difficulty_ladder.py` are imported afresh by every worker the driver
  spawns; editing any of them mid-sweep would silently change what the
  remaining rolls measure. All new code goes in new files.
- Nothing runs on the machine until the sweep and its chained ablation stage
  finish. Source authoring only until then.

## 7. Order of work

1. Write the pre-registration (axes, α grid, β derivation, λ rescaling rule,
   B1 re-tuning rule per α, claim statements, falsification criteria) — freeze
   with hashes **before** any run.
2. Profile the deployed pipeline over the reachable `(K,H)` grid (16 × 33 = 528
   cells); fit `ĉ`; report residuals. Requires an idle machine.
3. Implement the cone and the injected-latency roller in new files;
   differential-test that α = 1 with `ĉ` reproduces the measured-latency rolls
   within the profile's own spread.
4. Retrain: OURS and B2 at hold = 1; B2_bohn. Two policy seeds each.
5. Re-tune B1 per α by the frozen rule, on the tune split only.
6. Roll the α × ladder suite; adjudicate once.
