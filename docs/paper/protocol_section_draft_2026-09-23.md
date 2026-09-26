# Protocol section draft (IROS)

Status: DRAFT, 2026-09-23. Written from C111–C117 and the frozen pre-registrations
while the test sweep is mid-roll. It contains **no test-split result**, by
construction: everything below is design, procedure and disclosure, all of which
were fixed before the test rolls and none of which changes when they land.

The numbered subsections map onto what an IROS reviewer checks in order:
what was compared, whether the baseline was given a fair chance, what was frozen
and when, and what went wrong.

---

## 1. What is being compared

Three arms, differing **only** in how the MPPI sample count `K` and horizon `H`
are chosen at each control cycle:

| arm | K | H | source |
|---|---|---|---|
| B1 (nominal MPPI) | fixed | fixed | tuned over a 16-point grid, frozen before any comparison roll |
| B2 (RL-tuned H) | fixed | learned per cycle | SAC policy, two independently trained seeds |
| OURS (joint K/H) | learned per cycle | learned per cycle | SAC policy, two independently trained seeds |

Everything else is held identical across arms: the same scenes, the same roll
seeds, the same `max_steps`, the same reward, the same safety layer, and — the
one that took a rewrite to get right — the **same plant model**.

> **A confound we removed rather than reported.** An earlier three-method runner
> set `dynamics_mode = 'nominal'` for the baseline and `'residual'` for the
> learned arms, giving the baseline a different plant model on top of its fixed
> `(K,H)`. Its 64.15 mm against RL-H's 56.06 mm therefore confounded two changes
> at once, in the direction that flatters our method. Every roll in this paper
> uses `prediction_mode = 'residual'` for all arms; it is a module constant, not
> a flag, because changing it would invalidate every roll on disk.

## 2. Compute is physical, not bookkeeping

The simulator runs with `delay_mode='measured'`: the actuation latency applied
to the plant each cycle **is** the wall-clock time the planner spent. Drawing
more samples does not merely cost seconds of experiment time — it delays the
command, and the plant moves during the delay.

Two consequences that belong in the protocol rather than a footnote:

1. **Machine exclusivity is a reproducibility condition.** Any process competing
   for CPU, memory bandwidth or page cache changes the simulated physics. This is
   not a statement about experiment hygiene; it is a statement about what the
   numbers mean. (C116)
2. **Read-only analysis is not safe.** During the test sweep, a niced, read-only
   trace audit perturbed rolls for 160 s. Low CPU priority was no protection
   because the binding resource was memory bandwidth and page-cache eviction —
   decompressing ~4.6 GB of traces on a 7 GB machine already running twelve
   workers. Affected rolls were quarantined by a mechanical rule (execution
   window overlapped a named wall-clock window), fixed before any affected result
   file was opened.

**The quarantine rule is arm-blind, which is not the same as arm-symmetric.**
Because the rule keys on *when a roll executed* and the workers interleave arms
freely, the resulting losses are uneven: the first excursion took 3 rolls from
`nominal_K064_H24` and 2 from each of the other five arms, and the second took 2
from every arm but not at the same scene keys (the joint-K/H arms lost
`a0_s17_seed13330003` where the rest lost `a0_s18_seed13330002`). We state this
rather than rounding it to "symmetric" because the asymmetry has a real and
bounded consequence:

- **Paired comparisons are immune.** `paired()` intersects the two arms' key sets,
  so a key missing from either arm is dropped from both. An uneven quarantine
  shrinks `n`; it cannot tilt a difference.
- **Per-cell unpaired means are not immune.** A group table averages each arm over
  the rolls that arm actually has, so an unbackfilled asymmetry would compare cell
  means computed over different scene sets.

Every quarantined roll is therefore re-rolled before adjudication, and the
analysis refuses to report any claim from a run whose coverage banner is not
complete. *(Status at the time of writing: the re-roll pass is queued behind the
live sweep and has not yet run. This paragraph states the protocol, and the
protocol is enforced mechanically by `coverage_check()` — not by our
remembering to do it.)* The quarantine is a deletion of *contaminated* rolls,
never of unfavourable ones: the rule was written before the excursions, reads
only wall-clock timestamps, and has no access to any outcome field.

A guard (`sweep_guard.py`) now refuses to start any competing job while
measured-delay rolls are alive.

## 3. The scenario ladder

Claims about "where the advantage appears" require a difficulty axis that was
defined before the results, not chosen after them. The ladder varies three
things independently, each over its own levels:

| axis | what varies | levels | why it is separate |
|---|---|---|---|
| A | the **distribution** of turning along the path, with **total turning matched across levels** | 0–4 | isolates heterogeneity from sheer difficulty: level 4 is not "more turning", it is the same turning bunched differently |
| B | the **amount** of turning | 0–4 | the conventional difficulty knob |
| C | corridor **width** along the path | 0–3 | clearance, which the previous suite held constant by construction and therefore could not express |

That is **14 axis–level cells**, 13 scenes each, **182 scenes** total — with
**tune seeds 1–3 (42 scenes) and test seeds 11–20 (140 scenes) disjoint**. The
pre-registration was frozen with per-file hashes before any comparison roll.

Axis A is the one that makes the ladder more than a difficulty sweep. Because
total turning is held equal across its levels, an arm that does better at A4
than A0 cannot be explained by "A4 is easier" — the path demands the same
cumulative rotation. Any difference is attributable to how the turning is
*distributed*, which is precisely the regime where a fixed horizon should be
mismatched and an adaptive one should pay off.

Tune-split numbers select; they are never reported as results. The test split is
adjudicated exactly once.

## 4. Baseline tuning — why the baseline is strong

B1 was tuned to **its own** best over a 16-point `(K,H)` grid on the tune split
and frozen at `K=64, H=24`. Two deliberate choices:

- The selected `K=64` differs from the learned arms' mean `K≈170`. Because a
  reader will reasonably ask whether the comparison is really about *allocation*
  or just about *budget*, we report both the tuned baseline and a **K-matched**
  baseline at `K=128`, and **lead with whichever is less favourable to our
  method**.
- The grid is not re-run or re-selected for any later experiment. It is a frozen
  measured frontier on the same scenes.

## 5. Statistics

Comparisons are **paired** on the matched key `(axis, level, scene_seed,
roll_seed, policy_seed)`. A learned arm is only ever compared against another
arm's roll from the *same trained network*; B1 has no policy seed and is
broadcast, because it is deterministic given `(scene, roll_seed)` and
replicating it per policy seed would record the same number twice rather than
add evidence.

- Percentile bootstrap, 10,000 resamples, seed 20260923, **resampling pairs** —
  breaking the pairing would reintroduce the between-scene variance the design
  removes.
- Sign-flip permutation p-values (the null is symmetry about zero, and the
  differences are not Gaussian over a ladder with hard-scene blowups).
- Holm–Bonferroni, with **success and RMSE in separate families**: correcting
  them jointly would treat one claim as two.
- Cliff's delta alongside every interval, because a difference can be reliable
  and tiny.
- Both policy seeds are always reported separately. A sign disagreement between
  seeds **is** the result and is not resolved by preferring the pooled number.

## 6. Disclosed train/deploy mismatch

The deployed learned policies were trained with a **5-cycle decision hold** — the
trainer selects one `(K,H)` and applies it for five consecutive cycles — and are
evaluated at **1-cycle hold**, the roller re-querying the actor every cycle.
Because the actor forward pass sits inside the end-to-end latency meter, and
because latency is physical here, the learned arms pay inference cost five times
more often per episode than they did in training.

Both identified mechanisms — distribution shift from deciding 5× more often, and
5× more inference cost inside the measured latency — run **against** the learned
arms. Neither can have manufactured an advantage for our method. What they do
mean is that our compute figures are an **upper bound** on what learned joint
allocation intrinsically costs, and we report them as a measurement of *this
deployed configuration* rather than of the method in general. (C117)

The magnitude is not established, and we say so. Fixing it requires a hold-1
retraining, which is a separate pre-registered experiment.

## 7. What the policy was actually optimizing

The environment's reward contains a compute price term whose coefficient
`lambda_compute` is 0 in every configuration — but the SAC trainer **does not use
the environment reward**. It computes its own, containing

```
resource = -0.05 * sum(measured_e2e_s)
```

over the cycles of one decision, which carries roughly **17% of the tracking term
at the median decision** in the deployed policies' own training records.

We flag this because an earlier draft of our own analysis asserted the opposite —
that compute was never charged — and built a causal story on it. The correct
statement is narrower: *a compute price was present during training and was not
sufficient to prevent sample over-allocation.* Whether a larger price would
change the allocation is a genuinely open question and is tested separately, at
prices calibrated by a rule frozen before its own inputs were read.

## 8. Deviations

Every deviation is recorded in a dated artifact with its detection method, its
disposition, and what it does **not** establish:

| deviation | disposition |
|---|---|
| concurrent sweep contention (77 s) | 13 rolls quarantined (3 from one baseline arm, 2 from each of the other five); re-roll queued, not yet verified |
| read-only trace audit contention (160 s) | 12 rolls quarantined, 2 per arm but at differing scene keys; re-roll queued, not yet verified |
| false "compute is unpriced" mechanism | corrected in place; the pre-registration built on it was **superseded, not edited** |
| decision-hold mismatch | disclosed; magnitude unmeasured; direction argued and adverse to us |

Frozen documents are never edited to match later discoveries. When a frozen
pre-registration turned out to rest on a false premise, we left it frozen, wrote
a supersession notice beside it, and wrote a successor that inherits the
parameter grid verbatim — so the grid could not be re-chosen with knowledge of
where the anchor lay. A subsequent error *in the successor* was likewise handled
by errata rather than edit.

---

## Open items before submission

- [ ] IROS submission date — deliberately not guessed; fill from the official call
- [x] §3/§4 scene and grid counts: verified against `ladder_manifest.json` and
      `b1_selection.json` (182 scenes over 14 axis–level cells; 16-point grid;
      `K=64, H=24` selected) — one error found and fixed (axis C has 4 levels, not 5)
- [x] §2/§8 quarantine counts: verified against both deviation records — the
      "arm-symmetric" claim was **false** and is corrected here and in all four
      artifacts that carried it
- [ ] §2/§8 "all re-rolled": currently a **forward claim**. The backfill pass has
      not run. Confirm from `backfill_verification.json` before submission; if any
      key is still missing, the sentence must change, not the data
- [ ] §6 magnitude: fill in if the hold-1 retraining runs, otherwise keep as disclosed-unmeasured
- [ ] §7: the 17% figure is median-based; do not quote it beside the mean-based 4.87% without saying which is which
- [ ] Results section is NOT drafted here — it cannot be, until the test split is adjudicated once

## A note on how two errors in this document were found

Both were caught by doing arithmetic on numbers that were already in the file
being cited: axis C's level count, and 13 ÷ 6. Neither needed new data. Both had
survived because each document quoted the one before it rather than re-deriving
from the source — the same failure mode as the "compute is never charged"
premise corrected earlier the same day. The practical rule that follows is
narrow enough to actually apply: **any quantitative claim that crosses a
document boundary gets recomputed from its own source at least once before it
reaches the paper.**
