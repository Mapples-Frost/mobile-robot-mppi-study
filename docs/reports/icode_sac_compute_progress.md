# ICODE + SAC compute allocation: live development record

Status: in progress, 2026-09-08. Branch `codex/icode-sac-compute`.
This is a new prospective continuation of the joint K/H research line. Old
physical_tradeoff and physical_tradeoff_v2 decisions remain unchanged.

## Completed model screen

288 episodes, 54,323 physical control cycles; all 12 scene families and both
speed caps, one paired development seed, two prediction models, six budgets.
Every treatment computes the same frozen shadow ICODE reliability context.
Sensing is ideal simulated pose/twist plus simulated scan. These are MuJoCo
results, not robot hardware or noisy odometry validation.

| Prediction model | K | H | Success / 24 | Collisions | Mean Q | Compute seconds / episode |
|---|---:|---:|---:|---:|---:|---:|
| Nominal | 16 | 12 | 10 | 1 | 3.649 | 0.715 |
| Nominal | 16 | 20 | 14 | 2 | 3.008 | 0.697 |
| Nominal | 64 | 20 | 17 | 2 | 2.468 | 0.738 |
| Nominal | 64 | 32 | 16 | 1 | 2.373 | 0.965 |
| Nominal | 128 | 20 | 15 | 2 | 2.865 | 0.801 |
| Nominal | 128 | 32 | 14 | 2 | 3.028 | 1.150 |
| ICODE | 16 | 12 | 9 | 1 | 3.862 | 1.916 |
| ICODE | 16 | 20 | 14 | 1 | 2.781 | 2.550 |
| ICODE | 64 | 20 | 15 | 2 | 2.856 | 2.935 |
| ICODE | 64 | 32 | 16 | 1 | 2.344 | 4.213 |
| ICODE | 128 | 20 | 16 | 2 | 2.665 | 3.563 |
| ICODE | 128 | 32 | 17 | 2 | 2.438 | 5.207 |

Q = duration / 30 + 2 * final_distance / 4.5 + 3 * timeout + 10 * collision.
Lower is better; compute does not enter Q. Higher success is not proof of lower
collision risk. Ranking follows the previously frozen success-first rule, then
collision count and Q, selecting ICODE K128/H32 as the inactive-axis reference.
This rule and the one-seed screen do not establish a population optimum.

ICODE changes task outcomes but costs substantially more. Moderate-speed large
block, offset wall, mixed, and sparse slalom contexts remain difficult under
nearly every budget; this motivates investigating shared controller tuning if
SAC cannot recover task performance. No failed context is removed from reporting.

## Actual SAC round

Four arms: joint K/H, H only, K only, joint K/H with masked dynamics context.
Each receives 40,000 physical cycles (complete final episode), common ICODE,
common scenario order and separate exploration RNG. Replay stores unrounded
continuous actions; execution maps to integer K/H and holds five cycles.
Full details, seeds and training reward are frozen in the protocol.

The initial queue failed before training because `queue.py` shadowed Python's
standard library and broke Torch import. Renamed to `run_stages.py`; all four
tests then passed. Failure log is retained under `queue_190142/00.log`; no
training results were produced by the failed launch. Corrected launch uses
commit `530243b`, with per-stage source snapshots and resumable replay/optimizer.

The first joint K/H arm completed 200 episodes, 40,017 physical cycles and 8,052
SAC decisions. Training behavior (stochastic policy, changing contexts, not an
evaluation comparison): 106/200 successes and 12 collisions. First40 episodes:
27 successes, 2 collisions, Q2.42, compute3.99s/episode; last40:23 successes,
2 collisions, Q2.88, compute3.17s/episode. Rounded training summaries describe
learning diagnostics only; differing episode contexts invalidate a simple causal
first-versus-last interpretation. H frequently approaches its lower limit while
K remains variable. Four checkpoints and full replay were saved.

H-only completed 215 episodes,40,227 cycles,8,110 decisions:130 successes and
16 collisions during stochastic training. First40:23 successes/4 collisions,
Q3.23,compute3.83s;last40:24 successes/3 collisions,Q2.88,compute3.79s. Again,
these are nonpaired training diagnostics. Its horizon migrated toward both
upper and lower limits at different learning stages, motivating explicit
checkpoint evaluation rather than assuming the final network is best.

K-only completed227episodes,40,127cycles,8,099decisions:155successes and
15collisions during stochastic training. First40:27successes/4collisions,
Q2.69,compute4.78s;last40:28successes/3collisions,Q2.40,compute5.25s. These
are not matched evaluation samples and cannot establish K-only superiority.

Masked joint completed216episodes,40,022cycles,8,054decisions:134successes and
12collisions. First40:22successes/2collisions,Q2.99,compute4.16s;last40:
28successes/3collisions,Q2.42,compute3.22s. These remain training diagnostics.

All four arms completed858training episodes,160,393physical cycles and
32,315SAC decisions. Development validation completed1,152episodes:16saved
checkpoints and8fixed model/budget combinations,24contexts,2paired seeds.
No final test seed has been used. No learned-method advantage or publication
readiness is claimed at this stage.

Complete validation: best fixed ICODE K128/H32 has36/48successes,2collisions,
Q1.966; selected joint full checkpoint30,193cycles has30/48successes,4collisions,
Q2.861. K-only final checkpoint has34/48successes,3collisions,Q2.307. Joint
allocation is not supported by this initial round. Leave-one-environment-seed-out
ICODE context selection gives4.25%Qheadroom and1.18sless computation per episode,
but this is a post-hoc development diagnostic with only two seeds and known
memory-related timing limitations. See the generated `icode_sac_compute_results.md`
and `icode_sac_development_notes.md` for full results and limitations.

Common MPPI tuning is now running. Subsequent13kgICODEdomain-matched training,
observable command-history input and60k-cycle matched SAC arms remain prospective.

## Evidence paths

- Protocol: `docs/protocols/icode_sac_compute.md`.
- Raw screen: `research_artifacts/icode_sac_compute_2026-09-08/model_screen/`.
- Descriptive summaries and PNG/PDF figures: `research_artifacts/icode_sac_compute_2026-09-08/diagnostics/`.
- Audit: `diagnostics/audit.json`, passed on all2298completed episodes,
  430896cycles and32315continuous action blocks. Scope is source
  snapshot hashes, summary arithmetic, budget legality and measured/physical
  readiness accounting; independent physical replay is a separate later check.
- No final test seed 9090301–9090304 has been used.
