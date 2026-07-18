# L28 Gate/RL prior inference profiling results

Date: 2026-07-15

Status: **development gate failed; behavior equivalence passed; cross-K confirmation not run**

## 1. Question

L26 showed that the gated RL sampling prior could improve difficult-scene behavior, but its additional inference cost prevented it from satisfying the pre-registered compute gate. L27 removed inference entirely when the outer gate coefficient was exactly zero. L28 asks a narrower engineering question:

> When the RL prior is active, can redundant critic diagnostics and a repeated actor evaluation be removed without changing any planning decision, while reducing enough end-to-end planner time to close the compute gap?

This is a runtime optimization experiment, not a new control-method comparison. Its design and decision thresholds were fixed in `docs/rl/37_l28_inference_profile_prereg_2026-07-15.md` before the formal run.

## 2. Conditions

| Condition | Selected target critics | Unselected online-critic diagnostics | Reuse actor base action |
|---|---:|---:|---:|
| `reference_full_diagnostics` | yes | yes | no |
| `selected_critic_only` | yes | no | no |
| `selected_critic_base_reuse` | yes | no | yes |

All conditions retain the L27 zero-alpha fast path. The implementation defaults preserve the reference behavior; the optimizations are activated only through explicit configuration flags.

## 3. Experimental design and data integrity

- MPPI samples: `K = 100`.
- Scenes: four clean/control scenes and three blocking scenes.
- SAC checkpoint seeds: `20260721`, `20260722`, `20260723`.
- Development episode seeds: `20282001`--`20282010`.
- Total episodes: `3 conditions x 3 checkpoints x 10 seeds x 4 clean scenes` plus `3 conditions x 3 checkpoints x 10 seeds x 3 blocking scenes = 360`.
- Observed steps: `79,734`.
- Paired behavior steps per candidate: `26,578`.
- Duplicate episode keys: `0`.
- Protected L25--L28 seeds used: `0`.
- Bootstrap: two-stage checkpoint/episode resampling, `20,000` replicates, fixed seed `20260782`.

The formal jobs initially outlived the shell orchestration timeout, but all three processes completed normally and wrote the expected 120 episodes per checkpoint. The merged audit verifies the complete expected key set; no formal job was rerun.

## 4. Exact behavior equivalence

Both optimized implementations are step-for-step identical to the full-diagnostic reference on every pre-registered behavior field.

| Check | Selected critic only | Selected critic + base reuse |
|---|---:|---:|
| Steps compared | 26,578 | 26,578 |
| Executed `v` maximum absolute difference | 0 | 0 |
| Executed `omega` maximum absolute difference | 0 | 0 |
| Goal-distance maximum absolute difference | 0 | 0 |
| Outer gate alpha maximum absolute difference | 0 | 0 |
| Correction gate alpha maximum absolute difference | 0 | 0 |
| Selected consensus LCB maximum absolute difference | 0 | 0 |
| Selected target Q-value maximum absolute difference | 0 | 0 |
| Step behavior mismatches | 0 | 0 |
| Episode success mismatches | 0 | 0 |
| Episode collision mismatches | 0 | 0 |

This establishes that the removed online-critic evaluations were diagnostic-only for the selected target-critic gate, and that reusing the already-computed deterministic actor action did not alter the chosen control sequence in this implementation.

## 5. Component profile

The following values are means over blocking-scene steps for which the RL prior was active.

| Condition | Active prior | Actor | Advantage critics | Decoder | MPPI batch rollout | MPPI cost | Planner total |
|---|---:|---:|---:|---:|---:|---:|---:|
| Reference | 2.573 ms | 0.529 ms | 0.883 ms | 0.613 ms | 2.733 ms | 0.473 ms | 7.832 ms |
| Selected critic only | 2.258 ms | 0.527 ms | 0.588 ms | 0.603 ms | 2.725 ms | 0.474 ms | 7.592 ms |
| Selected critic + base reuse | 2.190 ms | 0.527 ms | 0.539 ms | 0.593 ms | 2.700 ms | 0.470 ms | 7.513 ms |

Removing the unselected diagnostics reduces the measured advantage-critic component by about 33%. The combined implementation reduces it by about 39%. Actor and decoder inference then become the largest remaining active-prior components, so critic pruning alone cannot remove most of the prior overhead.

## 6. Pre-registered timing decisions

| Candidate | Active-prior reduction | Required | Blocking planner reduction | 95% nested-bootstrap CI | Required | Result |
|---|---:|---:|---:|---:|---:|---|
| Selected critic only | 12.26% | >=15% | 3.07% | [0.53%, 5.34%] | positive CI | fail prior threshold |
| Selected critic + base reuse | 14.89% | >=20% | 4.07% | [1.94%, 6.05%] | >=5% and positive CI | fail both magnitude thresholds |

The speedup is positive and statistically stable under the pre-registered hierarchical resampling: both confidence intervals exclude zero. It is nevertheless smaller than the practical thresholds fixed before looking at the formal data. L28 therefore receives the decision `development_fail`.

## 7. Scientific interpretation

L28 supports a limited claim:

> Selected-critic inference and actor-base reuse are behavior-preserving optimizations that produce a reproducible, modest speedup.

It does **not** support the stronger claim that these changes solve the L26 compute problem. A statistically non-zero speedup is not automatically a practically sufficient speedup. The pre-registered thresholds deliberately preserve that distinction.

The result does not invalidate the gated RL-prior research direction. It says that the present exact diagnostic-pruning strategy is insufficient by itself. The behavior result from L26 and the zero-alpha efficiency result from L27 remain separate evidence; L28 must not be used to retroactively change either conclusion.

## 8. Gate consequence

Because the development gate failed:

1. no cross-K confirmation was run;
2. no sealed L28 seeds were opened;
3. no additional optimization condition was added after observing the results;
4. the L28 thresholds were not weakened;
5. the reference implementation remains the default-compatible behavior.

Any follow-up runtime optimization, such as decoder caching or inference-context changes, requires a new pre-registration and a new development seed block. It must not be appended to L28 post hoc.

## 9. Artifacts

- Formal merged results: `results/research_platform/rl/l28_inference_profile_development_multiseed_20260715_v1/`
- Decision record: `development_gate.json`
- Episode-level data: `episodes.csv` and `episode_profiles.csv`
- Step-level behavior and component data: `profile_steps.csv`
- Paired timings: `paired_timing.csv`
- Per-checkpoint timing: `checkpoint_timing.csv`
- Component table: `component_summary.csv`
- Publication figure: `fig_l28_inference_profile.pdf` and `fig_l28_inference_profile.png`
- Formal configuration: `configs/rl/sac_mppi_inference_profile_l28.yaml`
- Runner: `experiments/rl/run_inference_profile_ablation.py`
- Merger and gate evaluator: `experiments/rl/summarize_inference_profile_ablation.py`
- Figure generator: `experiments/rl/analyze_inference_profile_results.py`

## 10. Reproduction commands

Each checkpoint was run independently, then merged:

```bash
.venv/bin/python experiments/rl/run_inference_profile_ablation.py \
  --config configs/rl/sac_mppi_inference_profile_l28.yaml \
  --training-seed 20260721 \
  --output-dir results/research_platform/rl/l28_inference_profile_seed20260721_dev20282001_10_20260715_v1
```

Repeat with checkpoint seeds `20260722` and `20260723`, then run:

```bash
.venv/bin/python experiments/rl/summarize_inference_profile_ablation.py \
  --input-dir results/research_platform/rl/l28_inference_profile_seed20260721_dev20282001_10_20260715_v1 \
  --input-dir results/research_platform/rl/l28_inference_profile_seed20260722_dev20282001_10_20260715_v1 \
  --input-dir results/research_platform/rl/l28_inference_profile_seed20260723_dev20282001_10_20260715_v1 \
  --output-dir results/research_platform/rl/l28_inference_profile_development_multiseed_20260715_v1

.venv/bin/python experiments/rl/analyze_inference_profile_results.py \
  --input-dir results/research_platform/rl/l28_inference_profile_development_multiseed_20260715_v1
```
