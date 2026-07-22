# L270 Observation Aliasing Diagnosis Protocol

## Question and stopping rule

L269 found delayed recovery value, but its preregistered Stage-1 Gate narrowly
failed: only 55.6% of fixed L268 states favored recovery by H=40 versus the
frozen 60% threshold. L270 therefore asks whether the frozen 69D observation
maps states requiring materially different recovery steering to nearly the same
representation. L270 is diagnostic only. It cannot initialize or update the
formal Actor, Critic, normalizer, reward, ICODE, MPPI, HSS, Actor/Traditional
fusion, maps, or safety chain.

## Frozen inputs and exclusions

- Windows-native execution under `D:\Projects\mobile-robot-mppi-study` only.
- Source checkpoint: L262 seed 20262611 step 6000, SHA256
  `48e662fd728c0d1d62b1ff79681e5cdd924b120d9411c8df6e062a814b4d42a7`.
- L263 accepted fixed states and H40 counterfactual grid returns are hash-pinned.
- L268 accepted recovery chains are hash-pinned. Their manifest-defined
  train/validation/test split is used without reassignment.
- L269 negative summary is hash-pinned and must report `horizon_gate_fail`.
- L258 and final Hairpin, S-Chicane, and Infinity artifacts are forbidden.

## Part A: fixed-state nearest-neighbor conflicts

For every accepted L263 state, the oracle action is the H40 `actor_follow` grid
action with maximum realized MuJoCo return. Ties are resolved before seeing any
aliasing statistic by lower action L2 norm and then lexical action ID.

Distances use the checkpoint-normalized 69D observation. For each state, the
three nearest other states are retained. A pair is a steering conflict when
either both oracle angular actions have magnitude at least 0.25 and opposite
sign, or their normalized angular-action difference is at least 0.75. The
primary conflict statistic is evaluated on pairs at or below the frozen 25th
percentile of all retained neighbor distances. Within-scene and cross-scene
statistics are reported separately. State-neighbor pairs are descriptive, not
independent experimental replicates.

## Part B: grouped privileged-information probes

The independent unit is an L268 recovery chain. Each accepted chain contributes
at most 12 deterministic, evenly spaced steps after step two, so long chains do
not dominate. All feature arms use exactly the same eligible rows and teacher
actions. The target is the recorded normalized recovery action, with angular
MAE and steering-sign accuracy as primary responses.

The manifest split remains frozen: 72 train, 18 validation, and 18 test chains,
balanced across the six training scenes. No step from one chain may appear in
more than one split. Preprocessing is fitted on train rows only.

Five feature arms are compared with identical fixed model families:

1. `current_69d`: the frozen observation only;
2. `far_preview`: 69D plus body-frame route points at 2.4, 3.2, and 4.8 m;
3. `three_frame_history`: the current and preceding two 69D observations;
4. `progress_segment`: 69D plus normalized route progress, remaining distance,
   and active segment index (the segment index is privileged diagnostic data);
5. `all_privileged`: all additions above.

Far preview is computed from the configured public path and recorded robot pose;
it does not query obstacles or future outcomes. Truth pose is used only to
reconstruct this candidate observable and is never supplied directly. A fixed
Ridge probe measures linear decodability. The primary nonlinear probe is a
fixed Random Forest with three preregistered model seeds; no hyperparameter
selection is allowed. Test results are read only after every arm and seed has
completed validation evaluation.

The recovery dataset stores `observation[t]` and `action[t]` before the MuJoCo
step, whereas audit row `t` stores truth after that step. Candidate features for
retained step `t` therefore use audit row `t-1`; the frozen minimum retained
step of two guarantees a valid predecessor. Any audit-step ordering drift is a
fail-closed error.

## Frozen Gate and interpretation

Observation aliasing is supported only if all of the following hold for at
least one privileged arm relative to `current_69d`:

- close-pair oracle steering-conflict fraction is at least 0.15;
- median test angular-action MAE improves by at least 15%;
- median test steering-sign accuracy improves by at least 0.10;
- validation angular-action MAE improves by at least 10%;
- all three model-seed MAE directions improve;
- at least four of six test scenes improve;
- no test scene angular MAE regresses by more than 10%.

If multiple arms pass, the fixed simplicity order is far preview, history,
progress/segment, then all-privileged. Passing authorizes only preregistration of
a small observation intervention; it does not authorize Actor training. If the
nearest-neighbor conflict criterion passes but no privileged probe passes, L270
reports unresolved representation/model mismatch. If the conflict criterion
fails, observation aliasing is not supported and the next audit is independent
versus shared/scene-conditioned Critic representation. All failures are kept.
