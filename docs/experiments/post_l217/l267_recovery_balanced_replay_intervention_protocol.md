# L267 Recovery-Balanced Replay Intervention Protocol

Status: preregistered development intervention. This is not sealed or formal
paper evidence. No L267 outcome has been inspected when this protocol is
frozen.

## Causal question

L263 showed that the frozen L262 Critic does not rank actions consistently with
short MuJoCo returns. L264--L266 excluded a Bellman-target implementation error
and found that the failure is already present for actions with local replay
support. The same replay contains 2145 off-path transitions but only five
explicit positive-recovery transitions and two complete recovery chains.

The strongest current explanation is therefore sparse complete-recovery
coverage. L267 tests the intervention claim:

> Holding the algorithm and total replay size fixed, does adding balanced,
> complete recovery trajectories repair the Critic's in-support action ranking?

Only a successful paired intervention may elevate this explanation from a
strong association to causal development evidence.

## Frozen components

The following remain byte- or configuration-equivalent to L262:

- reward and termination semantics;
- 69-dimensional student observation and fitted normalizer;
- 256x256 direct Actor;
- twin 25-quantile Critic and target update;
- group-robust Actor objective and scene-balanced replay sampling;
- Actor/Traditional fusion, Value-Consistent ICODE, role-aware reliability,
  HSS, MPPI cost, maps, LaserScan, scan guard and safety arbitration.

There is no BC loss and no privileged teacher field enters a student
observation or inference checkpoint. Final Hairpin, S-Chicane and Infinity
geometry, trajectories, seeds and outcomes are forbidden.

## Frozen initialization

All Critic-only arms start independently from:

`results/research_platform/rl/l262_coverage_gated_value_seed20262611_6k/checkpoints/step_000006000.pt`

SHA256:
`48e662fd728c0d1d62b1ff79681e5cdd924b120d9411c8df6e062a814b4d42a7`

The Actor, entropy temperature and normalizer are frozen throughout the
Critic-only phase. No arm is initialized from another L267 arm.

## Stage 1: complete recovery chains

### Independent chain unit and allocation

- Six frozen L261 training geometries only.
- Exactly 18 accepted chains per scene: 12 train, 3 validation, 3 test.
- Total accepted chains: 108 (72/18/18 by split).
- Split is assigned at the whole-chain level before collection. No transition
  from one chain may occur in more than one split.
- Within each scene, the schedule blocks on side (left/right), severity
  (mild/moderate/severe), heading error (small/large), and feasible curvature
  anchor. The seeded schedule is archived in the manifest.
- Collection plan seed: `20262967`; collection episode seeds begin at
  `20264000` and are disjoint from L234--L266 and sealed seeds.

Severity offsets are frozen at 1.0, 2.0 and 3.1 m. Heading-error magnitudes are
0.15 and 0.65 rad, with signs balanced by the schedule. Feasible progress
anchors are chosen from the interior 10--90% of the reference and recorded.

### Teacher and acceptance contract

The privileged offline polyline teacher uses ground-truth pose solely to
choose normalized direct-control actions. Its fixed parameters are inherited
from the qualified L243 teacher: lookahead 0.35 m, cruise speed 0.28 m/s and
yaw gain 1.8. The unchanged safety chain executes every proposed command.

A chain is accepted only when:

1. reset CTE is greater than 0.75 m;
2. CTE later becomes less than 0.25 m;
3. CTE stays below 0.25 m for 15 consecutive control steps;
4. progress after re-entry increases by at least 0.10 m;
5. no collision or boundary violation occurs;
6. all observations, actions, rewards and diagnostics are finite.

The maximum episode length is 240 steps. Each frozen stratum receives at most
eight deterministic attempts. Failure to fill any scene/split/severity quota
stops Stage 1; quotas are not weakened after viewing outcomes. All failed
attempts remain in the raw audit and are never silently deleted.

Five bounded perturbation profiles are assigned before collection: none,
normalized v +/-0.08, and normalized omega +/-0.10. Perturbed actions are
clipped to [-1, 1] and actually executed in MuJoCo; no counterfactual action is
stored with an unrelated next state.

Student replay shards contain only 69D observation, normalized action, reward,
next observation, terminated-only done, constraint cost, group and chain ID.
Privileged pose/path fields live only in a non-training audit sidecar.

## Stage 2: paired Critic-only intervention

### Arms and blocking

Critic seeds are `20262971`, `20262972`, `20262973`. Each seed is a block and
receives both arms from the identical L262 checkpoint. Arm order is generated
with seed `20262968` and archived before training.

- **Control:** the original 6000-transition L262 replay.
- **Recovery-balanced:** exactly 6000 transitions, 1000 per scene. Within each
  scene, 300 transitions are sampled only from complete train-split recovery
  chains and 700 are sampled from that scene's original non-recovery replay.

Both arms use scene-balanced minibatches, batch size and optimizer settings
from L262, and exactly 6000 Critic updates. Sampling is seeded and with
replacement when required; source-chain IDs and duplicate multiplicities are
reported. Actor weights, Actor optimizer, log alpha and alpha optimizer must
remain bitwise unchanged. Target critics retain the frozen tau update.

The true independent replicates for the development comparison are the three
paired training seeds, not individual transitions or actions.

### Frozen diagnostic sets

1. All 26 accepted L263 states and their 27 fixed actions, with H40
   actor-follow returns and frozen L265 support labels.
2. Validation/test recovery-chain start states. Teacher-recovery, frozen Actor
   and fast-forward candidate actions receive fixed H40 MuJoCo returns before
   any Critic result is read.
3. Results are stratified by scene, CTE severity and action support.

### Critic Gate

All stability conditions must hold: finite checkpoints and metrics, absolute
mean Q <= 500, mean quantile spread <= 100, and no Actor/alpha mutation.

The recovery-balanced arm passes only when all are true at 6000 updates:

1. aggregate in-support Spearman is >= 0.05;
2. paired median improvement over control is >= 0.20 and at least 2/3 seed
   blocks improve;
3. recovery-vs-forward pairwise accuracy is >= 0.60 and improves over control
   by >= 0.15 on at least one of the two frozen diagnostic sets, without a
   decrease greater than 0.05 on the other;
4. in-support Top-3 agreement improves over control by >= 0.10;
5. no individual training scene has a negative Spearman change below -0.10.

No p-value is claimed from three seeds. This is a fail-closed development
effect-size Gate.

## Stage 3: conditional small-budget SAC

Stage 3 is prohibited unless Stage 2 passes. If it passes, each treatment
Critic checkpoint initializes its matching Actor run. The Actor is then
unfrozen for exactly 6000 recovery-balanced SAC updates for seeds
`20262981`, `20262982`, `20262983`; no arm or seed initializes another.

The frozen L261 validation Switchback, Wave and Loop-exit maps compare the
unchanged initial Actor with the three post-intervention Actors. Report success,
completion, CTE, goal distance, off-corridor ratio, mean velocity,
omega--preview-curvature correlation, turn-sign accuracy and collision. Stage
3 is a development screen only and stops after the fixed budget regardless of
outcome.

## Stop and reporting rules

- Engineering progress may be monitored, but no intermediate ranking or
  validation outcome may change collection, sampling, updates or thresholds.
- Stage 2 failure stops all Actor training. Preserve raw artifacts and write a
  short Gate status only.
- Stage 2 success permits only the fixed Stage 3 budget; it does not permit a
  large-scale or sealed experiment.
- Preserve every failed chain, seed and arm. No seed filtering, deletion or
  outcome-based checkpoint selection is allowed.

