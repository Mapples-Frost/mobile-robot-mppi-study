# L268 Reachability-Capped Recovery Replay Intervention Protocol

Status: preregistered development intervention. No L268 recovery chain,
Critic result, or Actor result has been generated or inspected at freeze time.

## Why L267 stopped

L267 correctly rejected local-normal pseudo-CTE states, but its formal Stage 1
then failed closed at the first S-bend severe/large-heading stratum. Its severe
target was the side's geometric maximum (5.70 m). Three candidates violated the
boundary on their first executed step and the remaining five failed the frozen
realized-CTE tolerance. Seven previously accepted chains and every failed
attempt remain preserved under the L267 raw directory. L267 is not resumed,
relabeled, or used for training.

This is a collection-feasibility failure, not a method-performance result. It
shows that geometric free-space maximum is not the same as a recoverable state
under a finite horizon and unchanged safety chain.

## Causal question and frozen method

L268 asks the same isolated question: holding algorithm and replay size fixed,
does balanced complete-recovery coverage repair the Critic's in-support action
ranking? Reward, termination, 69D observation/normalizer, 256x256 Actor, twin
25-quantile Critic, group-robust objective, scene-balanced replay, ICODE, MPPI,
Actor/Traditional fusion, maps, sensors and safety chain remain unchanged.
There is no BC loss. Final Hairpin, S-Chicane and Infinity are forbidden.

All arms independently initialize from L262 step 6000, SHA256
`48e662fd728c0d1d62b1ff79681e5cdd924b120d9411c8df6e062a814b4d42a7`.

## Stage 1: complete recovery chains

- Six unchanged L261 training geometries.
- Exactly 18 accepted chains per scene: 12 train, 3 validation, 3 test.
- Whole-chain splits; total 108 chains (72/18/18).
- Seeded blocking on globally projected side, map-side-relative severity,
  heading-error magnitude, curvature anchor and action perturbation.
- Plan seed `20262969`; episode seeds start at `20265000`.

Candidate states use a fixed 81x81 field lattice and global nearest-path
projection. The lattice is inset 0.35 m beyond the robot collision radius so a
reset is not accepted on a boundary tangent. On each scene/side, the target
range starts at CTE 0.80 m and ends at the smaller of (a) 0.03 m below the
geometry-only maximum and (b) the preregistered reachability cap 3.50 m. Mild,
moderate and severe are 0%, 50% and 100% through this range. Realized MuJoCo
reset CTE must match target within 0.08 m. Exact maxima and targets are archived.

The reachability cap is fixed before L268 collection from the unchanged teacher
speed and finite horizon, not tuned to success: 600 steps at 0.28 m/s provides
substantially more path length than 3.50 m after turning and the 15-step stable
hold. The 600-step maximum and 16 deterministic attempts per stratum are frozen.

The teacher remains the L243 ground-truth polyline teacher (lookahead 0.35 m,
cruise 0.28 m/s, yaw gain 1.8). The unchanged safety chain executes every
command. A chain is accepted only if it starts above CTE 0.75 m, reaches below
0.25 m, remains there 15 steps, gains at least 0.10 m progress after re-entry,
has no collision/boundary violation, and is finite. Perturbations remain none,
v +/-0.08 and omega +/-0.10, clipped and actually executed.

Student shards contain only 69D observation, action, reward, next observation,
done, constraint cost, group, chain ID and step. Privileged fields stay only in
audit sidecars. Any quota failure stops L268 and is retained.

## Stage 2: paired Critic-only Gate

Seeds `20263071/72/73` form paired blocks; arm order seed is `20263068`.
Control is the original 6000-transition L262 replay. Treatment is also exactly
6000 transitions: 1000 per scene, comprising 300 train-split complete-recovery
transitions plus 700 original non-recovery transitions. Recovery sampling is
scene-then-chain-balanced so each of the 12 train chains in a scene contributes
25 rows; replacement is allowed only within a chain that has fewer than 25
rows, and all multiplicities are archived. Both arms use identical
initialization, batch 256, scene-balanced sampling and exactly 6000 Critic
updates. Actor, Actor optimizer, log alpha and alpha optimizer remain bitwise
frozen.

Diagnostics are the frozen L263/L265 counterfactual set plus validation/test
recovery starts with H40 MuJoCo returns. The treatment Gate is unchanged from
L267: finite/stable Q, aggregate in-support Spearman >=0.05, paired median
Spearman improvement >=0.20 with at least 2/3 seeds improving,
recovery-vs-forward accuracy >=0.60 with >=0.15 improvement on at least one set
and no >0.05 decrease on the other, Top-3 improvement >=0.10, and no scene
Spearman change below -0.10. Individual transitions are not replicates.

## Stage 3: conditional small SAC

Stage 3 is prohibited unless every Critic Gate condition passes. If it passes,
the matching treatment Critic initializes three independent 6000-update SAC
runs (`20263081/82/83`). Evaluation uses only frozen L261 validation Wave,
Switchback and Loop-exit and reports success, completion, CTE, goal distance,
off-corridor ratio, speed, omega-curvature correlation, turn-sign accuracy and
collision. No large training, sealed test, seed filtering, outcome-based
selection, or failure deletion is permitted.
