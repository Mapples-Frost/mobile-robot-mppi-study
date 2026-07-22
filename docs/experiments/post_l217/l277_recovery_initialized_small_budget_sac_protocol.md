# L277 Recovery-Initialized Small-Budget SAC Probe

## Question

Does the L276 recovery-sequence Actor initialization survive a short, fully
reinforcement-learning continuation and improve independent L261 validation
tracking without changing the method, reward, observation, planner, or safety
chain?

L276 passed before this protocol was frozen. L277 is a development Gate, not
paper evidence and not authorization to inspect final Hairpin, S-Chicane, or
Infinity outcomes.

## Paired initialization

Three L277 runs are paired one-to-one with the three L276 3,000-update Actors.
Each run imports the complete numerical SAC state (Actor, both Critics, both
target Critics, alpha, and 69D normalizer) but intentionally starts fresh
optimizers, replay, counters, and RNG state. Thus the only demonstration-derived
information entering L277 is encoded in the L276 Actor; no recovery trajectory,
teacher action, oracle return, privileged field, or held-out transition enters
SAC replay.

The full-agent initialization mode and every source checkpoint SHA256 are
frozen before L277 runs. Ordinary resume is forbidden because it would import
old replay and counters. Actor-only initialization is also forbidden because it
would discard the paired L262 Critics used by L276.

## Frozen training

- three paired seeds: `20263311/20263312/20263313`;
- exactly 6,000 Windows CUDA environment steps per seed;
- checkpoints and validation at step 0, 3,000, and 6,000;
- the six L261 training scenes and three L261 validation scenes only;
- deterministic six-scene blocking inherited from L262;
- fresh replay must cover all six scenes before learning is eligible;
- Critic updates may begin after step 500;
- Actor updates remain frozen through step 3,000 and occur only in the second
  complete six-scene block;
- reward, 69D observation, SAC architecture/objective, ICODE, MPPI, HSS,
  Actor/Traditional fusion, and the safety chain remain unchanged.

No seed, scene, checkpoint, threshold, or initialization may be replaced after
outcomes are viewed. L258, sealed seeds, and final Hairpin/S-Chicane/Infinity
maps are forbidden.

## Frozen validation Gate

Only the 6,000-step checkpoint is the treatment endpoint; 3,000 is a
pre-Actor diagnostic milestone. Relative to the paired step-zero L276 actor,
L277 passes only if all checks hold:

1. all three runs are complete at exactly 6,000 steps, contain replay and
   3k/6k checkpoints, resolve to CUDA, and remain finite;
2. the validation grid contains exactly three scenes at steps 0/3k/6k for each
   seed and uses only the preregistered validation seed bases;
3. no seed gains a validation collision and no seed loses a validation
   success;
4. no seed's mean completion regresses by more than `0.02`, and the median
   paired completion change is nonnegative;
5. the median paired CTE improves by at least `0.05 m` **or** median paired goal
   distance improves by at least `0.10 m`;
6. at least two of three seeds improve either mean CTE or mean goal distance;
7. at least two of the three validation scenes improve paired mean CTE;
8. provenance proves the exact one-to-one L276 full-agent initialization and
   fresh-replay/reset-counter semantics.

Failure stops before expansion and retains raw data plus a short status only.
Pass authorizes preregistration of a larger independent-seed validation study;
it does not authorize final-map tuning or checkpoint selection.

