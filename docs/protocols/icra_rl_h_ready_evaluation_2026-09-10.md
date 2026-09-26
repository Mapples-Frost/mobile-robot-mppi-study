# RL-H matched-readiness fresh-scene evaluation protocol

This evaluation begins only after all six `rl_h_ready_training_2026-09-10`
checkpoints and the independent training-data audit pass. It is a new,
serial, fresh-scene evaluation. It does not alter the earlier 210-episode
joint-SAC evaluation or reuse its scene seeds.

The ten contexts are two existing routes (`single_turn`, `reverse_turns`) by
five new environment seeds `13620001` through `13620005`. Every method sees
the same contexts and the same standard mass, sensing, safety, physical
readiness delay, and control interval. Method order is randomized once using
seed `13601001` and all episodes run serially. Evaluation uses deterministic
actors and final checkpoints only.

The learned methods are all three original residual-predictor joint SAC and
H-only checkpoints (policy seeds 12901001, 13101001, 13101002), all three
matched-readiness nominal RL-H checkpoints, and all three matched-readiness
residual RL-H checkpoints (policy seeds 13501001, 13501002, 13501003). The
RL-H policies retain their own trained terminal-value network. For each
RL-H checkpoint, fixed H=16 and fixed H=32 controls reuse that same terminal
value network and predictor mode; they isolate learned horizon selection from
the value augmentation. Residual K128/H32 and nominal K128/H32 remain strong
static controls. This gives 26 methods and 260 serial episodes.

Original joint/H-only SAC uses its recorded 50D actor mapping and hold=5.
RL-H uses its recorded 19D actor, K=100, H in [8,50], and one decision per
physical cycle. It has a different network, reward, decision frequency, and
terminal-value augmentation from original SAC. It is therefore a
literature-inspired local MPPI adaptation, not an exact author reproduction
or a same-architecture ablation.

Primary descriptive outcomes are success, collision, route tracking RMSE,
and summed measured readiness compute time. Results are reported by route,
policy initialization, and shared scene seed. The five scene seeds are not
multiplied into fifteen independent scenes because each learned policy sees
the same five scenes. All fixed-H value controls remain attached to their
policy initialization. The post-run audit must replay raw original-SAC and
RL-H actor outputs, verify action mappings, source/checkpoint hashes,
physical timing, and recompute tracking metrics before any conclusion.
