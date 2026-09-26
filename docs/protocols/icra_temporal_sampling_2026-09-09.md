# Temporal sampling intervention — 64 episodes

The48episode shared cost/recovery factorial is complete and raw-audited. No
variant reliably resolves compound navigation: successes range0–2/6. This
rejects an easy universal fix by the tested influence radius/recovery prefix.
All safety configurations matched; negative results remain in the record.

New hypothesis: independent per-time perturbations fail to generate sustained
turning sequences in a constrained passage. Test a change in proposal temporal
structure, not another budget or geometry search. Use the same original .7m
soft influence and zero recovery prefix, physical parameters, collision costs,
guard, action bounds, controller timing, and fixed K/H within each model.

Four samplers:
- iid: exactly the original sampling code;
- coarse: Gaussian perturbations constant in blocks of5 steps (0.5s);
- mixture: half iid, half block-constant perturbations;
- guard_mixture: original iid until three consecutive previous safety feedback
  events reduce forward command; then the same half-mixture. This uses past
  observable safety decisions, not simulator geometry or future outcomes.

All use the same per-coordinate marginal noise scale, clipping and untouched
mean candidate at index0. Rollout counts remain unchanged: nominal K64/H40 and
ICODE K16/H40. Mixing different temporal covariance is an experimental weighted
sampling optimizer, not claimed to preserve an exact path-integral measure;
the existing shared importance correction is disabled for all variants. These
are non-learned prototypes and not reproductions of published algorithms.

Four layouts: open, alternating_gates levels0/1, gate_then_turn level0. Fast
.65cap and two fresh seeds9990101/9990102. 4x2x2x4=64 episodes. Eight methods in
random order within paired environment blocks; measured execution is serial.
All failures and all methods retained; geometry selection is development based
on previous failures. No checkpoint selection, training or final-test claim.

Primary question: does temporal structure improve navigation at matched sample
count, and does feedback-triggering outperform always-on correlation? Record
success, collisions, Q, compute, guard fraction and trajectories. Include the
open control to expose gratuitous performance loss. If all correlated modes
fail, do not scale training; inspect whether subgoal/geometry representation
places the task outside the current controller's capabilities. Any gain needs
new-seed verification and literature positioning before being a paper claim.
