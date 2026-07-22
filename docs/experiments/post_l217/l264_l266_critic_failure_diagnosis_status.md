# L264–L266 Critic failure diagnosis status

Status: complete, read-only diagnosis; no training or controller change.

## Frozen evidence

- Checkpoint: L262 step 6000, SHA256
  `48e662fd728c0d1d62b1ff79681e5cdd924b120d9411c8df6e062a814b4d42a7`.
- Replay: 6000 transitions, six L260 training scenes.
- L263 counterfactual library: 26 accepted states, 702 candidate actions;
  the one rejected reset remains preserved.
- Final Hairpin, S-Chicane and Infinity artifacts were not used.

## Result

The frozen decision is
`in_support_ranking_failure_with_sparse_recovery_coverage`.

1. L264 passed. The float32 NumPy Bellman target matches the tensor captured
   from the actual `SACAgent.update` path exactly. Static checks also confirm
   the configured gamma, entropy sign, elementwise twin minimum, termination
   semantics and target soft update. A float64 re-evaluation differs by at
   most `1.2804813e-6`, which is the expected rounding/operation-order scale
   and is retained as a secondary diagnostic rather than treated as an
   implementation mismatch.
2. L265 does not support action extrapolation as the primary cause. Even the
   in-support actions have mean Spearman `-0.1250`; therefore the Critic's
   ranking failure is already present where replay has local action support.
3. L266 finds only five explicit positive-recovery transitions out of 6000
   (`0.0833%`) and only two complete off-path-to-corridor recovery chains.
   Those two chains cover only S-bend and curvature-ramp; four of six scenes
   have no complete recovery chain.

## Causal interpretation

The replay contains many off-path states (2145/6000) but almost no transitions
that demonstrate how a corrective action returns the robot to the corridor.
Consequently, the Critic receives abundant failure-state data but insufficient
action-conditional recovery evidence, and cannot learn a reliable recovery vs
fast-forward ranking. The Actor then optimizes against that wrong ranking.

This diagnosis does not establish an Actor-gradient, reward, or 69D
observation defect. It also does not claim that the core RL+ICODE+MPPI method
is invalid.

## Unique next recommendation

Before any further SAC training, preregister a targeted recovery-data
collection protocol that balances complete recovery events by scene and
off-path severity. Do not modify the reward, Actor/Traditional fusion,
observation, MPPI cost, or safety chain in the same intervention.

Per the protocol, implementation and training must wait for human review.
