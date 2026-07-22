# L262 Coverage-Gated Value Learning Engineering Screen

Status: preregistered development engineering screen; never formal paper
evidence.

## Frozen question

Was L261's path-adherence regression caused by updating the Critic and Actor
before replay contained all six training geometries?

## Single treatment

- Inherit the complete L261 reward, network, batch, action, map, initial-state,
  and value-objective configuration.
- Set `replay_require_all_scenes=true` and delay Actor updates until step 3,000,
  after the first deterministic six-map block.
- Run a second identical six-map block so learning occurs from scene-balanced
  replay.  Total budget is one Windows CUDA seed (`20262611`) x 6,000 steps.
- Initialize only from L257 seed20262333 step30000, SHA256
  `fc9166f5c3010156a7c9fad4cb9d155ac222447506ff2ebdfe6c2205250a1547`.
- Final Hairpin, S-Chicane, and Infinity maps remain forbidden.

## Fail-closed gate

At 6k all of the following must hold:

1. Complete checkpoint/replay/provenance and no exception, OOM, NaN, or Inf.
2. Every replay group has at least 500 records; no update is recorded before
   all six groups are present; Actor updates start no earlier than step 3,000.
3. Absolute mean Q <= 500, quantile spread <= 100, and Actor loss <= 1,000.
4. Zero validation collision regression; mean validated completion regression
   <= 0.01; mean CTE and goal-distance increases each <= 0.10.
5. No individual validation scene's CTE increases by more than 0.50 m.
6. At least one useful signal: completion +0.02, CTE improvement 0.05 m, or
   goal-distance improvement 0.10 m.

Failure stops expansion. Preserve raw data plus a short status only; do not
plot, write a detailed negative report, inspect final held-out maps, or tune
from intermediate validation.
