# L261 Offline Actor Value-Stability Engineering Screen

Status: preregistered engineering screen; never formal paper evidence.

## Frozen question

Does bounding invalid path progress and reward scale prevent the L260 critic
and Actor divergence without behavior cloning or changes to online ICODE,
MPPI, Actor/Traditional fusion, MPPI cost, or safety?

## Design

- Windows native only; one development seed `20262601`.
- Actor-only initialization from the frozen L257 selected checkpoint
  `seed20262333/step_000030000.pt` with SHA256
  `fc9166f5c3010156a7c9fad4cb9d155ac222447506ff2ebdfe6c2205250a1547`.
- Six qualified L260 training geometries and three independent L260 validation
  geometries. Final Hairpin, S-Chicane, and Infinity are forbidden.
- Budget: 3,000 steps. Each training geometry receives one deterministic
  500-step block. Validation occurs at 0, 1,500, and 3,000 steps.
- No new supervised learning or BC anchor.
- Quantile critic, network, batch size, action contract, and group-robust Actor
  objective remain unchanged.

## Engineering changes

- Synchronize reference progress to a mid-path reset before applying the live
  projection window.
- Limit live projection advance to 0.25 m per control step.
- Cap reward-bearing progress at 0.10 m per step and grant none outside the
  0.75 m route corridor.
- Cap cross-track error used by the reward at 1.50 m and scale all offline
  training reward terms by 0.10.
- Report corridor-validated completion as the primary completion metric and
  retain raw projection completion as a diagnostic only.

## Fail-closed gate

Do not continue to 10k/20k unless all conditions hold at 3k:

1. No exception, OOM, NaN, or Inf; replay/checkpoint/provenance complete.
2. Absolute mean Q <= 500, quantile spread <= 100, Actor loss <= 1,000.
3. Zero validation collision regression.
4. Mean validated completion is not lower than initialization by more than
   0.01; mean cross-track RMSE and goal distance each increase by no more than
   0.10.
5. At least one useful signal: validated completion improves by >= 0.02,
   cross-track RMSE improves by >= 0.05, or mean goal distance improves by
   >= 0.10.

If the gate fails, retain raw data and a short status only. Do not plot, write
a detailed negative report, expand budget, or inspect final held-out maps.
