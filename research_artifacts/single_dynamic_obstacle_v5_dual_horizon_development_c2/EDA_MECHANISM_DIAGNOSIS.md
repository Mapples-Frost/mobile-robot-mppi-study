# C2 mechanism diagnosis

## Integrity and gate

- C2 is repeated, outcome-informed development, not confirmatory evidence.
- All 16 paired jobs completed, stderr is empty, and all 91 manifest hashes independently match.
- Freeze commit: `5263ef2aae41a023df2a534bd644c57e1be225ad`.
- Bundle SHA-256: `ea872bd1ecd669af51a2fa69c378013f7b9d865b74ba482be060d10654494f00`.
- C2 failed only the two-conversion gate. Every safety, clearance, progress, stuck, zero-speed, direction-switch, oscillation, and 600-rollout check passed.

## Quantitative diagnosis

On `740200006`, increasing urgency reserve from 2 to 3 raised active guarded cycles from 32 to 38 and reduced final goal distance from 0.314809 m to 0.307774 m. The episode remained collision-free, aligned, and capped at 0.35 m/s through the final cycle, but missed the 0.300 m tolerance by 0.007774 m.

The one-step reserve amendment therefore moved the endpoint in the predicted direction by 0.007036 m but not far enough. A reserve of 4 would extrapolate to approximately the tolerance boundary and is too sensitive to deterministic actuation variation. C3 pre-freezes urgency reserve 5, providing a two-step robustness margin while remaining only 0.5 s of the 40 s episode. Authority reserve remains 0 and every safety gate retains priority.

## Final-attempt rule

C3 is the third and final attempt in the dual-horizon family. It changes no code and only urgency reserve 3 to 5. If C3 fails any frozen gate, this family closes and another global review is mandatory. If it passes, the candidate freezes immediately and proceeds only to new, disjoint, unopened held-out qualification.
