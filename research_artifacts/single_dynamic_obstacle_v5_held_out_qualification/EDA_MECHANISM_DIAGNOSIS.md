# V5 held-out qualification diagnosis

## Integrity and evidential status

- This was the one-shot, unopened held-out engineering qualification frozen at commit `481ebc322c56a60a10a1dd996e07820fc9a6a300`.
- All 32 paired blocks and 64 episode jobs completed; there were no structural failures and stderr is empty.
- No controller outcome was opened until all 64 jobs were complete.
- An independent recomputation matched all 355 manifest file hashes and the canonical bundle hash.
- Sealed registry SHA-256: `3a606928a5621128a411c91eae075b9ffc1a808124fcf04c7b839a57a0b07eea`.
- Result bundle SHA-256: `a69c434c1bd3dd40ec9b42b1fa567545b3e46f99cb33d1e358e7287295c0ddd0`.
- Scope remains engineering qualification, not formal effect estimation.

## Frozen-gate result

The candidate failed the qualification contract. Safe success was identical at 19/32 for both arms: there were no safe-noncompletion conversions, no lost V4 successes, and a paired net gain of 0 rather than the required +2. ID and OOD net gains were each 0.

Safety did not hold. V4 had 10/32 collisions and the candidate had 11/32. OOD seed `750200043` was the sole new paired collision: V4 remained collision-free with minimum clearance 0.233008 m, while V5 collided at step 225 with minimum clearance -0.003261 m. The candidate therefore failed both the no-new-paired-collision and collision-count-not-worse gates.

Mean minimum-clearance change (-0.003795 m), mean conflict-window q05-clearance change (-0.000630 m), zero-speed-under-risk reduction (+2.87%), stuck-step reduction (+5.55%), and three-phase oscillation change (0) all stayed within their frozen bounds. Total direction switches increased by one and failed its zero-increase gate. The increase came from ID seed `750100038` (+2 switches), partially offset by OOD seed `750200043` (-1).

## Mechanism diagnosis

The deadline supervisor activated in only eight of 32 candidate episodes and produced no held-out success conversion, so the development-panel effect did not generalize. The new-collision trace exposes a more serious eligibility problem. On OOD seed `750200043`, the supervisor activated at t=21.5 s with reported temporal closing rate 0.281 m/s, TTC 1.972 s, temporal risk alpha 1.0, and scan clearance 0.243 m. It applied forward motion before the episode entered repeated dynamic-escape control and collided one second later.

This trace is inconsistent with the intended contract that renewed temporal hazard, closing, TTC, risk, and clearance vetoes have absolute priority over deadline mobility. It does not by itself prove that the two supervisor cycles were the sole cause of the later collision, but the deterministic paired divergence and unsafe eligibility are sufficient to reject the candidate. The early t=2.3 s activation on the same episode also shows that the `prior dynamic escape` latch can authorize deadline behavior long before a post-conflict terminal-recovery context.

The global conclusion is that a deadline computed only from remaining goal distance and episode time is not a reliable post-conflict recovery selector. A historical escape latch plus a short clear streak does not establish that the current hazard has ended. The development panel overrepresented late safe non-completions where this pressure helped; the controller-independent held-out cohort revealed sparse benefit and a hazardous false release.

## Frozen decision

Per the preregistered stop rule, single-obstacle v5 tuning is closed. There will be no threshold amendment, seed replacement, C4/C5 attempt, or formal v5 server run. V4 remains the defensible single-obstacle result and the C3/held-out contrast is retained as negative evidence about deadline-only recovery.

Work now moves to the three-dynamic-obstacle study. Any reuse of recovery logic there must begin from a global state-machine review and must require current, causal hazard-clearance evidence; the rejected C3 deadline supervisor is not promoted as the multi-obstacle default.
