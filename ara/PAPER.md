# Research Artifact Manifest

- Working title: Cross-Layer Uncertainty-Gated Learned Dynamics and Policy-Guided MPPI
- Target venue: ICRA 2027
- Submission target: 2026-09-15
- Repository: `mobile-robot-mppi-study`
- Current evidence stage: Amendment 17 remains the frozen conditional environment baseline. Residual Stage 1 and six bounded repairs are retained negative results. Stage 2 task-aware conservative fine-tuning passed 3/3 offline blocks and the frozen three-seed development matrix completed 12/12 goal reaches with zero collisions, but did not establish speed superiority or checkpoint ordering. Stage 3 static-buffer CUDA Graph rollout plus parallel shield planning passed the current-machine development runtime gate with 12/12 goal reaches, zero collisions and maximum residual-controller P95 97.19 ms; the 2.81 ms margin is not a hard-real-time guarantee. Stage 4 rejected direct frozen L217 Actor/HSS integration, and Stage 5's safe-rejection probe failed its overall gate. Dynamic-task Actor adaptation added causal lidar deltas, bidirectional control, collision-exposure DAgger and per-action exploration. After the first expanded V5 Amendment 6 matrix failed, a bounded Pareto active-traversal repair removed new collisions but missed uncensored step gates. Amendment 2 then passed all 11 registered gates on 24 new arm-order-balanced single-obstacle development pairs: success improved 20/24 to 22/24, collisions fell 3/24 to 1/24, there were zero new Candidate collisions and zero lost Source successes, and outcome-aware efficiency improved 7741 to 7615. A fresh seed-730100255 real-time confirmation reached the goal without collision, exercised causal temporal escape for 17 cycles, used exactly 600 rollouts on all 300 decisions and achieved Candidate P95 83.54 ms. Formal paired tests remain underpowered (binary exact p=0.500), raw steps usually favor Source, and older matrix P95 reached 115.15 ms. The retained Candidate is qualified only for this bounded development scope; sealed seeds remain unopened and multi-obstacle work is paused.

## Layers

- `logic/`: falsifiable claims, problem framing and implementation heuristics.
- `trace/`: chronological research decisions, failed gates and session provenance.
- `evidence/`: pointers to immutable experiment artifacts and reports.
- `staging/`: observations not yet promoted to claims or decisions.
