# Residual-Dynamics Stage 1 Amendment 5 Preregistration

Status: **frozen before Amendment 5 closed-loop outcomes**  
Date: 2026-07-23  
Scope: nominal-noninterference residual safety shield  
RL status: **disabled**

## Single intervention

Amendment 5 retains the Amendment 4 structure mask, causal innovation gate and
stall fail-safe, then places the residual MPPI behind a concurrent nominal MPPI
baseline. Both controllers use identical MPPI configuration and random seeds.

The residual plan is replayed under nominal dynamics and accepted only when:

1. its residual-view probability is below the frozen hard threshold;
2. its nominal-view probability is below the frozen hard threshold;
3. nominal-view maximum probability does not exceed the nominal plan;
4. residual-view and nominal-view positions differ by no more than `0.10 m`,
   the existing frozen probabilistic safety margin;
5. nominal-view terminal distance is no more than `0.02 m` worse than the
   concurrent nominal plan, matching the existing Stage 1 completion margin.

Any failure returns the exact concurrent nominal plan. Both internal
controllers then observe the command actually executed by the safety layer.
No simulator truth or future obstacle state enters the decision.

This first implementation deliberately computes both complete planners so that
the fallback contract can be tested exactly. Compute is a registered secondary
gate; shortlist acceleration is authorized only after the safety screen.

## Development screen

- observed development seed `730100003`;
- three frozen checkpoint blocks plus one shared nominal;
- randomized schedule seed `730199994`;
- collision stops the screen;
- no sealed seed is authorized.

Required:

- zero collision increase in every residual block;
- median completion and clearance noninferiority;
- all three raw masked residuals retain positive H36 prediction improvement;
- reliability authority reaches at least 0.5 in every block;
- the shield accepts residual plans for at least 1% of steps in every block,
  preventing a nominal-only pass;
- forecast and artifact contracts pass;
- total planner p95 at most 150 ms.

Passing remains a development result because the obstacle seed is already
observed.

