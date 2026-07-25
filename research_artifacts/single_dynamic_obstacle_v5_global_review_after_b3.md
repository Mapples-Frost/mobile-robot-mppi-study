# Global review after deadline B3

## Trigger

The single-horizon online deadline-feasibility family reached the predeclared stop threshold of three unsuccessful local attempts:

1. **B1** combined deadline supervision with inherited A5 recovery overrides. It improved success and collision outcomes but failed oscillation and zero-speed-risk gates; attribution showed most oscillation occurred without deadline activation.
2. **B2** isolated the deadline supervisor from A5 overrides. It passed every safety and mechanism-regression gate but produced only one of two required challenge conversions.
3. **B3** removed the two-step reserve. It again passed every other gate but produced only one conversion; activation became later because the reserve parameter also determines urgency.

All three attempts and their failures are retained with independent hashes, commits, and checkpoint tags. No held-out qualification has been opened.

## What is established

- The B1 oscillation regression belongs to the inherited recovery override package, not to deadline supervision.
- Isolated deadline supervision is safe on the repeated development panel: B2 and B3 introduced no collision, lost no V4 success, preserved clearance, and added no direction switch or three-phase oscillation.
- The isolated mechanism has a real task signal: it consistently converts `740300251`, reduces aggregate final distance, and reduces stuck/zero-speed exposure.
- The remaining miss is not generic immobility. On `740200006`, the robot is aligned, clear, moving, and only centimeters outside tolerance at the deadline.
- A single `reserve_steps` variable is structurally overloaded. It controls both the urgency denominator and final authority availability, which creates an unavoidable timing tradeoff under parameter-only tuning.

## Rejected directions

- Do not weaken risk thresholds, TTC gates, clearance margins, or collision boundaries.
- Do not increase the 0.35 m/s safety speed cap.
- Do not reintroduce the A5 recovery overrides that caused B1 oscillation.
- Do not change the 0.30 m goal tolerance or 400-step deadline.
- Do not count the repeated panel as held-out or inferential evidence.
- Do not continue B-series reserve, margin, or threshold sweeps.

## New mechanism direction

The admissible next family is **dual-horizon deadline supervision**:

- an **urgency horizon** retains the two-step reserve when computing required speed, preserving B2's earlier activation;
- an independent **authority horizon** uses zero reserve solely to determine whether a clear, aligned, post-conflict terminal cycle may still receive bounded forward authority;
- required speed remains capped by the existing 0.35 m/s dynamic-escape maximum;
- the existing online risk, TTC, closing, hard-violation, front-clearance, heading, and renewed-hazard guards remain authoritative;
- steering remains unchanged and no rollout candidate is added.

This separates two causal roles that B3 demonstrated cannot be represented by one parameter. It is a new mechanism family (`dual_horizon_deadline_supervisor`), not B4. Its first attempt is C1. The B1-B3 gate remains unchanged so the structural change cannot pass by relaxed criteria.

## Advancement rule

C1 may run only after unit tests prove:

1. frozen V4 has zero configuration changes;
2. the candidate differs only in declared deadline fields;
3. urgency and authority horizons are independently computed;
4. the supervisor stays inactive on renewed hazard;
5. the final clear cycle may remain eligible while urgency time is exhausted;
6. 600 rollouts, horizon 36, dt 0.1 s, 400 steps, and 0.30 m tolerance are unchanged.

If C1 passes the repeated development gate, the next step is a newly frozen 32-seed paired held-out engineering qualification (16 ID, 16 OOD), selected only by the controller-independent conflict certificate from seed ranges reserved away from all development and future formal registry ranges. Formal server execution still requires explicit user confirmation.
