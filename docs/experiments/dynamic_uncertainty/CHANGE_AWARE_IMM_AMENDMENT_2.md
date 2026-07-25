# Change-Aware IMM Amendment 2

Status: frozen after Confirmation 1 failed and before Pilot 3 outcomes are inspected  
Date: 2026-07-23

## Confirmation 1 result

Amendment 1 selected `persistent_975` on fresh Pilot 2 seeds and then evaluated
56 untouched confirmation units. Twelve of thirteen frozen Gate checks passed.
The sole failure was:

```text
true-change recall = 0.193548 < 0.20
```

The failed result is not rounded upward. Confirmation 1 artifacts remain at
`research_artifacts/change_aware_imm_v3_development_amendment1/`.

Other confirmation results, retained for diagnosis:

- P2--P4 NLL ratio to ordinary IMM: 0.7910;
- post-change 0--1 s NLL ratio: 0.9313;
- post-change 1--3 s NLL ratio: 0.7929;
- P1 ADE ratio: 1.0021;
- P1 false triggers: 0.7792 per minute;
- area ratio: 1.0264;
- median matched delay: 0.975 s.

## Mechanistic diagnosis

The persistent-only detector requires two high NIS values in three available
observations. An abrupt change can create one extreme innovation, after which
the Kalman update absorbs much of the residual and the second exceedance never
arrives. This failure mechanism is testable without changing V3 or using event
labels online.

## Amendment

Add an OR branch:

```text
trigger = persistent moderate NIS OR one extremely large NIS
```

The single-exceedance threshold is still computed from the same current
innovation and has no access to future or true event labels.

Frozen Pilot 3 candidates:

| Candidate | Persistent threshold/rule | Single threshold |
|---|---|---:|
| dual_975_999 | 7.377759, 2 of 3 | 13.815511 |
| dual_99_999 | 9.210340, 2 of 3 | 13.815511 |
| dual_975_9995 | 7.377759, 2 of 3 | 15.201805 |

The single thresholds are the 99.9% and 99.95% chi-square quantiles for two
position measurements.

Everything else remains frozen: V3, ordinary IMM, response inflation, dropout
guard, pilot selection rule, all feasibility limits, and all confirmation Gate
thresholds.

Pilot 3 uses fresh development seeds 730100054--730100056. Confirmation 2 uses
fresh seeds 730100057--730100063. No Pilot 1, Pilot 2, or Confirmation 1 unit is
reused.

