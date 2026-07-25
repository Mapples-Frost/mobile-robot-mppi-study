# Held-Out Predictor Qualification Preregistration

Status: frozen before held-out ID/OOD trajectories are generated  
Date: 2026-07-23  
Platform: native Windows only

## Objective

Evaluate the completely frozen `dual_975_999` Change-Aware IMM against:

1. deterministic CV;
2. Gaussian CV Kalman;
3. ordinary four-mode IMM.

No predictor parameter, NIS threshold, response inflation, V3-ID parameter, or
Gate threshold may change after held-out generation begins.

## Independent unit and pairing

The independent unit is a seed. For each seed, all four predictors share the
same truth and observation histories. Process, noise profile, forecast origin,
and horizon are repeated conditions within seed and are not counted as
independent replicates.

Primary paired statistics first average P2--P4 results within seed, then compare
predictors across seeds.

## Power and sample size

On the completed development confirmation, the seed-level paired P2--P4 NLL
difference had `d_z = 4.607`. To account for winner's curse, the planning effect
is only 35% of this estimate: `d_z = 1.613`.

For a two-sided paired test with family-adjusted alpha 0.025 and 90% power,
8 seeds are required. Sensitivity:

| Fraction of development effect | Planning d_z | Required seeds |
|---|---:|---:|
| 35% | 1.613 | 8 |
| 50% | 2.304 | 6 |
| 65% | 2.995 | 5 |

Each split uses 10 seeds, giving two seeds of reserve beyond the conservative
calculation.

## Held-out ID

- Seeds: 730200001--730200010.
- Frozen V3 process configuration.
- Four processes x two ID noise profiles x ten seeds = 80 trajectories.

## Held-out OOD

- Seeds: 730300001--730300010.
- Four processes x two OOD profiles x ten seeds = 80 trajectories.
- The V3 implementation and physical limits remain unchanged.

OOD shifts are registered before generation:

- cruise speeds move from 0.48--0.92 to 0.62--1.05 m/s;
- dwell times shorten from 0.35--1.35 to 0.20--0.80 s;
- hybrid intervention gaps shorten from 2.50--6.50 to 1.50--4.00 s;
- hybrid dropout count increases from 2--5 to 4--7;
- dropout durations increase from 0.40--0.90 to 0.60--1.20 s;
- `process_shift` uses process acceleration std 0.10 and observation std 0.075;
- `sensor_shift` uses process acceleration std 0.06 and observation std 0.125.

This tests faster decisions, denser changes, higher process disturbance, and
sensor degradation without introducing a new algorithm-visible process label.

## Statistical analysis

Primary endpoints:

1. Change-Aware minus ordinary-IMM P2--P4 NLL in held-out ID;
2. the same paired difference in held-out OOD.

For each split:

- aggregate P2--P4 and both profiles within seed;
- report mean, SD, median, paired standardized effect `d_z`;
- compute a 95% seed-cluster bootstrap CI with 10,000 resamples;
- compute the exact two-sided sign-flip permutation p-value over 10 seed
  differences;
- apply Holm correction across the two primary split tests.

ADE, FDE, coverage, area, event windows, observation status, and detection
metrics are preregistered secondary outcomes. No time-step-level p-value is
allowed.

## Frozen Predictor Qualification Gate

All of the following are required:

1. 80/80 ID and 80/80 OOD trajectories complete with unique keys;
2. all finite/PSD/probability/replay and no-leakage invariants pass;
3. ID P2--P4 Change-Aware NLL is lower than ordinary IMM, with bootstrap upper
   CI below zero and Holm-adjusted p at most 0.05;
4. OOD P2--P4 Change-Aware NLL is lower than ordinary IMM, with bootstrap upper
   CI below zero and Holm-adjusted p at most 0.05;
5. ID P2--P4 NLL ratio to ordinary IMM is at most 0.95;
6. OOD P2--P4 NLL ratio to ordinary IMM is at most 0.98;
7. ID P2--P4 NLL ratio to Gaussian CV is at most 0.90;
8. OOD P2--P4 NLL ratio to Gaussian CV is at most 0.98;
9. ID macro 90% coverage lies in [0.68, 0.95];
10. ID macro 95% coverage lies in [0.72, 0.98];
11. ID and OOD P2--P4 90% coverage are not lower than ordinary IMM;
12. ID and OOD overall ADE are at most 1.05 times ordinary IMM;
13. ID P1 ADE is at most 1.05 times ordinary IMM;
14. ID and OOD post-change 1--3 s NLL are lower than ordinary IMM;
15. ID P1 false NIS triggers do not exceed 3 per trajectory-minute;
16. no held-out or sealed outcome is used for tuning or exclusion.

Failure preserves all artifacts and stops before collision-probability work.
