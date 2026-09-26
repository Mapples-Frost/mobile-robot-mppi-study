# Pre-outcome engineering amendment v1a

The initial `profile/` run at commit405af5b failed during environment construction,
before any timing or task samples: existing safety_recovery_prefix_steps=8 rejects
H=5. Preserve that failed directory. Do not weaken the safety prefix.

For planner experiments replace v1's H grid5,10,20,40 with **8,10,20,40** and action
Hmin=8. Independent prediction-error probes retain H1,5,10,20,30,40 because they
do not instantiate a planning horizon. All gates and other treatments unchanged.
New output `profile_v1a/`. This is an engineering correction before outcomes,
not a relaxed scientific criterion. Manifests hash both protocols.
