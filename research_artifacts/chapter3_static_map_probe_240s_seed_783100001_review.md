# Chapter 3 240 s probe — retained failure on superseded geometry

- Seed: `783100001`
- Episodes: 1 development episode
- Result: collision at step 1895 (189.5 s)
- Robot pose: `(1.748, 2.399)`
- Nearest dynamic centre distance: 0.396 m
- Nearest static footprint clearance: 0.796 m
- Forecasts at collision: 2
- Maximum predicted collision probability at collision: 1.0
- Tracker association and forecast were valid
- Executed response before impact: reverse at -0.35 m/s with a turning command

Diagnosis: this is not a static/dynamic classification failure. Dynamic C was
correctly tracked and assigned maximum collision risk, but it approached faster
than the fixed reverse escape could separate. The same map audit also found
that four static boxes violated the frozen 0.04 m dynamic-loop jitter clearance
contract. This geometry revision is therefore superseded before further
development.

The replacement map moves only those four static boxes, raises the worst-case
dynamic/static separation to 0.102 m, and keeps the static route feasible. The
controller intervention replaces front-obstacle reverse-only escape with a
committed full-speed lateral forward arc while retaining reverse for severe
rear-half-plane danger.

The raw result, trajectory, resolved configuration, provenance, stdout and
stderr are retained without rerunning or overwriting.
