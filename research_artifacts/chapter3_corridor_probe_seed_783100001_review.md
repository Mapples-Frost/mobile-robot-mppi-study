# Chapter 3 hard-corridor probe — retained failure

- Seed: `783100001`
- Episodes: 1 development episode, 1200 control steps
- Intervention: 0.50 m reference half-width, 0.25 m planning
  footprint, hard path-boundary candidate filtering
- Result: no collision and no boundary violation, but timeout
- Maximum path progress: 0.399 m (2.45%)
- Trajectory length: 1.396 m
- Stuck steps: 1103 / 1200
- Minimum boundary margin: 0.129 m
- Mean candidate feasible fraction: 0.394
- Minimum candidate feasible fraction: 0.00167

Decision: reject the hard route-tube mechanism. It prevents the previously
observed static-corner collision but destroys mobility near the start. The
replacement mechanism uses exact static-map geometry in the rollout cost and
does not impose a route corridor.

The raw result, trajectory, resolved configuration, provenance, stdout and
stderr are retained without rerunning or overwriting.
