# Complex static + three dynamic MuJoCo scene v1

## Purpose

This scene is a development environment for complex closed-loop simulation. It
tests global route following through static clutter while three independently
randomized moving obstacles repeatedly cross the only useful route. It is not a
formal comparison registry and it does not authorize outcome claims.

## Static geometry

The 10.0 m by 6.4 m arena contains four alternating wall gates and five
cylindrical clutter objects. The gates create a genuine S-shaped route rather
than a straight start-to-goal corridor. The frozen polyline is 22.661 m long
and has at least 0.08 m offline clearance after inflating static obstacles by
the 0.25 m robot footprint.

The task corridor half-width is 0.85 m. This gives the local controller room to
depart visibly from the reference line and pass a moving obstacle, while the
static walls still prevent arbitrary shortcuts.

## Dynamic conflicts

Three 0.25 m radius cylinders use independent ping-pong motion:

1. a vertical crossing through the first upper gate;
2. a diagonal crossing through the central lower gate;
3. a vertical crossing through the final lower gate.

Every motion segment intersects the frozen route geometrically. Periods are
25.0, 18.5 and 24.0 s. At reset, each obstacle independently receives phase
jitter, period scaling and two-dimensional endpoint jitter. These sampled
plant parameters and obstacle identities are not exposed to the controller;
only causal simulated LaserScan observations are available.

The shortest worst-randomized free gap is at least 2.683 s, enough for a 1.4 m
crossing at the 0.6 m/s action limit plus a 0.35 s actuator margin. This avoids
turning physical impossibility into an algorithm failure.

## Frozen evaluation fields

Future development runs should record, per episode:

- success and collision;
- time and steps to goal;
- path length and final goal distance;
- minimum static and dynamic clearance;
- static boundary violations;
- zero-speed exposure during active risk;
- stuck steps and direction switches;
- reverse commands and three-phase oscillations;
- conflict traversal time at each of the three gates;
- exact rollout count and planning-time health metrics.

The three conflict windows must be defined from the frozen route/motion
geometry and reused across arms for a paired seed.

## Development discipline

Scene geometry, motion uncertainty and controller changes must not be tuned in
the same iteration. First validate this scene visually and geometrically. Then
compare controller versions using paired seeds and common randomized obstacle
motions. Formal server execution requires a separate sealed registry and the
user's explicit confirmation.
