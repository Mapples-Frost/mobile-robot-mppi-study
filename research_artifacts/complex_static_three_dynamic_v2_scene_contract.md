# Complex static + three dynamic MuJoCo scene v2

## Purpose

This scene is a development environment for complex closed-loop simulation. It
tests global route following through static clutter while three independently
randomized moving obstacles repeatedly cross the only useful route. It is not a
formal comparison registry and it does not authorize outcome claims.

## Static geometry

The 12.0 m by 8.0 m arena contains four alternating composite gates, five
primary crates, eight asymmetric inner islands and six scattered entry/exit
blocks. Each former straight gate wall is split into four offset, rotated,
unequal-height segments. The 39 static collision bodies therefore form jagged
non-parallel boundaries rather than a regular grid. Every room requires local
path-shape decisions rather than merely following the central polyline. The
gates retain a genuine S-shaped route. The frozen polyline is 25.403 m long and
has at least 0.08 m offline clearance after inflating static obstacles by the
0.25 m robot footprint.

The task corridor half-width is 0.85 m. This gives the local controller room to
depart visibly from the reference line and pass a moving obstacle, while the
static walls still prevent arbitrary shortcuts.

## Dynamic conflicts

Three directionally oriented multi-part moving carriers use independent
ping-pong motion. Each carrier combines a primary chassis with two offset,
rotated boxes, producing distinct arrow, L and asymmetric T silhouettes. Their
conservative planar envelope radius is 0.38 m:

1. a vertical crossing through the first upper gate;
2. a diagonal crossing through the central lower gate;
3. a vertical crossing through the final lower gate.

Every motion segment intersects the frozen route geometrically. Periods are
12.0, 10.5 and 11.5 s, making the three obstacles 43--52% faster than v1. At
reset, each obstacle independently receives phase
jitter, period scaling and two-dimensional endpoint jitter. These sampled
plant parameters and obstacle identities are not exposed to the controller;
only causal simulated LaserScan observations are available.

The shortest worst-randomized free gap is at least 2.017 s, enough for a 1.0 m
crossing at the 0.6 m/s action limit plus a 0.35 s actuator margin. This avoids
turning physical impossibility into an algorithm failure.

## Visual contract

The scene uses a dark blue checkerboard floor, deep navy arena walls, alternating
blue-gray and purple-gray gate walls, forest-green primary crates, cyan/blue
rotated low barriers and terracotta secondary crates. Boundary, gate and island
heights are deliberately layered so room structure remains legible from the
top-down camera. The dynamic obstacles remain high-saturation orange, yellow
and magenta so their identities remain visually distinct without exposing
identity to the controller. No scene obstacle uses a cylinder primitive; the
robot wheels remain cylinders because they are part of the vehicle model.
Reduced exposure, a soft fill light and a full-arena camera make depth, wall
height and route structure easier to inspect. Visual irregularity comes from
collision geometry itself rather than non-colliding decoration.

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
