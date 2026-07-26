# Chapter 1 static-A* geometry contract correction

## Correction to the preceding audit

The earlier `chapter1_static_reference_feasibility_audit.md` is preserved as a
failed diagnostic, but its conclusion that the scene was topologically
infeasible is withdrawn.

The audit reused `planning.static_astar._point_clearance`, which interpreted a
segment's configured `thickness` as a half-width. In the simulator,
`model_factory`, and the independent `scene_feasibility` auditor,
`thickness` is the **full** MuJoCo box width and the centreline extent is
`0.5 * thickness`. A* therefore inflated every segment wall by an extra half
thickness and falsely reported the spiral corridors as closed.

## Independent checks

Using the simulator-consistent geometry contract:

- the original, visually approved 29-point reference is retained unchanged;
- its length is `36.7434982326 m`;
- its sampled minimum footprint clearance is `0.0850402844 m`, above the frozen
  `0.08 m` margin;
- the original static obstacles and wall thicknesses are retained unchanged;
- all three dynamic carriers retain their route intersections;
- static-only A* finds a route from the nominal start to the goal;
- static-only A* also finds a route from the prior failed run's post-detour
  position `(-6.2786, -3.8150)` to the goal.

## Minimal repair

Only the static-A* segment-clearance implementation is changed to use
`0.5 * thickness`, matching the simulator and existing independent feasibility
auditor. A focused regression test fixes the numerical contract:

`0.50 m centreline distance - 0.10 m half-width - 0.25 m robot radius
= 0.15 m clearance`.

No map geometry, reference, dynamic trajectory, controller parameter, or paper
algorithm is changed. This is an implementation-consistency repair of the
existing soft static-reference infrastructure.
