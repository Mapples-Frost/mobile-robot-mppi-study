# Irregular spiral + three dynamic MuJoCo scene v1

## Role

This is the primary complex-map candidate. The preserved multi-gate S map is a
separate second map. This scene changes topology rather than decorating the
same layout.

## Geometry

An irregular decagonal boundary encloses four nested, non-concentric polygonal
rings. Each ring has one deliberately omitted wall segment, and successive
openings alternate sides. The route therefore enters from the outer boundary,
travels almost a full circuit between the outer and middle rings, reverses its
winding direction between the middle and inner rings, circles again between
the inner and core rings, and reaches a goal inside the core.

The arena uses 42 static wall segments. Segment endpoints, lengths, headings,
thicknesses, heights and colors vary; there is no rectangular room grid. The
frozen 29-point route is 36.743 m long. Offline inflation by the 0.25 m robot
footprint retains at least 0.08 m clearance, and an independent 0.05 m occupancy
grid confirms start-to-core connectivity.

## Dynamic conflicts

Three asymmetric multi-box carriers patrol along the three ring openings. Each
motion segment intersects the frozen route and remains at least 0.04 m from all
static geometry under every corner combination of endpoint jitter. Period,
phase and endpoint uncertainty are sampled once per episode and hidden from the
controller. The controller receives causal LaserScan observations only.

The carriers have periods of 9.5, 10.0 and 8.8 s. A conservative 0.53 m
traversal width at the 0.70 m/s action limit plus a 0.30 s actuator allowance
requires a 1.057 s free window; every worst-randomized motion retains more than
that amount.

## Use

This contract freezes map topology before controller development. Controller
changes must use paired seeds and may not change ring openings, dynamic motion
programs or route geometry in the same iteration. This is a development scene,
not authorization for formal server execution.
