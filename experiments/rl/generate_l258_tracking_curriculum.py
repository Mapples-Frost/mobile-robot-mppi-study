"""Generate leakage-safe procedural path configs for L258."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path


FAMILIES = ("s_curve", "hairpin_like", "double_loop", "curvature_ramp", "corridor")
PHYSICS_ROLES = ("seen", "unseen")
FORBIDDEN = (
    "mujoco_l247",
    "mujoco_l248",
    "mujoco_l249",
    "mujoco_l250",
    "mujoco_l251",
    "mujoco_l252",
    "mujoco_l253",
    "mujoco_l254",
    "mujoco_l255",
    "mujoco_l256",
    "hairpin_final",
    "s_chicane",
    "infinity",
    "sealed",
)


def _rotate_scale(points: list[tuple[float, float]], rng: random.Random):
    angle = rng.uniform(-math.pi, math.pi)
    scale = rng.uniform(0.72, 1.18)
    ox, oy = rng.uniform(1.0, 2.0), rng.uniform(1.0, 2.0)
    ca, sa = math.cos(angle), math.sin(angle)
    rotated = [(scale * (ca * x - sa * y), scale * (sa * x + ca * y)) for x, y in points]
    min_x = min(x for x, _ in rotated)
    max_x = max(x for x, _ in rotated)
    min_y = min(y for _, y in rotated)
    max_y = max(y for _, y in rotated)
    fit = min(5.2 / max(max_x - min_x, 1e-6), 5.2 / max(max_y - min_y, 1e-6), 1.0)
    center_x = (min_x + max_x) * 0.5
    center_y = (min_y + max_y) * 0.5
    return [
        [round(3.25 + fit * (x - center_x), 5), round(3.25 + fit * (y - center_y), 5)]
        for x, y in rotated
    ]


def _family_points(family: str, rng: random.Random) -> list[tuple[float, float]]:
    amp = rng.uniform(0.65, 1.15)
    if family == "s_curve":
        return [(i * 0.65, amp * math.sin(i * 0.62)) for i in range(11)]
    if family == "hairpin_like":
        return [(0.0, 0.0), (2.4, 0.0), (2.4, 0.8), (0.35, 1.05), (0.35, 2.1), (2.7, 2.35)]
    if family == "double_loop":
        return [(i * 0.48, amp * math.sin(i * 0.9)) for i in range(15)]
    if family == "curvature_ramp":
        return [(i * 0.58, amp * (i / 10.0) ** 2 * 2.0) for i in range(11)]
    if family == "corridor":
        return [(0.0, 0.0), (1.2, 0.0), (1.8, 0.7), (2.8, 0.7), (3.5, 1.35), (4.6, 1.35)]
    raise ValueError(f"unknown family: {family}")


def generate(output_dir: Path, *, seed: int, routes_per_family: int = 2) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    routes = []
    for family in FAMILIES:
        for replica in range(routes_per_family):
            points = _rotate_scale(_family_points(family, rng), rng)
            role = PHYSICS_ROLES[(len(routes) + seed) % len(PHYSICS_ROLES)]
            name = f"l258_generated_{family}_{seed}_{replica}"
            path = output_dir / f"{name}.yaml"
            lines = [
                "include: ../../mujoco_l218_serpentine_polyline.yaml",
                "",
                "# l258_generated: procedural route; no held-out route data used",
                "experiment:",
                f"  name: {name}",
                "  max_steps: 1400",
                f"  physics_domain_role: {role}",
                "task:",
                "  type: polyline",
                "  points:",
            ]
            lines.extend(f"    - [{x}, {y}]" for x, y in points)
            lines.extend(["scene:", f"  name: {name}", ""])
            path.write_text("\n".join(lines), encoding="utf-8")
            routes.append({"family": family, "replica": replica, "role": role, "path": str(path)})
    manifest = {
        "protocol": "L258",
        "generator": "generate_l258_tracking_curriculum.py",
        "generator_seed": seed,
        "routes_per_family": routes_per_family,
        "families": list(FAMILIES),
        "physics_roles": list(PHYSICS_ROLES),
        "routes": routes,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--routes-per-family", type=int, default=2)
    args = parser.parse_args()
    generate(args.output_dir, seed=args.seed, routes_per_family=args.routes_per_family)


if __name__ == "__main__":
    main()
