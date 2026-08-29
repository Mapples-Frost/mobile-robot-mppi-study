#!/usr/bin/env python3
"""BC Policy Real-Car Test Script for Scout Mini"""
import sys, json, math, time, random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from bc_policy import BCPolicy, TRACK_WIDTH, COLLISION_RADIUS, V_MAX, OMEGA_MAX, DT, GOAL_THRESHOLD
import numpy as np

SCENES = {
    "sparse": {
        "start_state": (0.0, 0.0, 0.0), "goal": (6.0, 0.0),
        "obstacles": [(2.2,0.9,0.55),(2.2,-0.9,0.55),(4.2,0.9,0.55),(4.2,-0.9,0.55)],
        "bounds": {"x_min":-0.5,"x_max":6.5,"y_min":-1.8,"y_max":1.8},
    },
    "dense": {
        "start_state": (0.0, 0.0, 0.0), "goal": (6.0, 0.0),
        "obstacles": [(1.0,0.55,0.33),(2.1,-0.35,0.36),(3.2,0.60,0.36),(4.3,-0.50,0.36),(5.1,0.40,0.30),(2.7,1.35,0.28),(3.8,-1.35,0.28)],
        "bounds": {"x_min":-0.6,"x_max":6.6,"y_min":-1.5,"y_max":2.2},
    },
    "narrow": {
        "start_state": (0.0, 0.0, 0.0), "goal": (6.0, 0.0),
        "obstacles": [(2.8,0.9,0.65),(2.8,-0.9,0.65),(4.3,0.9,0.65),(4.3,-0.9,0.65)],
        "bounds": {"x_min":-0.5,"x_max":6.5,"y_min":-1.5,"y_max":1.5},
    },
}

def wrap_angle(t):
    while t > math.pi: t -= 2*math.pi
    while t < -math.pi: t += 2*math.pi
    return t

def check_collision(state, bounds, obstacles):
    x, y, _ = state
    if x < bounds['x_min']+COLLISION_RADIUS or x > bounds['x_max']-COLLISION_RADIUS: return True
    if y < bounds['y_min']+COLLISION_RADIUS or y > bounds['y_max']-COLLISION_RADIUS: return True
    for ox, oy, orr in obstacles:
        if math.hypot(x-ox, y-oy) < COLLISION_RADIUS + orr: return True
    return False

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', default='sparse', choices=['sparse','dense','narrow'])
    parser.add_argument('--episodes', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model', default=None)
    args = parser.parse_args()

    model_path = args.model or str(Path('/home/pi/rlmppi/model/sparse_multiscale_bc_policy.pth').resolve())
    print(f'Model: {model_path}')
    policy = BCPolicy(model_path)
    scene = SCENES[args.scene]
    start = scene['start_state']; goal = scene['goal']
    obstacles = scene['obstacles']; bounds = scene['bounds']

    np.random.seed(args.seed); random.seed(args.seed)
    successes = collisions = timeouts = 0

    for ep in range(args.episodes):
        cs = [start[0]+random.gauss(0,0.001), start[1]+random.gauss(0,0.001),
              wrap_angle(start[2]+random.gauss(0,0.001*5))]
        cs = [max(bounds['x_min']+0.1,min(bounds['x_max']-0.1,cs[0])),
              max(bounds['y_min']+0.1,min(bounds['y_max']-0.1,cs[1])), wrap_angle(cs[2])]

        collided = False
        for step in range(400):
            vl, vr = policy.infer(tuple(cs), goal)
            v = (vl+vr)/2; omega = (vl-vr)/TRACK_WIDTH
            x,y,theta = cs
            nx = x+v*math.cos(theta)*DT
            ny = y+v*math.sin(theta)*DT
            ntheta = wrap_angle(theta+omega*DT)
            if check_collision((nx,ny,ntheta), bounds, obstacles):
                collided = True; break
            cs = [nx, ny, ntheta]
            if math.hypot(cs[0]-goal[0], cs[1]-goal[1]) < GOAL_THRESHOLD:
                successes += 1; break
        else:
            timeouts += 1
        if collided: collisions += 1
        status = "SUCCESS" if (successes > 0 and not collided) else ("COLLISION" if collided else "TIMEOUT")
        print(f"  Ep{ep+1:2d}: {status} | steps={step+1}")

    total = successes + collisions + timeouts
    print(f"\nResults: {successes}/{total} ({successes/total*100:.0f}%) success")
    print(f"Avg infer time: {policy.get_average_infer_time()*1000:.2f}ms")

if __name__ == '__main__':
    main()
