"""One-step engineering-only migration smoke, not formal timing evidence."""
import sys,pathlib,json,time,platform
ROOT=pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/bohn2021_reproduction'))
from conservative_canonical_reset import make_env
import numpy as np
out=[]
for task,h in [('vehicle',25),('pendulum',30)]:
 start=time.perf_counter();env=make_env(task,2609269901,aligned=True,scaled_obs=True)
 obs=env.reset();begin=time.perf_counter();result=env.step(np.array([float(h)]));elapsed=time.perf_counter()-begin
 assert np.all(np.isfinite(result[0])) and np.isfinite(result[1])
 out.append(dict(task=task,horizon=h,observation=np.asarray(obs).tolist(),next_observation=np.asarray(result[0]).tolist(),reward=float(result[1]),done=bool(result[2]),engineering_step_seconds=elapsed,construction_reset_step_seconds=time.perf_counter()-start))
 if hasattr(env,'close'):env.close()
p=ROOT/'docs/bohn2021_takeover/server_smoke.json';p.write_text(json.dumps(dict(passed=True,scope='engineering portability only, no learned terminal, no efficacy or speed claim',test_accessed=False,python=platform.python_version(),episodes=out),indent=2));print(p.read_text())
