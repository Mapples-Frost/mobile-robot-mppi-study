"""Run already-declared queue tail early; run.py locks prevent duplicate training."""
from concurrent.futures import ThreadPoolExecutor
from optimized_suite import train

jobs=[('pendulum',0,h) for h in range(50,4,-5)]+[('vehicle',0,h) for h in [50,45,40,35]]
with ThreadPoolExecutor(max_workers=6) as pool:
    results=list(pool.map(train,jobs))
assert all(r['exit_code']==0 for r in results),results
