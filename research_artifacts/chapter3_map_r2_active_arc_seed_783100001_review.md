# Chapter 3 map R2 fixed active-arc probe — retained failure

- Seed: `783100001`
- Result: dynamic collision at step 1026 (102.6 s)
- Robot pose: `(1.437, 2.534)`
- Nearest dynamic centre distance: 0.453 m
- Nearest static footprint clearance: 0.988 m
- Forecasts at collision: 2
- Maximum predicted collision probability: 1.0
- Temporal TTC at collision: 0.300 s
- Tracker association and forecast: valid

The six-turn/eight-straight fixed escape transaction activated at 101.6 s and
turned right at 0.75 m/s. Dynamic C continued down-left across that arc. The
fixed bearing-side heuristic therefore entered the obstacle's future path even
though tracking and collision probability were correct.

Decision: reject the fixed active-arc mechanism after this safety regression.
The final bounded mechanism attempt moves forecast-relative escape candidate
generation into standard MPPI, evaluates every candidate with the same frozen
probabilistic risk model, and allows the final safety layer to execute only the
selected vetted candidate. If that attempt does not remove the collision, this
local mechanism family stops.

The raw result, trajectory, resolved configuration, provenance, stdout and
stderr are retained without rerunning or overwriting.
