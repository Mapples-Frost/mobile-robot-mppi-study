# Gate 1c preregistration: learned covariance inside hybrid ICODE-MPPI

Date frozen: 2026-07-18

The frozen SAC sequence proposal failed Gate 1b against an equal-compute
hybrid-no-RL control. That negative result is retained; confirmation seeds are
not opened. Gate 1c changes the RL action, not the research direction: the
already independently confirmed LinUCB policy selects an exploration
covariance, while ICODE predicts every rollout and the new hybrid optimizer
keeps shifted/conventional anchors.

The learned contextual covariance occupies exactly 30% of K=100 candidates.
The comparator uses the strongest globally fixed covariance in the same 30%
source. The remaining 70%, elite update, two iterations, ICODE checkpoint,
costs, safety chain and physical plant are identical. Thus a difference cannot
be attributed merely to the new optimizer.

Development uses two held-out route geometries, two physics domains and seeds
22410301--22410303. The independent unit is a complete episode. The primary
Gate requires zero success/collision regression, cross-track RMSE noninferiority
within 2 mm, and a strictly negative hierarchical 95% interval for time to
goal. Planner compute and jerk are reported. No claim about SAC, terminal value,
dynamic obstacles, or formal stability is permitted from this Gate.
