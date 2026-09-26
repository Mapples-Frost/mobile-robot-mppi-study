# L62 Sealed Residual-Structure Confirmation Preregistration

L62 is an exact replication of L61 on the ten seeds that were sealed before L61:
`21860791`--`21860800`. No model, checkpoint, plant, path, MPPI, safety, cost,
threshold or analysis change is permitted. The L61 three-way design and structure
gate are inherited verbatim. The only changes are the explicit confirmation flag,
seed list, randomized schedule seed, bootstrap seed and output directory.

Confirmation requires the full inherited gate: both learned residuals improve over
nominal without safety/task regression, and ICODE beats the parameter-matched MLP
overall and on the unseen path with positive hierarchical-bootstrap lower bounds.

Even a passing result remains scoped to one fixed high-dynamic MuJoCo plant using
clean ground-truth state. It is not cross-plant, odometry, obstacle, RL or real-robot
evidence.

