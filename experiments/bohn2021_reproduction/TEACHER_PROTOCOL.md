# Teacher-data horizon-value pilot

The user authorized one round of training with improved data and explicit reward
design. The unit is a complete independently generated scene. All prior banks
remain observed research data; this experiment uses new seeds 26091840--43.

## Reward and value meaning

The physical stage cost is the previously audited cart-pole energy/tracking/input
objective. Add 0.4905 to set upright equilibrium cost to zero, retain 0.003 H,
and divide the negative total by 0.6. On physical failure retain the existing
10 times remaining steps penalty and also charge the remaining 0.4905 offsets.
Consequently the complete undiscounted shaped return is exactly the negative
of (old total cost + 600*0.4905)/0.6, even for early failures. No reward clipping,
teacher-action bonus, extra reward for shortening H, or arbitrary failure dropping.
This preserves the original total-cost preference; it does not guarantee safety.

Discounted teacher targets use gamma 0.97. A full 600-step episode is finite:
no bootstrap after its end or physical failure. Both discounted shaped return
and undiscounted decomposed cost are reported; they are different objectives.

At a shared state, compare seven first-step horizons (5,10,20,25,30,40,50), then
follow the same frozen H5/H30 causal rule for 59 more steps. The branch is a fork
of the exact simulator/MPC/RNG state. The future reference after the anchor's
50-step visible preview is held constant for label simulation. This removes
privileged future information but introduces an explicit forecast approximation.
Add a discounted local Riccati physical tail and settled-H5 computation tail,
unless the branch terminates. The local physical tail is an approximation to
the remaining finite-horizon value, not an exact optimal value or soft-SAC Q.

Supervise costs relative to the H30 branch. Subtract the known first-step H-cost
difference for neural regression, and add it back at inference. No costs are
clipped. Train a 19-input, 64x64 ReLU critic with seven outputs, constraining
the H30 difference to zero. Choose the minimum predicted relative cost directly.
No separate stochastic actor or SAC fine-tuning is used in this controlled pilot.

## Data and training

- Eight training scenes, four labelled validation scenes, four new student-state
  aggregation scenes, twelve final holdout scenes; 600 steps each.
- Reference levels independently vary in magnitude 0.35--0.85 m; two reversal
  times are randomized. Initial position, velocity, angle and angular velocity
  have larger perturbations than in the preceding plateau screen.
- Twelve scheduled anchors per surviving scene cover initial recovery, plateaus,
  approach to each reference change, movement and recovery. All seven actions
  are labelled at each anchor. Early failures remain recorded.
- Three initialization seeds 0,1,2, each 3000 Adam updates (3e-4, batch64),
  SmoothL1 regression plus a 0.1 weighted ranking loss. Final checkpoints only.
- The predetermined first seed's initial student generates aggregation states;
  all seeds are then retrained from matching initialization on the same augmented
  dataset. This is one supervised dataset-aggregation round, not on-policy SAC.
- Features use current state, remaining episode fraction, and the same visible
  50-step reference: ten five-step mean changes, next visible change distance and
  magnitude. No future event calendar or privileged physical state is provided.
- Export to a NumPy inference model and verify actions against the trained model.

## Comparisons and reporting

Retain initial and augmented students for all seeds. Re-select fixed H using
validation coarse 1,5,...50 and integer refinement within +/-4 of its best value.
Evaluate every student, the frozen H5/H30 rule, selected fixed H, and H30 on all
twelve holdout scenes. Do not select students by holdout or training seed.

Report label MAE, decision regret, near-optimal action fraction, complete-episode
cost components, reward, tracking RMSE, H and failures. Include all branch and
prefix simulation cost in the data budget. At most two active MPC workers; fork
children run one at a time per worker. Save all trajectories and source hashes.

This is a supervised method on a new task distribution. It cannot isolate data
quality from architecture, reward representation, and training changes relative
to the old SAC experiments. Nor does it prove globally optimal control. Final
holdout uses new scene draws from this family; broader task transfer is untested.
