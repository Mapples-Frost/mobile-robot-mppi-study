# Paper baselines and progressive scenario design

User steering: prioritize baselines from papers, include a related-work horizon
adaptation method, and start from a scenario all methods can solve before
changing one factor at a time. Supersedes any plan to expand ad-hoc selector grids.

## Baselines to implement and qualify

1. ICODE-MPPI, Song et al., arXiv2605.03260. Retain local differential-drive
   adaptation, but document equation/code differences and smoothing. Existing
   scores are adaptation evidence, not exact original-vehicle reproduction.
2. Bohn et al., Reinforcement Learning of the Prediction Horizon in Model
   Predictive Control, IFAC-PapersOnLine54(6),314–320 (2021),
   DOI10.1016/j.ifacol.2021.08.563, arXiv2102.11122. This is the selected related
   work H-adaptation baseline. Read full algorithm, state/reward/action mapping,
   training and evaluation conventions; reproduce a small original task first
   where feasible, then document MPPI adaptation. Train fresh H-only policies
   for the current tracking task; do not relabel old KH or point-goal actors.
   A simple manually chosen H rule is NOT this baseline.
3. Retain fixed-H/fixed-K nominal and ICODE as controls, with competitive budgets
   selected on development data. Turn/heading/periodic belong to ablations,
   not the primary literature-baseline table.

The two cited papers are primary sources:
https://arxiv.org/html/2605.03260v1
https://arxiv.org/html/2102.11122v1
Related ADP arXiv2510.05330 adapts dynamics discretization; do not confuse that
with selecting the number of H prediction steps at fixed dt.

## Common solvable anchor

Start with a low-speed straight-to-rounded-bend route, radius1.2m, a2m approach
straight, a90-degree arc, and a1m exit straight. Speed cap.35m/s, original
plant and ideal sensing, generous fixed-step time allowance60s. All methods see
the same known reference and have the same physical action/safety constraints.
This is a NEW planned anchor, not yet validated. Previous late_turn_low success
only motivates it. Require >=19/20 successful paired development episodes and
zero collisions for each primary method before proceeding. Use seeds10710001–20
only for feasibility; failure triggers debugging/common feasibility correction,
never selecting a weak baseline checkpoint. Declare any changes before new tests.

## Progressive axes, varied separately from anchor

| Axis | Planned levels | Question |
|---|---|---|
| Speed cap m/s | .35,.45,.55,.65 | Does anticipatory dynamics allocation help as turn demand increases? |
| Bend radius m | 1.6,1.2,.8,.6 | Where does curvature expose model need or model-switch failure? |
| Approach straight m | 0,1,2,3 | When does avoiding complex-model use on easy segments save work? |
| Sustained turning | onebend,two alternating bends,four alternating bends | Does the gain disappear when complex dynamics are continuously needed? |

Use identical exit lengths and report path length/duration explicitly. Never
pool different task lengths into total-compute savings without per-cycle and
distance-normalized costs. If converting to time-budget tests, separately
validate a simulator that executes previous commands during compute delay;
current fixed-step measurements cannot support deadline-robustness conclusions.

Freeze exact geometry generation and selected training protocol after reading
H-baseline details, before collecting comparative test outcomes. Planned test
seeds10720001–10 are held out from training, threshold choice and anchor screening.
Three independently trained H policies and three residual model blocks; disclose
crossed/shared dependence. Equal development search budgets and information.
Report every level, including unfavorable continuous-bend results. Do not stop
the ladder at the most favorable point or train on held-out failures.

Primary outcomes: paired tracking RMSE, measured planner+selector compute,
success/collision, achieved speed, model invocation and selected H. Show curves
and raw paired values to locate the onset and loss of an advantage. No assertion
that advantages will monotonically increase. Main current task is baseline
qualification and anchor feasibility, not immediately launching another large grid.
