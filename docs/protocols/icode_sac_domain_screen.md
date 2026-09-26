# ICODE training-domain transfer screen

Prospective2026-09-08, before any result of this screen. Source inspection found
that L57ICODE was trained on L56's13kg,weak-drive,low-friction plant, while the
current compute campaign uses9kg,stronger-drive,higher-friction physics. Both
use40msactuator delay. Neither is relabeled as universally matched/unmatched
reliability; these labels describe physical configurations and training provenance.

Screen all12families,both speeds,two plants,two prediction models,three budgets
(16,20),(64,24),(128,32),paired development seed9091301:288episodes. Use the same
common planner settings selected by the preceding shared-backbone budget check,
same geometry,safety,cost,100msphysical period,and readiness accounting. Randomize
all12treatments within each context with seed9091300. Use observable command
history for innovation; all treatments pay the same context overhead.

The13kgplant override is copied exactly from
`configs/research/mujoco_high_dynamic_fixed_plant_l56.yaml`:mass13,inertia
[.13,.19,.27],wheel_mass.38,torque_limit1.55,kp.38,ki1.20,integral_limit2.50,
delay.04,torque_slew25,wheel_friction[.62,.005,.0008],floor[.55,.0025,.00025].
Prediction nominal time constants stay.18/.12. No extra actuator state enters
policy/context. The common state/action representation and physical geometry
are unchanged. Sensors remain ideal simulated state/scan.

Before outcomes, designate13kgas the next observable-SAC training domain because
it is the ICODE pretraining plant. This is an in-domain study, not robustness to
unknown physics. The9kgresults are retained as transfer evidence. Every fixed,
single-axis,joint and masked comparator in the next training/evaluation round
uses the same13kgplant. Do not give only the proposed method a better model or
different physics. If ICODE still does not improve tasks, state it explicitly.

Within13kgICODE cells, choose the inactive-axis fixed pair among candidates with
collision count<=K64/H24; rank success descending,collisions ascending,Qascending,
compute ascending. Record the resolved setting before SAC starts. This is a
development rule, not a claim of collision noninferiority. Original frozen
rounds and their negative outcomes remain unchanged. Final909030xseeds untouched.
