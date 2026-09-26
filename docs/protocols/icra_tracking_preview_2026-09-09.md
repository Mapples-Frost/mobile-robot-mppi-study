# Task-sensitive model selection — 42 development episodes

Tracking30 completed and audited. Mean RMSE: nominal.06123m, ICODE.04203m,
turn.04654m, innovation.05365m, periodic.05118m. Mean per-episode compute:
1.075,7.356,3.386,3.328,4.259s respectively. This shows useful accuracy/compute
tradeoffs, but one-step innovation underperforms the simple turn heuristic.
It is not evidence that our selector is a sufficient new paper contribution.

Test two task-sensitive candidates while retaining ALL five previous controls,
with unchanged inherited K100/H36, physical/safety/task settings and timing scope.
Three paths x two fresh seeds10410101/10410102 x seven modes =42. No threshold
selection after outcomes; this is the last small selector-candidate screen
before choosing a single design for independent verification.

Preview: project current sensed pose on the known reference without updating
reference progress; sample tangents over next.6m; use maximum absolute angle
from current heading. Enter ICODE at.25rad and leave below.12rad. This anticipates
turning demand; it does not access future robot motion.

Lateral: carry the previous selected control sequence, shifted by one. Roll
nominal and residual for10steps from current sensed state using those same
commands and fixed-delay average-command approximation(.04/.10). Use normals
of the known reference at current progress+max(abs(v),.1)*dt*j to project the
two predicted position differences. Enter ICODE when maximum absolute lateral
disagreement>=.01m; leave below.005m. Initialization uses zero previous commands.
This is a task-sensitive disagreement heuristic, not calibrated error probability
or a formal safety bound. Every proxy rollout and projection is timed.

Turn, innovation, periodic and fixed predictors are unchanged algorithmically.
No actor training or model fitting. Log proxy values and actual model choice;
all source/config/seed records retained. The residual is an existing published
structure, not our contribution. A positive candidate must beat simple controls
in an accuracy-compute tradeoff and survive other model training seeds and fresh
path/seed verification. All paths/episodes and failures remain visible.
