# Obstacle anchor feasibility:40 paired development episodes

Purpose: verify geometry and shared controller feasibility before any claim
about superiority or increasing mismatch. One1.8m nominal free-width rounded
corridor, radius1.2m reference,2m approach and1m exit. Physical box walls enter
MuJoCo collisions and lidar, not merely drawing. Speedcap.35,600steps,dt.1.
Safety unchanged. Wall overlap and discretization mean actual minimum clearance
must be checked from geometry; label1.8m nominal until numerical verification.

20 reserved development seeds10710001–20, nominalK100H36 and previewK100H36;
one fixed pretrained residual20261201, no model selection or new learning.
Goal >=19/20success andzero collisions for EACH method. BaselineRL-H does not
yet exist: this is initial anchor qualification only, not completion of the
all-method common-success requirement. RL-H must subsequently pass this anchor.
No efficacy inference or publication success claim from these development runs.
If either method fails, inspect clearance, guard interventions and local
controller before changing geometry equally for all methods. Do not tune only
our method. Retain all initial failures. No test seeds used.

Frozen sources/configs/checkpoint and raw per-step artifacts. Serial isolated
workers, randomized pairedmethod order. No heavy training during measured runs.
Ideal pose sensing remains, physical lidar obstacles enabled; no occlusion/noise
robustness claim. Fullmodel and learnedH comparisons still pending.
