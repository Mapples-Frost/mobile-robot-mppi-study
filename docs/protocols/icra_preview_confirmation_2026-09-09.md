# Frozen preview selector confirmation

Selection uses the complete development42 comparison. Preview RMSE .041166m
vs full .040325m, compute4.812s vs7.143s. Turn .045576m/3.019s.
Choose preview alone for independent confirmation; no further threshold tuning.
This is a heuristic contribution candidate, not established literature novelty.

135 serial episodes: three paths (mirrored sweep, stretched chicane, existing
reverse-S), three fresh episode seeds10510101/2/3, three pretrained model seeds
20261201/2/3, five modes nominal/full/turn/periodic/preview. Randomized method
order within scene/episode/model blocks; fixed schedule seed10510200.
The two transformed paths are unseen geometries, not new independent task families.
All three model checkpoints are existing pretrained models, no new training.
Nominal repeats across model blocks measure timing repetition, not independent
trajectory replicates. Analyze paired episode blocks and disclose shared models.

Preview unchanged .6m lookahead, angle on.25rad/off.12rad. Turn on.25/off.15rad.
Periodic uses 6/10 ICODE cycles, approximately matching development preview60%
usage. This controls invocation budget; actual measured runtime remains primary.
K100/H36; same safety and physical settings; fixed-step simulation, measured
compute including selection overhead, no wall-time injection into dynamics.

Primary accuracy criterion: mean paired preview-minus-full RMSE <=.003m, a
prospective practical tolerance; require >=20% mean measured compute reduction.
Also report all paired effects versus turn and matched periodic, every path and
model, success/collision and failures. Bootstrap by episode seed within path,
keeping methods and model blocks together; limited path/model counts constrain
inference. Do not equate lack of significance with equivalence. No selection of
best path/model or omitted failures. Failure to pass is retained and triggers
reassessment, not threshold adjustment on these data.

Sources, checkpoints, configs, raw trajectories and selector logs are retained.
Numerical mechanism audit and figures follow after timing completes.

Transformed sweep uses x1/y-1 and chicane x1.15/y1.1. Both have a common420-step cap across all methods; reverse-S retains360. Compare methods within each path and report duration, not pooled unequal-cap success alone.
