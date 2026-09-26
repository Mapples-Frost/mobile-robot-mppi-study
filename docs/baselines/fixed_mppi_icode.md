# Fixed MPPI and fixed ICODE-MPPI controls

B1 uses the existing MppiController with nominal dynamics. B2 changes only the
prediction model to the existing frozen ICODE residual correction. K/H, scene,
task reward, objective, sampling, control period, action limits, guard and seeds
are shared within each comparison. Neither baseline reads reliability features
to choose K/H. These are common-backbone controls, not new published methods.

The fixed-wrapper equivalence tests compare controls and trajectories exactly at
the old default600/36. The new bounded development reference is256/20; old results
and default configurations remain unmodified. Best fixed budget is to be selected
only after development landscape, never from formal test outcomes.
