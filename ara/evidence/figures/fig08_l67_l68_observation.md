# Figure 08: L67/L68 observation-domain residual-structure confirmation

- Vector artifact: `results/research_platform/rl/l68_residual_structure_observation_confirmation_20260716_v1/figures/fig_l67_l68_observation_robustness.pdf`
- Raster artifact: `results/research_platform/rl/l68_residual_structure_observation_confirmation_20260716_v1/figures/fig_l67_l68_observation_robustness.png`
- Generator: `experiments/rl/plot_residual_structure_observation.py`
- Development input: `results/research_platform/rl/l67_residual_structure_observation_20260716_v1/residual_structure_observation_summary.json`
- Confirmation input: `results/research_platform/rl/l68_residual_structure_observation_confirmation_20260716_v1/residual_structure_observation_summary.json`
- Raw-odometry stress table: `results/research_platform/rl/l68_residual_structure_observation_confirmation_20260716_v1/figures/table_l68_wheel_odometry_stress.csv`

Panel A reports the sealed primary ICODE-versus-MLP reduction for each bounded
observation domain. Panel B compares hierarchical intervals between development
and sealed confirmation. Panel C shows that the raw wheel-odometry development
success ordering reversed under sealed seeds, preventing an unsupported
localization-robustness claim.
