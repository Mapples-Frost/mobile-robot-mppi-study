# ICRA deadline pivot — 2026-09-09

User priority: obtain a feasible experiment and a defensible paper dataset for
the September 15 deadline. This supersedes the previous open-ended sequence of
mechanism diagnostics and automatic expansion into matched-model training.
Existing frozen experiments and their conclusions remain unchanged.

## Working question

Can observable online planning-budget adaptation improve the navigation /
measured-compute tradeoff in a clearly specified operating region compared
with a well-tuned fixed budget? Joint K/H, ICODE, and reliability features are
candidate components, not mandatory conclusions. H-only is an eligible simpler
method if it is better supported. Publication or acceptance is not guaranteed.

## Immediate execution

Run the already specified geometry screen, unchanged: 480 episodes, four
families, three geometry scales, two speeds, two fresh environment seeds,
both full/masked training initializations, and six fixed model/budget controls.
Protocol: docs/protocols/icode_sac_geometry_screen_2026-09-09.md.
This is targeted development. Its geometry selection was informed by previous
results. Retain every cell; it is not final test evidence.

Do not automatically start the large matched-model training factorial after
this screen. First make the bounded decision below. Run timing experiments
serially, without concurrent training. Preserve frozen source/configuration,
checkpoint hashes, raw trajectories, and failures.

## Decision by September 10, 18:00 Asia/Shanghai

Use the entire screen to select at most one operating-region hypothesis and
one primary method. Compare to the strongest development-selected fixed
controller, not just K256/H24. Evaluate both initializations separately.
Development prioritization: repeated direction of task-quality and compute
benefit across initializations and adjacent geometry settings; no observed
collision increase; meaningful practical compute reduction (target 15%) or
task improvement at comparable compute. These are prioritization criteria,
not significance or safety noninferiority claims.

If joint adaptation is unstable, test a bounded H-only candidate using the
existing trained policies before any new RL campaign. If reliability features
remain unsupported, omit reliability as a claimed contribution. If ICODE is
retained, include a tuned nominal controller and matched nominal adaptive
control; its extra inference cost counts. No more than one new training round
after the selection decision. If no candidate survives, record that fact and
reassess submission scope rather than continue unbounded seed searches.

## Remaining calendar (work targets, not completed evidence)

- September 9–10: screen, select mechanism/region, check novelty against primary
  literature, freeze the method and exact validation protocol.
- September 11–12: independent paired validation including fixed, single-axis,
  and essential ablation controls. Use fresh environment seeds and at least
  three training initializations for any newly claimed learned algorithm.
  Choose episode count using pilot variance and measured runtime before launch;
  retain enough time for completion. Report uncertainty at episode/scene and
  training-seed levels; control cycles are not independent replications.
- September 13: one bounded generalization axis; real-robot demonstration only
  if hardware access and the existing stack are ready. Simulation-only results
  must be identified as such. Prepare complete tables and figures.
- September 14: full manuscript, limitations, references, reproducibility and
  submission checks. No new mainline architecture changes.
- September 15: final review and submission buffer.

The minimal desired evidence package is one primary tradeoff comparison,
one mechanism/ablation comparison, and one generalization experiment, all
generated from traceable records. Existing development results support method
selection and limitations, not fresh confirmation after selection.

## Handoff snapshot

Read task 01a07b93-9142-7e31-b87e-b63f18a8c3d4 and local morning/mechanism reports.
Prior 624+960 episodes are complete. High-budget compute savings of 12–28%
do not establish superiority over tuned fixed budgets. Reliability effects
reverse across initializations. At inspection no experiment Python process was
running. Geometry screen existed but had no output directory.
