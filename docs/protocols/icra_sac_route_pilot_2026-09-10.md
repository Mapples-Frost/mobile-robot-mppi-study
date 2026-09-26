# Original SAC, explicit route-task adaptation: bounded learning pilot

Preserve original50D causal observable/reliability context,128x128SAC architecture and spec hyperparameters, raw continuous replay, K16..256step16/H8..40 mapping, hold5, fixed inactive axesK128/H32, and measured command-readiness delay entering MuJoCo. Full ICODE and calibrated sent-command history remain. No rule policy or residual surrogate in learned arms.

Four arms: joint full, H-only full, K-only full, joint with dynamics input17:33zeroed. One initialization12901001 and10000physicalcycles perarm (finish finalepisode), equal training budget. This is a bounded developmental learning check, not final multi-seed validation. Use same episode schedule RNG12902001 and environment seed12910000+episodeindex for each arm; twoqualifiedshapes,originaldirection,standardmass,block0. All arms retain causal context computation time evenwhenmasked. No training/evaluation concurrency. Retain all failures and finalchecks only; no bestcheckpoint selection.

Task reward is explicitly adapted from oldpointgoalprogress to curve progress. Perdecisionblock:

    reward = 2*(progress_after-progress_before)/route_length
             - block_cycles/episode_cap
             - sum(dt*(lateral_error/0.05m)^2)
             - 0.05*sum(measured_command_readiness_seconds)
             - 10*collision - 3*timeout_without_success.

Progress is projection of completed observedpose onto suppliedreference; lateralerror is recordedpostexecutiontaskcost/truth, used only for trainingreward. 0.05m is a declared development normalization scale, not a user-specified tolerance or safety guarantee. It gives tracking a meaningful scale unlike unnormalizedengineeringcost where tracking was0.3–1.3%ofsum. No tuning of this scale orcomputeprice duringthepilot. Speed/stagecostdiagnosticsare retainedbutnotsecretlyaddedtothisreward. Taskrewarddiffersfromoldpointgoalstudy andfromRL-Hbaseline; comparisonprotocol must disclose these differences.

Useoriginalspec learningstarts500decisions,batch128,twoSACupdatesperdecision,scene-balancedreplay,automaticentropy. A10000cyclepilot has roughly2000decisions; limitedtraining may be inconclusive. Checkpoint/resume recordsoptimizer,replay,RNG,episodes andallcomponents. Firstrun short engineeringvariant16decisions andverifyrewardreconstruction/finitegradients/actualhold5beforefullpilot. Then evaluatefinalpolicies withfreshseeds, strongfixedK/H, K-only/H-only/masked andsimple rule; matchedliteratureRL-H must executein samephysicaldelayenvironment beforecomparativeclaim. No claim untilfairfullcomparison, no automaticlarge expansion basedontrainingreturn.
