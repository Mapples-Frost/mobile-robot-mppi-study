# Repair baseline domain coverage before interpreting narrow failures

Timeoutdiagnostic8 showednominal28/36 andpreview2/2completion with120s,
RL-H0/2stillstuck. OriginalRLtrainingonlywidth1.8m makesnarrowfailure OOD.
TrainonefreshSAC-H seed11101001 for30000steps with uniformlyrandom widths
1.8/1.4/1.0 each episode, samemass1, speed.35, reference and safety. No test
seed reuse; trainingepisode seeds11110000+index. Same reward coefficients,
H8–50,19features,valueaugmentation as firstbaseline. Cap1200steps, timefeature
andremainingconstraintpenalty normalizedaccordingly forALL subsequentcomparisons.
This trainingdomain amendment is development, not evidence originalpaper fails.

Finishcurrentepisodeafter30k,savefinal,no testselection. Sharedterminalvalue
remainsaugmentation not originalquadraticreplacement, explicitlyreport. This
doesnot claimoptimal training. If baseline stillfails, inspecttrainingcoverage,
valuebehavior and objective; don't turn undertraining intooursadvantage.
Logwidthperepisode; boxgeometry deterministicfunction frozeninsource. Allother
trainingsettings unchanged, rawreplay/checkpoints/source snapshots retained.

Aftertraining, developmentqualification mustcover ALL3widths using newseeds,
fixedH controls andlearnedH, common120s cap. No concurrenttraining+timedeval.
Do not runa newmismatchgrid untilqualified. Original60s dataretainedseparately.
