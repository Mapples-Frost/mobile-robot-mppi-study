# Current status correction: evaluation and audit COMPLETE

260 episodes complete; authoritative audit is rl_h_ready_evaluation_2026-09-10/analysis/audit.json (passed=true). Older RUNNING entries below are historical. No executor needs restarting.

Statistical correction: context-level intervals are Student-t intervals (df=4), NOT bootstrap. Single-turn joint versus fixed residual Pareto count is 1/5. Crossing zero is not evidence of equivalence. Two route families do not isolate a causal complexity effect, and unstable H-only comparisons leave the incremental value of K unresolved. The paper is not yet established as submission-ready. Next work must assess a bounded test of that incremental value and manuscript gaps; avoid broad sweeps and do not manufacture favorable results.

# Historical continuation: matched-readiness RL-H training audit PASS; fresh-scene evaluation RUNNING

All six matched-readiness RL-H runs are complete and independently audited. Each completed its final episode after 10k physical cycles: nominal/residual seed13501001 10114/10152 cycles, seed13501002 10097/10150, seed13501003 10059/10203; SAC updates 9560--9704. `training_audit.json` passed current-source hashes, final/resume consistency, every saved 19D action/H mapping, raw postexecution reward reconstruction, physical timing, episode record and finite update metrics. This is provenance only, not a performance conclusion.

Frozen protocol `icra_rl_h_ready_evaluation_2026-09-10.md` and serial runner `evaluate_rl_h_ready.py` now evaluate 260 fresh episodes: two existing routes x five new seeds13620001--5, all3originaljoint+Honly initializations, all3 nominal and residual matched-readinessRLH checkpoints, matched terminal-value fixedH16/H32 controls per RLH checkpoint, plus strong residual/nominalK128H32. New route seeds, deterministic final checkpoint, one serial randomized schedule. RLH is explicitly a local literature-inspired 19D/K100/H8..50/value adaptation with a different reward/architecture/frequency from original50D SAC, never exact author reproduction or same-architecture ablation. Evaluation started cleanly (first8/260 at verification); `audit_rl_h_ready_evaluation.py` waits for completion and will replay raw metrics, source hashes, originalSAChold/map, RLH actor/map, fixed controls and physical timing. No new training/candidate branch is active. Do not edit runner/protocol/adapter/frozen source while the queue runs.

# Latest heartbeat09:31UTC: matched-readiness RL-H training STARTED after qualifications PASS

Nominal6engineeringcases and rawrewardauditPASS, fourfullfixedroutesallsuccess0collision. ResidualprevioussamePASS. Implemented experiments/icra_rl_ready_training/train.py under protocol docs/protocols/icra_rl_h_ready_training_2026-09-10.md BEFOREtraining: 3seeds13501001/2/3 xnominal/residual,10kphysicalcycleseachfinishfinalepisode, only2routesstandardmassoriginal, predictorblock0. Keeps19D32x32SAC,K100,H8..50,hold1,gamma.97,tau.005,lr3e-4,autoentropy,replay50k,batch128,warmup500,1update/cycle;value32stepbootstrap8updates/episode. Differentreward/arch/updatefrequency from50Djoint explicitlydisclosed. No timingcomparisonconcurrent.

Both80cyclesmokesDONE65SACupdateseach,finitevalueupdates; queue independentlycheckedactorparameterchange>0,mapping/19D/mode/physicalreadiness. Smokeartificialcutoffnotnavsuccess. Saved smoke_audit.json. ACTIVE start_rl_h_ready_training.py session33438, firstseed13501001nominalstarted, firstepisode241cyclessuccess. All6serial, completionmarker all_training_completed.json; queue_status finalizedproperly. Savesrawrows/config/update metrics/source+residualweighthashes/resumeactorcriticoptimizers/replay/RNG/scenes. Do NOTedit train.py orfrozenadapter/sourceswhileactive.

Stage rl_h_ready_training_2026-09-10. RNGscene=seed+1000; env=13510000+(seed-13501001)*10000+episode. Bothmodespairedsceneorder, behaviorlengthcanvary. Randomwarmupselectioninsideactorcallbacktimed; traininggradientoutsidephysicalcycle. NEXT preparefresh evaluationprotocol/runner afterALL6traincomplete; includeall3originaljoint/H-onlyinits andstrongfixednominal/full, bothRLHpredictorsall3inits, preservingold210. Needfullrawtrainingreward+32stepvalue/replay auditingaftertraining and beforepublication. Thisremovespredictor+delaytransfergap bynewtraining, doesnotclaimliteratureexactreproduction.
# Latest heartbeat09:18UTC: learned210 COMPLETE/all audits PASS; nominal RL-H qualification ACTIVE

210/210success0collision; raw57726cycles/6628learneddecisions auditPASS, postchecks_completedPASS (newtrainingreward/configparity/budgetvariation/groupedinit+scene summaries). ResidualRLH engineering6casesDONE, fourfullfixedroutesALLsuccess0collision, actor40bothtruncatedinterfaceonly; independentrewardauditPASS. Allpriorprocessesexited atcheck.

Ran report_independent_route_verdict.py; docs/reports/icra_route_sac_independent_verdict_2026-09-10.md and desktop原SAC独立评估210/00_本轮结论.md plusroot00_最新进度.md updated. Full210data retained. Meanacross3inits:single joint10.55294mm/3.29625s vsfull128H32 10.55497/4.08036 =>19.2167%saving/error-.00203mm;reversejoint12.47697/4.72577 vs16.12978/5.65628 =>16.451%saving/-3.65281mm. Honly10.9632/3.66615 single,12.78643/4.54851reverse. JointnotuniformlybetterHonly; reverseinit13101001 Honlymean-dominatesjoint. Masked9.53149/3.70166single,12.57049/4.72767reverse =>no stable dynamicsdescriptorbenefit. Nominalmuchcheaper. LiteratureRLHtransfer confounds predictor+delay, explicitlydisclosedinverdict. Notpaperready/no robustnessclaim.

New protocolicra_rl_h_nominal_ready_qualification_2026-09-10.md and qualify_rl_h_nominal_ready.py clonedengineeringwithONLYcompute_allocation.mode nominal, same6cases/seed13410001, independentstage. py_compilePASS. ACTIVE session47140(firstsingleH16success), auditwaiter67954. Runsafterallprevious210/audits/residualqualificationverifieddone. No trainingyet. NEXT verify nominalqualification+rewardaudit then implement boundedRLH nominal/residual matchedreadinesstrainingwith newprotocol,budgets,seeds, original19D32x32value/32step settings. Need distinguishsamearchitectureH-onlyablation vs literature-inspiredadaptation (reward/arch differ). Frozenadapter/environment.py now hashed byqualifications, do notedit; extendnewtrainerfolder instead.
# Latest heartbeat08:58UTC: eval158/210; additional RL-H transfer confound FOUND

Eval1653364 active158/210, primaryaudit/postchecks/RLHqualification/rewardaudit waiting, no concurrentheavyjob. Staticread oldgeometryRLHtraining run.py and manifest12401001 shows planner.prediction_mode=nominal (no compute_allocation), whereas current rl_h_transfer RouteObservableEnv uses residual mode. Thus transfer confounds BOTH readinessdelay and predictor change, not delay alone. Wrote docs/reports/icra_rl_h_predictor_transfer_correction_2026-09-10.md. Existing transferdiagnostic exclusion fromfairbaseline claims remains; finalreports MUST disclose both changes. Frozen210 unchanged.

New ReadyHEnvironment qualification currently residual too; valid interfaceengineering only. Before newRLHtraining explicitly choose/train nominal baseline + residual predictor-control version (protocol/budget required), avoid silently moving originalnominalbaseline intoresidual. Same50DH-only controls architecture/reward/residual for mainjointincrement. No newtraining yet. Do not infer relativeperformance from partial158 results.
# Latest heartbeat08:47UTC: independent timed evaluation100/210 active; RL-H raw reward audit queued

All8newtraining complete marker verified in prior userstatus turn; training process exited. queue_status.json is stale index7 running, completion marker and process state authoritative. EvalPID1653364 now100/210, no concurrenttraining. Existingprimaryaudit1653456/postchecks1684151/RLHqualification1746013waiting. No partial-effect inference or newresultsclaims.

Added audit_rl_h_ready_qualification.py and started waiter session20115; only afterRLH engineeringcompleted marker, validates sourcehashes and independently reconstructs postexecution pathprojection/speed/accepted-command delta/failurecost/reward for all6cases, Hrawmapping/timers/physicaldt. py_compilepassed. No model/module imports for rewardarithmetic. Writes rl_h_ready_qualification/reward_audit.json. Preserves fixed_routes_qualified flag; auditpassed doesnotmean navigation qualification if false. No changes frozen evaluation/training sources. Next inspect complete210, postchecks, thenRLHengineering + rawreward audit; matched-delayRLHtrainingstillnotstarted.
# Latest heartbeat08:24UTC: final independent training arm active; RL-H qualification queued after all audits

Verified queue index7 seed13101002 masked active PID1715070; evaluation/audit/postchecks waiters alive. Last of8 training runs, no overlap with timed eval. New docs/protocols/icra_rl_h_ready_qualification_2026-09-10.md frozen before engineering data: two routes x fixedH16/32 full episodes plus actor40cycles smoke, seed13410001, K100, existing19D actor+value weights onlyforinterface, all measured physical readiness. No training or superiority claim.

Added qualify_rl_h_ready.py, py_compilepassed, started waiter on independent evaluation analysis/postchecks_completed.json. After all210 and primary+postchecks, executes6engineering cases serially; outputs rl_h_ready_qualification_2026-09-10 with source/checkpoint hashes, configs, rawrows, finite19D/actionmap/timer/physicaldt checks. Fixed-H still runsactorsoNOTfixedbaseline timing evidence. Checks reward=-.3cost identity only; independent reward reconstruction remains required BEFORE newtraining. No newtraining queued. Stagecost commanddelta uses accepted commands, not delayed actual input; explicitprotocol. Actor40 is artificialcutoff, not navsuccess. On anyqualificationfailure preserveoutputs and inspect, don't expand blindly. Existingtraining/eval frozensources untouched.

NEXT inspect210 completion and all audit outputs, interpret by policyinit and sharedscene seeds; inspect newRLH adapter qualification/errors after marker. Fix baseline qualifiedtraining gap only afterengineering/rewardaudit/protocol. Currentpositivefixedsurrogate still irrelevanttooriginallearnedKH claim.
# Latest heartbeat08:12UTC: second initialization active; RL-H readiness adapter drafted, NOT qualified

Verified queue index5 seed13101002 H-only active PID1684341; queue1620469, eval1653364, audit1653456 and postchecks1684151 alive. All four first-init arms and second-init joint completed by queue progression. No timed evaluation yet. A pgrep command failed from Windows/WSL regex quoting; used ps successfully, no experiment failed. Initial guessed geometry train.py absent; actual source run.py read.

Implemented isolated experiments/icra_rl_ready_adapter/environment.py (py_compile only, NOT imported/smoked/trained). ReadyHEnvironment shares RouteObservableEnv measured physical readiness, retains local literature-inspired 19D/H8..50/K100/value augmentation and actor inference inside timing; skips unused 48D neural reliability context to avoid needless baseline cost. Keeps previous local RL-H stage reward (not joint reward-matched), reconstructs after-execution route cost rather than hardcoded goal. No active source changed, no heavy concurrency.

NEXT after210 and postchecks finish: qualify new RL-H adapter with real actor + value, all finite features and mapping/timer/cost reconstruction, fixed H common-success smoke; inspect accepted-vs-physical delayed control reward semantics explicitly. Then freeze protocol, original RLH hyperparameters and independent initialization budgets BEFORE training. Preserve clear distinction: same-architecture50D H-only ablation is resource/objective matched; literature-inspired19D+value RLH differs architecture/reward/Hrange/K100 and must be disclosed, not called exact paper reproduction. Existing RLH transfer210 results remain unchanged, cannot silently swap checkpoints. New adapter is preparation only; no matched-delay baseline available yet.
# Latest heartbeat08:00UTC: independent training healthy; postchecks queued

Verified actual processes: independent queue PID1620469, index3 seed13101001 masked running PID1653754; first three arms completed by queue progression. Evaluation PID1653364 and primary audit PID1653456 waiting, no concurrent timed evaluation. No new outcome claim.

Added prepare_independent_postchecks.py generating audit_independent_training_rewards.py (both new policy seeds, correct schedule RNG13102001/13202001, all four arms) and assess_independent_policy_variation.py (exact config parity and executed budget ranges for all12policies). Added finish_independent_postchecks.py, started waiting on evaluation analysis/completed.json; then serial audits and paired summaries by scene seed AND policy initialization. Keeps five shared scenes distinct from three initialization repetitions; no pseudoreplicated15scene claim. Scripts py_compile passed. No frozen trainer files edited. Desktop outputs go to 原SAC独立评估210. Need inspect postchecks_completed after210 primary audit, full results and fair matched-delay RL-H gap still pending. Current learned replication remains only one predictor block, standard mass/two routes; no robustness or paper readiness claim.
# Latest heartbeat07:43UTC: independent training progressing;210fresh evaluation+audit queued

Verified13101001jointdone10287cycles,queueindex1H-onlyactive. Same8runqueue82541, no newtrainingconcurrent. Wroteprotocolicra_sac_route_independent_evaluation_2026-09-10.md. QUEUED evaluate_route_sac_independent.py session99087 waitsall8trainingmarker,then210serial2routesx5freshseeds13320001-5x21methods (3initsx4learned+9controls). All12finalcheckpointsincluded, baselinesonce/context. Savedmethodids arm_sPOLICYSEED. All3residualcheckpoints nowexplicitlyincludedin evalmanifest sources; predictorstillblock0. Sharedphysicaldelay/strongnominalscopeandRLHtransferlimitationsunchanged.

QUEUED audit_route_sac_independent.py (generated/py_compilepassed) waits210thenfull210rawmetrics/configsource/12actor/hold5/map checks,42method-routecells,360jointpairedcontrasts (sameinitH/K/masked andshared9baselinecontrols),copiesdesktop原SAC独立评估210. Don'ttreat15policy-seed combinationsas15independentscenes. Nextneedaggregatebyinitwithinsceneseed andperinitconsistency, exactconfigparity, rewardauditsfornewreplicatesandsmokes; figure workaftertiming. Generatorinitialduplicateprotocolreplaceassertfailedbeforecreatingrunner; correctedandcompiledsuccessfully, nofailedexperiment/dataortraininginterruption. No editsoldfrozen/trainingfolders. Needmatched-delayRLH training/qualification separatelybeforepaperclaim; notqueuedyet, don'trunconcurrentlywithtimedeval.

# Latest heartbeat07:30UTC: independent SAC replications STARTED, same frozen settings

Protocolicra_sac_route_independent_2026-09-10.md writtenbeforetraining. Newisolatedexperiments/icra_sac_route_independent/train.py copiedpilottrainerONLYparameterizing policy/replay/schedule/envseed andoutputpath,importsoldadapterunchanged; checkedrewardfunctionbyte-identical. Newpolicyseeds13101001/13101002,4arms each,10000cyclesplusfinalepisode;replayseed=policyseed+1;scheduler13102001/13202001;env13110000/13210000+episode. Same residual predictorblock0 (NOT3modelblocks),50D/128x128/originalmapping/hold5/rewardunchanged. Allfourarmsschedulepairedwithineachinit. Old12901001policiesretainedasthirdinitialization, no selection.

80cycles/16decisions smoke newseed13101001 passedfiniteactualSACupdates, truncatednotnavigationresult. ACTIVE start_route_sac_independent.py serial8runs;first13101001jointstarted. Stage sac_route_independent_2026-09-10, queue_status.json. Nootherheavyprocess. Oldpilotfrozensourcesunchanged. Do NOTeditoldadapterfolderornewtrainerfolderwhilequeueactive;wholefoldershashed. No performanceclaimfromtraining.

NEXT preparefreshsame-delayevaluation onceALL8complete: plannednewseeds13320001-5 x2shapes x(12learnedpolicies from3initsx4arms +5fullfixed+2nominal+rule+RLHtransfer)=210episodes. Baselinesrunoncepercontext, nottriplicatetoinflateN. KeepRLHtransferlabeluntilmatched-delaytrainingqualified. Need protocol/run/auditpreparedbeforetimedexecution. Independenttrainrewardauditneedsreplicate-specificschedule/envseed. NeedmatchedRLHtrainingafterthisqueue orotherwiseavoidconcurrenttimedeval. Goaloriginaljointnotyetcomplete; no hyperparameterretune tohide maskedadvantage.

# Latest heartbeat07:14UTC: learned route78 COMPLETE, auditsPASS; masked mean-dominates full joint

All4trainingdone (~10kcycles each); all78evalsuccess0collision. audit21592cycles/1325learneddecisionsPASS; independenttrainingreward(all4arms)PASS; configequality78within6contextsPASS; actual learnedbudgetvariation confirmed. Noactiveprocessaftercompletion. Data desktop原SAC学习评估78 with00_本轮结论.md copiedfrom docs/reports/icra_route_sac_pilot_verdict_2026-09-10.md.

MeansRMSEmm/compute s:single joint10.323/3.353,Honly10.422/3.501,Konly11.069/4.361,masked10.190/3.254,fixed128H32 9.822/4.061. Reverse joint12.961/4.765,Honly14.129/4.631,Konly15.571/5.842,masked12.097/4.635,fixed128H32 15.845/5.616. Jointvsfixedsaves17.4/15.2% with+.501/-2.884mm. MaskedmeanbetterboththanfulljointBOTHshapes; no dynamicscontextbenefitclaim. JointnotuniformlybetterthanHonly (reversefasterHonlybutlessaccurate). Nominalmuchcheaper; RLH remainsOODtransferdiagnostic fromfixedsteptraining. Full26cells preserved, no cherry-pickedcandidate.

Actualjointadaptationnotcollapsed(singleexampleK80..144,H18..32,29switches;seeall24recordsconfig_and_budget_variation.json). Oneinitialization3evalseeds isdevelopmentonly. Training1e4cycles~2kdecisions limited. NEXT freezeindependentreplication withsame reward/arch/hold/mapping andALL4arms, additionalinitializations/newseeds; avoidretuningbasedonpilot. Needmatchedreadiness-delayRLH training/qualification beforepaperbaselineclaim. Don'tswapmaskedidentityintofulloriginalmethod, don'trevivefixedsurrogate756asKHproof. Coreoriginaljointgoalstillnotcomplete; keepautonomousworkfocused.

# Latest heartbeat07:02UTC: three training arms done, masked8038; full reward audit queued

Verified joint10041/2021,Honly10204/2057,Konly10177/2054(cycles/decisions)complete. Masked~8038cycles/1620decisions activePID1514707;queue1451806 healthy. Evaluation1514258 andaudit1514375waiting, no concurrenttiming. No finalperformanceclaimfromtrainingreturns.

Added audit_route_training_rewards.py waiter (afterALL78timedeval completed) independentlyreconstructsALLtrainingblocks fromrawsensed-before/sensor-after/truepostpositions anddeterministicgeometrycontextschedule:routeprogress,totalpathlength,trackingnormalized.05m,resource/time/failureterms,rawcontinuousactionmap/hold5,50Dmaskedchannels,episode/cycle/decisioncounts. Writesstage reward_audit.json +desktop. Doesnotimportadaptercostfunctionforarithmetic. Thisisadditionalprovenance check, notnewexperiment. Mustinspect anyfailure; don'tloosenlimitsblindly. No edittrainingfolder. Continuewatchtransition to78eval thenall26cellcomparisonandfreshlearnedpolicyinterpretation. Fullmatched-delayRLHstillmissing; keeptransferlabel.

# Latest heartbeat06:47UTC: joint training done, H-only active;78eval+audit queued

Jointpilotcomplete10041cycles/2021decisions;Honly~7112cyclesatcheck, Konly/maskedfollowin61561queue. No heavyconcurrentevaluation. Newprotocolicra_sac_route_pilot_evaluation_2026-09-10.md:78episodes2shapes3freshseeds13020001-3x13methods learned4+full5fixed(64H16/32,128H16/32,256H32)+nominal64/128H32+oldpathrule+RLHtransfer. QUEUED evaluate_route_sac_pilot.py session87392 waitsALL4trainingmarker;py_compilepassed. No edittrainingfolder.

Evalallmeasuredreadinessdelay;SAC50Dhold5,ruleandRLHpercycle; fixednominal skipsunusedneuralcontext via subclasszero-placeholder(returnnotfedactor), toavoidweakslowbaseline. Otherfullfixedretainsharedcontextcost. RLH19D+terminalvaluefromoldgeometryfixed-steptrainingexplicitTRANSFERdiagnostic, notfairfinalbaseline. Latermatchdelaytrain/qualifybeforeRLHsuperiorityclaim. Fullconfigmethodsmaplearned50Drawactions; no residualsurrogate.

QUEUED audit_route_sac_pilot.py after78:source+residualhashesagainsttrainingmanifest,learnedactor/hold/mapreplay,rawtrackingRMSE/timers/success/physicaldt,all26cells+72jointpairs+desktop. Ruleconsistentloggedbudgets/RLHrawactionmappingcheckedbutnotindependentfullrule/19Dactorreplay; scopeexplicit. Evalmanifestdoesnotitselfsnapshot3residualweights, auditchecksunchangedtraininghashes instead. Needadditionalexactconfig/physicalfairness auditandtrainingrewardreconstructionaftertimedrun. Evaluationresultsnotyetavailable. Currentgoalnotcomplete; processactuallytraining. Do notduplicateortrainconcurrentlywithfuturetimedeval.

# Latest heartbeat06:30UTC: original SAC interface PASS; bounded learned route pilot ACTIVE

check_sac_interface.py PASS120physicalcycles: originalinitialize(spec)50D128x128SAC KH/K/H,8rawcontinuousdecisionsheld5eachperarm, actualSACgradientupdateperarmfiniteandactorchanged, inactiveaxes128/32 preserved. Rewardscaleinspect4engineeringroutes: unnormalizedtrackingonly.31–1.27%ofstagecost,time75.1–83.2%; do notclaimthisalonecausedpriorlearningfailures. Outputs sac_route_adapter_engineering/sac_interface andfull_routes/reward_scale.json.

NEW protocolicra_sac_route_pilot_2026-09-10.md fixedbeforetraining: 4armsjoint/Honly/Konly/masked,original50D/128x128/mappingK16..256H8..40/hold5/SACsettings,seed12901001,10000physicalcycleseachcompletefinalepisode. Sameepisoderng12902001,envseed12910000+episode,2shapesoriginalstandardmassblock0. Measuredreadinessdelayentersplant. TaskrewardEXPLICITadaptation 2*routeprogress/length -cycles/cap -sum(dt*(cte/.05m)^2) -.05summeasuredseconds -10collision -3timeout. .05misdevelopmentnormalizationscale,notuserspecifiedtolerance or guarantee; no midpilot tuning. OriginalpointgoalrewardNOTused. Fullcontexttimechargedmaskedtoo. No final/multiseedclaim.

train.py newisolatedrunnercheckpoint/resume/rawrows+blockreward+actions+RNG; smoke80cycles/16decisionsfiniteupdatespassed. start_route_sac_pilot.py independentlyauditedsmokehold5/maprawactions/rewardcomponents thenACTIVE session61561 serialjoint,h_only,k_only,masked. AtlaunchSTART ARM joint,nootherheavyjobs. Stage sac_route_learning_pilot_2026-09-10. Do NOTeditanyfilesinexperiments/icra_sac_route_adapter whiletraining (wholefolderhashed). Manifest's inheritedspecname remainsoldhistoricalstring butoutputarm/seed/axisareexplicit andcorrect; do notinterpretnameasselectedoldcheckpoint. No oldfrozenfileschanged.

NEXT whiletraining: preparefairfreshseed evaluation underSAMEreadiness-delay/contextcontract withstrongfixedbudgets+singleaxes/masked+rule. LiteratureRLH currentlytrainedfixedstepnotreadinessdelay; must discloseOODoradapt/requalifytrainingbeforemainclaim. Existingnominalplaincostlowerthanlegacyshadowcontextnominal; don'tclaimfastestoverallbychargingunneededcontextonlytobaselines. No timedcomparisonuntilALL4trainingdone. Trainreturnnotfinalperformance. GoaloriginallearningKHnowactuallyrunning; preserveallnegativeoutcomes. Automaticheartbeatcontinue.

# Latest heartbeat06:13UTC: route adapter IMPLEMENTED; full measured-delay routes PASS

New isolatedexperiments/icra_sac_route_adapter/adapter.py loadsPhysicalTradeoffV2sourceinPRIVATE module namespace andbindspathcostfunction; originalmodulecostguardunchangedverified. Subclasspreservesold48Dcausalcontext+speed/time=50D, CommandHistory.04, originalcore.map_action. Pathcostdeclaredcte²+.2speederr²+.002acceptedcommanddelta²+.02time, collisionweight2,resourceprice0 forengineering; NOToldpointgoalMPPIcost. Configusesoldcomputeallocation settings pluscommoncorridorplanner/plant/task/sensing. No SACtrainingyet, no claimsfromrules/surrogate.

smoke.py PASS16cycles:50Dfinite,zero/injected-zero physicalequivalence, originalmap endpoints,componentarithmetic,legacynamespacepreserved. qualify.py full4measuredreadiness-delay episodes (2shapes xK64/H32,K128/H32,seed12810002) ALLsuccess0collision0deadlineviolation. SingleRMSE9.068/10.571mm meanreadiness14.28/17.79ms;reverse17.531/16.790mm,14.92/17.53ms. Engineeringonlysingle seed, nottimedcomparativepaperresult. Rawrows/sourcesmanifest sac_route_adapter_engineering/full_routes. check_zero_against_unwrapped.py PASS12cycles each wrappedzero vsdirectunwrappedtask maxstate5.63e-15. Nothingrunningaftercompletion.

NEXT: actor/replay/hold5 smoke withoriginal50D128x128SAC andmapped16..256,8..40; defineexplicitroute-task trainingreward+resourceprice+timeouts undernewprotocol beforetraining. Need auditrewardcomponent scale toavoidtime/computepenaltydominatingaccuracybyconstruction. Do notreusehardcodedold(4.5,0)progressreward. Important: copiedCommandHistorycurrentlyusesmeasured_e2e_s evenzero/injected physicalmode; measuredtrainingmodeconsistent butzero-modecontextmaydiffer. ZeroequivalenceclaimPHYSICS only; ifzero-modepolicycomparisonplanned, correct/explicitlyversionhistorytoselectedknownphysicaldelay andverify, notchangepastrecords. Allnewcost/taskchangesexplicit, oldfrozenfilesunchanged. OriginalSACidentityclear, newtaskadaptationrequiresnewtrainingdata.

# Latest heartbeat06:00UTC: original SAC/corridor compatibility audited; concrete adapter gaps identified

Read core.py,observable_env.py,train_round.py,axis_transfer.py,run_observable.py,PhysicalTradeoffV2Env/base andoriginalindependent spec. OriginallearnedSAC50D=48+speed+time;128x128; joint2D/Konly/Honly1D;K16..256step16,H8..40step1,hold5,rawcontinuousreplay. OriginalObservableEnv causalCommandHistory calibrated.04. RecentRLH19D/32x32/H8..50/valueaugment isdifferentbaseline, notplug-inreplacement.

Hardincompatibilities:core.observed_distance hardcodesgoal(4.5,0);dense_reward2progress/4.5 inappropriatecurrentpaths;PhysicalTradeoff.executed_stage_cost explicitlyraisesforpath_preview/path_boundary, whilecorridorsusepreview; oldreadinesslatencyentersplant, recentcorridortimingdoesnot. New50Dtaskadaptation/retrainingrequired, notzeropadloadoldactor. Wrotedocs/reports/icra_sac_corridor_compatibility_2026-09-10.md copieddesktop原SAC与走廊环境兼容性.md withmatrix+reusablecontracts.

Noactiveprocess/newtrainingatend. NEXT actualengineering: independenttaskadapterpreservingoldSAC/budgetmap/hold/causal50Dcontext andreadinessdelay; replacepointgoalreward/stagecostwithdeclaredroutecost, smoke fixedbudgets/featurecausality/zerodelay equivalence/commonreachability beforetraining. Keepoldfrozenfilesuntouched, don'tappendrulesandcallitSAC. Neednewprotocol beforedata/newtraining. Userauthorizesautonomousboundedwork, noapprovalneeded. Don'tkeepdoingonlystatusreports; progressadapterwhenfeasible.

# Latest heartbeat05:46UTC: user explicitly challenges relevance; return to learned jointKH identity

Latestuser“这和我们论文有啥关系” afterclarifyingtwoideas. Explainedfixedsurrogate756notjointKHproof; directionwasmiscommunicated. Readoriginaldocs: earlierBetaPPO(NO-GO),laterphysicaltradeoff/SACICODEjoint+singleaxis+maskedcontext. NEWcausalrule60 isNOToriginallearnedSACmethod. Don'tswitchidentityagainorclaimnewrulefulfillsoriginalalgorithm. Wrotedocs/reports/icra_original_method_alignment_2026-09-10.md copieddesktop论文主线与方法身份.md. Needrestorelearnedjointscientificquestion andexplicitversionedinterface, notautomaticnewtrainingwithoutcompatibilityaudit.

audit_joint_increment.py DONE: all6pairedshape/seedjointfasterthanH-only, accuracybetteronly2/6. MeanRMSEpenaltyjoint-vsHonly+1.445mmsingle,+.876mmreverse. JointvsKonlybetterbothall6;vsfixedK64single2/3both,reverse3/3. 5044adaptivecyclesZEROwithin1e-10rad of15degthreshold; doNOTinventthresholdtieproblem. Actualearlierfloat32auditdifference wasnearestsegmentatvertex, correctedfullprecisionauditalreadyPASS. Outputsanalysis/joint_increment_and_switching.json copieddesktop在线KH60. Noactiveprocess/newexperiment.

NEXT concreteboundedtask: inspectoriginalSACjoint observationbuilder/actions/reward/latencycontract+checkpoints versusnewcommon-successgeometry environment; produceexplicitcompatibilitymapping andidentifywhichcomponentscanbeusedunchanged. Separateoriginallearnedmethod,rulebaseline,andfixedsurrogateengineering. Iftrainingnewversion, freezeprotocolandfairfixed/Konly/Honly/RLH/rulecontrols andtrainingbudget first. Userautonomousauthorizationcontinues; noapprovalneededforboundedresearchbutno silentmethodreplacement. Keepallpreviousfailedgates anddata.

# Latest heartbeat05:17UTC continued: causalKH60 COMPLETE/audited; real adaptive positive tradeoff, not full gate pass

All60success0collision; raw16691cyclesauditPASS incl actualK/H andindependenttriggerreplay. Auditfirstfailed atindex50step79 becausefloat32policyxynearpolylinevertexchangednearestsegmenttie/variation by7.5deg whilebudgetstillshort. CorrectedAUDITONLYusespreviousfullprecisionstate underverifiedgroundtruthpose/noodomnoise/zerolatency +assertagreesfloat32obswithin5e-7; alltriggerdecisionsrecomputeexactly. Frozenrun/controller/results unchanged. No newqueryorhiddenstatewasintroducedtothepolicy. Noactiveprocessaftercomplete.

Singleturnjoint RMSE9.860mm/1.724s vsfull100H36 13.052/2.069; reverse14.965/2.282 vs17.431/2.902. Realjointreduction24.5%/14.2%RMSE and16.68%/21.36%compute. H-only8.415/1.794 and14.089/2.393: moreaccuratebut~4–5%morecompute thanjoint. K-only12.973/2.020 and19.181/2.799. FixedK64H36 11.399/1.777 and18.539/2.527, sojointmeanbetterboth vsK64bothshapes. Nominal/RLHstillmuchcheaper. Don'tclaimjointdominatesH-only orKnecessaryfromthese3seeds.

Predeclaredjointgate singleturnFAIL>=20%computesaving (actual16.68%), reversePASS. Cannotrenameallpassorretuneonthesameseeds. Resultsareoneblock3newdevseeds,notindependentmulti-modelconfirmation. Full20cells/gates/pairedfactorialeffects desktop/在线KH60, currentlyfilename固定KH完整结果.md. Next worthwhile: counts/dwelltimes/numericalboundarysensitivity ofobservabletrigger, perseedjointvsHonly/K64 comparisons, confirm exactactualfullK/Hbudget instrumentation ifneeded. Then decideboundedfreshvalidation scope honoringfailedgate; no automaticlargeRLtraining/promotion. Useroriginaljointgoalnowhasactualpilotdata, unlikefixedsurrogate756, butnotcompletepaperclaim.

# Latest heartbeat05:17UTC: phase/seed evidence complete; actual causalKH60 ACTIVE

kh_phase_evidence.py complete: atK100 H36 improveswholeRMSEall6shape/seedpairs;atK32singleturn2/3better butone+7.50mm reversesmean,reverse0/3better. Phases: longerH improvesentryallseeds, oftenworsensbend/connector; descriptiveonly, notcausalmediation. Allrecordsseed_and_phase_evidence.json copieddesktop/KH识别42.

ACTIVE causal_kh_pilot.py session29689,~30/60atcheck. Protocolicra_causal_kh_pilot_2026-09-10.md frozenbeforelaunch: fullmodelonlinejoint32H16vs100H36, H-onlyK100fixed,K-onlyH36fixed. Sameobservabletrigger: observedposeprojection onto suppliedreference; maxwrappedtangentvariation next.56m >15degrees requests short. No hiddenmass/shape/seed/futurestates. Constantsnotretuned. Counterintuitive shortnearcurvature motivatedbypriorphasefailure butmayfail; notnewtheory/optimality. Fullresidualmodelalways; NOsurrogate. Newseeds12720001-3,2shapes,standardmass,original,block0. Tenmethods5fixedfullbudgets+nom28+matchedRLH+3adaptive=60. Common10successgatepassed. LogsactualK/Hdecision andinputpolicyobservation; schedulingtimeincluded. PreserveinitialcfgbeforemutableKchanges.

QUEUED assess_causal_kh_pilot.py session91782 after60: rawconfig/K/H/RLactor/RMSE/timer/sourceaudit, independenttriggerrecompute fromrecordedfloat32policyxy*5 (mayneedprecisioncheckifboundary mismatch), all20cells andpairedfactorial effects+jointgates (>=20%saving<=1mmvsfull100H36bothshapes,success,andnosingeaxisdominates). Outputsdesktop/在线KH60. ScriptgeneratedfromassessKH42; scopedtextupdated,butfilelabel固定KH完整结果.md retained. Do noteditrunningfrozensources orlaunchheavyanalysis concurrently. Finishwholebatch beforeinterpretation. A failedpolicydoesnotjustifyretuningonthesesameseeds.

# Latest heartbeat05:03UTC: full-model KH factorial42 COMPLETE/audited

Implementedindependentkh_factorial.py fromserialgeometryrunner,protocolicra_kh_factorial_2026-09-10.md frozenbeforeexecution. FourfullICODEK32/100 xH16/36 treatments +fullK64H36,nominal28,matchedRL-H;twoexistingcommon-successshapes,standardmass,originaldirection,block0,3freshdevseeds12620001-3. All42success0collision; assess_kh_factorial.py audit11703cyclesPASS exacttemplates/Koverrides,actualH,RLactor,rawRMSE/timers/sourcecheckpoints. RuntimeHoverrideisactual16/36 evenbasecfgstoredhorizon36; don'treadbasestoredHastreatment. Noactiveprocessaftercompletion.

Singleturn RMSEmm/compute:K32H16 15.291/.858;K32H36 15.548/1.633;K100H16 15.837/1.087;K100H36 12.714/2.131;K64H36 11.429/1.870;nom28 14.133/.522;RLH16.668/.641. Reverseturns:19.254/1.242;20.847/2.219;19.750/1.610;17.805/2.905;K64H36 18.233/2.598;nom28 20.095/.729;RL22.256/.929. MeanH36helpsatK100 butnotK32inbothshapes; interactionisdevelopmentclue,notstatisticalproof/onlineadaptation. K64dominatesK100H36onsingleturnmean,nominaldominateslowKsettingssingleturn. Preserveallstrongcontrols.

Outputkh_factorial_2026-09-10/analysis all14cells andperseedfactorial_effects.json copieddesktop/KH识别42. NEXT inspectpairedperseedconsistency beforeexpanding, testcausalobservablewithinroutebudgetbenefit (notshape-labeloracle), strongK-only/H-onlycontrolsifnewpolicy. Currentfixedsurrogate756doesnotprovejointKH. Do notlaunchlargeRLbasedonly3seedmeans; currentboundedfactorialidentifiesmissinginteractionaxisandcanfail.

# Latest heartbeat04:51UTC: fixed-K evidence reanalysis complete; H interaction unidentified

inspect_fixed_budget_frontiers.py analyzedALL12geometrycells onlyfullICODEK32/64/100 atH36. AccuracybestcountK32=1,K64=3,K100=8; computecheapestK32=12; meanPareto membership12/9/8. All108pairedseedrows preserved (3methodpairs x12cells x3seedmeans over3blocks). Outputkh_scope_diagnosis_2026-09-10/fixed_K_frontiers.json +固定预算证据诊断.md copieddesktop/KH证据范围. This isposthocfixedbudgetevidence, notonlineadaptation; modelmasscontextnotobservablepolicyinput. Noactiveprocess/newexperiment.

Importantmissingidentification: fullmodelH isalways36 in756, so nofull-modelH effect orKxHinteraction. NominalH28/36andRL-Hwithterminalvalueaugmentation cannotreplacefactorial. Nextreasonableboundedwork: predeclare smallsame-modelKxHdevelopment inalreadycommon-successgeometry withfreshseeds, strongK64control,nominal/RLH; NOTa hugeRLcampaign ornewclaimsfromexistingfixedsurrogatedata. Earlierautonomousauthorizationpersists; neednoextraapprovalforboundedtests. Do not selectposthoccelloracleasdeployablemethod. Originaljointgoalunproven; don'trelabelfixedaccelerations.

# Latest heartbeat04:39UTC: user challenges missing K/H adaptation; scope audit complete

Latestuserasked论文支持then“什么意思，你没有调KK.H?” Explicitly explained ourpositive756 is FIXED K100/H36, notjointK/H; ~35–38%savingcomesresidualbatchapproximation. Priorpivotwasnotclearlycommunicated; don'tcontinuepresentingasoriginaljointbudgetcontribution. Needhonoruserexpectationswithoutinventingresults orassumingsilentacceptanceofpivot.

audit_geometry_budget_scope.py completed all108our+108RLepisodes: oursconfiguredK100andallactualH36,0episodesHchanges;RLconfiguredK100,actualH8–50,108/108episodesswitchH. Outputsgeometry_confirmation/analysis/budget_scope_audit.json +研究范围更正_KH.md copieddesktoproot. No newKHexperiment,noactiveprocess. Readpriorprotocoldeadlinepivot andtemporal+historicaljointfailureentries; priorcompoundfulljoint2/24and7/24 vsH-only6/24and3/24,alltimeoutnotcollision; no blanketjointsuccessclaim. Currentpositivefixedsurrogatevaluepreserved.

NEXT meaningfulKHwork must first auditwhetherbudgetquality/computeorderingactuallychangeswithobservablecontexts, usingmatchedfixedfullK/H controls incommon-successdomain; don'tjustappendpolicyandreusefixeddata. This is a necessary hypothesis notyettestedinthecurrentcorridorstudy. UserdidnotexplicitlyordernewlargeRLtraininginlastquestion, so avoidunreviewedcostlycampaign orclaimscopeagreed. Existingautonomousmandatecontinuesforbounded,evidence-basedwork. Do noterasepriornegative results orreslabelcurrent756. No finalpaperreadinessclaim.

# Latest heartbeat04:19UTC: geometry756 COMPLETE;12/12gates PASS; full figures delivered

All756success0collision, ours108/108. Raw211162cycles auditPASS; independent378mirrorconfigpairs+seedaveragingPASS. All12percell gatesPASS: full100compute savings34.8447–37.7050%; RMSEdifferences-.419353 to+.567521mm. Do not generalize beyondthisstage oreraseprior720failedcell/new162mirrorlight+2.77mm.

finalize_geometry_evidence.py DONE, threePNGvisuallychecked/layoutQApassed andPDF/SVGexported; complete84cellreport+756CSV/profile+pairedmeans+allrawanalysisfiles copiedandhashverified to C:/Users/lenovo/Desktop/ICRA_2026-09-10_最新720对比/跨场景756; open00_当前可核验结果.md. verified_claims.json hasallcomparisons: vsnom28accuracybetter12/12 by12.87–31.42% butcompute2.51–2.67x; vsnom36accuracybetter12/12 butcompute2.18–2.30x; vsRLHaccuracybetter12/12 by12.55–46.00% butcompute1.70–2.20x. vsfullK32betterRMSEANDlesscompute12/12, compute14.54–18.93%less. vsK64faster12/12,betterRMSE10/12 (two worse up to2.81%). Thesearecellmeans,notindividualseedwins/statisticalequivalence.

Noactiveprocess aftercompletionverified. Positive dataset substantive; informuserwithfilepathandqualifiednumbers. Next meaningfulwork: consolidatecurrentfrozenmethod+allnegativeablations into paper-ready evidence argument, strongerclosest-efficiencybaseline/novelty assessment (RTNeuralMPC,KoopmanMPPI,ADO primary leads), verifyactualruntimeK/rollout semantics ifneeded. Do not endlesslyredesignmerelytoeliminatesmallprecisiontradeoffs; no unfoundednoveltyclaim. Userautonomousresearchgoalremains; don'tpauseheartbeat. Needactualmoreusefulfollowup thanidenticalgrids. Currentpostivegeometryfigcaptions statependingvectorreviewhistorically butvisual_review.json nowPASS andvectorsdelivered; cancleanthatcaptiononnextartifactupdate. FinalpaperreadinessNOTestablishedby12/12numericalgatesalone.

# Latest heartbeat04:01UTC: geometry648/756 still executing normally

Verified648/756progress; activeevaluationPID750364, rawaudit750514,pairedmirror799753,andplot882790waiting. No trainingprocess or concurrentheavyanalysis. No completedwholebatch/analysisyet; don't prematurelyinterpretincompletecells. Allrequired nextaudit+figurework queued, no duplicatejob or implementationchange needed. Once756complete review all12gates/84cells/pairedseed evidence, inspect3PNG+grayscale,exportvectors/copydesktop. Usergoal remainsactualpublishableevidence, not completion count.

# Latest heartbeat03:49UTC: geometry402/756 healthy; full figures queued

RealPID750364 evaluation ongoing402/756 atcheck; rawaudit750514 andpairedmirror799753waiting. No trainingorheavyanalysis concurrently. Addedplot_geometry_confirmation.py session20747 waits mirror_and_seed_evidence.json (thus afterall756+bothaudits), then all756CSV/profile, two7method6celltradeofffigures byshape and12cellseedmeanpairedplot. Compilespassed; no sourcechangeinrunningexperiment. Figurepreviews+layoutQA saved geometry_confirmation/analysis/figures; must later inspectPNG/grayscale,exportPDF/SVG,copydesktop aftervisualQA. No interimperformanceconclusion, no batchrestartneeded. Continueafterwholedata; preserveall12gates/failures/strongcontrols.

# Latest user status: geometry756 ACTIVE, common7 PASS, all RL complete

Verified three matching RL policies done (third30319steps/105episodes). No trainingprocess remains. geometry_confirmation activePID750364, standardrawauditPID750514 andpairedmirrorwaiterPID799753 waiting. Latestprogress172/756 at firstcheck; common7methods allsuccess0collision. Do not use singlecommonseed numbers asaggregateadvantage. Stage executingserially, no newheavyanalysis or sourceedits. Continuefull756, thenqueuedaudits andall12cellcomparisons; no interimaccuracybasedselection. Figuresfor756notyetqueued/created; existing720+162figureson desktop. Goalnotcomplete, no needrestartoraskuser.

# Latest heartbeat03:26UTC: RL two seeds done, third active; paired/mirror check queued

Verified realprocesses: trainingqueue19717 healthy withseed12401003, geometryconfirmation80117 andrawaudit72714 waiting. RLseed12401001complete30087steps/107episodes;12401002complete30266/109;third~17473stepsatcheck. No error or timedjob yet, no duplicate launches. Addedgeometry_paired_evidence.py session20065 waitsforauditcomplete: independentlymatches378mirroredfullconfigpairs (walls matched by reflectedposition since sideorderingreverses), reportsALL72comparisoncells as3seedmeans over3blocks, keeps all216seedrows and no9independentscenes/pvalues. Copiesdesktopaftercompletion. Noheavytaskconcurrenttraining. Nextcheckhandoffinto756commonanchor; don'tcelebratebeforematchedRLsuccessgate. Three12cellfigures/fullnoveltygapsremainafteraudit. Researchgoalincomplete, processesactuallyactive.

# Latest update: geometry12 PASS/render checked; matched RL ACTIVE;756+audit QUEUED

geometryqualification12/12success0collision, actual4panelMuJoCorender visuallychecked (walls/reference/robot/openexit), visual_review.json saved, desktop 新场景_MuJoCo.png. Training8stepsmoke passed. ACTIVE qualify_geometry_and_train.py --train session19717 serial RL seeds12401001/2/3 x30k. Latestseed1~13320steps, no failure. Entireexistingmatchedarchitecture/reward;new geometry/source/protocol frozen inindividualtrainmanifest. No timedcomparison untilall3finish.

QUEUED geometry_confirmation.py session80117 waits rl_geometry_mass/all_training_completed.json, then common7methodsingleturnanchor first (stopandpreserve ifanyfails), then756 total:2shapes x2directions x3mass x3blocks x3freshseeds12520001-3 x7methodsnom28/36,matchedRLH,full100/32/64,frozen100. Protocolicra_geometry_confirmation_2026-09-10.md written; source+checkpointfreeze atstart. No recenter method inthisstage. Newgeometry singleturn differsslightly inwallmesh fromold720; allmethodsusecommongeometry. Allnongeometricinheritedsettingspreserved. Route lengths differ, comparemethodswithinshapenotrawpoolcausality. H36exceptnom28/RLH, allEuler andsharedsafety.

QUEUED audit_geometry_confirmation.py session72714 waits756, exactdeclaredtemplate+geometry/sourcehash/checkpoints/rawRMSE/success/timers/RLactor/H/residual108 audits, all84cells +12percell1mm/25%gates+all648pairs, meanpercyclecomputems andepisodep95, fullreport+desktopcopy. Bothscriptspy_compilepassed. Need lateractualK/runtime sampling andstrongcontrol interpretation, newfigures+pairedseedaveraging, geometryreflection verification, model novelty review. Do not duplicate or edit activefrozen scripts. Aftercompletionassesswhole12cells, not just favorableS-turn. Maintain10minheartbeat(activeverified); userrequiresautonomousprogress. Data goal NOTachieved merelybyqueuedwork.

# Latest update: recenter162 rejected/audited, figures delivered; newgeometry qualification ACTIVE

recenter162 complete162/162success; rawaudit35627cycles passed exact inheritedconfigs/source/checkpoints/RLactor/H/timer/RMSE/refinement. weighted meanRMSE11.188 vsfrozen11.371mm butcompute1.588vs1.280s (~24%overhead), uniform11.070/1.594. Weighted improves4/6cells but notuniformmean and failsdeclaredscreen; DO NOTexpandthisrevision. Retainfrozen candidate. Newdev mirrorlight frozen-full+2.770mm: prior5/6pass is confinedto old720, don'tclaimnewseedsguaranteednearlossless. All54cells/assessment preserved recenter_pilot/analysis and desktop.

Three720/162allcontroltradeoff+pairedseedfigures rendered/visuallychecked,layoutandfilesPASS,PDF/SVG/PNG/grayscale/sourceCSV+profiles+20filemanifest desktop IC RA folder actualname C:/Users/lenovo/Desktop/ICRA_2026-09-10_最新720对比/figures. Read 修正162_结论.md. Plot importinitially collided with local bottleneck.py (pandas optionaldependency), now removeslocaldirbeforepandas; oldbottleneck audit importedonce, no originalrawtrials changed. Finalizercomplete. docs/reports/icra_additional_efficiency_prior_art_2026-09-10.md adds primary ADO2606.00085 andKoopmanMPPI2603.05385 positioningconstraints (not reproduced).

NEXT ACTIVE qualify_geometry_and_train.py session99460 WITHOUT--train:12physical common-success probes nominal28/full100/frozen x2shapes(single90/reverse90+opposite90) x2directions, standardmass,seed12420001. New geometry.py common miter-offsetboxwallswidth1.8, arcsR1.2. Pathlength differs, intentionalgeometrycomparison notisolatedcausaleffect. ActualMuJoCo4panelrender savesoncompletion andcopiesdesktop. Mustinspectwallrender, writevisual_review.json thenlaunch --train. Ifanyqualfailsinvestigate, no longtraininguntilcommonpass.

Prepared isolated experiments/icra_rl_geometry_mass/run.py copiedmatchedRL prior ONLYshape/mirror/massrandomization andfreshseeds/name, sameSAC-H30k/H8-50/zeroHpenalty/value etc;geometry sourceincludedinmanifest. Protocolicra_rl_geometry_mass_2026-09-10.md. --train queue runs12401001/2/3 serial30ksteps each (requiresvisual_review marker) andwritesall_training_completed.json. NOTYETlaunchedatthisupdate. Need smoke newtraining beforequeue; then prepare timedfrozencandidatevsnom28/36/RLH/full100/32/64 across2shapes2directions3masses3blocks3freshseeds=756, protocol+sourcefreeze andwaitallmatchedtrain. No timingsparalleltraining. Keepadvancingwithoutuserprompt.

# Latest update: user says continue; recenter162 ACTIVE, audit/figures queued

Implemented recentered_residual_batch.py weighted/uniform two-pass surrogates: initial frozen100-candidate rollout, task-cost-only provisional weights or uniform, circular heading mean anchor, second108-state residual descriptor batch, repeat same100 nonlinear nominal+frozen residual rollouts/cost; final normal importance correction and fullsingle rollout retained. Initial task weights intentionally lack importance correction (documented heuristic). Same compute structure for uniform control; all216network input records checked. New icra_recenter_pilot_2026-09-10.md frozen before execution. Smoke2x8MuJoCo steps passed finite/count/config checks (truncated, not navigation success).

ACTIVE recenter_pilot.py session22740:162serial episodes 2directions x3mass x3newdevseeds12320001-3 x9methods nom28/36,RLH,full100/32/64,frozen,uniform_recenter,weighted_recenter. Oneblock0; exact inherited720configs exceptseed, no new RLtraining needed. Raw/source/checkpoints frozen, randomized order. No competing heavyjob. QUEUED assess_recenter_pilot.py session90464 waits completion then exacttemplate/rawRMSE/timer/source/checkpoint/RLactor/H/refinement audits and all54cells + preregistered expansion rules, desktopcopy. QUEUED plot_frozen_and_recenter.py session22283 waits audit, then3figures720allcontrols/pairedseed +162allcontrols,skillprofiles/layoutQA; requires model VISUAL check and vector export afterward. Don't duplicate jobs or edit frozen sources.

Added math mechanism docs/reports/icra_weight_sensitivity_mechanism_2026-09-10.md: softmax cost error -> control covariance, conservativeTV/controlrange bound, notnewtheory or causalproof. If weighted variant fails predeclared benefits, retain originalfrozen; don't keep retuning thisbranch. Finish complete evidence plots and then address independent variedgeometry/novelty gaps. User explicitly wants continuous action and final data, not idle analyses. Current latest desktop contains720tables only; new162willcopyautomatically afteraudit.

# Latest update: latest720 full-table/paired evidence exported to desktop; bounded revision plan

export_frozen_evidence.py DONE: current full48cell table, allpair data/gates copied to C:/Users/lenovo/Desktop/ICRA_2026-09-10_最新720对比/当前结果与后续计划.md. Separate from old package; oldmanifest unchanged. User asked next actions. Plan retain currentbaselinecandidate and 5/6gates as tradeoff, don't obsess over arbitrary gate. Next bounded mechanistic revision: test weight-recentered residual anchors vs nominal-anchor frozen and unweighted recentering control, then existing fullK100/32/64/nom/RLH. NOTimplemented or launched yet. Need exact protocol + smoke before new devseed comparativebatch; not choose conditions based on wins. Need full frontier/paired figures, novelty assessment and variedgeometry evidence before paper. Actual priorweight diagnostic30 complete; no active simulation now. Continue work autonomously from this specific plan.

# Latest update: actual-weight diagnostics30 COMPLETE; no gate relaxation

User asked if one failed cell can be ignored: explained it is 15 paired trials (3 model blocks x 5 seeds), all navigation success, RMSE11.950->13.763mm (+1.813mm, about15.2%) with35.24%compute saving; not one failed episode. Keep failed predeclared1mmgate, can frame accuracy-compute tradeoff with transparent limitation. Do not discard entire line solely for this numerical gate; contribution/baseline evidence still necessary.

diagnose_frozen_weights.py completed30 real MuJoCo replays of all standard-mass local episodes, both directions, every5cycles exact-full vs approximate SAME actual candidates/cost/importance corrections. All30 full state traces match historical within1e-9. No actions/RNG altered; extra instrumentation invalidates timing. Stage frozen_actual_weights_2026-09-10 contains protocol,30logs,completed. Original entry/bend/exit mean weightTV .00706/.01105/.01802, mean abs preoverride deltaomega .00165/.00318/.00520rad/s. Mirror .00704/.01123/.00925 and .00179/.00331/.00303. Original exit has greater action-weight sensitivity despite SMALLER unweighted trajectory error (.443mmvsentry.945/bend1.000mm). Diagnostic supports weight-sensitivity mechanism to investigate, NOT causal proof or final applied action difference. No new improved candidate yet; no active process aftercompletion.

Also audit_frozen_probe_arrays.py passed90/90 savedNPZ discrepancies. Saved all42 mean comparator tradeoffs in mirror confirmation analysis/all_control_tradeoffs.json. NEXT: per-seed weighting-sensitivity and counterfactual actuator/terminal effects; consider globally specified cost-sensitive correction ablation, not direction/seed-specific patch, preserve strong fullK32/K64+nom/RLH controls. Any new candidate needs fresh development seeds and separate protocol, no benchmark speed claims from instrumented replay. Update desktop report with all6cells; existing desktop package outdated.

# Latest update: phase60 and recorded-state approximation90 COMPLETE

frozen_phase_diagnosis.py analyzed ALL standard-mass 60 trajectories (two directions, five seeds, three blocks, full/local). Exact whole RMSE and additive phase MSE checks pass. Original entry/bend/exit mean phase RMSE gaps -.275/+1.112/+2.526mm; mean MSE contributions -1.323/+9.230/+46.360mm2. Exit dominates the added squared error; phase RMSEs are not additive. Mirror corresponding gaps -.184/-.561/+.273mm. Output mirror confirmation analysis/phase_diagnosis.json.

probe_frozen_recorded_states.py completed90 offline synthetic-bank probes: all30 local trajectories, median timestep of each geometric phase, matched previous executed command, H36/K100, fixed Gaussian sigma [.12,.35], full learned vs frozen approximate rollout. Output frozen_recorded_state_probe_2026-09-10 includes protocol, all90 NPZ, results, completed. Original entry/bend/exit mean prediction discrepancy .825/.560/.320mm; mirror .706/.789/.285mm. This does NOT support the simple claim that exit has the largest local approximation discrepancy. It is NOT historical candidate replay, NOT physical accuracy, no speed benchmark, not a new navigation win. No causal fix established. No active process after completion.

NEXT: verify NPZ recomputation then instrument actual candidate banks/cost weights at fixed development replays (all seeds, keep original seeds diagnostic only) to test action-weight sensitivity and cumulative drift. Need cost/action discrepancy rather than just unweighted trajectory error before adding anchor refresh. Frozen source/results/gates unchanged. User expects continuation; do not report a fixed accuracy gate or claim new closed-loop success. Desktop latest updates still pending.

# Latest update: reflection360/config equivalence PASS; failed cell decomposed

Ran experiments/icra_deadline_analysis/decompose_frozen_mirror.py. All 360 matched original/mirror full saved configs agree exactly after prescribed geometry reflection; double reflection identity passes. Outputs analysis/failure_decomposition.json. Original mass1 gap +1.813mm: 10/15 paired trials worse, all three block mean gaps positive (+1.868/+2.537/+1.034mm). Seeds12220003/5 mean gaps +3.674/+5.146mm (all three blocks worse); other three seed means -.284/+.488/+.041mm. This is descriptive shared-seed diagnosis, not 15 independent scene replicates or causal proof. Retain all seeds and failed threshold. No simulation/training running at process check; this diagnosis finished. NEXT: trajectory phase decomposition across ALL five seeds then instrument fixed candidate-bank approximation diagnostics to test anchor bias/action sensitivity before changing surrogate. Do not launch blind grid or claim precision issue fixed. Latest desktop package still needs these results.

# Latest update: mirror720 COMPLETE/audited;5of6numericalgates pass, oneaccuracygate fails

rawauditpassed160604cycles. assessment_completed.json: originalmass.5/+0.407mm/35.08%saving pass;originalmass1/+1.813mm/35.24% FAIL1mmaccuracythreshold;originalmass1.5/-0.928mm/35.22%pass. Mirror.5/+0.345mm/35.22%pass;mirror1/-0.203mm/35.14%pass;mirror1.5/+0.294mm/36.95%pass. Allcandidatefull100successpreserved. No mean-dominatingcontrolinANYof6cells, butdoesNOTmeanuniformdominanceours. Do NOTrelax1mmthresholdposthoc orclaimallgatespassed. Computationbenefitstable,accuracygaplocalizedyetreal.
Noactiveprocess. NEXT: fullreflection/configaudit, per-model/seedpaired decomposition of originalmass1 errorgap; approximationerror sample audit todecidewhetherneedanchorrefresh/correction vsnoise. Publishfull6cells withfailedgate. Addfiguresandlatestdesktopstage. Anyrevisedcandidatehasnewprotocol/freshconfirmation,noteditingfrozenoutputs. Userautonomousredesignrequestpersists; continuefromfailureevidence, noblindgrid.
# Latest update 09:56: allmatchedRLdone; original275/360, mirrorfollows

RLmirrorseeds12101001/2/3completed30189/30164/30145steps,all3trainingprocessesfinished. frozen_mirror_confirmation timedexecutor active original275/360thenmirror360. No failuresdetected. Existingrawauditwaiting.
Addedassess_frozen_mirror.py waiter(compilationpassed):after720rawaudit, separatelyeachdirection360rowschecksbudgets/zeroth108-inputbatch/callcounts,all24cells,fullpairedcomparison,candidatefrozen<=1mm/>=25%predeclaredgates,mean-dominatingcontrols. Outputsanalysis/original andanalysis/mirror reports+combinedassessment_completed.json. Itchecks5newseeds and3blocks. Needlaterfullreflectiongeometry/configaudit andplotsaftertiming. Noheavyjobcompeting. No interimperformanceconclusion.
# Latest update: user demands continuedredesign; fixedcandidate mirror720 queued, matchedRL ACTIVE

Do notstopatexplainingpositive270result. Fixedlocal_frozen chosenoncefromablation; nowfreshseed/mirrorconfirmation, NOTchoosebestorderbypayload. ACTIVE experiments/icra_rl_mirror_mass/run.py session43354:3seeds12101001–3,30keach,bothreflections+mass.5/1/1.5,Euler/.35/1.8m/120s, otherwisepriormatchedRL.8stepsmokepassedincludingmirror. SamezeroHpenaltyadaptation,notoriginalpaperexactreplication.
QUEUED frozen_mirror_confirmation.py waitsALL3trainingcomplete,thenoriginal360+mirror360=720serial (2directions3masses3blocks5freshseeds12220001–5,8methodsnominal28/36,RLH,preview,full100/32/64,local_frozen). Samefrozenzerothordercode;reference/wall/initialposey reflectionviafrozentrainmodulehelper, physicsunchanged. Protocolicra_frozen_mirror_confirmation_2026-09-10.md:<=1mmmeanerrorincreaseand>=25%computesavingvsfull100eachdirection/mass,successpreserve,strongK32/64+nom/RLHreported. Prior270selectiondata notindependentproof. Twohourbudgetperdirection.
QUEUED audit_frozen_mirror_confirmation.py waits720done thenrawsource/checkpoint/actor/H/metrics/timers. Needaftertimingfullreflection/config equivalence,approximation/budgetaudit,pairedcellreportandplots. Noheavyanalysisduringtimedrun. Userwantsbrightdata butdon'tfudgeorclaimnovelty; RTNeuralMPCpriorartgapremains. Autonomouscontinue.
# Latest update: confirmation270 COMPLETE; zeroth-order simpler ablation strongest overall

All270success0collision,standardrawaudit+assessmentcomplete. Affinepassespredeclared3mm/20%gatesallmasses with24.7–29.5%saving, butlocal_frozen mean-dominatesaffine atmass.5/1; affinebetteraccuracyheavy. Do NOTannounceaffineasuniformwinner.
Simplerlocal_frozen fixedcandidate(withoutper-massselection):RMSE9.780/10.999/13.638mm vsfull1009.415/11.380/13.876;compute1.480/1.679/1.633s vs2.353/2.755/2.624. ~37–39%saving,meanerrchanges+.365/-.381/-.238mm. Versusdirectfull32:errors10.543/11.913/15.393mm,compute1.789/2.065/1.974, so frozenmeanbetterbothall3masses. All27frozensuccess. STILLnotdominatingnominal/RLcompute, no independentlargeconfirmation/generalization yet; candidatechosenfromablationrequiresfreshconfirmation, priorRTNeuralMPCnoveltygap remains.
Noactiveprocessatcheck. Userasksstatus; giveactualpositiveswithablationlimitation. NEXT act: verifyfrozencandidatecost/approximationandfullprotectedconfigequivalence, generateall10methodplots andcopylatestdesktop. Then predeclarefreshheldoutconfirmationoftheONEfrozenvariant(notmixmass-dependentbestchoices), withfullK32/K64,nominal/RLHstrongcontrols. Need geometrygeneralization withfairRLtraining/qualification ifchanged; don'thideOODfailures. No newtesthasstarted yet. Keepautonomousworkflowmoving.
# Latest update 08:59: matched seed3 nearing completion; gate assessment queued

Matchedseed12001002complete30136steps;seed12001003~26569atcheck. Trainingqueue26112healthy;confirmationwaitsall3complete. Addedassess_local_confirmation.py waiter: after270rawaudit, checksallK32/64/100+Eulerconfigs, localcallcount/batch1188vs108, all30method/masscells, pairedlocal-versusALLcontrols, predeclaredper-mass3mm/20%saving/successgatesandexplicitmean-dominatingcontrols. Outputsanalysis/完整确认报告.md,predeclared_gates.json,paired_effects.json,assessment_completed.json. Doesnotverifyfullcandidateapproximationerror;anchoroffsetisnotbound. Noheavyconcurrenttimingstarted. Do notduplicatejobs.
# Latest update 08:47: matched RL seed2 training healthy; closest prior-art identified

Seed12001001done30081steps;seed12001002~16k atcheck;seed3followsinsame26112queue. Confirmation/auditwaitershealthy,no concurrenttiming. Do notduplicatejobs.
ReadprimaryRTNeuralMPC arxiv2203.07747v3 fullHTML: alreadyparallel localapproximations toseparateneuralmodelcostfromonlinegradient-basedMPC. Our sampling-basedimplementationdifference aloneNOTprovennovel. Saved docs/reports/icra_local_approximation_prior_art_2026-09-10.md withsource/DOIandexplicitnoveltygap; GPUlinearizationerrorboundsarxiv2607.01203abstractonly leadnotfullyreviewed. No activeprotocolchanges. Keepconfirmationrunningtoassessnumericalvalue,butdon'tequatepositiveperformancewithpublishableinnovation.
# Latest update 08:39: local63 promisingtradeoff; matchedRL training ACTIVE,270confirmation queued

local63all success+standardrawauditpassed. Localaffine vsfull100 compute~2.59s vs3.44–3.72,RMSE.00855/.01183/.01282 vs.00801/.00995/.01318m(mass.5/1/1.5). Versusfull32 .01013/.01183/.01266m and2.79/2.74/2.83s:localcheaper,errornear. Oneblock3seedsNOproof; notdominatingnominalcompute.
ACTIVE RLmatchedtraining experiments/icra_rl_mass_euler/run.py session26112:threefresh12001001–3 sequential30ksteps each, Euler,width1.8,mass.5/1/1.5perepisode,.35cap,zeroHpenalty,otherwisepriorSAC/value.8stepsmokepassed. Sourcefrozen/protocolicra_rl_mass_euler_2026-09-10.md. No timedjobrunsuntilALL3complete.
QUEUED local_residual_confirmation.py waitall3checkpoints then270serial3massesx3modelblocksx3newdevseeds11920001–3x10methodsnominal28/36,fixed36value,RLH,preview,fullICODE100/32/64,local_affine,local_frozen. Newzerothorderablationfrozen_residual_batch.py (anchorbatch108inputs instead1188, noJacobian), finalfullrolloutstillshared. Protocolicra_local_residual_confirmation_2026-09-10.md prerecords<=3mmerrorincrease and>=20%computesavingvsfull100per-mass, successpreservation, inspectstronglowK/zerothdominance. Notnewnovelty/guaranteeclaim. Do notmodifyanysourcewhilewait/execute frozenstage.
QUEUED audit_local_residual_confirmation.py waitstage then270rawactor/H/hash/metrics/timer audit, newRLcheckpointpaths12001001–3. NeedadditionalactualK32/64,approximationquality/config andpercellpairedgateanalysisaftertiming. Trainingqueue cancompletewithinminutes; monitorthenact, no parallelheavyanalysisduringtiming.
# Latest update 08:19: multifidelity72 negative/audited; local surrogate63 ACTIVE

Multifidelity72all success,standardrawauditpassed. audit_refinement_indices.py PASSED4214calls/18MFepisodes, deterministictop32/randomsubset+configuredK32checks. Top32 errors.01334/.01627/.01638m vsfull32 .01086/.01162/.01442 andtopcost~3.2svsfull32~2.8s: rejectscreeningcandidate, notexpand. Allpreserved.
Newcandidate local_residual_batch.py: no candidate discard. Perplanmean-controlnominalanchortrajectory, onebatchedresidualqueryatH36 x(1+2*5stateprobes)x3controlbasis=1188inputs, centralFD.01stateJac fordrift/gain; approximateresidualforall100Eulerrollouts, finalsingleweightedtrajectoryexactfullmodel. Recordsmaxanchorxydeviation, notrustregionguarantee. No corecodechange. This is familiar localapproximationtechnique, NOTnoveltyclaim.
probe_local_residual_batch.py 3fixedbanks(onepredeclaredexistingmodel) smoke maxlearned-modeltrajectorydiscrepancy<.001m. Singlecalltimingsunwarmed(firstfull.064s), DO NOTclaimtheseasspeedup/physicalaccuracy. Data/probe retainedlocal_residual_probe_2026-09-10.
ACTIVE local_residual_pilot.py:63serial3massx3newdevseeds11820001–3x7methodsnominal28/36,RLH,preview,fullICODE100/fullICODE32,local_affine. Commonwidth1.8,.35speed,Euler,H36exceptbaselineH,120s,oneblock. Protocolicra_local_residual_pilot_2026-09-10.md. QUEUEDaudit_local_residual_pilot.py waitsfor63 thenrawaudit. Needapproximationquality/budget/source/configaudits andsamebudgetcomparisonbeforeexpansion. Currentadaptiveheuristicoldcandidatesallnegativesretained. Userrequiresautonomousredesign; noidleafterbatch.
# Latest update: user requests self-continuation/redesign; multifidelity72 ACTIVE

Automationicode-k-h UPDATED ACTIVE10min; prompt explicitlydoesnotstopatoldartifactpackage, latestdirection/completebatch/evidence-basedredesign, no fabricatedprogress. Userdoesnotwanttocueeachiteration.
Newmultifidelity_screen.py heuristic:100nominalcandidatecosts, refine32nominaltopcost orrandom32withfullICODE, largefinitecostexcludesunrefinedfromupdate; originalimportancecorrectionremainsafter. Finalsequencefullmodelrollout. Biasedsubsetestimator,notunbiasedfullMPPI/noveIclaim. Boundarycandidatefilterassertdisabled; safetyunchanged.
Newbudgetenvironment copiedpriorRLzero sourceONLYconfigure_budget nowusescfgnum_samplesinstead100 sofullK32controlactuallyK32. Smoke24realMuJoCocycles passedtop/random/direct32;topindicesandunique32verified, actualK32verified. No originalsrcchange.
ACTIVE multifidelity_pilot.py:72serial,width1.8/speed.35,mass.5/1/1.5 x3freshdevseeds11720001–3 x8methodsnominal28/36,RLH,preview,fullICODE100/fullICODE32,top32/random32refinement. AllEuler,H36exceptspecifiednominal/RLH,onepredeclaredmodelblock. Protocolicra_multifidelity_pilot_2026-09-10.md; source/checkpointsfrozen. CompareagainstlowKANDrandomsubsetbeforeclaimingbenefit.
QUEUED audit_multifidelity_pilot.py waitscomplete thenstandard72rawaudit. NeedADD refinementtopindices/randomdeterminism/count, actualK32config/runtimeaudit,andwholematchedconfigaudit aftertiming. Fullnegativehistory72retainedandscoreauditpassed; don'texpandit. Newcandidatecanfail, thenredesignbasedonmechanism; do notpausewaitinguserafterbatch.
# Latest update: history72 COMPLETE and score replay audited; no stable advantage

All72success0collision. Standardrawauditpassed. Newaudit_history_scores.py independenthistory/delayedcommand reconstruction passes387scoreupdates/9history20episodes; retrospective scorelogicimplementedasdesigned. ICODEusagefraction averages~.292(light),.593(normal),.748(heavy); selectorrespondstopayload butnotstablecontrolbenefit. Atmass1 history20 .01325m/2.679s vsinnovation.01138/2.575 (dominated); mass1.5history.01459/3.097 vsoldpreview.01443/1.859 (dominated). Lowmasshistory.01029/1.917 slightlybetterthanpreview.01107/1.940 butnotgeneral. Do NOTexpandhistory20 aswinner. Noactiveprocess.

Userexplicit '只要不理想就重新设计' persists. Need next redesign target objective mismatch: retrospective model-prediction accuracy isn't guaranteed MPPI action improvement. A bounded diagnostic of past20 rank vsfuture20rank on recordedcontrolwindows canmeasureselectorpredictionrelevance, withoutusingfutureinformation incontroller. Alternatively cost-aware multi-fidelity sampling/refinement is a new implementationcandidate but requires strongsamebudget/lowKcontrols and noveltycheck; don'tclaimnewpaperidea simplybecausecoding. Preserveallnegativehistoryresults. Currentstatusreply should reportcompleted72, two dominatedcells, scoreauditpass; notclaimrunningnewexperimentuntilactuallylaunched.
# Latest update: regime probe COMPLETE; mechanism-supported history-selector72 ACTIVE

model_regime_probe108physicaltraces/1080errorrows completeandrawerrorauditpassed. AtH36leftturnmass.5 nominalposition.030m vsICODE.1255; mass1.5 nominal.1214 vsICODE.0334. Rightturnsimilar .030vs.1425 and.1214vs.0472. ShortH10doesNOTnecessarilyshareordering. Speed63previoushypothesisnotconfirmed; no speedexpansion. Entiremodelprobe/reportpreserved. Nominalmodelblockrepeatsnotindependentreplicates; probeNOTnavigationwin.

New independent heuristic history_model_selector.py: retrospective20stepnom/fullEulerprediction ofpastobservedtrajectory underknownaverageddelay/safety-sentcontrols, refresh5cycles. Scorexy²+(.25heading)²; defaultnominal; alternatewinsifscore<.8other. Causalnohiddenmass. Codefrozenwithnewpilot. Doesnotclaimnovelpapercontributionyet.
ACTIVE history_selector_pilot.py session99373:72serialcases,width1.8,speed.35,mass+inertia.5/1/1.5,3freshdevseeds11620001–3,onepairedmodelblock,8methodsnominal8/28/36,RLH,fullICODE,preview,innovation(single-step),history20. ALL Euler,K100,common120s+safety. Protocolicra_history_selector_pilot_2026-09-10.md. RLmassOODdiagnosticnotmainclaim. Needfullresultsbeforeexpansion, compareinnovationandfixedstrongcontrols.
QUEUED audit_history_selector_pilot.py waitscomplete then72rawRMSE/actor/H/hash/timer/masschecks. ADD independenthistoryscore/delay/selection replay andmodel-selectionfractionsaftertiming; notyetimplemented. Hypothesisrequiresgoodselection ANDusefulperformancetradeoff; noautomaticdynamicmassscenes. Newuserrequestedredesignandcontinue; actualexecutionresumed, noidleafterartifactbundle.
# Latest update: redesigned speed63 COMPLETE; model-validity probe ACTIVE

speed63all success/rawauditpassed/commongatepassed. Actualp90speedincreases(.33->.43->.46) forALLmethods. v*omega is kinematic demand proxy, NOT independently measuredlateralaccel; analysis wordingcorrected+rerun. Hypothesis ofmonotonicresidualbenefit NOTsupported: atcap.65 nominal8 .01890m/.488s vspreview.01926/1.578 andfullICODE.02063/3.364. RL-H OODstillstrongbutnotqualifieddomaincontrol. Do notexpandthisspeedgrid or claimvictory; report full21cells underanalysis/速度需求报告.md.

NEXT ACTIVE model_regime_probe.py session55248: mechanismtestbeforeanyswitchingnavigation, freefloorMuJoCo commonconstantcontrols, mass+inertia ratios.5/.75/1/1.5,3models3newdevseeds11520001–3,3commands(.3,0/±.45),20warmup+36predictionsteps,Euler.108physicalprobes1080errorrows. No navigation/successclaim, nohiddenmasspassedtoplanner. Protocolicra_model_regime_probe_2026-09-10.md. Goal findwhethernominal/fullaccuracyordering actually crossesdomains; do NOTdesignswitchingexperiment withoutthisnecessaryevidence. Allcasesretained.
QUEUED analyze_model_regimes.py waits30minthenrawerrorrecompute108files, all24H36cells, audit/report. Reviewcomponentwise(allunits separate) andmodel-level consistencybeforeaction. No guaranteednewcontribution. Userexplicitcontinue: proceedfromevidence, notidle becauseoldpackagecomplete.
# Latest update: USER explicitly requests redesign and continue; speed-demand pilot ACTIVE

Fresh instruction supersedesidleartifactcomplete status. Appliedexperimental-design. New hypothesistest: increase actualdynamictrackingdemand withoutnarrowingwalls. speed_demand_pilot.py session3307:63serial cases, fixedwidth1.8m, speedcaps.35/.50/.65, freshdevseeds11420001–3, onepredeclaredmodelblock, sevenmethods nominal8/28/36,ICODE,preview,turn,RL-H. ALL Euler,K100,sameplant/safety/120s. Source+protocol+checkpoints frozen. Actualmean/p90speed andp90|v*omega|recorded toverifyintervention ratherthancaponly. Protocolicra_speed_demand_pilot_2026-09-10.md.
RL-H trainedonly.35, highspeedOODdiagnosticNOTcontributionevidence; if hypothesis supportsnextstage, matchedspeedtrainingrequired beforeformalRLcomparison. No resultpromised. Firstgate.35all3successpermethod; noexpansionifgatefails.
QUEUED audit_speed_demand_pilot.py session76697:63source/actor/H/rawRMSE/timers. QUEUED analyze_speed_demand.py waitsforrawaudit then21cells+all3seedpairedresidual-minusnominaldifferences, actualspeed/lateraldemandmonotonicchecks,common-successgate; outputsanalysis/速度需求报告.md,demand_analysis_completed.json. Reviewwholebatchthenact; don'tpauseafterbatchjustbecauseartifactsdone. Neverlaunchcompetingheavyworkduringtiming.
# Latest update 03:20: paired scene-seed direction evidence added

paired_obstacle_evidence.py computespairedpreview-minusALL5controls within3modelblocks thenaveragesperseed; reports15cells5seedeach, no pseudoreplicated15independentsamples,no pvalues. PreviewcomputehigherthanALLnominal/RLcontrols onALLseeds atALLwidths. Errorvaries: vsnom28width1.8 previewbetter3/5, width1.4only1/5; vsnom36width1.0previewbetter2/5. Mean-dominanceisNOTevery-seeddominance. vsfullICODE previewcomputealwayslower, errorworseall5wide/mid,better3/5narrow. Addedpaired_evidence JSON/ChineseMD+indexlink; refresheddeliverymanifest PASS74files. Noactiveprocess/newexperiment.

Allrequiredartifactworkcomplete; originalstrongnovelpaperclaimunmet. Keep scientificscopehonest, no needlessmoregrids/audits. Heartbeat10minactive,userasleep; onlynewjustifiedworkornewusersteering, no fakeprogress. Desktopstart00_先看结论.md includeslatestnuance.
# Latest update 03:08: complete evidence package VERIFIED (71 files), no running job

verify_morning_package.py PASS:71files source/desktopSHAparity, allrelativeMarkdownlinksvalid, counts270+135+384+36+72 andexistingrawauditstatuseschecked. delivery_manifest.json existsbothcopies withexplicit status: evidencepackagecomplete, publishablecorecontributionNOTestablished. Unifiedindexincludesreplay+draftlinks, fixedliteralbacktick-n. Desktop C:/Users/lenovo/Desktop/ICRA_2026-09-10_成果包/00_先看结论.md isstartpage. Noactiveexperiment/analysisprocess.

Do not falselysay897newovernighttrials:135+384werepre-existing; newovernightcomparative270+36+72=378, plusinstrumentedprofile3(nonbenchmark). RLbaseline360alsoalreadydonebeforeuserbedtime. All scopes inreports. Reportpackageisfinished; researchobjectiveisnot. If continuingautonomously, onlydo substantivelyjustifiednextwork; do NOTrepeat audits/grids or fabricate progress to satisfyheartbeat. No newmethod currentlysupported. Usermayreviewmorning; heartbeatremainACTIVE10minasrequested, no routine notificationswhileunchanged.
# Latest update 02:59: methods/results research draft CREATED

Usedml-paper-writing skill; saved研究草稿_Methods_Results.md toartifact+desktoppackage. CoherentEnglishmethods/results draft withnominal/residual/selector equations, cost-accounting model, upper-boundlimitation, RL-H adaptationdetails, all270obstacle/135+384tracking/36+72integratordiagnostics, inputsemanticscorrection andexplicitunsupportedclaims. NOTsubmission-ready/fullnoveltypaper. Sourcesreverifiedwebprimaryarxiv1509.01149(WilliamsMPPI),2102.11122(BohnRLH),2605.03260(ICODEMPPI); directlinks, no fabricatedBibTeX. No newexperiment/activeprocess.

Fullnightdeliverableexists ondesktopincludingresults/figs/replay/draft/provenance. Corepublishablecontributionstillunproven; don'tmanufactureone. Remaining worthwhilecleanup: unifyindex/sourceartifactanddesktopcopies, verifyallrelativefilelinks/reportnumbers/audits, addmachine-readablecompletionmanifest. Userexplicitlyrequestedcontinuousnightwork; avoidrepeatinggridsorclaimingnewprogresswithoutaction. Heartbeatstill10minactive.
# Latest update 02:44: actual MuJoCo four-method replay CREATED and verified

render_comparison_replay.py complete: width1.0,firstdevseed11220001,block0,nominal28/RL-H/preview/fullICODE, synchronized planarpose replay withphysicalwallsvisible. Usesrecordedtraces,NOTnewphysics; wheelfullarticulationnotreconstructed;8xspeed,holdterminalposeafterfinish. GIF decoded/midpoint/finalframes visuallychecked. All4finish. Provenance includestracehashes/runindices/predeclaredcontextchoice. Desktop/mujoco/comparison_replay.gif withmid/finalPNG,replay_provenance.json,linkedfrom00_先看结论.md. Noactiveprocess.

Mainresearchconclusionsunchanged. No strongnewcandidateevidence; avoidmanufacturingsuccess. Remaining usefulovernightwork: coherent methods/experiments researchdraft grounded inallverifiedresults, paired effectuncertaintyifappropriatewithoutpseudoreplication, sourcecitation check. Fullmanuscriptmustnotinventnovelcontribution; writeexplicitdevelopmentreport/draftwithgaps. Use relevantpaperskillifdrafting. Userasleep, heartbeat10minactive, no routine notification.
# Latest update 02:34: consolidated decision index written; prior canonicalization found

Desktop00_先看结论.md nowlinksallreports/actualMuJoCographs/fullcontrols/corrections andexplicitlyseparateswhatcanbecitedfromunsupportedclaims. Sourceaudit showedpositioncanonicalizationalreadyexistsinconfigs/research/expanded_navigation_development_l218.yaml andconfigs/rl/residual_conditioned_expanded_maps_l219.yaml (x/ytrainingmeans), so don'trepackageasnewmethod. No new experiment or activeprocess thisturn.

Currentoriginalgoalofpublishablecorecontributionunmet; completeevidencepackageexists. Avoidendlessgrids/unsupportednovelty. Useful remaining overnightdeliverables: a reproducible actualMuJoCo trajectory replay video with visible walls fromfixedselectedepisode (labelreplay,notnewexperiment), and coherent methods/experiments draft constrainedtoverifiedclaims, if it helpsuserreview. Needpaperwriting skillforactualmanuscript. Do not claim success or pause heartbeat withoutuserrequest. Continuefromevidence; userasleep,no routineupdatesneeded.
# Latest update 02:19: input-alignment audit COMPLETE; double-delay hypothesis REJECTED

IMPORTANT correction toprevioushandoff: datasetcontrol_t is APPLIED interval-average delayedcontrol, NOTcommanded. validationmetadata control_source=applied; all744entriescontrol_t==applied_control_t exactly; model_versiondynamic_unicycle_5_applied_input_v1. Builderbuild_l38_onpolicy_residual_dataset.py appliessourceexplicitly. Runtime_prediction_controls averagesprevious/currentcommand with.04/.1 delayfraction, hence inputsemanticsalign. Do NOTclaimdoublecompensationbug orchangecommand_delay_s onthatbasis.

Corrected audit_training_discretization.py assertions+scopeandChinesereport; reranresultssame, sincealwaysusedrawcontrol_t. Added 控制输入语义更正.md desktop/training_discretization explainingearlierwordingerrorandunchangednumbers. No new test/unseenread. Noactiveprocess.

Both simplesourcemismatchhypotheses(Euler-onlytraining,double-delaylabels) disproven. Currentcandidatehasconditionaltrackingtradeoffbutno broadstrongbaselinewin. Next meaningfulwork should be a consolidated evidence/claim decision and bounded method hypothesis using actual sensitivity, not another speculativeimplementationfix. Desktopfullpackage containsallnegativecontrols. Userwantsnightwork; stilltimeformethoddesignbutdon'tfabricatecontribution or runrandomgrids.
# Latest update 02:07: training-discretization hypothesis checked; simple mismatch explanation REJECTED

FrozencheckpointsALLtrainRK4withderivative/one-step/36-step weights.25/1/4. Derivativelabelsverifiedexactfinite-difference, residualtarget=observed-nominal exact, butNOTEuler-onlytraining. audit_training_discretization.py verifiesdatasetvalidationSHAandcheckpointSHA; usesALL19nonoverlap36stepwindowsfrom3validationepisodes,3models,RK4/Euler,commandedcontrols, horizons1/5/10/20/36. No test/unseenread. H36positionRMSE RK4 .05709/.05252/.05367m vsEuler .06028/.05517/.05644: RK4betteropenlooppositiononall3models. Eulerbetterclosedloopcannotbeexplainedsimplybybetteropenlooppositionfit. VelocitychannelEulerbetter; reportcomponentwise,noall-stateclaim.

Outputs morning_package/training_discretization/{训练离散化核对.md,results.json,provenance.json} copieddesktop. Noactiveprocess. This killsanunsupportedmechanismstoryratherthanclaimingnewcontribution.
NEXT possible concreteaudit: trainingrolloutusescontrol_tdirectly; runtimeplanner hascommand_delay_s=.04 and _prediction_controls mixesprevious/currentcommand. Determinefromdatasetcollector/provenancewhethercontrol_t is commanded/sent/applied and whethertraininghandlesdelayelsewhere before claimingdoublecompensation. Currenthypothesisonly. A validation-only inputalignmentcomparisoncouldbejustified aftersourceaudit; preserveoriginalmodels/safety, no controllerretuningyet. Continueevidence-groundedwork, no geometrygrids or publication claims. Desktoppackagecompletedbutoriginalgoalofstrongpaperstillunmet.
# Latest update 01:55: both integrator diagnostics COMPLETE, audited, figures packaged

tracking72all success; rawauditpassed. finish_integrator_diagnostics.py audits54matchedEuler/RK4configpairs acrossobstacle36+tracking72:onlyplanner.integrator andselector_mode differ, pass. Exportedpercelltables,sourceJSON,pairedmetricchanges,PNG/PDF/grayscale,Chinese积分器完整报告; bothfigures visuallychecked. Copied desktop/integrator_sensitivity. Fixedwidthlabel formatting(1.8 previouslyPath.stemtruncated to1) beforefinalcopy. Noactiveprocess.

Trackingfreshdevseed11320001 x3models: EulerfullICODE vsRK4preview bettermeanerror+compute ALL4cells: alternatinghigh47.97mm/3.354s vs51.23/9.942; alternatinglow39.65/3.437 vs41.70/10.111; late_turnhigh28.68/3.049 vs34.23/3.773; late_turnlow21.52/3.218 vs22.33/4.192. Same-Eulerpreviewretainsconditionaltradeoff(e.glate_turnlow22.50mm/1.498vsfull21.52/3.218). TheseareONEevaluationseed,n=3modelblocks, notrobustfinalproof. Don'tclaimold35%savingfalse; itwasconditionalonsharedRK4. Do NOT claimselectoroptimalwhencheaperintegrationunderminesit.

NEXT worthwhile independent work: inspect residual training target/discretization provenance. Euler improves error aswellascost inthese cells; possible finite-step-derived acceleration labels vs continuous-RK4rollout inconsistency, but currently ONLYhypothesis. Read frozen training/config/source beforeasserting, then bounded open-loop common-commandprobe onlyifmechanismsupported. This mayexplainresults; do notinventnewpapercontribution orrunanothergeometrygrid. Desktoppackage alreadytruthfulandcomprehensive, userasleep expectscontinuednightwork; no routine notificationneeded.
# Latest update 01:35: obstacle integrator36 DONE; tracking sensitivity72 ACTIVE

Integrator36all success,rawaudit14513cyclespassed. Wide fullICODEEuler .01091m/3.275s beatspreviewRK4 .01745/4.221; mid fullEuler.02447/3.838beatspreviewRK4.03099/7.532. Eulerforpreviewalso faster. This is integrator confound/sensitivity, notnewmethod. Only2reuseddevseeds1model; don'tovergeneralize. Chineseintegratorreport copieddesktop/integrator_diagnostic. Supplemental config/integrator/selector-alias audit stillneeded!

ACTIVE tracking_integrator_diagnostic.py session99600:72serial cases (existinglate_turn/alternating eachlow/high x3models x1freshdevseed11320001 x6methods fullICODE/preview/nominal eachRK4/Euler). K100H36samephysics/safety. No geometrysearch. Protocolicra_tracking_integrator_diagnostic_2026-09-10.md. Instantiates configs once; runnerinprocessserial, no workerrebuildgrid. Freeze contains originalrun+selector+configs+models. Hypothesis checksintegrationconfound in existingpositive trackingclaim before keepingit. No newalgorithmclaim; allnegativesretained,2hourbudget.
QUEUED audit_tracking_integrator.py waits thenraw72sourcehash/trajectoryRMSE/selector/timeraudit,outputdeadline_analysis/tracking_integrator_diagnostic_2026-09-10. Alias _euler strippedonlyforselectorreplay. Need supplemental fullconfigintegratormatch & percellpairedreport/figs aftercomplete. Avoid pooledonlyconclusions. Prior36executiondone, no competingheavywork.
# Latest update 01:18: runtime profile DONE; integrator sensitivity36 ACTIVE

profile_residual_runtime.py complete3instrumented existingwideepisodes. FullICODE64,224residualcalls,8.752s cumulativeof14.655totalinstrumented; preview18,720calls2.583sof7.823total; nominalnoresidual. These are cProfiletimes NOTbenchmarkcomparisons. Production already batchedTorchScript, RK4fourmodelcallsperstep. No baseline was accidentallyunbatched. Profiles/sourceconfig refs copied desktop/runtime_diagnostic.

ACTIVE integrator_diagnostic.py session98924:36serialdiagnosticruns,3existingwidths x2reuseddevseeds11220001–2 xblock0 x6methods nominal36/icode/preview eachwith RK4/Eulerplanner. K100,H36,120s,safety/physicalMuJoCointegratorunchanged. Protocolicra_integrator_diagnostic_2026-09-10.md. This is symmetricimplementationaudit, notnewalgorithm/confirmation. Hypothesis cheaperintegrationmayachievefullmodelcomputecompetitivewithselector. Needretainnegativefindings.
QUEUED audit_integrator_diagnostic.py waits then36rawmetrics/H/hash/timer audit. Need alsocheckplannerintegrator matchesvariant andsameplant/safety/otherconfig,selector transition rules forEuler aliases. Reviewcomplete3widthresultsbeforefurtherdecision; no largegrids. Priorovernightpackage+tracking519alreadycompleteondesktop. Do notruncompetingheavyworkwhiletimedexecutoractive.
# Latest update 01:01: tracking evidence package COMPLETE, all controls retained

package_tracking_evidence.py reads verified135confirmation+384mechanism summaries, requiresexistingrawauditpassed; exports all519compactsource records, fullper-scenemethodtables, paired preview differences againstEVERYcomparator, summary/audithashprovenance, data profile. Figures confirmation3panels andmechanism8panels showallseedpoints+means, PNG/PDF/grayscale. VisualQA passed layout; stablemethodcolors/shapes applied acrossbothfigures. Copied tracking/ into desktopICRA_2026-09-10_成果包 withChinese跟踪证据报告.md. No new performance experiments. No active process.

Evidence remainslimited: confirmation supports35.1%compute vsfullICODEwith~0.58mmmeanRMSEincrease; mechanism stronglowKcontrols invalidateglobaldominance. Figure late_turn showsusefultradeoff; alternatinglow/high defeatpreview with32/64 respectively; simpleturn occupiescheaperfrontier. Obstacle270 furthernegative. Next must derive a mechanism-based boundedmethoddecision or scopepaper honestly, not manufacturebroadadvantage. Desktopreport explicitly saysnewcontribution/independentconfirmationmissing. Nightuserexpectscontinuedmeaningfulwork; consider source-level audit of compute bottleneck/modelselection overhead and matchedbudget mechanism before any additionalrun. Do not repeatexistinggrids.
# Latest update 00:48: overnight270 COMPLETE/audited; desktop package CREATED; tracking consolidation NEXT

All270success0collision. Audit107284cycles passed(source/checkpoint,actor/H,RMSE,timer). Supplemental package_overnight.py audit passed36387selectorcycles fromloggeddemand, fullconfig differences outsideplanner/labels zero. Not independent demand/plant replay. All experiment processes finished. ActualMuJoCo renders show physicalwalls(allgeomgroupsenabled), PNG visualQA passed; scatter allmethods/seeds +mean PNG/PDF/grayscale visualQA passed.

NEGATIVE conclusion: preview is mean-error/compute dominated by nominal28 atwidth1.8(.01711m/3.481s vs.01687/1.098), width1.4(.03060/5.473 vs.02807/1.420), bynominal36 atwidth1(.04480/18.796 vs.04094/4.347). RL-Hall45success. FullICODEmayimprovewideaccuracy butcostly. Do NOT extend obstaclegrid or claimwins. Timingvaluesbatchspecific; fixed-step no hardwaredeadlineclaim.

Created morning_package under overnightstage, copied fully to C:/Users/lenovo/Desktop/ICRA_2026-09-10_成果包. Contains 早晨报告.md, all270source_data JSON/CSV, raw/supplemental audits, sourcehashprovenance, actualMuJoCo3widthPNG/configs, fullscatterPNG/PDF, protocol, baseline/literature scope docs. Chinese report explicitly explains no broadadvantage, oldtrackingpositive limited35.1%compute/0.58mm, paper gaps. This is a truthful interim evidence package, not submission-ready.

NEXT autonomous work: consolidate already audited preview_confirmation135 and preview_mechanism384 into properly scoped figures and paired comparisons, include ALL low-K and reactive controls/negative cells in desktoppackage. Do not treat baseline repair as ourcontribution. Existing strongerlowK cells can defeatpreview: alternating_high full64; alternating_low full32. Need compare candidate novelty and determine narrowclaim with evidence before any furtherexperiment. Only propose/run a bounded new intervention if mechanism supports it; no random scenehunting. User wants overnight sustainedwork, heartbeat10min remainsactive. No useraction needed.
# Latest update: overnight execution AUTHORIZED; 270-run direct comparison ACTIVE

User going to sleep, explicitly requests continuous overnight work and wake every10minutes. Automation icode-k-h UPDATED ACTIVE interval10minutes, same thread; prompt now latest-direction-aware and morning desktop成果包. No routine nighttime notifications; continue actual work after each batch until deliverables assembled.

ACTIVE overnight_comparison.py session35370:270 serial runs, existing widths1.8/1.4/1.0 x5fresh DEVELOPMENT seeds11220001–5 x3pairedtrainingblocks x6methods nominal8/28/36, strengthenedzero-penaltyRL-H+value,preview36,alwaysICODE36. K100,120s,commongeometry/safety/mass1. Protocol icra_overnight_comparison_2026-09-09.md. No new geometry search. Two-hour batch timeout. Snapshot sources/model checkpoints, preserve config/trace/selector.json per episode. Selector must log preview/icode choices. Actual feature+policy+selector costs included, fixed-step physics not compute-delay simulation.

QUEUED audit_overnight_comparison.py waits for completed stage, then checks source/checkpoint hashes, fixedmass/inertia, actor/H, rawRMSE, timer; writes analysis/comparison.json,report.md,audit.json. Need review all6methods/allwidths/allblocks, then add selector/config audit, paired effect summaries and figures/desktopreport. If negative don't force obstacle superiority: scope existing tracking35%compute claim properly, include negative obstacle evidence and remaining novelty gap. Do not stop after audit/report; produce actual visible morning package. Do not launch competing heavy work while timed executor active.
# Latest update: all three zero-penalty baseline replications COMPLETE

replication_completed.json confirms3trainingseeds/360evaluations, all per-seed actor/hash/H/timer audits passed. No running experiment process at final check. RL+value:45/45success,zero collisions across3widths x5shareddevelopmentseeds x3trainingseeds. All9cells5/5. MeanH about30–37. RMSE averaged overtrainingseeds: width1.8=.01754m,1.4=.02779m,1.0=.04683m. These are15unique scene/seed contexts repeated across3trainedpolicies, not45independent scenes. This improves baseline credibility, not evidence ours wins. Replication_cells.json includesall8controls; must review complete controls and then implement same-condition ours comparison. Stop further baseline-training expansion. No new comparative executor launched yet. No submission readiness claim.
# Latest update 23:39: seed2 audited; common/mass config comparison COMPLETE

Seed11101002 qualification120 audit passed52,168cycles. RL+value all three widths5/5, zero collisions; meanH36.04/35.25/31.68, RMSE.01726/.02987/.04967m. Seed11101003 training still active (~24ksteps at check), then120eval+audit queued same session12056. Wait full3seed results before final baseline assessment.

Ran audit_common_config_equivalence.py entirely during third-seed TRAINING, completed before its timing evaluation: no protected plant/safety/scene/sensors/reference/runtime config differences across matched common/mass runs. Full720savedconfigs audited. Differing paths are experiment/factorial labels, checkpoint, prediction_mode, selector_mode and residual gate configuration. Preview reliability/support gate enabled=false; nominal omits these entries. Delay-context nested enabled=true exists under disabled reliability gate; do not infer active privileged input from this field alone. This is saved-config audit only, not runtime intervention equivalence. Outputs common_and_mass_2026-09-09/analysis/config_equivalence.json; session54796 finished. No other analysis process remains.
# Latest update 23:26: baseline seed2 trained; serial qualification healthy

Seed11101002 completed30,554steps/75episodes; qualification began normally, seed3 follows in same continue_reward_replications.py session12056. No duplicate executor. No new complete performance conclusion yet.
Prepared audit_common_config_equivalence.py for full saved-config comparison of matched common/mass methods; NOT run during active timing. Run after replication executor finishes. This closes a previously recorded audit gap beyond mass/inertia checks. It reports all differing paths and protected differences, not automatic overall fidelity approval. Review unexpected safety/plant/reference settings before any further ours comparison. Existing result claims remain unchanged.
# Latest update 23:20: reward intervention COMPLETE/audited; two baseline replications ACTIVE

Zero-H-penalty training seed11101001:120 evaluations complete, audit PASSED51,239cycles (checkpoint/actor/H/timer). Source equivalence checked: stage/protocol name and .001H->0H only. RL+value succeeds5/5 at all widths (original narrow3/5). MeanH wide31.30/mid32.73/narrow30.85 vs~10. RMSE improvements wide10.62mm/mid20.74mm; narrow66.52mm is confounded by prior timeouts and not a pure quality effect. Full report reward_intervention_report.md retains all8methods; fixed16value narrow0/5. This supports reward sensitivity of our adaptation, NOT intrinsic paper failure and NOT ours superiority. All no-value fixed36 trajectories reproduce RMSE exactly across stages; their compute shifted0.12–0.19s, so don't overinterpret between-batch timing deltas.

Figure plot_reward_intervention.py: all5pairedseeds eachwidth, success/timeout fill symbols; PNG visually checked, PDF and grayscale exported; data_profile.json. Desktop ICRA_RLH_reward_diagnostic.png and .md copied. Figure is development baseline diagnostic, not paper proof.

ACTIVE continue_reward_replications.py session12056: train seed11101002(30k)->evaluate120->audit, then11101003 same sequence. Two-hour bound, original sources frozen. Protocol icra_rl_horizon_reward_replication_2026-09-09.md. Evaluators *_11101002.py/*_11101003.py differ only train/outputseed. Same15developmentcontexts intentional; no independent test claim. Original positive-penalty counterparts NOT trained, so extra runs assess baseline stability rather than replicated causal intervention. At finish writes replication_cells.json/replication_completed.json for3seeds360evals. Review ALL fixed controls before ours comparison; no broad geometry grid. All prior executor/audit sessions finished.
# Latest update 23:10: reward-ablation training COMPLETE, qualification ACTIVE

30,181 training steps /72 episodes completed. evaluate_reward_ablation.py is the sole timed executor (original session52589); 120 development evaluations running. finish_reward_ablation.py (session28412) waits for width_cells.json, then actor/hash/H/timer audit and full paired 120-episode comparison; outputs audit.json, paired_reward_intervention.json, reward_intervention_report.md. Waiting helper uses no heavy compute until qualification ends. Do not duplicate these jobs.

First deterministic RL+value episode width1.8 chose mean H30.83, range9–45 (old near10), succeeded with RMSE.01824m; one episode only, do not extrapolate. Whole narrow-width results required before deciding. Underlying intervention changes reward coefficient only; value-training visitation changes too, so not actor-only causal isolation. Need review complete paired report and preserve all results. No more geometry grids.
# Latest update 23:05: reward diagnosis COMPLETE; controlled ablation ACTIVE

Offline diagnose_horizon_objective.py replayed all 120 reward traces exactly. Outputs rl_horizon_diverse_2026-09-09/objective_diagnostic/{report.md,cells.json,episodes.json,manifest.json}. At width1.8, discounted task cost H8=.760 vs H36=.744, but H penalties .266 vs1.199. Thus observed objective favors short H. Discount gamma.97 gives ~3.3s effective span; terminal timeout credit concern is a hypothesis, not proof. This analysis is descriptive, not causal retraining.

NEW controlled diagnostic experiments/icra_rl_horizon_reward_ablation/run.py changes ONLY horizon penalty .001->0.0 from diverse baseline (plus independent stage/protocol names). Same seed11101001,30ksteps,three widths,120s cap, gamma/value/safety. Protocol docs/protocols/icra_rl_horizon_reward_ablation_2026-09-09.md. 8-step real MuJoCo smoke passed. Production training started, followed by evaluate_reward_ablation.py in SAME sequential shell: 120 matched development evaluations. Do not launch competing timed jobs. All originals frozen. Need actor/hash/timer audit on completion (copy audit_diverse replacing module directory; coefficient irrelevant there). Evaluate H shift, success, task cost and compute; one training seed diagnostic, not confirmation. No current publication-readiness claim.
# Latest: diverse RL-H qualification COMPLETE and audited

Training seed11101001 completed 30,040 steps / 67 episodes. Qualification: 120 episodes across widths 1.8/1.4/1.0, five fresh development seeds, eight learned/fixed-H variants, common 120 s cap. Audit PASSED: 55,392 cycles, checkpoint hash, deterministic actor replay on recorded observations, H mapping, compute aggregation. This is not independent plant replay or a complete implementation-fidelity audit.

Widths 1.8 and 1.4: every method 5/5 success. Width 1.0: RL-H with and without learned value each 3/5; fixed H8/28/36/50 with value and H36 without value each 5/5; H16 with value 1/5. All zero collisions. Learned H stays approximately 10 across widths. Broader training improves narrow performance but does not qualify this adaptation as a strong adaptive-horizon baseline. Do not interpret this as an intrinsic limitation of the original Bohn paper. No ours evaluation on these fresh qualification seeds yet.

Next priority: inspect reward/action distribution and terminal-value fidelity before more RL training; consolidate already audited positive tracking evidence and negative obstacle evidence into figures. Do not launch another broad geometry/mass grid or claim submission readiness. Current positive claim is selective residual-model use reducing compute versus always-on ICODE in tracking; it is not a demonstrated obstacle-navigation or joint K/H advantage. All prior negative results remain part of the record.
# Active execution handoff

## Latest22:29 timeoutdiagdone; broader-domainRL-H training ACTIVE

120sdiagnostic:preview2/2success(63.2/71.2s),nominal28/36both2/2,
RL-H0/2stillnearwall with~3.1–3.3mremaining. Distinguishesours/nominal60s
timeoutfromRLdeadlock, notoursdominance. RLonlywide-trained cannotbeusedas
strongnarrowbaseline. No additionalnarrowfailurehunting.

NEW experiments/icra_rl_horizon_diverse/run.py: seed11101001,30kcycles,
randomwidth1.8/1.4/1.0perepisode,mass1,common1200stepcap,samereward/model
andH8–50,32stepvalueaugmentation. Freshtrain11110000+episode, noheldoutreuse.
Real8stepsmoke passed. Source copiedfromseparatelongenv; originalfrozenrunsintact.
Protocol icra_rl_horizon_diverse_2026-09-09.md. Stage
rl_horizon_diverse_2026-09-09/seed11101001. Don'teditsourcewhiletraining.
evaluate_diverse_rl_horizon.py waits then120devqualification:3widths x5fresh
seeds11120001–5 x8learned/fixedH/valuevariants,common120s. Writeswidth_cells.json
aswellaspooledsummary. Needactor/rawauditafterward. No newmassgriduntilqualified.
Allpriorwidth/timeoutprocsdone. Figures and honestclaimscoping stillpending.

## Latest22:21: width360 COMPLETE/audited, negative;8-run timeout diagnostic ACTIVE

Width1.8/1.4allmethodsall success;1.4nominal28beatsours error+compute. Width1m:
preview0/30,RL-H0/30,nominal36zero/30,nominal28six/30 (2uniqueepisode successes
x3repeatedblocks). All0collision. Guardcensus: RL~4700/5500near-bodyhardstop
cycles per15runscells;preview/nominal mostlyfrontslow. Nooursadvantage established.
Do NOT use smallRMSE of stoppedrobot asnavigationquality. CurrentRLbaseline
trainedonlywideanchor; narrowOODfailure not intrinsicpaperlimitation evidence.

ACTIVE width_timeout_diagnostic.py uses separate long_anchor_environment.py
(copy ofbaseline env with600->1200 time/cap/penalty) forALL4methods, twoexisting
devseeds11010001/2,block0,mass1,width1.0:8diagnosticruns, nofreshconfirmation.
Protocol icra_width_timeout_diagnostic_2026-09-09.md. Safetyunchanged. Need
inspectresults, then distinguishslowcompletion vsdeadlock. No morebroadgrids.
Outputs width_timeout_diagnostic_2026-09-09. Widthaudit/clearancepostprocessors
allfinished. Figures/claimscope andstrongerbaseline domaintraining stillpending.

## Latest21:53 widthstudyhealthy; actualclearance postprocessor queued

Width1.8complete120; width1.4was20/120 atcheck. width_clearance.py waits for
wholewidthaudit, then signedboxdistance minusconservative.25bodyradius, per-
episode minima andverbatimguardreason counts; outputs clearance_guard_episodes/
cells.json. Do not interpret allguardreasonstrings asoverridewithout taxonomy.
This is geometricofflineclearance, notexactMuJoCocontactdistance/certificate.
No new heavywork duringtiming; no additionalexperiment launched.

## Latest21:44: mass480 COMPLETE/audited; width360 ACTIVE

Allcommon240 andmass480success0collision. Auditspassed. Atmass1 ->1.45,
RMSEnominal28 .01747->.01926;nominal36 .01827->.02101;preview .01637->.01788;
RL-H .02185->.02319m. Previewcompute~2.55–2.61s vsnominal28~.78–.80,
nominal36~.96–.99,RL~.64–.66. Modestaccuracygainbutmuchmorecompute;
NO overall advantage, no success-rate separation. Common-commandprobe1080rows
complete. MeanH36positionerror nominal~.07->.11m,ICODE~.05->.02m; other
channels notmonotonic. Need formatted exactprobeanalysis, don'tclaimallmismatch
increases withmass. Preserve allnegative results.

Now width_ladder.py ACTIVE, protocol icra_width_ladder_2026-09-09.md:
width1.8/1.4/1.0 xmass1/1.45 x3blocks x5freshdevseeds11010001–5 x4methods=360.
Same .35speed androundedreference, physicalwalls. Numerically checks dense
reference-to-wall clearance>.35 BEFORE eachrun; safety unchanged. Width1m may
stillnotseparate success, reportnullratherthanforceddifficulty. No testseedreuse.
Output width_ladder_2026-09-09. audit_width_ladder.py queuedafterwhole360.
Thisaudit checksrawRMSE,actor,H,mass,timer butnotminimumactualclearance/guard
summary; addthat aftertiming. Needfigures/visualQA stillpending. Do not launch
training/heavyanalysis alongside widthtiming. Decide contribution afterthis
bounded widthstudy ratherthananotherundirected grid.

## Latest check21:22: COMMON GATE PASSED; MASS480 ACTIVE

All common240 passed: everymethod/block20/20success0collision. Massphase91/480
atcheck, executorhealthy. Added probe_mass_mismatch.py queued after complete
navigation audit, then runs isolatedfreefloor common-command probes on4scales,
3residualmodels,3probeseeds,straight/left/rightcommands.20stepconstantwarmup,
36steprollout errors atH1/5/10/20/36. Offline truth only, no learnedcontroller
parameterprivilege. Writes rawactual/predictedtraces andseparateerrorchannels.
Not navigation efficacy data; no guarantee monotonicerror. Probe output
mass_prediction_probe_2026-09-09. No heavyprobe untilnavigationtiming/audit done.

## Latest check21:15: threeRLseeds done, commonphase active; audit queued

All3RL training+80qualification+audits complete. common_and_mass.py healthy,
153/240common atcheck. Queued audit_common_mass.py waits forwhole720complete,
thenhash/mass/inertia/rawRMSE/actor/H/timers andallcellreport. Does not read
interimtestoutcomes. Ifcommon gatefails, executor stops deliberately beforemass;
inspect gate and terminate orupdate waitingaudit rather than waiting8h blindly.
Need actualmismatch probe after timedmass, stillpending. No concurrentheavywork.

## Latest execution: common-anchor + mass ladder queued automatically

Seed2 qualification80done/audited; seed3training15229done, qualification was17/80
atlatestcheck. Existingqueue60103 handlesremainingqualification/audit.
NEW common_and_mass.py session11708 waits for three_seed_queue_complete.json
andall3audits, then serial common240 (20devseeds x3pairedmodelblocks x4methods:
nominal36,nominal28,RL-H+value,preview). Gate>=19/20zero collisions percell.
Only ifALLpass automaticallyruns mass480:10heldoutseeds10720001–10 x3blocks x
4methods x4mass/inertiascales1/1.15/1.30/1.45. No geometry/safety changes.
Protocol docs/protocols/icra_common_and_mass_2026-09-09.md. Sources/checkpoints
frozenwhenexecutorstarts, outputs common_and_mass_2026-09-09/common andmass.
No intermediate testselection. H28extra strongcontrol motivated bydevdata;
primarynominal36retained. Residualseeds20261201/2/3 paired diagonally toRLseeds.
All useSAME Environmentstep and common observationfeature timing; RL+value,
previewresidualfixedH36, nominalfixedH28/36. Fixed-step physics remains.
Inspecterrors ifqueuedprocess fails; do not stopawaituser. Aftermasscomplete
auditraw/configs/actor, summarizealllevels, then common-command predictionerror
probe needed toconnect mass intervention toactualmismatch; no claimmonotonicity.

## Latest execution: RL-H seed1 complete/audited; seeds2/3 serial queue ACTIVE

Seed10801001 trained15165steps,65episodes. Qualification80complete; actor/H/
checkpoint/compute audit passed18801cycles. All8methods10/10success,0collisions.
RL-H+value RMSE.021403m,compute.6115s,Hmean14.666; RL-Hwithoutvalue .021344/
.5194,H14.631. fixed16+value .020255/.5832 (dominates RL+value in pooled means);
fixed28+value .016125/.8233; fixed36novalue .019646/.9006. Anchor stable,
but no claim baseline has converged optimally or RL beatsfixedH universally.

ACTIVE continue_rl_horizon_seeds.py orchestrates seed10801002 then10801003,
each15000steps->80qualification->audit SERIAL. Param evaluator
evaluate_rl_horizon_seed.py and audit_rl_horizon.py. Queue writes
rl_horizon_anchor_2026-09-09/queue_progress.json andthree_seed_queue_complete.json.
Do not launch duplicate training/analysis duringtiming. Training source/protocol
unchanged fromfirstseed. No more userpermission needed. After3seeds, all-three
methods commonanchor validation and mismatchladder are stillrequired.

## Latest execution: RL-H IMPLEMENTED, TRAINING + AUTOMATIC QUALIFICATION ACTIVE

User frustrated by repeated stops; do not await further prompts to continue.
Implemented experiments/icra_rl_horizon/run.py. Three targeted tests passed
(range/rounding;32-step bootstrap/terminal targets;raw replay+SACupdate), real
MuJoCo8-step smoke_v2 passed. Originalsmoke failed atH<8 because shared safety
recoveryprefix8; retained failure and changed local actionrange[8,50], explicitly
documented adaptation (originalpaper[1,50]). Safety prefix unchanged.

TRAINING session17088: seed10801001,15000cycles finishingepisode,32x32SAC,
gamma.97,500randomwarmup,batch128,rawactions inreplay. Stage
rl_horizon_anchor_2026-09-09/seed10801001. Latestobserved4894steps,21episodes,
training progressing. Save resume.pt everyepisode + fulltraces. Do not edit
source/protocol whileactive; sourcehashmanifest andsnapshot includeallsrc.
Terminalvalue NN32x32learns32-step taskcostreturns; addsdiscountedvalue toshared
MPPI terminal stabilizer, rather than originalpaper replacement/quadraticvalue.
This is explicit Bohn-inspired MPPI adaptation, NOT exactoriginalreproduction.
19observedstate/reference/lastcontrol/time/obstaclefeatures. Trainingcost
ct²+.2speederror²+.002commandchange²+.02,collisionremainingtime,timeout5,
horizoncost.001H,rewardscale.3. Fullscope inprotocol
docs/protocols/icra_rl_horizon_anchor_2026-09-09.md.

AUTOMATIC NEXT already started: evaluate_rl_horizon.py session16056 waits for
trainingcomplete then80serial devqualificationepisodes,10reservedanchor devseeds
x learnedH withvalue/withoutvalue + fixedH8/16/28/36/50withvalue +fixed36without.
No retraining ondevoutcomes yet; no testseed use. Sharedlearnedvalue in fixedH
controls is NOT separatelyrefitted originalfixedHvalues. Timers includeplanner+
Hpolicy but excludeobservationfeatures; disclose beforecomparing otherharnesses.
Outputs qualification_seed10801001/comparison.json. Need audit actualH/rawpolicy
replay/sourceconfigs andinspect all8methods, no cherry-picking. Afterqualification
continue baseline development /3seeds and commonanchor before mismatch ladder.
No concurrent heavytraining/timing; evaluator waitsfortrainingcomplete.
Original obstacleanchor40done andaudited. No need userpermission for nextsteps.

## Latest result: obstacle anchor40 COMPLETE, RL-H remains unimplemented

All40 complete; raw auditpassed. Nominal20/20success0collisions, RMSE.019393m,
compute.995s/episode; preview20/20success0collisions, RMSE.016699m,
compute2.633s/episode. Shared physical obstacle corridor is feasible for these
two controllers. Preview2.693mm lower error but2.65x computation; NOT overall
superiority and not all-three-method qualification. RL-H still no implementation
or training. No experiment/analysis process running now. Next required action
is implement/test/train the Bohn-based RL-H adaptation, including explicit
terminalvalue treatment, then evaluate SAME anchor. Do not defer baseline by
running additional nominal/preview scenario grids. Earlier anchor details below.

## Latest execution: physical-obstacle anchor40 ACTIVE

User says work and report only substantive results. Started
experiments/icra_obstacle_anchor/run.py session19535; stage
obstacle_anchor_2026-09-09. Waiting analysis obstacle_anchor.py session59949.
40paired developmentepisodes nominal/preview,20seeds10710001–20, model20261201.
RL-H not implemented/trained yet, so this is ONLY initialgeometry/backbone
qualification. Protocol docs/protocols/icra_obstacle_anchor_2026-09-09.md.
Physical box-wall1.8m nominal corridor, roundedradius1.2m, speedcap.35,
max600steps. Actual collision/lidar obstacles. Seedpool reserved for development.
Newgenerator make_obstacle_anchor.py frozen. Geometryaudit sampled1515reference
positions: minimumcenterline-to-wall.896436m, conservativebodyclearance.646436m,
clear of near-bodyguard.35m. Does NOT establish frontguard/closedloopfeasibility.
Scene-list missing-singleton-comma bug caught by --describe before freeze and
fixed; no collected results affected. No safety modifications.
Do not concurrently train/time othercontrollers whileanchor40 runs.
Aftercomplete inspect success threshold>=19/20 each and guard interventions;
if fail diagnose shared cost/guard/task before fullcomparative experiments.

Next baselineimplementation: MppiController.configure_budget(K,H) already exists
and preserves recedingsequence; no need mutating config manually. Existing
src/mobile_robot_mppi/rl/sac.py SACAgent and rl/replay.py ReplayBuffer available.
OriginalBohn mapping[1,50] rounded inenv, replaycontinuousaction,32x32policy,
gamma.97, terminalvalue32-stepbootstrap; originalperformance andremaining-time
constraint+horizoncost need explicit adapteddefinition. Do not silently omit
terminalvalue or call old49D pointgoal agent a reproduction. Fulloriginaltask
qualification or honest documented adaptation stillpending. Currentnextpriority
implementRL-H after geometry work, thenall-method common-successanchor.

## Latest steering: TWO primary baselines, mismatch-first obstacle study

User explicitly sets primarybaselines nominalMPPI and Bohn2021RL-H; ICODE
always-on now model ablation, not substitute primarycomparison. New controlling
protocol docs/protocols/icra_mismatch_question_2026-09-09.md. Mathematical bound
carefully labeled upperbound, not proof of actualfailure; exact scalar mismatch
and candidate-ranking reversal script mismatch_counterexample.py ran successfully
(20errorchecks). Outputs research_artifacts/icra_mismatch_mechanism_2026-09-09
are ANALYTICAL TOY ONLY, not MuJoCo or new efficacy evidence.
Full Bohnpaper read: horizon rounding/unrounded replay, performance+constraint+
horizon cost, state/timevaryingparameters, joint terminalvalue32-stepbootstrapping.
Do not omit terminalvalue silently; baseline implementation/training stillpending.
Next: implement credible RL-H adaptation and wide obstacle-lined common-success
anchor, then singlefactor mass/inertia mismatch, corridorwidth, diversegeometry.
Measure actual predictor mismatch; parameter scaling not guaranteed monotonic.
Current ours is modelselection, not jointKH or proven optimal. No experiments
running. Preserve earlier135/384results; no more broad ad-hoc grids.

## Latest user steering: paper baselines + progressive common-success scenario

User could not see image; copied actualMuJoCo montage successfully to
C:/Users/lenovo/Desktop/ICRA_MuJoCo_四种场景.png. Scene visualization source
experiments/icra_deadline_analysis/render_mechanism_mujoco.py. Images are actual
MuJoCo renders with visual-only cyan reference lines, no obstacles.
NEW priority protocol docs/protocols/icra_paper_baselines_progressive_2026-09-09.md.
Choose paper-based ICODE-MPPI adaptation and Bohn2021 RL prediction horizon
(arXiv2102.11122, DOI10.1016/j.ifacol.2021.08.563) as main related-work baseline.
Need implement/qualify H baseline from full paper; do not relabel old H agents
or manualH heuristic as reproduction. Original code search not yet successful.
Start a common-solvable low-speed straight-rounded-bend anchor, then change
speed/curvature/straightlength/turn duty separately. Concrete proposed levels
and dev/test seeds in new protocol. Anchor NOT yet executed. No new experiments
running. Next useful work: full H-paper algorithm/reward/action/training review,
baseline implementation, anchor feasibility. No new large ad-hoc grid.

## Latest update: mechanism384 COMPLETE and audited

All384 finished. Restarted raw audit after4h wait timeout (session18024), now
PASS. mechanism_cells.py also complete: paired physical/safety/sensor/action
config, K/H and speedcap checks PASS. All8methods48/48success0collision.
No executors/analysis currently running. Do NOT rerun384.

Mean RMSE m/compute seconds: preview .031436/3.089; full100 .030120/6.037;
full64 .031977/5.157; full32 .034356/4.315; periodic .034151/3.980;
turn .034656/1.566; heading .038454/1.256; nominal .041346/.844.
Preview versusfull100:1.316mm error increase,48.83% compute saving.
Versus64: .541mm lower error,40.10% compute saving. Versus32:2.919mm
lower error,28.41% saving. Versusperiodic:2.714mm lower error,22.38% saving.
These are descriptive pooled means, NOT population superiority proof.

Critical per-cell limits: alternating_high full64 .04575m/5.785s versuspreview
.05358/5.854 (64 dominates). alternating_low full32 .04383/5.025 versuspreview
.04656/5.864 (32 dominates);64 also lower error .03947 butslower6.126.
Late-turn both speeds favorspreview versusbothlowK baselines; arc benefits mixed
accuracy tradeoff; straight preview essentiallynominal, full100 moreaccurate.
Turn/heading generally cheaper butlessaccurate. Do NOT claim all-scenario
dominance or conceal alternating failures. Original135 remains separate.

Outputs deadline_analysis/preview_mechanism_2026-09-09: comparison.json,
audit.json,all_cells.json/all_cells.md,all_episode_metrics.csv. all_cells.json
includes model-wise rawpaired effects;2seeds/cell limits inference.
NEXT: create data profiles and publication figures for135and384 with visualQA;
inspect model consistency and speed/smoothness; then bounded fresh robustness
and appropriate external baseline work. Avoid another undirected large grid.
Strongest claim currently computation-accuracy allocation, not universal gain.
Open-space ideal-sensing fixed-step sim only; no hardware/latency guarantee.

## Latest update: user approved more data — mechanism384 ACTIVE

16:10 healthy18/384. Waiting mechanism_cells.py added downstream of complete
raw audit (8h timeout). Will verify paired plant/sensing/safety/action configs,
actual K32/K64/K100/H36 and speedcaps; export achieved speed, sent-command
derivative RMS, every scene-speed-method cell and model-wise paired effects.
Only2seeds/cell; no population significance claim. Command derivative is not
physical jerk. No heavy analysis until384 timing and raw audit complete.

16:05 check healthy10/384. Baseline scope report now includes source-to-equation
mapping: local MPPI exponential weighting/weighted perturbation update; residual
drift+gain normalization. Found material original-paper difference: original
mentions Savitzky-Golay smoothing, local controller instead has clipping and
terminal alignment logic. Do not claim exact reproduction or modify active384.
Future isolated smoothing-adaptation baseline requires explicit window/order
provenance and shared safety/action contract. No new baseline training yet.

New experiments/icra_preview_mechanism/run.py session22936; stage
preview_mechanism_2026-09-09. Waiting analysis preview_mechanism.py session86419.
384=4 geometries(straight,late_turn,alternating,arc) x2 speedcaps(.4/.65) x
3 modelseeds x2 fresh seeds10610101/2 x8methods(nominal,icode,icode32,
icode64,turn,heading,periodic,preview). Protocol icra_preview_mechanism_2026-09-09.md.
Sources frozen. Tested heading zero-preview versus preview trigger on actual
PolylineReference. Lower-K full predictors are budget brackets, not assumed
exact timing matches. Heading uses same.25/.12 thresholds but0m preview.
AllH36,450steps, fixed-step ideal-sensing open-space simulation; no safety changes.
Scene generator source also archived. First worker was alive constructing
configs (384 configs recomputed perworker, substantial overhead). Do not assume
no progress.json at startup means failed; inspect process. Runtime estimate may
exceed2h; analysis has4h wait timeout and can be safely restarted if it expires.
Do not modify frozen executor or snapshots to optimize active stage.

User explicitly requests stronger baseline reproduction and more favorable
scenarios. Both late-turn/alternation and simple straight/continuous arc retained;
no cherry-picking. Baseline provenance audit in
docs/reports/icra_baseline_reproduction_scope_2026-09-09.md: paper uses bicycle
model, current baseline is dynamic-unicycle structure adaptation. Underlying
author ICODE repo found, but no verified author ICODE-MPPI vehicle code located.
Do not claim exact reproduction. Need source/equation mapping and subsequent
appropriate external baseline work. No concurrent training/heavy plotting while
timing384. After384: raw audit, all8 scene/speed cells, model consistency,
strong fixedbudget/heading comparisons, achievedspeed/smoothness; figures also
still pending for completed135. All earlier135 audits passed and are done.

## Latest update: confirmation135 COMPLETE — positive bounded result

All135 complete. Raw audit, paired protected config audit, and preview demand
reconstruction7345cycles PASSED. All5methods27/27 success,0collision.
Mean RMSE m / compute s: full .048633/7.5867; nominal .068266/1.0522;
periodic60% .056300/4.8557; preview .049214/4.9257; turn .053580/3.3177.
Preview-minusfull error .581mm, conditional95% bootstrap [-.237,1.408]mm;
compute saving35.074%, interval[33.184,36.911]%. Both prospective mean gates
and supplementary CI gates pass. Againstperiodic:7.086mm lower error but1.44%
more compute; againstturn:4.366mm lower error but48.47% more compute.
These are fixed-step simulation measurements, not injected-latency/hardware
results. Three model seeds and three paths; transformed paths are not separate
task families. Nominal repeats acrossmodelblocks not independent.
All executors/postprocessors finished. Outputs deadline_analysis/
preview_confirmation_2026-09-09: comparison.json,paired_statistics.json,
paired_effects.csv,path_model_cells.json,timing_and_config_audit.json,
preview_reconstruction_audit.json. Strong outcome warrants mechanism ablation
and speed/mismatch robustness with fresh data, plus figure bundle; not yet
paper-ready novelty. Do NOT rerun135. Need profile/figures visual QA, per-path/
model scrutiny, then bounded prospective ablation (preview lead vs reactive
heading with invocation controls) and real-time validation gap assessment.

## Latest update: September9,15:10 — preview confirmation135 launched

15:47 check healthy124/135. Added waiting confirmation_preview_replay.py
after timing audit: recomputes preview tangent demand independently from known
path and sensed state, uses frozen reference projection and replays planner/
runner progress updates. Post-step truth ONLY for original runner reference
progress, never selector demand at prior cycle. Must inspect audit success;
if mismatch investigate missing reference updates, do not silence tolerances.
This is not physical/planner rollout reconstruction. Figures pending full data.

15:40 check: executor healthy,103/135. Added waiting confirmation_timing.py
in deadline_analysis; waits for paired statistics before reading data. Computes
per-episode100ms planner exceedances, p95/p99/max and checks paired physical/
task/safety/sensor/perception configs. This is measured planner timing only,
not a whole-control-loop deadline or hardware real-time guarantee. All three
postprocessors remain downstream of complete timing, no concurrent heavy work.

15:30 check: executor healthy. Targeted primary-source literature note added:
docs/reports/icra_preview_literature_position_2026-09-09.md. Multi-fidelity
switching for motion-planning efficiency already exists (Styler2018); adaptive
multi-model vehicle MPC also exists. Preview is currently a heuristic candidate,
not established novelty. Finish135 unchanged, then assess mechanism controls
and real-time validation gaps before any paper-ready claim. No heavy work ran
alongside timing and no intermediate outcomes used to change the design.

15:15 check: executor healthy; no duplicate launched. Added waiting
experiments/icra_deadline_analysis/confirmation_statistics.py, which starts
only after135 raw audit completion. Writes all path/model cells, paired effects,
seed-clustered percentile intervals (10000 resamples, fixed seed10510300),
and prospective mean gates plus additional CI checks. It preserves methods and
models together when resampling episode seeds within path; intervals conditional
on these paths/models, with only3 seeds/path. Nominal repeated model blocks
not treated as independent. No interim outcome comparisons performed.

Tracking preview v2 ALL42 complete, raw audit passed. Every method6/6success,
0collision. RMSE/compute: full .040325/7.143s; nominal .059041/1.021s;
turn .045576/3.019s; periodic .049686/4.059s; innovation .050164/3.168s;
lateral .044517/3.715s; preview .041166/4.812s. Preview nearly full accuracy
with32.63% less compute, but slower than turn; selected for fresh confirmation.

ACTIVE experiments/icra_preview_confirmation/run.py, stage
preview_confirmation_2026-09-09; waiting postprocessor
experiments/icra_deadline_analysis/preview_confirmation.py. Protocol
docs/protocols/icra_preview_confirmation_2026-09-09.md.135 episodes:
3 pretrained model seeds20261201/2/3 x3 fresh episode seeds10510101/2/3 x
3paths(mirrored sweep, stretched chicane, reverse-S) x5methods
nominal/full/turn/periodic60%/preview. Same K100/H36. Preview unchanged;
periodic six of ten cycles to match development preview60% invocation.
No new model training. New paths are transforms, not new task families.
Prospective mean accuracy tolerance versusfull .003m and compute saving>=20%.
All failures and strong-control effects must be shown. Nominal repeats across
model blocks are not independent trajectory replicates. No p-value peeking.
Postprocessor covers raw accounting; confidence intervals, per-path/model tables,
mechanism reconstruction, literature novelty and figures still pending.
Do not run concurrent heavy tasks while timed executor is active.
All earlier experiments/analyses finished. Check actual processes before restart.

## Latest update: September9,15:00 — tracking preview v2 active

Tracking selector30 COMPLETE, audited. All five methods6/6success,0collisions.
Mean RMSE m/compute seconds: nominal .06123/1.075; ICODE .04203/7.356;
turn .04654/3.386; innovation .05365/3.328; periodic .05118/4.259.
Simple turn is a strong control; novelty not established.

Original tracking_preview42 failed after1episode: project() requires XY but
selector supplied5D state. Original partial stage/source preserved unchanged.
NEW experiments/icra_tracking_preview_v2 fixes current[:2], repeats ALL42
same-seed paired episodes. Protocol docs/protocols/icra_tracking_preview_v2_2026-09-09.md.
Executor session44401 ACTIVE; analysis session41793 waits for completion using
experiments/icra_deadline_analysis/tracking_preview.py. Stage is
tracking_preview_v2_2026-09-09. Verified2/42 complete after API fix.
Actual PolylineReference unit test passed geometry, nonmutation, preview trigger,
numerical lateral disagreement, shifted-sequence branch and reset.
Three paths x seeds10410101/2 x seven modes nominal/ICODE/turn/innovation/
periodic/preview/lateral. K100/H36; fixed-step physics; measured runtime NOT
injected into physical delay. Preview known-path .6m angle on.25/off.12;
lateral10-step predicted normal disagreement on.01m/off.005. All overhead timed.
Postprocessor replays logged thresholds, hashes, raw RMSE and compute/history;
it does not reconstruct proxy predictions independently. No efficacy conclusion yet.
After full42 and audit, compare all7, choose only supported candidate for fresh
model-seed/path confirmation. Do not launch another broad selector grid.
Do not include original interrupted partial data in complete comparison.
All earlier executors/postprocessors are complete or failed, not active.

## Latest update: September9,14:30 — continuous tracking pilot running

ACTIVE experiments/icra_tracking_selector/run.py (session69362),30episodes on
sweep/chicane/reverse-S, fresh10310101/10310102, modes nominal/ICODE/turn/
innovation/periodic. New protocol icra_tracking_selector_2026-09-09.md. Resolved
inherited budget is K100/H36 (checked --describe); explicit planner seed per
episode. Uses ExperimentRunner fixed-step physics, measured planner+selector
time, NO wall-time physical injection. Known motor delay.04s remains. Extra
selector logs record observed state and past sent history. Two targeted wrapper
tests passed before launch (periodic/history/reset, evidence delay/innovation).
Stage output tracking_selector_2026-09-09. Postprocessor tracking_selector.py
waits, verifies source hashes, independent polyline RMSE, raw compute, sent
history and selector replay, then summarizes every method. Paired predictor
numerical reconstruction remains unimplemented. Do not use the point-goal audit.
All earlier stages are complete. If this method has a real accuracy/compute
signal, use a new frozen confirmation with other model seeds/path variations,
and compare simple periodic/turn and calibrated nominal. If not, don't claim
success or recycle only the favorable comparison. No model-parameter learning
or new training is part of this pilot; novelty still requires precise positioning.

## Latest result: model-selection30 complete (September9)

Model-selection30 raw audit and selector replay passed. All5methods6/6success,
zero collisions. Mean duration/compute: nominal16=20.90s/.828s; nominal64=
22.367s/1.008s; ICODE16=22.75s/4.873s; innovation=20.733s/2.298s; turn=
22.183s/3.139s. Innovation uses ICODE38.2% of cycles. Savings against always
ICODE do NOT establish superiority: same-count nominal16 has nearly identical
task duration and substantially lower compute. No strong contribution yet.
Do not promote this as a paper result or replicate only the favorable ICODE
comparison. All these executors are done, including postprocessor68367.

Important evidence found while reassessing task sensitivity: archived L59
(docs/rl/85_l59_high_dynamic_confirmation_results_2026-07-16.md) reports ICODE
cross-track RMSE .04117m vs nominal .05758m across240runs and3model blocks,
with allsuccess. This is OLD evidence, not new data, but indicates model utility
is measurable in continuous path tracking rather than tolerant point-goal
success. The residual itself is a reproduction of a published control-affine
ICODE structure (see L57 protocol), so it cannot be claimed as our new novelty.

NEXT bounded method experiment should test the new causal model selector on
continuous path-tracking accuracy/compute using the existing validated backbone,
with nominal, alwaysICODE, turn heuristic and periodic model switching controls.
Fresh seeds and a new protocol; do not copy archived scores into a new table.
Useful code/config: configs/rl/icode_high_dynamic_closed_loop_l58.yaml;
configs/rl/icode_high_dynamic_data_l56.yaml (4paths); experiments/rl/
run_cross_layer_factorial.py::_condition_config(base,scene_path,domain,condition,
rl_checkpoint,icode_checkpoint,episode_seed); runtime/experiment_runner.py.
ExperimentRunner.components['controller'] is MppiController; it can receive an
isolated plan/safety observer wrapper for model selection before runner.run.
Runner executes fixed simulated dt and separately measures planner compute;
wall-time is NOT injected into physical evolution. Preserve/report this timing
scope rather than silently claim readiness-delay effects. Include gate overhead
in diagnostics compute_ms via outer timer. Do not read actuator truth; reconstruct
the fixed.04s motor delay using past sent commands at dt.1. Explicitly set planner
seed to each fresh episode seed to avoid any inherited fixed RNG setting.
Primary metrics cross_track_rmse, completion, success/collision and measured
compute, not only a point-goal Q. Sample budget should initially use inherited
validated setting, not a new grid. Current generic point-goal raw audit is not
appropriate for ExperimentRunner artifacts; implement the correct trajectory/
compute and selector accounting audit. Verify actual model objects/selection
and keep both fixed predictors as strong baselines. This is pending implementation,
not another active experiment or established novelty claim.

## Latest update: September 9, 14:15 — actual model-selection pilot

ACTIVE: experiments/icra_model_selection/run.py, session6482.30episodes, protocol
docs/protocols/icra_model_selection_2026-09-09.md. On the common feasible known-map
backbone compare nominal16/40, strong nominal64/40, ICODE16/40, observed-turn
model switch16/40, and paired-past-innovation model switch16/40. Three new seeds,
two0.9m layouts,60s common cap. No actor training, no claim of novelty/efficacy.
All fixed controllers' unused shadow ensemble configs removed; innovation
selector pays for its extra paired predictions and history reconstruction.
Selection and actual model are logged each cycle. Nominal/ICODE objects cached;
warm-start carried between switches. Paired error uses only past observed state
and reconstructed sent commands, discarding actuator truth.

Postprocess experiments/icra_deadline_analysis/model_selection.py waits for
all30, runs finish_stage raw audit, replays model selection from recorded
observable yaw/error summaries, verifies logged mode, and adds actual ICODE-use
fractions. Independent reconstruction of the paired prediction itself is not
yet implemented. Need inspect source/invariants, complete outcomes and usage
before any new confirmation; do not equate prediction gain with control gain.
Prior feasibility8 and all earlier pilot executors are done; never restart.

## Latest result: September 9, feasibility calibration complete

feasible_backbone8 finished and raw-audited(1,849cycles). Both known-route
controllers4/4success, zero collisions/deadlines. ICODE16/40 mean22.075s task
duration,4.980s compute; nominal64/40 mean24.15s duration,1.508s compute. This
is ONLY four paired development cases per method, not an established model
gain or paper contribution. The common known-route, clearance-qualified
backbone is executable. All executors in the prior Latest update are complete.

NEXT WORK must move beyond feasibility checks: formulate and test a bounded
method-level hypothesis with this shared backbone and strong fixed controls.
Candidate to assess (not authorized result/established novelty): selectively
use costly learned dynamics only when causal prediction-error advantage or
known-route turning demand warrants it. This changes the adaptation axis from
K/H to model fidelity; ADP already studies fidelity, so a precise distinction,
primary-literature check and matched nominal/ICODE/heuristic controls are needed.
Do not start the old KH grid again. If implementing model selection, log actual
per-cycle model choice, charge all prediction/gating overhead, use observable
command history and no simulator actuator truth; freeze choices before new
seed validation. Current results do not justify discarding nominal baselines.

## Latest update: September 9, 14:02 — concrete geometry/safety mismatch

Route24 complete and audited: both known-route models3/6 versus direct1/6.
Narrow alternating route still fails. Ran clearance_probe.py (static poses,
not extra navigation episodes):0.7m gate half-width.35 equals near_body_stop_radius
.35.4/26 centerline poses have positive physical clearance(.10m at gate plane)
but guard stops translation; no proxy collision. The0.9m layouts have positive
physical(.20m) and proxy(.12m) clearance, no centerline guard stops. Actual
narrow-route terminal traces stop with near_body_hard_stop. Wide alternating
remaining failures have cleared the last gate and approach the final goal at
30s. This identifies a concrete test-design mismatch for the narrow family
and a too-short mission cap for some known-route wide episodes; preserve them.

ACTIVE8episode feasibility calibration: experiments/icra_feasible_backbone/run.py
(session54280), protocol docs/protocols/icra_feasible_backbone_2026-09-09.md.
Both models receive known route, two0.9m layouts, new seeds, unchanged guard/
physics/costs; all methods get60s mission cap. Q deliberately still normalized
by30s for compatibility, not fraction of timeout. Postprocessor finish_stage.py
feasible_backbone_2026-09-09 --expected8 waits in session27500. This establishes
a common executable backbone only, not any algorithm advantage. If it succeeds,
all later comparators share these route/safety/time settings. Do not interpret
this calibration as a paper result or keep expanding feasibility sweeps.

## Latest update: September 9, 13:51

Temporal64 complete and audited: all8samplers fail both alternating-gates
layouts on both seeds. Coarse sometimes improves gate_then_turn, but the
guard-triggered mixture has no extra gain. Do not scale it as a contribution.
New ACTIVE diagnostic: experiments/icra_route_feasibility/run.py (session12615),
24episodes, protocol docs/protocols/icra_route_feasibility_2026-09-09.md. Known-map
gate-center waypoint reference versus direct global goal, nominal64/40 and
ICODE16/40, three layouts and two new seeds. This is a privileged-route upper
bound for task feasibility, not a claimed fair method advantage. Fixed safety,
physics, costs and final goal retained; route index/waypoints logged. It tests
whether missing route guidance, rather than sampling budget, dominates failure.
Postprocessor finish_stage.py route_feasibility_2026-09-09 --expected24 waits
for completion and raw-audits all methods/cells. Do not analyze concurrently
with the timing executor. If route helps, share route guidance across all
comparators before any further adaptive method claim. If it fails, inspect
clearance/safety/reference consistency instead of another scheduler tweak.

## Latest update: September 9, 13:38

Controller bottleneck48 complete and audited: all8shared variants only0–2/6
successes, no collisions. No universal remedy from the tested soft-cost radius
and recovery prefix; preserve complete results. New mechanism to test is
temporal correlation of sampling perturbations, at matched fixed K/H.

ACTIVE: experiments/icra_temporal_sampling/run.py, protocol
docs/protocols/icra_temporal_sampling_2026-09-09.md, output temporal_sampling_2026-09-09
under the existing compute artifact. Session15186.64episodes (four layouts
including open, two models, two seeds, four samplers). iid exactly delegates
to original code; coarse uses block5constant perturbations; mixture uses
half iid/half coarse; guard_mixture triggers after3past forward safety reductions.
Rollout count, bounds, safety and cost unchanged. No RL, literature reproduction
or exact path-integral-law claim. Source isolated in new experimental directory.
Postprocessor experiments/icra_deadline_analysis/temporal.py waits until done,
then raw-audits and writes full methods, all layout cells and paired contrasts.
Inspect sampler invariant tests and complete raw outcome/trajectory evidence
before a next confirmation. Do not edit active frozen source or run heavy
analysis alongside timing. Prior pilot queues are all completed.

## Latest update: user asked to continue after negative grid

User explicitly rejected stopping at negative results and authorized further
experiment/method redesign. Ran failure_census.py over ALL288 compound runs.
243 were timeouts. Per-layout mean guard fractions over timeout final50cycles
range60.3–86.0%; no deadline misses there. Proposed speeds also shrink, so guard
correlation does not establish the sole cause. Census JSONs are in the compound
deadline_analysis directory. New hypothesis concerns the controller backbone:
overbroad soft obstacle cost and weak sampling-prior response to safety feedback.

ACTIVE: experiments/icra_controller_bottleneck/run.py (tool session8176),
protocol docs/protocols/icra_controller_bottleneck_2026-09-09.md. Small48episode
paired factorial: nominal64/40 and ICODE16/40, soft obstacle influence .70/.35,
safety-recovery prefix0/5, three diagnostic compound layouts, two fresh seeds.
Safety/physical rules identical; existing controller support used. This is
engineering diagnosis, not an alleged new algorithm or evidence of RL benefit.
At8/48 it was running normally. Postprocessor experiments/icra_deadline_analysis/
bottleneck.py is waiting in session71495; it audits all raw traces and verifies
identical safety/perception/action configs before generating report.md and CSV/
JSON under deadline_analysis/controller_bottleneck_2026-09-09. Do not run heavy
analysis while timing is active. Afterward inspect the complete factorial and
decide a bounded next method change; any backbone fix applies to baselines too.
Do not restart any prior completed480/288/480 stages.

## Latest update: September 9, 13:18 — budget explanation resolved

Door grid finished 480/480, raw audit passed over 100,750 cycles. Three current
batches total 1,248 episodes, all raw-audited. Door-grid fixed ICODE K16/H40:
12/12 success, zero collisions, Q0.668, compute3.659s. Full KH initialization1:
12/12, zero collisions, Q0.673, compute5.038s. Full KH initialization2:11/12,
zero collisions, Q0.983, compute5.923s. Fixed nominal K64/H40:12/12, zero
collisions, Q0.706, compute1.065s. Thus the selected fixed ICODE dominates both
full learned policies on point estimates; earlier door benefit is explained by
insufficient fixed-horizon coverage. These development selections need fresh
verification for positive fixed-controller claims, but current data do not
justify the existing learned scheduling contribution.

Do not rerun completed pilots or launch another geometry/budget sweep. No
experiment executor remains scheduled after the grid. Further work should
be a bounded method/claim redesign grounded in these results and related work,
not more seeds of the unchanged KH policy. User has not yet approved or asked
to abandon the ICRA objective; continue working toward it autonomously.
Figure PNGs fixed_grid and fixed_and_adaptive were inspected; layout legible.
Shortening verbose K-only legend names is a cosmetic analysis change; no data
or frozen experimental source is modified. Full earlier bundle visual QA is
still not complete. Door grid analysis is under deadline_analysis/
door_budget_resolution_2026-09-09; report.md has all 40 methods and references.

## Latest update: September 9, 12:40

12:56 heartbeat: door grid was running normally at 246/480; do not restart.
Added experiments/icra_deadline_analysis/door_grid.py and after_door.py. The
latter is waiting (tool session 61866) to audit the complete 480-row schedule,
output all fixed/learned contrasts and 32-setting annotated heatmaps plus a
fixed-versus-adaptive scatter plot. It deliberately starts only after timing
is complete. Its code has not executed yet; inspect errors and visually review
PNGs before treating that bundle as done. Output uses deadline_analysis/
door_budget_resolution_2026-09-09. Both task-best and cheapest count-matched
fixed settings are exposed as development selections, not independent winners.

Both first stages finished (480+288=768 episodes); raw audits passed over
66,377+81,282 cycles. Analysis bundles exist. Four analysis invariant tests
were run manually and passed. Full visual QA is still pending; inspected
transition alternating-gates, transition gate-then-turn and geometry door-offset
trajectory PNGs have legible layouts. The alternating-gates example stalls near
x=1.7m before the second gate for all six displayed policies. This is an example,
not yet a complete failure-location census.

Compound pilot task results are poor: full joint seeds succeed 2/24 and 7/24;
H-only 6/24 and 3/24; fixed ICODE 3/24,4/24,3/24; fixed nominal 5/24,4/24,3/24.
Every method has zero collisions, so failures are timeouts. Do not portray this
as successful transfer. The first geometry stage's door-only 12/12 versus
fixed ICODE10/12 remains promising but needs a stronger fixed-budget comparison.

NEW ACTIVE STAGE: experiments/icra_door_budget/run.py, output
research_artifacts/icode_sac_compute_2026-09-08/door_budget_resolution_2026-09-09.
Protocol docs/protocols/icra_door_budget_2026-09-09.md. Forty methods: 32 fixed
model/K/H combinations plus eight existing KH-full/masked/H/K final policies,
three widths x two speeds x two new seeds = 480 episodes. Tool session 21801.
Do not run heavy analysis concurrently. This resolves the fixed-budget and
single-axis explanations, not another general geometry sweep. It is explicitly
development; final confirmation remains separate. The previous three executors
are complete and must not be restarted. The generic analysis script currently
accepts only the first two stages; add a separate appropriate 40-method grid
analysis (readable heatmaps/small multiples, not a 40-item legend) after this run.

User instructions: work autonomously toward an ICRA paper by September 15;
explore new designs; report substantive results, not repeated progress. No
guarantee of publication and no fabricated/hidden results. This is active work,
not a request to keep writing plans. The old automatic matched-model training
priority has been superseded.

## Running chain

Workspace: /home/mapples/projects/mobile-robot-mppi-study in Ubuntu-20.04 WSL.
Python: .venv/bin/python. Run all measured-timing experiments serially.

1. experiments/icode_sac_geometry/run.py — 480 episodes; output
   research_artifacts/icode_sac_compute_2026-09-08/geometry_validation_2026-09-09.
2. experiments/icra_transition_pilot/after_geometry.py — waits for stage 1,
   checks new scene openings and checkpoint compatibility, then runs 288 new
   compound-scene episodes via run.py in the same directory. Output
   research_artifacts/icode_sac_compute_2026-09-08/transition_pilot_2026-09-09.
3. experiments/icra_deadline_analysis/after_pilots.py — waits for both stages,
   then tests pairing/sign logic, audits raw data and generates full CSV/JSON/
   Markdown and PNG/PDF bundles. Output under the artifact's deadline_analysis.

The analysis queue was already running when the test command was added to its
source on September 9. Its loaded code may omit that new test command. Run
test_analysis.py after both timing stages if not shown in the queue output.

WSL restarted around 11:16 local time, killing earlier tool sessions. No
experiment process remained. Saved 70 episodes were retained, and all three
executors were resumed. Tool sessions after resume: geometry 10001, transition
waiter 50280, analysis waiter 76102. These IDs are provisional; inspect actual
processes rather than assuming they persist across app restart.

Heartbeat automation icode-k-h was updated using the app tool: now targets
current task 01a08421-4835-7d00-aca5-27255a9c63af, every 15 minutes, and follows
the deadline pivot. Old task 01a07b93-9142-7e31-b87e-b63f18a8c3d4 is no longer
the target. Keep the monitor quiet without actionable change; do not start
duplicate queues. Resume missing processes from saved results after inspecting
errors. Never overwrite frozen snapshots or change active experiment source.

## Required follow-through

- Resolve actual runtime exceptions before assuming success. New postprocessing
  code has not yet been executed against completed stages.
- Read all methods and family/geometry/speed cells, including failures and both
  training initializations. A cheap controller that collides is not a win.
- Inspect generated PNGs visually; layout audit alone is insufficient. Data
  profile and machine layout audit are included. Run strict figure compliance
  check after adjustments. Mark visual QA complete only after inspection.
- Trajectory examples are prespecified by first seed, fast speed and central
  geometry (or first level). The selection is independent of outcome. Crossing
  trajectory overlay is explicitly omitted because obstacles move; complete
  numerical crossing results are still included.
- Decide next experiment from full results. Candidate methods are not yet
  successful new contributions. These pilots transfer existing policies and
  do not establish newly trained algorithm performance or reproduce literature.
- To claim joint scheduling, add K-only; to claim model contribution, compare
  trained nominal adaptive policy; to claim superiority over fixed budgets,
  tune a broader fixed bank including cheap K64 configurations on development.
- Then freeze a small independent verification protocol with new seeds and
  enough independent training seeds. Do not let a third round of screening
  consume the writing/submission window. September 10 is the selection target.

## Literature checked during current work

Bounded lookup, 2026-09-09. Full text now opened for both:

- https://arxiv.org/pdf/2102.11122 — Bohn et al. learns state-dependent horizon
  with SAC. A local H-only policy is related, not automatically a reproduction.
- https://arxiv.org/html/2510.05330v1 — Adaptive Dynamics Planning learns temporal
  discretization using TD3 and evaluates BARN navigation. This is a relevant
  novelty comparison. Variable fidelity already exists; joint K/H alone must
  not be declared novel without a precise distinction and supporting evidence.

The earlier cs.gmu.edu author-PDF link redirected. The arXiv full text resolves
that retrieval gap. Avoid copying reported paper scores into our experiment
tables; protocols, robots, and outcome aggregation differ.

## User-requested app setting

C:/Users/lenovo/.codex/config.toml now has model_context_window=400000 and
model_auto_compact_token_limit=360000. Values were verified on disk; hot reload
was not claimed. This does not change the experimental model or checkpoints.









































# Latest handoff: 2026-09-10 matched-readiness evidence complete

The six fair-execution RL-H training runs and the 260-episode fresh evaluation are complete. Evaluation contains 26 methods across two route families and five shared fresh scene seeds per route: three joint SAC initializations, three H-only initializations, three residual RL-H initializations, three nominal RL-H initializations, matched fixed-H16/H32 controls for each RL-H value network, and fixed K128/H32 controls for residual and nominal models. All 260/260 episodes completed without collision; the raw audit passed (including 3,313 original-SAC decisions and 16,514 RL-H decisions).

The analysis unit is five independent scene seeds per route, with policy initializations averaged within scene. Do not report 15 initialization-by-scene rows as independent scenes. With n=5, the smallest attainable two-sided exact Wilcoxon p-value is 0.0625; report paired effect sizes, Student-t 95% CIs (df=4), per-scene directions, and initialization consistency rather than significance claims.

Main evidence: on reverse_turns, joint versus fixed residual K128/H32 improves RMSE by 3.88 mm and reduces compute by 1.11 s; all five scenes are jointly better. Joint versus residual RL-H improves RMSE by 4.20 mm with an approximately equal compute time (delta +0.05 s, CI crossing zero). Joint versus H-only is mixed and not a stable superiority claim. On single_turn, joint saves 0.79 s versus fixed residual K128/H32 but RMSE is essentially tied within uncertainty. RL-H adaptive versus its own fixed-H32 control is small and initialization-dependent.

The matched RL-H result is an execution-interface control, not a fully reward-matched competitor: RL-H was retrained with reward -0.3*task_cost and no compute-price term, whereas the original joint SAC spec fixes compute_price=0.05. The six training runs passed provenance, reward reconstruction, action mapping, physical-delay, checkpoint/resume, finite-update, and source-hash audits. Do not call this a reproduction of the literature implementation.

Publication direction is now fixed: complexity-aware joint K/H budget allocation for the compute-tracking trade-off. Use reverse_turns as the mechanism/primary result, single_turn as the boundary condition, and RL-H as a fair execution-interface comparison plus local horizon-adaptation diagnostic. State that all tested episodes were collision-free; do not claim safety robustness. Do not launch another broad grid or alter frozen protocols/checkpoints unless a narrowly pre-registered confirmation is required.
