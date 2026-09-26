# Bøhn 2021：实测延迟驱动的浅树策略搜索进度

更新：2026-09-26 05:11（北京时间）。这是阶段进度，不是复现成功报告。精确运行状态以结果目录的 status.json、条件 progress.json、实际进程及原始日志为准，本文不是持续刷新的状态页。

## 同步结论

关联任务“推进 Bøhn 2021 复现”的历史会话曾停留在旧门控验证运行阶段。当前工作区已经推进到新方法。旧有限门控的两任务×两固定比较器共四组验证门槛全部失败；旧测试继续封存。失败状态诊断已完成原始轨迹、实际计时、独立复核、图表和交付包审计，不能将其手工规则收益归为学习结果。

新方法用分类交叉熵直接搜索深度2决策树，每控制步选择H，含第0步。它属于改进方法，不是原论文SAC。目标、两任务、三训练种子、两类固定比较器、控制/安全门槛和封存确认要求保持预定协议。暂不开展移动机器人联合K/H。

## 已完成并有记录支持

- registration.json 冻结726个既有源码、配置、继承模型及证据文件。核心六份训练源码在本次同步中逐一复核哈希不变。
- 新建18个场景库、556个输入。去重审计对照81个历史场景库、1,231个历史唯一输入，无精确重复；该检查不证明分布上的统计独立。
- 策略/边界合成检查36项、独立学习排名与概率更新检查50项通过，共86项。后者文件使用 checks:50 字段，不应把JSON键数误当作检查数。
- 冒烟覆盖两任务×三种子×固定/测试树，共12条件、48回合、3,720控制步、3,872求解、104次重试。
- 冒烟独立积分1,860次，最大状态误差约2.091×10⁻⁸；因果上下文、H路由、成本、日志时间扣除、计数与重复重放审计通过。
- 冒烟中的树是预定仪表测试树，不是优化结果；其失败保留，不能作为方法优劣证据。
- 正式串行训练进程PID 1694525运行中。05:09核实实际命令和进程存活；05:10日志已推进到车辆种子1第三代（generation2），最新完整条件g2_c01。车辆种子0已经完整结束，训练内选出g3_c09，fit.json确认learned_tree_selected=true。六个作业完成1个；其余尚未完成。这只是训练内选择，尚无正式验证效果结论。日志：training_1790365288991336049.log。

结果根目录：

research_artifacts/bohn2021_reproduction_2026-09-17/results/latency_tree_2026-09-26

## 本次新增实现与状态边界

新增十个Python文件，未改写已登记的训练源码：

|文件|用途|
|---|---|
|latency_tree_evaluation_spec.py|90个初始验证条件、继承固定模型清单、登记及策略冻结、严格安全比较|
|latency_tree_evaluate.py|共用计分初始状态、各自终端、所有原始步记录、串行双重复正式计时|
|latency_tree_baselines.py|独立seed0完整H网格与三种子共用终端网格提名，仅补训缺失独立模型|
|latency_tree_fixed_train.py|复用已审计原固定H训练记录器，仅替换实验路径与验证库文件名|
|latency_tree_evaluation_audit.py|独立动力学、成本、计数、路由、跨终端初始状态与精确重放检查|
|latency_tree_effect.py|10,000次配对场景bootstrap、全部主比较与C5消融、正验证后的封存登记|
|latency_tree_effect_review.py|独立重算固定提名、统计量与通过条件，不导入生产验收函数|
|latency_tree_evaluation_checks.py|正反例、阈值边界、零/负成本、极小失败率增量、配对/重复一致性及随机对照检查|
|latency_tree_training_delivery.py|验证前冻结全288候选、所有条件/回合、失败和重复执行，以及实际新预算和继承模型预算|
|latency_tree_training_delivery_review.py|独立逐行复核交付CSV，重新遍历求解原始日志核对预算和完整覆盖|

上述Python实现已写入和静态阅读，**尚未运行其合成检查、尚未登记为正式评估、尚未进行新评估冒烟**。不能据此宣称执行正确或效果通过。当前训练依赖实际时间排名，因此这些Python检查、模型加载和重型审计都延后到训练退出后。

另新增 latency_tree_native_pipeline.ps1。它使用原生PowerShell等待训练状态完成且实际PID退出，再按顺序运行后续阶段。语法、进程识别、空/非空参数传递已作原生检查。每一步非零退出都会停止并保存异常；没有自动覆盖部分记录或重启训练。该控制器的源码另行哈希登记，Python检查必须通过后才允许正式评估登记。当前控制器的启动状态以 native_pipeline_v2_launch.json 和 native_pipeline_v2_status.json 为准；无v2后缀的旧状态已明确标为被替换，不再代表活跃控制器。

2026-09-26 04:07:30已启动该原生控制器，Windows PID 34488；04:08交叉核实实际进程和状态文件均处于 waiting_for_training_exit，stderr为空。九份源码登记校验通过，登记SHA-256为5c71be0a2f62b505cd66971f44417eff440aeae9a15f8fbe44e8921611c37c91。首次直接调用因未签名策略被拒绝、未启动任何子流程；后改用只对该进程生效的执行选项，自检和启动通过，未修改系统执行策略。隐藏窗口运行不提供断电/主机睡眠后的自动恢复；出现中断仍须先审计。

04:16发现并修复执行依赖缺口：原流程没有在验证前生成“全候选结果＋实际预算”的冻结交付。按原协议补上此交付及独立复核。旧控制器在仍等待、已完成实验阶段为0、无实验子进程时停止；只有已识别的控制台宿主conhost.exe，训练进程未动。旧源码、登记、日志、原状态和停止记录保存在 delivery_order_amendment。修改只涉及新增交付依赖及登记，计分rollout函数与科学specification文本逐字一致，六个既有训练源哈希不变，统计门槛实现未改。

04:19:44启动v2原生控制器PID 26188，04:20核实存活、waiting_for_training_exit、stderr为空。v2登记11个源文件，登记SHA-256为123f58bb2cdba1482526c312ecedd8770f19c8921b792980248f5b7443d11a02。原生语法及前九阶段依赖顺序检查通过；新增Python检查尚待训练退出后执行。正式评估入口也强制核验训练交付和独立复核，避免仅靠控制器顺序保证。

## 最终预算收尾准备（04:31新增）

另新增 latency_tree_final_budget.py、latency_tree_final_budget_review.py 及原生 latency_tree_budget_finish.ps1，未改动v2登记的11份源码。它们在整条评估流水线完成且实际退出后才执行，属于后处理账目，不改变模型选择、科学门槛或测试准入。

预算按场景生成、训练冒烟、搜索/复选、新评估冒烟、正式验证、固定终端补训、计时和获准的独立确认分别统计。固定终端补训的转移/reset可计数，但未记录逐次NLP尝试；构造器内部求解也未仪表化。因此工具区分“有日志的求解总数”与“整个研究求解总数（未知）”，不把缺失项记零。继承模型历史训练和本轮新训练分列；离线积分按实际保留的每次审计记录统计，初始验证被完整审计再次检查属于重复计算成本，不能去重成只算一次。

独立复核重新枚举全部计数文件，逐行重数求解日志和补训转移/reset记录，检查重复attempt标识、未分类目录、未完成条件、训练阶段冻结预算的一致性及独立确认准入。原始测试轨迹不会由预算工具读取；若已获准完成确认，则读取其汇总和审计凭据并明确标记。任何不能解释的计数差异均停止，不能声称预算完整。未保存凭据的重复审计工作仍属未知，不作绝对全CPU预算声明。

预算控制器04:30:22启动，Windows PID 7108，04:31核实存活、waiting_for_pipeline_exit、stderr为空。其3个源码另行登记于 budget_finish_registration.json，SHA-256为7da3cb7fdfd303e0368e6df2fa35a928b04c0398e92b1bd6a7cef352fa80e1c9。原生语法、checks→register→ledger→review阶段顺序及存活/不存在进程识别检查通过。Python计数正反例、正式汇总及独立复核仍待流水线退出后执行；不得把源码准备或原生检查当成实际账目通过。

运行状态分别见 budget_finish_launch.json、budget_finish_status.json；不代替总体目标的完成审计。预算通过之后仍需全量结果图表、最终中文总结和交付包检查。

## 全种子图表实现准备（04:48新增）

新增 latency_tree_figure_data.py 和 latency_tree_figures.py。只新增后处理源码，未改动既有训练、评估和预算控制器。04:47复核六份核心训练源码、v2登记11份源码、预算登记3份源码的SHA-256均匹配。一次人工核对命令误用绝对路径查训练登记表，因不存在该键报错；查明该表使用相对路径后重新核对通过，并非源码变化。

图表数据入口独立读取经审计的结果和独立效果复核，不导入训练或验收实现，不重算或改变主效果门槛。覆盖两任务×两比较器×三种子的12项主比较，以及C5和组合策略的全部21项诊断；逐条件检查完成凭据、摘要哈希和全场景索引，并根据原始计时摘要再核对两次延迟比值。准备全部场景、时延重复、候选和比较CSV，并运行SciPilot数据剖析，留下报告供选图复核。

验证图表拟包含14张：成本与实测延迟点区间图、物理＋约束成本、两任务全部配对场景成本差、安全指标、完整固定H网格、回合内H切换、平均H、C5/组合诊断、全部288训练候选。独立固定终端完整网格仅seed0，共用终端网格全部三种子；seed1/2独立终端的补充评价只覆盖提名H，不能声称完成三种子独立全网格。训练候选只作训练内、经过选择的描述，固定回退必须显示，重复候选不剔除。

95%区间沿用已独立复核的10,000次配对场景bootstrap；不把种子、重复计时或控制步当新增独立场景。成本百分比区间仅为绝对差区间除以观测固定平均成本绝对值，不冒充重新bootstrap的随机分母比值。零分母保持未定义；百分位区间可能不包住点估计，采用直接绘制端点避免负误差棒。延迟图显示两个单独重复，坐标范围同时容纳区间、点估计和重复。C5/组合始终单列，不能解锁主效果。

入口必须确认训练完成、两个原生控制器完成且实际退出、实验Python进程不存在、预算独立复核通过；之后才能登记图表源码、检查和准备数据。登记与科学验收分开，图表不能解封测试。数据剖析复核记录存在后才允许预览；实际查看全部彩色/灰度图并记录八项视觉检查通过后才允许PDF/SVG/300dpi PNG导出。预览轮次保留；如改源码，须保留登记修订，不能静默覆盖。灰度副本直接由同尺寸PNG生成，避免外部助手灰度选项重新裁剪主图。

**上述两份Python只写入并作静态审查，尚未执行语法/合成检查、数据剖析、绘图或视觉QA。** 准备状态及源快照另存figure_preparation；没有新绘图控制器，也没有替换或重启三个现有进程。后处理依赖实际数据，运行时仍需依据错误和剖析报告修复，不得把源码存在当成已交付图表。

仅在整条实验和预算流水线实际退出后，顺序执行：

~~~bash
.venv/bin/python experiments/bohn2021_reproduction/latency_tree_figures.py --mode register
.venv/bin/python experiments/bohn2021_reproduction/latency_tree_figures.py --mode check
.venv/bin/python experiments/bohn2021_reproduction/latency_tree_figures.py --mode prepare --split validation
~~~

之后先查看validation_figures下全部剖析报告，按profile_review_template.json写真实复核记录，再运行preview。对每张彩色和灰度图实际读图，依visual_review_template.json留下真实记录后再export；不能自动填“通过”。封存确认获准且实际完成后，才可对test采用同一流程。若上游失败，先审查失败和部分记录，不能绕过门禁绘制正式效果图。

## 最终报告与独立表格核对准备（05:04新增）

新增当前方法的逐项核对报告 `docs/reports/bohn2021_latency_tree_requirements_2026-09-26.md`，保留历史已冻结报告，不改写旧负结果。新表逐项区分已验证、待执行、证据不足与协议未完整实施，明确原SAC重建、新学习树、C5手工规则和整个用户目标的不同结论范围。

另新增 `latency_tree_final_report.py` 与 `latency_tree_final_report_review.py`。前者只在训练、评估及预算控制器实际完成退出后登记执行；生成全部主比较、全部21项C5/组合诊断、固定模型来源、逐项要求CSV与中文报告。后者不导入报告生成器或生产验收实现，独立核对精确失败率、共同路线、全部CSV/Markdown显示值、模型来源、预算和图表实物。bootstrap仍依赖已有独立效果实现，不称重新进行bootstrap或积分。

固定模型清单区分评价H和终端训练H，检查全部两任务×两比较器×三种子来源、15k训练凭据和model.zip哈希。图表要求明确给出最终导出目录，不自动挑选最后一次或最有利快照；核验剖析/彩色灰度复核链、全图集合、PNG真实尺寸/DPI/灰度类型、SVG无嵌入位图，PDF仅核对格式与已有合规报告，不能声称完成新的PDF字体全面检查或实际读图。

环境入口在实验结束后采集原实验Python3.7与审计解释器的包清单、解释器散列、CPU/OS和作者工作树，比较既有依赖要求和归档提交。任何不一致在报告中显式列出；此后验清点不是干净环境重建，也不证明整个训练期间包从未变化。旧runtime-freeze含本地构建URL，不能直接作为可移植锁文件。

特别保留协议偏差：每条件完整墙时与进程CPU没有完整仪表化；固定终端训练逐次NLP与构造器内部调用未知。自动报告不会补零或由局部时间推断。即使验证和确认都通过，输出仍保持goal_complete=false，要求最终科学解释、包级审查、实际可复运行性及用户目标逐项核验。

**两份新Python尚未执行语法检查、合成检查、环境采集、报告生成或真实独立复核。** 本轮仅源码准备和静态阅读，没有新增控制器、替换现有登记、打开测试或运行竞争Python。源码与准备记录保存在`report_preparation/`。05:04重验核心6份、v2登记11份、预算登记3份和上一轮绘图2份源码哈希不变；实际三个进程存活，两个stderr仍为空。

流水线和预算实际退出后，先后执行：

~~~bash
.venv/bin/python experiments/bohn2021_reproduction/latency_tree_final_report.py --mode register
.venv/bin/python experiments/bohn2021_reproduction/latency_tree_final_report.py --mode check
.venv/bin/python experiments/bohn2021_reproduction/latency_tree_final_report_review.py --check
.venv/bin/python experiments/bohn2021_reproduction/latency_tree_final_report.py --mode environment
~~~

全部实际图表导出与读图检查完成后，用`--mode build --validation-figures <明确的final目录>`生成新编号快照；只有本轮获准完成独立确认时才提供`--test-figures`。随后对该`final_delivery_XX`执行`latency_tree_final_report_review.py --folder <明确的交付目录>`。这些入口没有自动接入正在等待的实验控制器，不能假定会自行生成最终交付。

## 会话同步与执行接口静态核对（05:11新增）

已读取关联会话“推进 Bøhn 2021 复现”的任务目标及最近记录。其最后可见进度仍是旧门控验证8/84条件、270回合；该历史状态不能覆盖当前工作区的新方法进度。当前目标仍为active，尚未完成。

05:09核实训练PID 1694525及两个Windows控制器PID 26188、7108的实际命令匹配。评估控制器处于waiting_for_training_exit，预算控制器处于waiting_for_pipeline_exit，二者completed_stages均为空、stderr均为0字节。05:10再次读取日志，训练仍产生新的完整候选记录。训练和评估状态均未标记访问测试。

核对了原生流程各阶段调用参数与对应Python命令行入口，以及检查、评估登记、训练审计、策略冻结、完整训练交付及独立复核、新评估冒烟、验证、固定模型补全、双重复计时和条件式确认的执行顺序。本次静态核对未发现明显参数或依赖冲突；不等于Python检查或实际执行通过。工具输出曾让报告换行写法显得可疑，逐字符检查确认源码只有一个反斜杠，属于正确Python换行字面量，无需修复。

05:10—05:11原生哈希复核：训练核心6份、v2评估流程11份、预算流程3份均与各自登记一致。正式评估登记、评估检查结果、全训练审计、训练交付manifest、验证效果、确认登记和最终预算独立复核均尚未生成。未启动竞争Python、未修改冻结源码或登记、未重启任何已有进程。

## 预定后续顺序

1. 等六个训练作业完成并实际退出。运行新评估与交付合成检查；通过后登记评估实现、随机种子、固定模型清单。
2. 全训练轨迹审计与学习选择独立复核；冻结六个最终策略。生成全288候选、条件、回合、预算CSV及中文训练报告，并由独立实现复核覆盖和原始求解计数；交付manifest和复核通过后才允许新评估。若选到固定回退，明确记录学习未成功，仍保留完整评价。
3. 新评估冒烟16条件×2场景×2重放，共64回合，覆盖两任务三种子、独立终端和C5两臂。独立审计通过才运行验证。
4. 新验证每任务64场景，初始90条件共5,760回合。独立固定基线完整H5/10/…/50的seed0网格；共用终端完整10H×三种子。按严格安全条件下最低原始总成本提名。
5. 提名H若缺seed1/2独立终端，各按原设置训练15k步并逐转移审计；复用已存在且合格的模型。附带有/无终端各64验证回合仅作最终诊断，不选checkpoint。随后完成新终端的正式配对验证。
6. 全轨迹审计后，对主学习、两类已选固定及倒立摆C5/组合进行两个随机顺序的串行计时重复。保留毛/净决策延迟、日志扣除、reset边界，非时间轨迹必须与控制评估精确一致。
7. 每任务全部三种子、两比较器使用共同的成本或延迟路线：成本改善≥3%且配对95%差值上界<0；或实测决策延迟改善≥10%、比值95%上界<1、两次重复均更快且总成本NI≤2%。两路线都另要求成功、约束、初始/最终失败率不劣、物理+约束成本NI≤2%及回合内H变化。
8. 独立实现重算所有主比较和21项C5诊断比较。C5/组合不进入主门槛。仅完整正验证且复核通过才冻结并运行每任务128场景独立确认；同样标准再验一次。否则保持测试封存。

场景是bootstrap重采样单位；同一抽样索引在三训练种子之间共享。两次计时重复和控制步不当成新增独立场景或训练种子。区间条件化于已经训练好的模型，不能推断广大种子总体。

## 预算与未完成事项

训练上界4,920回合/738,000控制步，另556次场景生成reset/热身；这是保守上界，不是已经执行的实际预算。继承模型训练、新策略搜索、重复候选、基线补训、额外诊断、正式验证/计时、离线积分须分别披露。构造器未仪表化调用标为未知，不记为零。源码审查另确认，每条件完整墙时和进程CPU时间没有被仪表化；最终只报告全训练控制器墙时、实测决策/求解时间及明确的未知项，不从局部求解时间推算CPU成本。

全部训练结束后的真实预算总表、全候选报告、最终冻结策略、正式验证、两次实际计时、独立确认、图表与最终交付尚未完成。现阶段总体目标未达成；不得将实现完成、单个条件改善或手工C5结果包装为复现成功。

## 运行入口

以下为顺序入口；**当前串行训练仍在运行时不要手工执行**。原生控制器登记并启动后负责执行，避免重复运行。

~~~bash
.venv/bin/python experiments/bohn2021_reproduction/latency_tree_evaluation_checks.py
.venv/bin/python experiments/bohn2021_reproduction/latency_tree_training_delivery.py --check
.venv/bin/python experiments/bohn2021_reproduction/latency_tree_evaluation_spec.py --register
.venv/bin/python experiments/bohn2021_reproduction/latency_tree_audit.py --phase train
.venv/bin/python experiments/bohn2021_reproduction/latency_tree_learning_audit.py
.venv/bin/python experiments/bohn2021_reproduction/latency_tree_evaluation_spec.py --freeze-policies
.venv/bin/python experiments/bohn2021_reproduction/latency_tree_training_delivery.py
.venv/bin/python experiments/bohn2021_reproduction/latency_tree_training_delivery_review.py
~~~

完整后续顺序、每阶段解释器、错误处理及测试门禁见 experiments/bohn2021_reproduction/latency_tree_native_pipeline.ps1。若任何阶段失败，先读取对应日志和部分记录再决定修复；不得盲目重跑。

整条实验流水线结束后的预算入口（已由独立原生收尾控制器等待执行，不要与控制器重复启动）：

~~~bash
.venv/bin/python experiments/bohn2021_reproduction/latency_tree_final_budget.py --check
.venv/bin/python experiments/bohn2021_reproduction/latency_tree_final_budget.py --register
.venv/bin/python experiments/bohn2021_reproduction/latency_tree_final_budget.py
.venv/bin/python experiments/bohn2021_reproduction/latency_tree_final_budget_review.py
~~~
