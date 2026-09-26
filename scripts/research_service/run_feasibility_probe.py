import sys,json
sys.path.insert(0,'/data/openai-agent/mobile-robot-mppi-study/scripts/research_service')
import orchestrator as w
result=w.execute(dict(script='experiments/bohn2021_aws/training_feasibility_probe.py',interpreter='legacy',args=[],method='engineering short SAC training feasibility; not formal ORIGINAL result',seed='2609268801',split='diagnostic',purpose='Measure actual learning updates, memory and 200 control steps per task without any validation/test access',config={'steps_each':200,'fixed_horizons':{'vehicle':25,'pendulum':30},'learning_starts':100,'batch_size':64,'buffer_size':2000},training_budget={'steps_max':400,'tasks':2},validation_budget={'episodes':0},test_budget={'episodes':0},artifacts=['research_artifacts/aws_diagnostics/training_feasibility_20260926/summary.json'],timeout_seconds=900))
print(json.dumps(result));sys.exit(0 if result['exit_status']==0 else 1)
