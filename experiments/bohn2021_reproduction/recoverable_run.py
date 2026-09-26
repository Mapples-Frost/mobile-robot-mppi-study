"""One-factor initial-distribution experiment with matched fixed-H baselines."""
import hashlib
import run
from optimized_runtime import install_terminal
from recoverable_runtime import install_distribution,make_env,prepare_config
from optimized_evaluate import evaluate

if __name__=='__main__':
    install_terminal('pendulum');install_distribution()
    run.make_env=lambda task,seed,fixed_horizon=None,**kw:make_env(task,seed,fixed_horizon);run.evaluate=evaluate
    original=run.write
    def write(path,data):
        if path.name=='manifest.json':
            data['forecast_refinement']={'observation':'Complete50step preview,56D'}
            data['recoverable_distribution']={'initial_state_bounds':{'pos':.3,'v':.5,'theta':.35,'omega':.5},
                'unchanged':'Dynamics,reward,constraints,reference process,terminal,training budget',
                'limitation':'Conservative near-upright distribution, not a formal viability certificate or original author distribution.'}
            data['config_path']=str(prepare_config());data['config_sha256']=hashlib.sha256(prepare_config().read_bytes()).hexdigest()
        original(path,data)
    run.write=write;run.main()
