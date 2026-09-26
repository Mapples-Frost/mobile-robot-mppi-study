"""Run author SAC with the explicitly improved reconstruction adapter."""
import argparse
import sys
import run
from optimized_runtime import install_terminal,make_env
from optimized_evaluate import evaluate

if __name__=='__main__':
    ap=argparse.ArgumentParser(add_help=False);ap.add_argument('--task',required=True)
    args,_=ap.parse_known_args()
    install_terminal(args.task)
    run.evaluate=evaluate
    run.make_env=lambda task,seed,fixed_horizon=None,**kw:make_env(task,seed,fixed_horizon)
    original_write=run.write
    def write(path,data):
        if path.name=='manifest.json':
            data['optimized_reconstruction']={
                'terminal':'PSD full quadratic over scaled current reference errors; matched TF/CasADi features',
                'terminal_target':'32-step joint fitted value learning; current/next references match the corresponding physical time',
                'mpc_terminal_reference':'Reference at selected horizon endpoint, not final trajectory goal',
                'solver':'Vehicle three deterministic initial guesses; pendulum one deterministic initial guess; feasible lowest NLP objective',
                'observation':'Original scaled state plus MPC-available forecast perturbation, reference previews, goal offset and remaining time',
                'reset':'Common zero-terminal H50 warmup',
                'cost':'Paper H proxy retained; extra solve count and timing must be disclosed, not claimed as wall-time speedup'}
            data['fidelity']='Core-hypothesis reconstruction with explicit method extensions; not exact original algorithm.'
        original_write(path,data)
    run.write=write
    run.main()
