"""Train the full-reference-preview adapter, preserving all other settings."""
import run
from optimized_runtime import install_terminal
from forecast_runtime import make_env
from optimized_evaluate import evaluate

if __name__=='__main__':
    install_terminal('pendulum');run.make_env=lambda task,seed,fixed_horizon=None,**kw:make_env(task,seed,fixed_horizon)
    run.evaluate=evaluate
    original=run.write
    def write(path,data):
        if path.name=='manifest.json':
            data['forecast_refinement']={'observation':'Original five features plus 50 relative reference previews and remaining time; 56D',
                'terminal':'Unchanged optimized PSD quadratic','entropy':.01,'fidelity':'Explicit method extension; not exact original paper.'}
        original(path,data)
    run.write=write;run.main()
