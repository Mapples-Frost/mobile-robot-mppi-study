"""Change only terminal initialization; continue the same joint fitted updates."""
import hashlib
import numpy as np
import run
import recoverable_runtime as distribution
from runtime import ART,imports
from optimized_runtime import install_terminal
from optimized_evaluate import evaluate
from riccati_terminal_probe import prior

if __name__=='__main__':
    distribution.OUT=ART/'results/prior_refinement'
    install_terminal('pendulum');distribution.install_distribution()
    import tensorflow as tf
    w,b,details=prior();get_variable=tf.get_variable
    def initialized(name,*a,**kw):
        if tf.get_variable_scope().name.endswith('mpc_value_fns/mpc_value_fn') and 'initializer' in kw:
            if name=='kernel':kw['initializer']=w.astype(np.float32)
            elif name=='bias':kw['initializer']=b.astype(np.float32)
        return get_variable(name,*a,**kw)
    tf.get_variable=initialized
    run.make_env=lambda task,seed,fixed_horizon=None,**kw:distribution.make_env(task,seed,fixed_horizon)
    run.evaluate=evaluate;original=run.write
    def write(path,data):
        if path.name=='manifest.json':
            data['forecast_refinement']={'observation':'Full50step reference preview,56D'}
            data['terminal_prior']={'initialization':'Discounted discrete Riccati linearization','joint_learning':True,**details}
            data['config_path']=str(distribution.prepare_config())
            data['config_sha256']=hashlib.sha256(distribution.prepare_config().read_bytes()).hexdigest()
            data['fidelity']='Explicit terminal initialization extension on reconstructed near-upright distribution.'
        original(path,data)
    run.write=write;run.main()
