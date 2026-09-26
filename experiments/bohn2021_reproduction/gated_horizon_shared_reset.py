"""Common primary-terminal initialization; own terminal restored for every scored action.

This adapter affects evaluation only. It preserves the author's unscored H50
reset step and the training distribution used by adaptive/matched-terminal arms.
Independently trained fixed baselines retain their own terminal during scoring.
"""
from runtime import imports
from min_q_eval_suite import model_dir
from paper_h_soft_probe import read
from run import weights_hash


def terminal(task,seed):
    source=model_dir(task,'fixed',seed);_,SAC,_=imports();model=SAC.load(str(source/'model.zip'))
    try:
        assert weights_hash(model)==read(source/'completed.json')['final_hash']
        return model.policy_tf.get_mpc_vfn_weights_and_biases()
    finally:model.sess.close()


def install(env,task,seed,own_terminal,independent_terminal=False):
    reference=terminal(task,seed) if independent_terminal else own_terminal
    reset=env.reset
    def shared_reset(**kwargs):
        env.set_value_function_weights_and_biases(*reference)
        try:return reset(**kwargs)
        finally:env.set_value_function_weights_and_biases(*own_terminal)
    env.reset=shared_reset
    return dict(reference_source=str(model_dir(task,'fixed',seed)),independent_terminal=independent_terminal,
        scope='Same primary fixed-terminal H50 reset step within each task/seed; restore own independently trained terminal before scoring. No learned horizon action runs during reset.')
