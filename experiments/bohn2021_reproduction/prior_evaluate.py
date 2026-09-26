import optimized_evaluate as ev
import recoverable_runtime as distribution
from runtime import ART

if __name__=='__main__':
    distribution.OUT=ART/'results/prior_refinement'
    distribution.install_distribution();ev.make_env=distribution.make_env;ev.main()
