import forecast_evaluate
import optimized_evaluate as ev
from optimized_runtime import install_terminal
from recoverable_runtime import install_distribution,make_env

if __name__=='__main__':
    install_distribution();ev.make_env=make_env;ev.main()
