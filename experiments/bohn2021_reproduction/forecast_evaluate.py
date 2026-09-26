"""Select the correct observation adapter when evaluating saved models."""
import json
import sys
from pathlib import Path
import optimized_evaluate as ev
from forecast_runtime import make_env

if __name__=='__main__':
    folder=Path(sys.argv[sys.argv.index('--model-dir')+1])
    if 'forecast_refinement' in json.loads((folder/'manifest.json').read_text()):ev.make_env=make_env
    ev.main()
