"""Explicit scenario-distribution reconstruction; plant and rewards unchanged."""
import json
from runtime import ART,imports
from forecast_runtime import make_env

OUT=ART/'results/recoverable_distribution'


def prepare_config():
    OUT.mkdir(exist_ok=True);path=OUT/'pendulum.json'
    if not path.exists():
        config=json.loads((ART/'configs/pendulum.json').read_text())
        bounds={'pos':.3,'v':.5,'theta':.35,'omega':.5}
        for name,bound in bounds.items():config['environment']['randomize']['state'][name]['kw']={'low':-bound,'high':bound}
        path.write_text(json.dumps(config,indent=2)+'\n')
    return path


def install_distribution():
    path=prepare_config();Env,_,_=imports();original=Env.__init__
    def init(self,config,*a,**kw):
        if str(config)==str(ART/'configs/pendulum.json'):config=str(path)
        original(self,config,*a,**kw)
    Env.__init__=init
