"""Verify the actual author step entry point without constructing/stepping a plant."""
import ast
import pathlib
import textwrap
import unittest
import numpy as np

ROOT=pathlib.Path('/data/openai-agent/mobile-robot-mppi-study')
SOURCE=ROOT/'research_artifacts/bohn2021_reproduction_2026-09-17/sources/gym-horizon/gym_let_mpc/let_mpc.py'


class BoundaryReached(Exception):pass


class AuthorActionTests(unittest.TestCase):
    def setUp(self):
        text=SOURCE.read_text();tree=ast.parse(text)
        cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='LetMPCEnv')
        func=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='step')
        end=next((x.lineno-1 for x in cls.body if isinstance(x,ast.FunctionDef) and x.lineno>func.lineno),len(text.splitlines()))
        source=textwrap.dedent('\n'.join(text.splitlines()[func.lineno-1:end]));namespace={'np':np}
        exec(compile(source,str(SOURCE),'exec'),namespace)
        self.step=namespace['step'];self.received=[]
        class FakeControl:
            def step(inner,h):self.received.append(h);raise BoundaryReached()
        class FakeEnv:pass
        self.env=FakeEnv();self.env.config={'environment':{'action':{'variables':[{'name':'mpc_horizon'}]}}};self.env.control_system=FakeControl()
    def test_scalar_horizon_reproduces_failure_before_controller_boundary(self):
        with self.assertRaises(TypeError):self.step(self.env,12.0)
        self.assertEqual(self.received,[])
    def test_one_element_action_reaches_controller_with_each_correct_horizon(self):
        for h in [12,15,35]:
            with self.subTest(h=h),self.assertRaises(BoundaryReached):self.step(self.env,np.asarray([float(h)],dtype=np.float32))
            self.assertEqual(int(self.received[-1]),h)


if __name__=='__main__':unittest.main()
