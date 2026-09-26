"""Load the pinned author SAC with only its actor Q1 target changed to min-Q."""
import ast
import hashlib
import sys
import types
from pathlib import Path

from runtime import ART, imports


SOURCE = ART / "sources/stable-baselines-horizon/stable_baselines/sac/sac.py"


class ActorTarget(ast.NodeTransformer):
    def __init__(self):
        self.replacements = 0

    def visit_Assign(self, node):
        self.generic_visit(node)
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            return node
        if node.targets[0].id != "policy_kl_loss":
            return node
        value = node.value
        if not (isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute)
                and value.func.attr == "reduce_mean" and len(value.args) == 1
                and isinstance(value.args[0], ast.BinOp)
                and isinstance(value.args[0].op, ast.Sub)
                and isinstance(value.args[0].right, ast.Name)
                and value.args[0].right.id == "qf1_pi"):
            raise RuntimeError("Author actor objective changed upstream")
        value.args[0].right.id = "min_qf_pi"
        self.replacements += 1
        return node


def source_sha256():
    return hashlib.sha256(SOURCE.read_bytes()).hexdigest()


def load_min_q_sac():
    imports()
    source = SOURCE.read_text()
    tree = ast.parse(source, filename=str(SOURCE))
    transformer = ActorTarget()
    tree = transformer.visit(tree)
    if transformer.replacements != 1:
        raise RuntimeError("Expected exactly one author actor objective")
    ast.fix_missing_locations(tree)
    module = types.ModuleType("bohn2021_min_q_sac")
    module.__file__ = str(SOURCE)
    module.__dict__["AUTHOR_SOURCE_SHA256"] = source_sha256()
    sys.modules[module.__name__] = module
    exec(compile(tree, str(SOURCE), "exec"), module.__dict__)
    return module.SAC
