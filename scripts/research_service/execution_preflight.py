"""Cheap source checks before registration or any scientific resource use."""
import ast
import subprocess


def validate_source(source, filename):
    # Compile only: never execute top-level code or import scientific modules.
    compile(source, filename, 'exec', dont_inherit=True)
    tree = ast.parse(source, filename=filename)
    for node in ast.walk(tree):
        # ROOT / "name_%s" % value formats a Path, not the string. This exact
        # precedence bug already caused an uninformative production failure.
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod) and isinstance(node.left, ast.BinOp) and isinstance(node.left.op, ast.Div):
            right=node.left.right
            is_string=isinstance(getattr(right,'value',None),str) or (type(right).__name__=='Str' and isinstance(getattr(right,'s',None),str))
            if is_string:raise ValueError('Path formatting precedence at line %s: parenthesize the formatted string before path division' % node.lineno)


def validate_interpreter(python, script, env):
    result = subprocess.run([python, '-c',
        "import sys; p=sys.argv[1]; compile(open(p, encoding='utf-8').read(), p, 'exec', dont_inherit=True)",
        str(script)], capture_output=True, text=True, timeout=30, env=env)
    if result.returncode:
        raise ValueError('Selected-interpreter parse check failed before experiment launch: ' + result.stderr[-1800:])
