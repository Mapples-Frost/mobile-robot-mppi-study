import pathlib
import sys
import os
import tempfile
import unittest
import execution_preflight as pre
TEST_ENV={k:v for k,v in os.environ.items() if k.upper() in ('PATH','SYSTEMROOT')}


class PreflightTests(unittest.TestCase):
    def test_unterminated_string_is_rejected(self):
        with self.assertRaises(SyntaxError): pre.validate_source('x="unterminated','probe.py')

    def test_invalid_top_level_return_is_rejected(self):
        with self.assertRaises(SyntaxError): pre.validate_source('return 1','probe.py')

    def test_known_path_formatting_failure_is_rejected(self):
        with self.assertRaises(ValueError): pre.validate_source('run_dir = ROOT / "name_%s" % stamp','probe.py')

    def test_correct_parenthesized_path_formatting_is_accepted(self):
        pre.validate_source('run_dir = ROOT / ("name_%s" % stamp)','probe.py')

    def test_checks_do_not_execute_top_level_source(self):
        with tempfile.TemporaryDirectory() as directory:
            p=pathlib.Path(directory)/'side_effect.py';marker=pathlib.Path(directory)/'marker'
            source='from pathlib import Path\nPath('+repr(str(marker))+').write_text("unexpected side effect")\nraise RuntimeError("This must never execute")\n'
            p.write_text(source);pre.validate_source(source,str(p));pre.validate_interpreter(sys.executable,p,TEST_ENV)
            self.assertFalse(marker.exists())

    def test_selected_interpreter_rejects_bad_file(self):
        with tempfile.TemporaryDirectory() as directory:
            p=pathlib.Path(directory)/'bad.py';p.write_text('def missing_colon()\n pass\n')
            with self.assertRaises(ValueError):pre.validate_interpreter(sys.executable,p,TEST_ENV)


if __name__=='__main__':unittest.main()
