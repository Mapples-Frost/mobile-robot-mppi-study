import json
import pathlib
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.append('/data/openai-agent/mobile-robot-mppi-study/scripts/research_service')
import orchestrator as worker


class ReceiptTests(unittest.TestCase):
    def test_preserves_verification_time_and_reuses_exact_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            base=pathlib.Path(directory);root=base/'repo';root.mkdir();state=base/'state';state.mkdir()
            value={'status':'verified','remaining_changed_files':0,'commit':'fixture','time':'2026-09-30T17:00:00+00:00','packages_this_run':[{'sha256':'a'*64,'verification':'github_server_sha256'}]}
            (state/'backup_status.json').write_text(json.dumps(value))
            with patch.multiple(worker,ROOT=root,STATE=state):
                first=worker.materialize_verified_backup();second=worker.materialize_verified_backup()
            self.assertEqual(first,second);result=json.loads(first.read_text())
            self.assertEqual(result['time'],value['time']);self.assertTrue(result['backup_verified'])
    def test_failed_backup_is_never_materialized_as_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            base=pathlib.Path(directory);state=base/'state';state.mkdir();(state/'backup_status.json').write_text('{"status":"failed"}')
            with patch.multiple(worker,ROOT=base/'repo',STATE=state):self.assertIsNone(worker.materialize_verified_backup())


if __name__=='__main__':unittest.main()
