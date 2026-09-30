import pathlib
import tempfile
import unittest
import sys
from unittest.mock import patch
sys.path.append('/data/openai-agent/mobile-robot-mppi-study/scripts/research_service')
import astra_reviewer as reviewer


class PauseTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.state=pathlib.Path(self.tmp.name);self.work=self.state/'astra_reviewer';self.work.mkdir()
        self.patch=patch.multiple(reviewer,STATE=self.state,WORK=self.work);self.patch.start();self.addCleanup(self.patch.stop)

    def test_both_provider_outage_pauses_only_with_user_authorization(self):
        reviewer.save(self.state/'astra_pause_policy.json',{'user_authorized':True,'condition':'both_endpoints_unavailable'})
        self.assertTrue(reviewer.pause_on_provider_outage('Both Astra endpoints unavailable: primary: 429; backup: timeout'))
        self.assertTrue(reviewer.load(self.state/'astra_pause_state.json')['paused'])
        self.assertEqual(reviewer.load(self.work/'status.json')['status'],'paused_by_user_condition')

    def test_primary_only_error_or_unrelated_bug_does_not_pause(self):
        reviewer.save(self.state/'astra_pause_policy.json',{'user_authorized':True,'condition':'both_endpoints_unavailable'})
        for message in ['primary: API_HTTP_429','All endpoints busy','TypeError: unrelated script error']:
            self.assertFalse(reviewer.pause_on_provider_outage(message))
        self.assertFalse((self.state/'astra_pause_state.json').exists())

    def test_no_implicit_permission_to_pause(self):
        self.assertFalse(reviewer.pause_on_provider_outage('Both Astra endpoints unavailable: failure'))


if __name__=='__main__':unittest.main()
