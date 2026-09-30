"""Isolated failover tests: no remote calls, no production state or credentials."""
import io
import json
import pathlib
import sqlite3
import tempfile
import unittest
import urllib.error
from unittest.mock import patch

import astra_routing as router


def answer(model='gpt-6-astra', effort='max'):
    return dict(model=model, reasoning={'effort': effort}, status='completed',
                usage={'total_tokens': 7}, output=[dict(type='message', content=[dict(type='output_text', text='OK')])])


class Response(io.BytesIO):
    def __init__(self, data):
        super().__init__(json.dumps(data).encode())


class RoutingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.original = {name: getattr(router, name) for name in ('BASE', 'STATE', 'WORK', 'ROUTE')}
        router.BASE = pathlib.Path(self.temp.name)
        router.STATE = router.BASE / 'state'
        router.WORK = router.STATE / 'astra_reviewer'
        router.ROUTE = router.STATE / 'astra_router'
        router.STATE.mkdir()
        secrets = router.BASE / '.secrets'; secrets.mkdir()
        self.fake_secret = 'sk-test-private-not-a-real-credential'
        for name, endpoint in [('reviewer.env','primary'),('astra_backup.env','backup')]:
            (secrets / name).write_text('REVIEWER_BASE_URL=https://%s.example/v1\nREVIEWER_API_KEY=%s\n' % (endpoint,self.fake_secret))
        with sqlite3.connect(router.STATE / 'research.sqlite') as c:
            c.execute('create table calls(id text primary key,timestamp text,model text,effort text,status text,usage text,duration real)')
        self.body = dict(model=router.MODEL, reasoning={'effort':router.EFFORT}, store=False,
                         input=[dict(role='user', content='test')],max_output_tokens=4096)
        router._wake.clear()

    def tearDown(self):
        for name,value in self.original.items(): setattr(router,name,value)
        self.temp.cleanup()

    def http_error(self, status):
        return urllib.error.HTTPError('https://primary.example',status,'simulated',{},io.BytesIO(b'upstream error'))

    def endpoints(self, transport):
        return [call.args[0].full_url.split('/')[2] for call in transport.call_args_list]

    def rows(self):
        with sqlite3.connect(router.STATE / 'research.sqlite') as c:
            return c.execute('select model,effort,status,usage from calls').fetchall()

    def test_primary_success_does_not_call_backup(self):
        with patch.object(router.urllib.request,'urlopen',return_value=Response(answer())) as transport:
            result=router.request(self.body)
        self.assertEqual(result['_research_route']['endpoint'],'primary')
        self.assertEqual(self.endpoints(transport),['primary.example'])
        self.assertEqual(len(self.rows()),1)

    def test_502_fails_over_with_identical_model_effort_and_context(self):
        with patch.object(router.urllib.request,'urlopen',side_effect=[self.http_error(502),Response(answer())]) as transport:
            result=router.request(self.body)
        self.assertEqual(result['_research_route']['endpoint'],'backup')
        self.assertEqual(self.endpoints(transport),['primary.example','backup.example'])
        self.assertEqual(transport.call_args_list[0].args[0].data,transport.call_args_list[1].args[0].data)
        self.assertEqual(router.state()['active_endpoint'],'backup')
        self.assertEqual([r[2] for r in self.rows()],['error','completed'])
        self.assertEqual(sum(json.loads(r[3]).get('total_tokens',0) for r in self.rows()),7)

    def test_recovery_switches_next_call_to_primary(self):
        router.failure('primary',router.EndpointError('old outage'))
        with patch.object(router.urllib.request,'urlopen',side_effect=[Response(answer()),Response(answer())]) as transport:
            self.assertTrue(router.probe_primary())
            result=router.request(self.body)
        self.assertEqual(result['_research_route']['endpoint'],'primary')
        self.assertEqual(self.endpoints(transport),['primary.example','primary.example'])
        self.assertTrue(router._wake.is_set())

    def test_failed_probe_keeps_backup(self):
        router.failure('primary',router.EndpointError('old outage'))
        with patch.object(router.urllib.request,'urlopen',side_effect=[self.http_error(502),Response(answer())]) as transport:
            self.assertFalse(router.probe_primary())
            result=router.request(self.body)
        self.assertEqual(result['_research_route']['endpoint'],'backup')
        self.assertEqual(self.endpoints(transport),['primary.example','backup.example'])

    def test_payload_error_does_not_reroute_or_mark_outage(self):
        with patch.object(router.urllib.request,'urlopen',side_effect=self.http_error(400)) as transport:
            with self.assertRaises(router.EndpointError): router.request(self.body)
        self.assertEqual(self.endpoints(transport),['primary.example'])
        self.assertEqual(router.state()['active_endpoint'],'primary')

    def test_other_model_is_rejected_even_if_provider_says_completed(self):
        with patch.object(router.urllib.request,'urlopen',side_effect=[Response(answer(model='gpt-5.5')),Response(answer(model='gpt-5.5'))]):
            with self.assertRaisesRegex(router.EndpointError,'model mismatch'): router.request(self.body)
        self.assertEqual(len(self.rows()),2)
        self.assertTrue(all(r[2]=='error' for r in self.rows()))

    def test_model_and_effort_pins_apply_to_every_call(self):
        with patch.object(router.urllib.request,'urlopen') as transport:
            for body in [dict(self.body,model='other'),dict(self.body,reasoning={'effort':'low'})]:
                with self.assertRaises(ValueError): router.once(body,'backup')
        transport.assert_not_called()
        self.assertEqual(self.rows(),[])

    def test_backup_finishing_cannot_undo_primary_recovery(self):
        router.failure('primary',router.EndpointError('old outage'))
        router.success('primary','primary_recovery_probe')
        router.success('backup','review')
        self.assertEqual(router.state()['active_endpoint'],'primary')

    def test_busy_probe_is_deferred_without_failure_or_usage_row(self):
        with patch.object(router,'once',side_effect=router.EndpointBusy('local lock')):
            self.assertIsNone(router.probe_primary())
        self.assertEqual(router.state()['active_endpoint'],'primary')
        self.assertNotIn('last_failure',router.state()['endpoints']['primary'])
        self.assertEqual(self.rows(),[])

    def test_both_endpoints_failed_stop_after_two_attempts(self):
        with patch.object(router.urllib.request,'urlopen',side_effect=[self.http_error(429),self.http_error(503)]) as transport:
            with self.assertRaisesRegex(router.EndpointError,'Both Astra endpoints'): router.request(self.body)
        self.assertEqual(transport.call_count,2)

    def test_secrets_redacted_in_state_and_events(self):
        router.failure('primary',router.EndpointError('error '+self.fake_secret))
        for name in ['status.json','events.jsonl']:
            self.assertNotIn(self.fake_secret,(router.ROUTE/name).read_text())


if __name__=='__main__': unittest.main()
