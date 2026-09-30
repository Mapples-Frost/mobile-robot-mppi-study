"""Isolated failover tests: no remote calls, no production state or credentials."""
import io
import http.client
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
        self.headers = {'Content-Type':'application/json'}


class StreamResponse(io.BytesIO):
    def __init__(self, events):
        super().__init__(events.encode('utf-8'))
        self.headers = {'Content-Type':'text/event-stream'}


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

    def test_incomplete_http_body_is_an_endpoint_outage(self):
        with patch.object(router.urllib.request,'urlopen',side_effect=[http.client.IncompleteRead(b'partial'),Response(answer())]) as transport:
            result=router.request(self.body)
        self.assertEqual(result['_research_route']['endpoint'],'backup')
        self.assertEqual(transport.call_count,2)

    def test_malformed_response_is_rejected_and_rerouted(self):
        with patch.object(router.urllib.request,'urlopen',side_effect=[Response([]),Response(answer())]):
            result=router.request(self.body)
        self.assertEqual(result['_research_route']['endpoint'],'backup')

    def test_stale_failure_cannot_undo_newer_primary_recovery(self):
        error=router.EndpointError('older transport failure')
        router.success('primary','primary_recovery_probe',error.observed_at+1)
        router.failure('primary',error)
        self.assertEqual(router.state()['active_endpoint'],'primary')
        self.assertTrue(router.state()['endpoints']['primary']['healthy'])

    def test_stale_success_cannot_undo_newer_primary_failure(self):
        error=router.EndpointError('newer transport failure')
        router.failure('primary',error)
        router.success('primary','review',error.observed_at-1)
        self.assertEqual(router.state()['active_endpoint'],'backup')
        self.assertFalse(router.state()['endpoints']['primary']['healthy'])

    def test_stream_completed_preserves_exact_tool_output(self):
        completed=answer()
        completed['output']=[dict(type='function_call',call_id='opaque-id',name='read_file',arguments='{"path":"a.py"}')]
        events=': keepalive\n\nevent: response.created\ndata: '+json.dumps({'type':'response.created','response':{'status':'in_progress'}})+'\n\ndata: '+json.dumps({'type':'response.completed','response':completed})+'\n\n'
        self.assertEqual(router.parse_stream(StreamResponse(events)),completed)

    def test_stream_does_not_publish_partial_text(self):
        events='data: '+json.dumps({'type':'response.output_text.delta','delta':'partial report'})+'\n\ndata: [DONE]\n\n'
        with self.assertRaisesRegex(router.EndpointError,'before response.completed'):
            router.parse_stream(StreamResponse(events))

    def test_stream_failure_is_not_completion(self):
        events='data: '+json.dumps({'type':'response.failed','response':{'error':{'message':'upstream failure'}}})+'\n\n'
        with self.assertRaisesRegex(router.EndpointError,'stream failure'):
            router.parse_stream(StreamResponse(events))

    def test_stream_malformed_json_rejected(self):
        with self.assertRaisesRegex(router.EndpointError,'Malformed JSON'):
            router.parse_stream(StreamResponse('data: broken-json\n\n'))

    def test_stream_handles_multiline_json_and_utf8(self):
        completed=answer();completed['output'][0]['content'][0]['text']='English output: café'
        frame=json.dumps({'type':'response.completed','response':completed},ensure_ascii=False,indent=2)
        events='\n'.join('data: '+line for line in frame.splitlines())+'\n\n'
        self.assertEqual(router.parse_stream(StreamResponse(events)),completed)

    def test_streamed_request_is_accounted_once_and_keeps_model_pins(self):
        completed=answer()
        events='data: '+json.dumps({'type':'response.completed','response':completed})+'\n\n'
        with patch.object(router.urllib.request,'urlopen',return_value=StreamResponse(events)) as transport:
            result=router.request(self.body)
        wire=json.loads(transport.call_args.args[0].data)
        self.assertTrue(wire['stream']);self.assertEqual(wire['model'],router.MODEL)
        self.assertEqual(wire['reasoning']['effort'],'max')
        self.assertEqual(result['output'],completed['output'])
        self.assertEqual(len(self.rows()),1)
        self.assertEqual(json.loads(self.rows()[0][3])['total_tokens'],7)


if __name__=='__main__': unittest.main()
