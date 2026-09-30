"""Isolated scheduler integration; subprocesses only write synthetic zero-usage receipts."""
import json
import copy
import pathlib
import subprocess
import sys
import unittest
from unittest.mock import patch
from test_execution_contract import ContractTests

# Support dependencies come from the production source, without invoking main/API.
sys.path.append('/data/openai-agent/mobile-robot-mppi-study/scripts/research_service')
import orchestrator as worker
import opus_lead as lead


class IntegrationTests(ContractTests):
    def test_real_scheduler_launch_receipt_glob_and_no_duplicate_handoff(self):
        dependent=copy.deepcopy(self.task);dependent.update(task_id='C0',dependencies=['B6'])
        self.plan['tasks'].append(dependent);self.ready=self.publish(self.plan)
        source = '''import pathlib, execution_contract as ec
ROOT=pathlib.Path(__file__).resolve().parents[1]
ec.runtime_snapshot(ROOT)
p=ROOT/'research_artifacts/diagnostic_smoke_20260101';p.mkdir(parents=True)
(p/'completed.json').write_text('{"G2":true}')
ec.record_outcome(ROOT,'scientific_result',{k:0 for k in ec.UNITS},{'G2':True})
'''
        (self.root / self.script).write_text(source)
        old = self.root / 'research_artifacts/diagnostic_smoke_20250101'; old.mkdir(parents=True)
        (old / 'completed.json').write_text('{}')
        import os
        os.utime(old, (1,1)); os.utime(old / 'completed.json', (1,1))
        ready_path = self.root / 'docs/PLAN_READY.json'; worker.dump(ready_path, self.ready)
        worker.dump(self.state / 'research_roles.json', dict(status='active', active_lead='claude-opus-5-5', lead_ready_path='docs/PLAN_READY.json'))
        real_popen = subprocess.Popen
        def isolated_popen(command, **kw):
            if command[1]=='-u':command[2] = str(self.root / self.script)
            kw['cwd'] = str(self.root)
            return real_popen(command, **kw)
        args = dict(self.args, interpreter='legacy', purpose='Isolated infrastructure integration',
                    artifacts=['research_artifacts/diagnostic_smoke_*/completed.json'])
        with patch.multiple(worker, ROOT=self.root, STATE=self.state, BASE=self.state, SERVICE=pathlib.Path(__file__).resolve().parent), \
             patch.object(worker,'identity',return_value={'commit_sha':'synthetic','dirty_state':''}), \
             patch.object(worker,'link_run',return_value={'status':'synthetic_no_cloudwatch'}), \
             patch.object(worker.subprocess,'Popen',side_effect=isolated_popen):
            result = worker.execute(args)
            self.assertEqual(result['exit_status'], 0)
            self.assertEqual(result['coordination']['handoff'], 'preauthorized_continuation')
            meta = worker.load(self.root / result['record'])
            matched = [x['path'] for x in meta['artifact_inventory'] if x.get('exists')]
            self.assertEqual(matched, ['research_artifacts/diagnostic_smoke_20260101/completed.json'])
            self.assertFalse((self.root/'docs/bohn2021_takeover/astra_reviews/NEXT_REVIEW_REQUEST.json').exists())
            self.assertEqual(worker.load(self.state/'active_experiment.json')['status'], 'idle')

    def test_native_tool_schema_and_publication_preserve_roles(self):
        from jsonschema import Draft202012Validator
        schema = next(t['input_schema'] for t in lead.TOOLS if t['name']=='submit_execution_plan')
        Draft202012Validator(schema).validate(self.plan)
        out = self.root/'docs/bohn2021_takeover/opus_lead'; out.mkdir(parents=True)
        work = self.state/'opus_lead'; work.mkdir()
        roles = self.state/'research_roles.json'; lead.save(roles,{'communication_language':'en','activated':'historical'})
        lead.save(work/'plan_drafts/audit.execution.json', self.plan)
        checkpoint = dict(audit_id='audit.execution',request={'request_id':'request-one'},started='synthetic')
        with patch.multiple(lead,ROOT=self.root,STATE=self.state,OUT=out,WORK=work,ROLES=roles,REQUEST=self.root/'docs/NEXT_REVIEW_REQUEST.json'), \
             patch.object(lead.evidence,'git',return_value='synthetic'):
            lead.publish(checkpoint,'Substantive synthetic report',[])
        ready = ec.read(out/'PLAN_READY.json')
        self.assertEqual(ec.plan_from_ready(self.root, ready), self.plan)
        self.assertEqual(ec.read(roles)['communication_language'],'en')


import execution_contract as ec
if __name__=='__main__': unittest.main()
