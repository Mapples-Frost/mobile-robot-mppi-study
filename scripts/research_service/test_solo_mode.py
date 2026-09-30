"""Temporary user-authorized solo mode; no API calls or scientific execution."""
import hashlib
import pathlib
import sys
import unittest
from unittest.mock import patch
from test_execution_contract import ContractTests
sys.path.append('/data/openai-agent/mobile-robot-mppi-study/scripts/research_service')
import orchestrator as worker
import execution_contract as ec


class SoloTests(ContractTests):
    def setUp(self):
        super().setUp()
        self.patch=patch.multiple(worker,ROOT=self.root,STATE=self.state,BASE=self.state,SERVICE=pathlib.Path(__file__).resolve().parent)
        self.patch.start();self.addCleanup(self.patch.stop)
        name='docs/bohn2021_takeover/solo_gpt55/ROLE_OVERRIDE_test.json'
        self.authority=self.root/name
        ec.write(self.authority,dict(requested_by='user',authorized_lead='gpt-5.5',mode='temporary_user_authorized_solo',final_test_authorized=False))
        ec.write(self.root/'docs/PLAN_READY.json',self.ready)
        self.roles=dict(status='active',active_lead='gpt-5.5',mode='temporary_user_authorized_solo',
            lead_ready_path='docs/PLAN_READY.json',user_authorization=name,
            user_authorization_sha256=hashlib.sha256(self.authority.read_bytes()).hexdigest())
        ec.write(self.state/'research_roles.json',self.roles)
        ec.write(self.root/'docs/bohn2021_takeover/astra_reviews/NEXT_REVIEW_REQUEST.json',{'request_id':'request-one'})

    def publish_solo(self):
        return worker.call_tool('submit_solo_execution_plan',dict(plan=self.plan,
            rationale='Synthetic evidence-linked bounded development plan: preserve all scientific constraints and resource limits, keep failed evidence, and defer independent final acceptance until the user restores reviewers.'))

    def test_solo_role_does_not_fall_back_to_astra_or_wait_for_absent_lead(self):
        ec.write(self.root/'docs/bohn2021_takeover/astra_reviews/NEXT_REVIEW_REQUEST.json',{'request_id':'new-outcome'})
        self.assertEqual(worker.current_roles()['active_lead'],'gpt-5.5')
        self.assertFalse(worker.awaiting_astra_analysis())

    def test_native_solo_publication_and_authorized_snapshot_work(self):
        result=self.publish_solo();ready=result['ready']
        self.assertEqual(ec.plan_from_ready(self.root,ready),self.plan)
        self.assertEqual(ready['primary_analyst'],'gpt-5.5');self.assertFalse(ready['independent_audit_passed'])
        snapshot=self.start(ready=ready)
        self.assertEqual(ec.verify_snapshot(self.root,self.state/'e1.json'),snapshot)
        roles=worker.current_roles();roles.update(active_lead='claude-opus-5-5',mode='restored')
        ec.write(self.state/'research_roles.json',roles)
        # A launched immutable authorization remains verifiable after role restoration.
        self.assertEqual(ec.verify_snapshot(self.root,self.state/'e1.json'),snapshot)

    def test_user_authorization_tamper_is_rejected(self):
        ready=self.publish_solo()['ready'];self.authority.write_text('{}')
        with self.assertRaises(ValueError):ec.plan_from_ready(self.root,ready)

    def test_solo_cannot_grant_final_tests_or_independent_acceptance(self):
        for field in ['final_test_authorized','independent_audit_passed']:
            with self.subTest(field=field),self.assertRaises(ValueError):worker.call_tool('update_state',{'state':{field:True}})

    def test_solo_publication_disabled_when_normal_roles_resume(self):
        roles=dict(self.roles,active_lead='claude-opus-5-5',mode='restored')
        ec.write(self.state/'research_roles.json',roles)
        with self.assertRaises(ValueError):self.publish_solo()

    def test_publications_cannot_be_overwritten_with_general_write_tool(self):
        with self.assertRaises(ValueError):worker.call_tool('write_file',{'path':self.roles['user_authorization'],'content':'{}'})


if __name__=='__main__':unittest.main()
