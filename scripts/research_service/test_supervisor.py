"""Supervisor self-tests without external API requests."""
import importlib.util,pathlib,tempfile,json,unittest
spec=importlib.util.spec_from_file_location('worker',pathlib.Path(__file__).with_name('orchestrator.py'));w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)
class SafetyChecks(unittest.TestCase):
 def test_escape(self):
  for name in ['../../.secrets/agent.env','/etc/passwd','.git/config']:
   with self.assertRaises(ValueError):w.safe_path(name)
 def test_redaction(self):
  for secret in w.REDACT:self.assertNotIn(secret,w.redact('before '+secret+' after'))
 def test_tools(self):
  self.assertEqual({t['name'] for t in w.TOOLS},{'list_files','read_file','write_file','run_experiment','update_state'})
 def test_protect_supervisor(self):
  with self.assertRaises(ValueError):w.call_tool('write_file',{'path':'scripts/research_service/orchestrator.py','content':'bad'})
 def test_secret_write(self):
  with self.assertRaises(ValueError):w.call_tool('write_file',{'path':'never-written.txt','content':w.REDACT[0]})
 def test_final_gate(self):
  with self.assertRaises(ValueError):w.call_tool('update_state',{'state':{'final_test_authorized':True}})
 def test_json_state(self):
  with tempfile.TemporaryDirectory() as d:
   p=pathlib.Path(d)/'state.json';w.dump(p,{'phase':'audit','queue':[]});self.assertEqual(w.load(p)['phase'],'audit')
 def test_model(self):self.assertEqual(w.SECRET['OPENAI_MODEL'],'gpt-5.5');self.assertEqual(w.load(w.STATE/'api_smoke.json')['selected_effort'],'xhigh')
if __name__=='__main__':unittest.main()
