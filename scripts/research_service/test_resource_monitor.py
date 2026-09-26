import unittest,tempfile,pathlib,json
import resource_monitor as m
class MetricsTests(unittest.TestCase):
 def test_cpu_denominators(self):
  before={'cpu_ticks':[10,0,10,80,0,0,0,0],'monotonic_s':10,'pid':9,'process_start_ticks':2,'process_ticks':100}
  after={'cpu_ticks':[30,0,10,160,0,0,0,0],'monotonic_s':12,'pid':9,'process_start_ticks':2,'process_ticks':100+m.os.sysconf('SC_CLK_TCK'),'logical_cpus':2}
  r=m.delta(before,after);self.assertEqual(r['host_cpu_utilization_percent'],20);self.assertEqual(r['process_cpu_percent_one_vcpu'],50);self.assertEqual(r['process_cpu_percent_instance'],25)
 def test_pid_reuse(self):
  a={'pid':1,'process_start_ticks':1};b={'pid':1,'process_start_ticks':2,'cpu_ticks':[0]*8};self.assertNotIn('process_cpu_percent_instance',m.delta(a,b))
 def test_missing_credit_is_not_zero(self):
  with tempfile.TemporaryDirectory() as folder:
   old=m.TELEMETRY;m.TELEMETRY=pathlib.Path(folder)/'telemetry'
   try:
    m.link_run(pathlib.Path(folder),'2026-09-26T10:00:00+00:00','2026-09-26T11:00:00+00:00')
    r=json.loads((pathlib.Path(folder)/'cloudwatch_snapshot.json').read_text());self.assertEqual(r['series']['CPUCreditBalance'],[]);self.assertEqual(r['collection_status'],'not_collected');self.assertFalse(r['finalized'])
   finally:m.TELEMETRY=old
 def test_charged_statistic(self):self.assertEqual(m.METRICS['CPUSurplusCreditsCharged'],'Sum')
 def test_actual_sample(self):
  with tempfile.TemporaryDirectory() as folder:
   sampler=m.ResourceSampler(m.os.getpid(),pathlib.Path(folder)/'samples.jsonl');sampler.sample();self.assertEqual(sampler.summary()['sample_count'],1)
if __name__=='__main__':unittest.main()
