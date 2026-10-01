import datetime as dt
import unittest
from progress_watchdog import evaluate,UNITS


class ProgressTests(unittest.TestCase):
    def setUp(self):self.now=dt.datetime(2026,10,1,2,tzinfo=dt.timezone.utc)
    def run_record(self,i,**usage):
        return dict(experiment_id=str(i),timestamp=(self.now-dt.timedelta(minutes=i+1)).isoformat(),status='complete',
                    coordination=dict(receipt_valid=True,actual_resources=dict({k:0 for k in UNITS},**usage)))
    def check(self,rows,active=None):return evaluate(rows,self.now,active or {'status':'idle'},{'status':'verified'})
    def test_many_completed_metadata_tasks_do_not_count_as_progress(self):
        result=self.check([self.run_record(i) for i in range(8)])
        self.assertTrue(result['alert_active']);self.assertEqual(result['measured_resources']['plant_steps'],0)
    def test_real_plant_transition_clears_the_stall(self):
        result=self.check([self.run_record(i) for i in range(8)]+[self.run_record(0,plant_steps=1,solver_calls=1)])
        self.assertFalse(result['alert_active'])
    def test_active_bounded_experiment_is_not_interrupted(self):
        result=self.check([self.run_record(i) for i in range(8)],{'status':'running','experiment_id':'active'})
        self.assertFalse(result['alert_active']);self.assertEqual(result['status'],'active_scientific_run')
    def test_unknown_usage_is_exposed_not_claimed_zero(self):
        rows=[self.run_record(i) for i in range(8)];rows[0]['coordination']={}
        result=self.check(rows);self.assertEqual(result['unknown_receipt_runs'],1)
    def test_solver_only_progress_keeps_the_closed_loop_warning(self):
        result=self.check([self.run_record(i,solver_calls=1) for i in range(8)])
        self.assertTrue(result['alert_active']);self.assertEqual(result['measured_resources']['solver_calls'],8)


if __name__=='__main__':unittest.main()
