"""Check fail-closed confirmation and64-scene inference without touching test outcomes."""
from pathlib import Path
from unittest.mock import patch
import numpy as np
import gated_horizon_confirmation as confirmation
from gated_horizon_report import interval,ratio


def main():
    checks=[];base=dict(validation_effect_passed=True,timing_audited=True,extra_baselines_audited=True)
    for key in base:
        gate=dict(base);gate[key]=False
        with patch.object(confirmation,'register'),patch.object(confirmation,'freeze_policies'),patch.object(confirmation,'read',return_value=gate),patch.object(confirmation,'write') as output:
            try:confirmation.freeze_confirmation()
            except AssertionError:pass
            else:raise AssertionError(key+' did not block confirmation')
            assert output.call_count==0
        checks.append(key+'_blocks_test_when_false')
    def times(scale):return [[dict(case=i,steps=i+1,decision_total_s=scale*(i+1)) for i in range(64)] for _ in range(2)]
    r=ratio(times(.8),times(1.))
    np.testing.assert_allclose([r[k] for k in ('mean_ratio','lower','upper')]+r['repeat_ratios'],.8,rtol=1e-14,atol=1e-14)
    checks.append('64case_paired_timing_same_rule')
    assert interval(np.full((3,64),-4.))==dict(mean=-4.,lower=-4.,upper=-4.)
    checks.append('64case_three_seed_scene_block_bootstrap')
    confirmation.write(confirmation.OUT/'confirmation_checks.json',dict(passed=True,checks=checks,
        source_hash=confirmation.digest(Path(confirmation.__file__)),check_source_hash=confirmation.digest(Path(__file__)),test_outcomes_accessed=False))
    print('Passed %d synthetic confirmation checks; no test outcomes accessed'%len(checks))


if __name__=='__main__':main()
