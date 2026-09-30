<!-- vehicle-v34c-contract-preflight-gate-constant-failure-20260930T090409Z -->
## Pending RESEARCH_LOG append: v34c zero-solve preflight operational gate repair

Astra direction remains the 20260930T084456Z report: first complete source-specific zero-solve preflight, then if passed and backed up, execute original fixed 24-call objective-vs-basin probe. This iteration executed one registered run of v34c (`20260930T090408_15cca729`) using the verified 09:03 backup context. It failed immediately with a stale request constant mismatch before scheduled cell setup. This is an implementation/provenance gate bug only; no mechanism evidence was collected.

Failure artifact: `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v34c_contract_preflight_v0_20260930T090409Z/failed.json`.

Versioned repair written: `experiments/bohn2021_aws/vehicle_true_variable_horizon_v34c_contract_preflight_v0b.py`, sha256 `a4749e2c4760991a68241eac139919a55e9c3969eaa7f5cc537aa0843ac7942d`. The repair only updates the Astra gate constants to the current report/request and changes output names; scientific design and budgets are unchanged.

Next bounded iteration: verify backup for the new wrapper/failure evidence, run v0b zero-solve preflight, then branch according to hard_pass/failure.
