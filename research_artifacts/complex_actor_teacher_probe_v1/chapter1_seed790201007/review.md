# Complex Actor teacher probe preflight failure

The first frozen teacher probe did not execute an episode. It failed on the
first planning cycle because the proposed 60-step teacher horizon exceeded the
unchanged 36-step Change-Aware forecast contract:

`ValueError: robot horizon cannot exceed obstacle forecast horizon`

This is a protocol incompatibility, not an outcome. The failure is retained
without overwrite. The correction must keep the deployed 36-step horizon and
increase only the training-teacher sample count; extending the predictor or
truncating risk evaluation would change the probability contract.
