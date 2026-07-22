# L283 Gate status

- Status: complete; 216/216 rollouts and 54/54 advantages are finite.
- Summary SHA256: `1c38c12501a0e9bdad55a3cd43f7ee0df1a365140473025e89200525749edbb2`.
- Maximum reset error: 0.
- Maximum closed-loop teacher action error: 0.
- Maximum teacher reward error: `5.9362710969068644e-08`.
- Frozen decision: `coupled_sequence_bottleneck`.

Closed-loop teacher recomputation did not attribute the remaining return loss
to velocity or steering alone.  A joint closed-loop recovery roll-in anchor is
the only next intervention authorized for preregistration.  Actor training is
not authorized until its dataset, leakage checks, tests, and resume smoke pass.
