# L282 Gate status

- Status: complete, engineering integrity passed.
- Summary SHA256: `da92e89515faea9c94cb0203ed2fb5fd4b45e1a200ba982e72d76fd003c7648e`.
- Coverage: 216/216 rollouts and 54/54 paired advantages.
- Reset error: 0; maximum teacher reward reproduction error:
  `5.9362710969068644e-08`.
- Frozen decision: `coupled_sequence_bottleneck`.
- Neither velocity nor steering alone recovered 50% of the teacher gap in four
  scenes for two seeds.  The full teacher sequence had a positive gap in four
  scenes for seeds 20263311 and 20263312.
- Actor training and final-map evaluation remain unauthorized.

L282 used recorded teacher components.  A hybrid rollout can leave the
recorded teacher trajectory, so L283 must first recompute the teacher action
closed-loop at the actual hybrid state before a coupled intervention is
allowed.
