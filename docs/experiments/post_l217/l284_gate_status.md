# L284 Gate Status

- Status: complete, negative combined Gate.
- Decision: `closed_loop_rollin_joint_anchor_gate_fail`.
- Summary SHA256: `85eb97367aef274e466d3f99c5bcda31b8b59f3020ac1d4bb43f30bc8a5c4a93`.
- Engineering: all three paired seeds reached exactly 6000 steps with 3001
  active anchor updates; dataset fingerprint and component weights matched the
  frozen protocol.

The closed-loop roll-in intervention improved median held-out recovery return
by `0.8102` relative to L281 for all three paired seeds.  Validation also moved
in the intended aggregate direction: median completion changed by `+0.0360`,
median CTE improved by `0.3500 m`, median goal distance improved by `2.3003 m`,
and two of three validation scenes improved CTE.

The preregistered combined Gate nevertheless failed.  Recovery-action test
RMSE deteriorated by a median `69.82%` relative to initialization, providing
only `7.59%` median improvement over the unanchored L277 control instead of the
required `20%`.  Seed `20263311` also regressed completion by `-0.0797`, beyond
the frozen `-0.02` limit.  Safety did not regress and four of six held-out
recovery scenes retained nonnegative return, but these facts do not override
the failed RMSE and per-seed completion checks.

Therefore the roll-in dataset repaired part of the sequence-return mismatch
without preserving the recovery action mapping robustly enough across seeds.
Larger validation, final-map evaluation, and further Actor expansion remain
unauthorized.  All raw results are retained; no seed, checkpoint, chain, or
threshold was filtered.
