# L269 Long-Horizon Credit Propagation Diagnosis Protocol

Status: preregistered development diagnosis. No L269 horizon or Critic result
has been generated or inspected at freeze time. L268 is retained as a valid
negative Gate: balanced complete-recovery replay moved all three paired
in-support Spearman estimates in the desired direction, but did not meet the
frozen aggregate, magnitude, held-out, Top-3, or scene-stability thresholds.

## Question and frozen method

L269 asks whether recovery value appears too late for the current one-step
Bellman target to propagate reliably to the initial corrective action. Actor,
reward, termination, 69D observation and normalizer, replay content, twin
256x256 25-quantile Critic architecture, group-robust Actor objective, ICODE,
HSS, MPPI, Actor/Traditional fusion, maps, sensors, and safety chain remain
unchanged. No Actor update is permitted anywhere in L269. Final Hairpin,
S-Chicane, Infinity, L258, and sealed seeds are forbidden.

The only source checkpoint is L262 step 6000, SHA256
`48e662fd728c0d1d62b1ff79681e5cdd924b120d9411c8df6e062a814b4d42a7`.
All L263 and L268 diagnostic inputs are hash-pinned in the L269 config.

## Stage 1: fixed-state horizon diagnosis

The independent unit is a frozen simulator state, not a rollout row. Two fixed
state blocks are used:

1. all 27 L263 counterfactual states, comparing the geometric recovery action
   with the frozen source Actor action;
2. all 36 L268 validation/test recovery-chain starts, comparing the recorded
   recovery action with the frozen source Actor action.

For the Q-consistent comparison, only the first action differs; both candidates
then follow the same frozen deterministic source Actor. Discounted MuJoCo return
is measured at H = 1, 5, 10, 20, and 40. The primary quantity is paired within
state: `Delta G_H = G_H(recovery) - G_H(actor)`.

L268 states additionally provide a sequence-level diagnostic. The accepted,
recorded recovery behavior is compared with a fresh source-Actor rollout from
the exact same reset for every prefix H and for the accepted chain's complete
length (maximum 600). This comparison is explicitly descriptive of trajectory
credit; it is not treated as an independent replicate at each step.

Stage 1 passes only if the L268 sequence-level block meets every condition:

- H40 recovery advantage is positive in at least 60% of states;
- positive-state fraction rises by at least 0.15 from H1 to H40;
- median H40 advantage is greater than median H1 advantage;
- at least 20% of states first become positive at H >= 10;
- complete-chain recovery advantage is positive in at least 60%;
- at least four of six scenes have majority-positive H40 advantage.

The thresholds and state hashes are frozen before L269 rollouts. If Stage 1
fails, L269 stops without Critic training. No horizon-specific result may be
used to relax this rule.

## Stage 2: paired Critic-only target probe

Stage 2 is prohibited unless Stage 1 passes. Four arms are compared within each
seed block: current one-step, 5-step, 10-step, and truncated TD(lambda) with
lambda 0.8 and maximum horizon 10. Seeds are `20263171/72/73`; arm order is
randomized once with seed `20263168`.

Every arm in a seed receives the exact same 6000 starting transitions from the
same recovery-balanced replay: 1000 per scene, with 300 chain-balanced recovery
starts and 700 original non-recovery starts. Sequence successors come only from
the same contiguous source trajectory. At an observed terminal, bootstrap is
zero. At an archived trajectory boundary that is not terminal, the target uses
the available shorter return and bootstraps from the last recorded next state.
Thus arm, not replay membership or start-state sampling, is the only treatment.

For n-step arm horizon n, each target quantile is
`sum(k=0..m-1) gamma^k r[k] + gamma^m (1-done) Z_target(s[m], pi(s[m]))`,
where m is the available length up to n. TD(lambda) is the frozen normalized
mixture of 1..10-step quantile targets: weights `(1-lambda)lambda^(n-1)` for
n < 10 and `lambda^9` for n = 10. This is an off-policy diagnostic probe, not a
claim of a formally corrected Retrace estimator.

All arms use identical initial Critic and optimizer, batch 256, 6000 updates,
learning rate, target updates, quantile loss, and seeded sample index stream.
Actor, Actor optimizer, log alpha, and alpha optimizer must remain bitwise
unchanged at 3000 and 6000 updates.

## Critic Gate and selection

Each multi-step arm is paired against one-step within seed. Eligibility requires
all of the following:

- aggregate in-support Spearman >= 0.05;
- paired median Spearman improvement >= 0.15;
- improvement direction positive in all three seed blocks;
- held-out recovery-vs-forward accuracy >= 0.65 and paired improvement >= 0.10;
- at least four of six L268 training scenes improve;
- no evaluated scene change below -0.10;
- paired Top-3 improvement >= 0.10;
- finite Q/TD/quantile diagnostics, absolute mean Q <= 500, and mean quantile
  spread <= 100.

If multiple arms pass, selection is lexicographic by simplicity:
5-step, then 10-step, then TD(lambda). There is no best-result selection.
Regardless of outcome, L269 ends after this Gate. Actor remains frozen and no
formal algorithm, reward, observation, fusion, or safety change is authorized.

