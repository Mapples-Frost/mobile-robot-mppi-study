# Gate 4A critic-confidence calibration result

Date: 2026-07-19

## Result

The originally proposed twin-critic disagreement signal failed the required
directional calibration and is excluded from the next closed-loop arm.

Ten frozen-policy MuJoCo validation episodes were evaluated. At every visited
state, the frozen SAC target critics were queried with the deterministic Actor
action. Discounted Monte Carlo return was then computed from the actually
executed episode. Episode, not timestep, was the independent unit.

The Pearson association between episode-mean twin disagreement and
episode-mean absolute critic/return error was **-0.942**. Greater disagreement
therefore did not identify greater error in this sample. Input support distance
was also non-informative (association **-0.235**).

The absolute critic/return errors are not interpreted as a final critic-quality
estimate because finite episodes and terminal truncation can bias Monte Carlo
return. They are sufficient for the preregistered directional question: neither
candidate score passed as an online ordering of critic error.

## Consequence fixed before the next run

The failed signals are not sign-flipped, threshold-tuned on control outcomes or
silently retained. The next exploratory remediation sets:

\[
c_Q = 1
\]

for the already frozen and independently competence-screened critic, while
retaining candidate-level ICODE dynamics confidence:

\[
c_k=c_{\mathrm{dyn},k}.
\]

This reduces the claim. The remediation tests **dynamics-reliability-weighted
terminal authority**, not calibrated epistemic uncertainty of the critic.
Critic support/disagreement interfaces remain explicit and disabled by config
so a future independently calibrated critic ensemble can re-enable them.

## First development result retained

The failed mapping was evaluated on 18 paired cells (seeds 40--42, two scenes,
three physics domains):

- fixed terminal final distance: 1.9178 m;
- failed conservative mapping: 1.9865 m (3.58% worse);
- jerk: 1.09% worse, with a strictly adverse seed-cluster interval;
- collisions: 0 in both arms;
- equal rollout budget: 100 in both arms.

This run is a negative result and is not reused as confirmation evidence.
