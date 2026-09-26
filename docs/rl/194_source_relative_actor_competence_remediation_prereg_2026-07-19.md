# L194 source-relative Actor competence remediation preregistration

Date frozen: 2026-07-19
Status: frozen after L193 failure and before implementation/evaluation
Parent: L193 innovation-anchor closed-loop development Gate

## Why this amendment is allowed

L193 is retained as a failed development Gate.  All twelve episodes succeeded
without collision, but Full Proposed exceeded the frozen ordinary-fixed
cross-track tolerance by 0.44 percentage points and the reliability mechanism
never left its high dynamics-confidence level.  The resulting guided fraction
was about 0.60 even though the guided source produced fewer elites per offered
candidate than the current Gaussian source.

This failure does not authorize changing the dynamics confidence mapping.
Instead it exposes a component already required by the frozen research plan
but absent from L193: a separately auditable Actor competence term
\(c_\pi\).  Dynamics confidence answers whether the rollout model is reliable;
it cannot answer whether the Actor is a better proposal source.

## Frozen mechanism

At MPPI decision \(t\), let \(E_{\mathrm{RL}}\) and
\(E_{\mathrm{G}}\) be the number of elite candidates originating from the
persistent RL-guided set and the current Gaussian set.  Let
\(N_{\mathrm{RL}}\) and \(N_{\mathrm{G}}\) be the corresponding numbers of
offered candidates, including reuse across the two paper iterations.

With a symmetric Beta prior:

\[
\hat y_s =
\frac{E_s+a}{N_s+a+b},
\qquad s\in\{\mathrm{RL},\mathrm{G}\}.
\]

The source-relative Actor competence observation is

\[
\tilde c_{\pi,t}
=
\operatorname{clip}
\left(
\frac{\hat y_{\mathrm{RL}}}{\hat y_{\mathrm{G}}},
0,1
\right).
\]

It is filtered causally:

\[
c_{\pi,t}
=
\gamma c_{\pi,t-1}
+
(1-\gamma)\tilde c_{\pi,t}.
\]

The reliability authority used for the next control decision becomes

\[
\alpha_t =
c_{\mathrm{dyn},t}\,
c_{\mathrm{support},t}\,
c_{\pi,t}.
\]

Here `c_support` remains only a hard Actor out-of-support veto in the
innovation-anchor variant.  No future outcome, MuJoCo ground truth, collision
label, or final episode metric enters this update.  The comparison uses costs
already computed by MPPI under the same ICODE model and the same candidate
budget.

Frozen defaults:

- initial \(c_\pi=0.50\);
- EMA decay \(\gamma=0.90\);
- symmetric Beta prior \(a=b=1\);
- low/medium/high authority thresholds remain 0.33/0.67;
- guided fractions remain 0.00/0.30/0.60;
- update is applied with a one-decision causal lag;
- fixed arms remain unchanged.

## Regression requirements

1. the feature is opt-in and default-disabled;
2. disabling it reproduces the existing reliability allocation exactly;
3. equal source yields drive raw competence toward one;
4. poorer guided yield reduces competence;
5. zero guided opportunities retain the previous state;
6. reset restores the frozen initial confidence;
7. no extra rollout, Actor call, critic call, or MuJoCo query is introduced.

## Development experiment

- scene: `high_dynamic_reverse_s_terminal_dev_l187`;
- physics domain: `nominal_seen`;
- new development seeds: 566, 567, 568;
- four randomized arms and the same checkpoints as L193;
- `K=100`, two paper iterations, horizon 36;
- terminal guidance radius 1.26 m and floor 0.30;
- all safety and perception paths unchanged;
- sealed L186 geometry and seeds 561--565 remain unused.

## Gate

The remediation Gate passes only if:

1. Full Proposed succeeds in all three episodes;
2. Full Proposed has zero collisions;
3. Full Proposed mean cross-track RMSE is no worse than ordinary fixed by more
   than 5%;
4. Full Proposed mean cross-track RMSE is no worse than value fixed by more
   than 5%;
5. Full Proposed mean control jerk is no worse than ordinary fixed by more
   than 10%;
6. source competence is finite, nonconstant, and changes the raw guided
   fraction away from 0.60 in every adaptive episode;
7. all arms retain identical rollout and iteration budgets.

Passing remains development evidence only.  It authorizes a separately
preregistered multi-domain Gate; it does not authorize opening sealed data.
