# L191 causal-innovation reliability amendment

Date frozen: 2026-07-19
Status: frozen before collecting any L191 transition
Parent: `docs/rl/190_path_aware_reliability_prereg_2026-07-19.md`

## Triggering development evidence

Both L190 calibrations passed validation but failed the complete held-out Gate.
Episode-level component analysis showed the same mechanism in ordinary and
value-aligned ensembles:

- causal innovation confidence ranked rollout error strongly in validation,
  test and unseen-development;
- the historical hard minimum of disagreement, support and innovation ranked
  validation/test error, but lost that ordering on reverse-S/hairpin;
- continuous Actor OOD confidence also reversed sign on held-out path geometry;
- the L190 data and thresholds are retained as negative development evidence.

This amendment is motivated by L190 and therefore cannot use L190 test/unseen
as confirmation.

## Single structural change

The frozen new fusion mode is `innovation_anchor`:

```text
before innovation is ready:
    dynamics confidence = min(disagreement, residual support)

after at least three completed transitions:
    dynamics confidence = causal innovation confidence

Actor support:
    hard veto only when Actor confidence reaches exactly zero
```

The soft continuous Actor penalty is removed because it confounded legitimate
unseen route geometry with policy incompetence.  Ensemble disagreement and
training support remain logged and retain authority during initialization.
No signal is deleted from provenance.

The default `conservative_min` implementation is unchanged for all historical
experiments.  `innovation_anchor` is opt-in and must be named in the runtime
configuration.

## Fresh evaluation

L191 repeats the frozen geometry-blocked L190 design with new episode seed
bases 22842000, 22843000 and 22844000.  It retains:

- 16 validation episodes on straight/turn;
- 16 test episodes on sweep/chicane;
- 20 unseen-development episodes on reverse-S/hairpin;
- disjoint scene/domain environments;
- the same Actor and two ICODE ensembles;
- the same quantile candidate grid and episode-level Gate;
- the same path representation and integrity checks.

The L186 geometries and seeds 561--565 remain sealed.

## Decision

Both ordinary and value-aligned L191 calibrations must pass validation, test
and unseen-development without threshold revision.  Failure keeps adaptive HSS
disabled.  Passing permits only a new closed-loop development factorial; it
does not by itself authorize sealed confirmation.

This score remains an engineering reliability ranking, not a calibrated
probability or a formal uncertainty guarantee.
