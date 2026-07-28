# Audit Record — Noise-Basis A/B Development Screen (PREPARED, NOT RUN)

**Created:** 2026-07-28T03:20:06Z
**Repo:** `mobile-robot-mppi-study-single-v6`, branch `codex/complex-static-three-dynamic-v1`
**Status:** protocol frozen, package validated, **no episode has been executed**

---

## 1. Why this is prepared rather than run

The analysis sandbox is a Linux environment separate from the Windows machine
holding this checkout. It has no MuJoCo and no PyPI access — verified, not
assumed:

```
python3 -c "import mujoco"      -> ModuleNotFoundError
pip download mujoco             -> No matching distribution found
urlopen https://pypi.org/simple -> Tunnel connection failed: 403 Forbidden
```

The Windows `.venv` in this repo **is** fully equipped — mujoco 3.2.3,
torch 2.13.0, numpy 1.26.4, pytest 9.1.1 — but it cannot be executed from the
Linux sandbox. The 24 episodes must be run on the Windows machine. Everything
that does not require MuJoCo is complete and validated here.

---

## 2. Files created

| SHA256 (16) | Path |
|---|---|
| `2dc83d0b33511ea7` | `configs/research/noise_basis_ab_development_v1.yaml` |
| `a1029951104edde7` | `experiments/dynamic_uncertainty/run_noise_basis_ab_development.py` |
| `d80277dd1822abb2` | `experiments/dynamic_uncertainty/analyze_noise_basis_ab_development.py` |
| `1d185c36b46e2626` | `scripts/run_noise_basis_ab.ps1` |

Additive only. No existing file modified.

---

## 3. Design

| | |
|---|---|
| Map | chapter1 — worst measured production coverage (1.8% vs 26.0% chapter3), and its margin-0 control ends by timeout rather than collision, so there is headroom on progress and no collision floor |
| Design | paired, common random numbers, 12 seeds, 24 episodes |
| Seeds | 791101201–791101212, disjoint from every previously opened chapter1 seed (verified programmatically) |
| Control | `noise_basis: "iid"` — bit-exact historical sampler |
| Treatment | `noise_basis: "ar1:2.0"` |
| Frozen | actor checkpoint, risk thresholds, ICODE, safety contract, MPPI cost terms, maps, obstacle motions, action bounds, 600 rollouts/cycle, H=36, dt=0.10 |

The two arms differ in exactly one parameter. Both preserve per-step marginal
variance exactly, so any difference is attributable to temporal correlation
rather than perturbation magnitude.

---

## 4. Decision rule — frozen before any result exists

Written into the protocol YAML and read by the analyzer at runtime. The analyzer
does not choose thresholds; it applies them. **Safety is evaluated first and can
veto regardless of efficacy.**

| Rule | Condition | Action |
|---|---|---|
| `FAIL_SAFETY` | ≥2 pairs where treatment collides and paired control does not | stop, seal evidence, close the mechanism family, no confirmatory |
| `PASS_TO_CONFIRMATORY` | no safety failure AND (net safe_success ≥ +2 pairs OR median paired goal-distance reduction ≥ 0.25 m) | pre-register and run n ≥ 20 confirmatory on fresh seeds |
| `FAIL_NO_EFFECT` | net success < +2 pairs AND median distance gain < 0.25 m | do not proceed; report that the offline gain did not transfer |
| `INCONCLUSIVE` | neither satisfied | do not proceed |

### Power statement, recorded up front

**Twelve pairs is a screening sample, not a confirmatory one.** It cannot
support a statistical claim and no p-value governs the decision. Its only job is
to decide whether an n ≥ 20 confirmatory is worth the compute. This is stated in
the protocol so that a favourable screen cannot later be presented as evidence
of efficacy.

---

## 5. Pre-execution validation (6 checks, all passing)

1. runner and analyzer parse as valid Python
2. protocol YAML well-formed, frozen, development-only, formal claims disabled,
   12 unique seeds, 600-rollout budget, H=36, all three decision rules present
3. A/B seeds disjoint from all previously opened chapter1 seeds
4. both arm basis specs parse through `build_basis`
5. the runner's own `_load_protocol` validator accepts the frozen protocol
   (executed in isolation, without the runner's MuJoCo-dependent imports)
6. analyzer thresholds match the protocol, and safety is evaluated before
   efficacy in source order

Not validated here: anything requiring MuJoCo — config resolution through
`build_complex_full_config`, `ExperimentRunner` execution, metrics schema.
**The first episode is therefore also a smoke test.** If it fails, the failure
will be in config resolution or runner wiring, and the stderr log is retained.

---

## 6. How to run

```powershell
cd D:\Projects\mobile-robot-mppi-study-single-v6

# optional: confirm the sampler regression tests pass in your venv first
.\.venv\Scripts\python.exe -m pytest tests\test_noise_basis_regression.py -q

# full screen: 24 episodes sequential, then the frozen decision rule
.\scripts\run_noise_basis_ab.ps1
```

`-Resume` skips completed episodes after an interruption. `-AnalyzeOnly`
re-applies the decision rule without re-running.

The driver refuses to overwrite existing evidence, refuses to start if another
measured episode is in flight, runs one episode at a time, retains stdout and
stderr per episode, and stops on the first failure with evidence intact.

**Smoke first if you prefer:** run one control episode by hand before committing
to all 24.

```powershell
.\.venv\Scripts\python.exe -m experiments.dynamic_uncertainty.run_noise_basis_ab_development `
  --map chapter1 --seed 791101201 --arm control `
  --output research_artifacts\noise_basis_ab_v1\seed791101201\control
```

---

## 7. What this run can and cannot establish

**Can:** whether the offline coverage gain (13.0% → 50.0%, not reproducible by
10× rollout budget) produces any closed-loop movement on a complex map, and
whether it causes a safety regression.

**Cannot:** any statistical claim, any generalisation beyond chapter1, or
anything about single-obstacle performance. It also does not resolve the three
open risks recorded in `noise_basis_integration_v1/AUDIT_RECORD.md` §7 —
importance-sampling correction under temporal correlation, rate-limit
feasibility, and interaction with the actor authority logic. Those must be
resolved before any formal experiment regardless of this outcome.

**If it fails**, that is a real result: the offline evidence chain was strong
and specific, and its failure to transfer would itself be informative and worth
sealing.

---

## 8. Commit instructions (run on Windows)

```powershell
cd D:\Projects\mobile-robot-mppi-study-single-v6

git add -- `
  configs/research/noise_basis_ab_development_v1.yaml `
  experiments/dynamic_uncertainty/run_noise_basis_ab_development.py `
  experiments/dynamic_uncertainty/analyze_noise_basis_ab_development.py `
  scripts/run_noise_basis_ab.ps1 `
  research_artifacts/noise_basis_ab_v1/AUDIT_RECORD.md

git diff --cached --check
git diff --cached --name-status

git commit -m "protocol: freeze noise-basis A/B development screen before execution

Additive only. No episode has been run.

Paired 12-seed screen on chapter1, control noise_basis 'iid' (bit-exact
historical sampler) versus treatment 'ar1:2.0'. Seeds 791101201-791101212,
disjoint from all previously opened chapter1 seeds. Actor, risk, ICODE,
safety, cost terms, maps and the 600-rollout budget are frozen; the arms
differ in one parameter.

Decision rule frozen in the protocol YAML before any result exists, with
safety evaluated first and able to veto: FAIL_SAFETY at >=2 pairs where the
treatment collides and the control does not; PASS_TO_CONFIRMATORY at net
success >= +2 pairs or median goal-distance reduction >= 0.25 m;
FAIL_NO_EFFECT otherwise. The analyzer reads the thresholds rather than
choosing them.

12 pairs is a screening sample and is recorded as such in the protocol: it
cannot support a statistical claim and only decides whether an n>=20
confirmatory is worth running.

6 pre-execution validation checks pass. MuJoCo-dependent paths are
unvalidated, so the first episode doubles as a smoke test."

git tag -a checkpoint/noise-basis-ab-protocol-frozen-20260728 `
  -m "Noise-basis A/B screen frozen before execution"

git rev-parse --short HEAD
```

Commit **before** running, so the protocol timestamp provably precedes the
evidence.
