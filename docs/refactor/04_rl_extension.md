# RL Extension Boundary

The recommended first integration is an RL sampling prior:

```text
observation + reference
→ RL mean/covariance
→ MPPI rollouts and cost optimization
→ safety arbitration
```

`RLPolicyPrior` accepts a framework-independent callable, so the MPPI package
does not depend on PyTorch, JAX, or an RL library.

The optional Gymnasium-style adapter returns proposed and executed actions
separately. This is essential when scan_guard modifies unsafe proposals.

Future levels are planner parameter adaptation, behavior-mode selection, and
finally direct action proposals. No RL level may bypass scan_guard or final
control limits.
