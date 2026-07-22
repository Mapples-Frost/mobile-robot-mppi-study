# L280 Gate Status

- Status: complete, positive internal-conflict diagnosis.
- Decision: `internal_anchor_conflict`.
- Summary SHA256: `d3b8fdd46fdd811b721fee32115bd2d27d9fc2f49844aab537d9d3ffa3c6cd78`.

All 36 preregistered seed/stage/scene gradient blocks completed on native
Windows CUDA with finite diagnostics.  The combined anchor did not show broad
opposition to the online SAC Actor gradient and did not dominate its norm.
Instead, recovery and source-distillation anchor gradients conflicted in the
linear-speed mean head for two of three seeds under the frozen 4/6-scene
coverage rule.  No component met the online-conflict rule.

This rules out blind anchor-weight reduction and SAC-vs-anchor gradient
projection as evidence-based next steps.  The only authorized intervention is
the preregistered L281 component-separated anchor: retain recovery supervision
for both action means while preventing source distillation from updating the
conflicted linear-speed mean component.  Actor training and final-map
evaluation remain unauthorized until the paired L281 Gate passes.
