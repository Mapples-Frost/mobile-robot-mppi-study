# L201 route-balanced value-ICODE remediation preregistration

Date frozen: 2026-07-19
Status: frozen after L199 offline Gate failed and before L201 collection

L199 proved that the path-conditioned value gradient is active: all members
improved validation rollout and value error. It failed test and unseen value
generalization because training contained only straight and single-turn route
families.

L201 changes only route coverage. Training contains the four L185 training
families (straight, turn, sweep, chicane). Validation uses reverse-S in seen
physics, test uses hairpin in seen physics, and unseen contains reverse-S and
hairpin only in the combined-unseen domain. Complete scene--domain episodes
remain disjoint.

Loss weights, learning rate, epochs, ICODE anchors, L185 checkpoint, path
feature re-encoding, confidence, and the L199 offline Gate remain unchanged.
No L198 holdout trajectory enters the dataset.

All three members must pass the existing rollout and value-improvement Gate on
test and unseen before any closed-loop experiment is authorized. Failure ends
this remediation; it does not permit post-hoc loss-weight search.
