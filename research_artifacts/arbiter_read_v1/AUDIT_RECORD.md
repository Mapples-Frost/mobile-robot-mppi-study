# Audit Record — Full Read of `safety/arbiter.py`

**Created:** 2026-07-28T06:57:13Z
**Repo:** `mobile-robot-mppi-study-single-v6`, branch `codex/complex-static-three-dynamic-v1`
**Status:** read-only. No code, config, or result modified.

---

## 1. Structure

`safety/arbiter.py` is 1,641 lines containing a single class, `ScanGuardArbiter`:

| member | lines | role |
|---|---|---|
| `__init__` | 12–418 | configuration parsing and validation only |
| `reset` | 419–442 | clears the state-machine latches |
| `arbitrate` | 443–1641 | the entire decision path, one ~1,200-line method |

All override behaviour lives in `arbitrate`.

---

## 2. The override path — six mutually exclusive branches

`reason` is read from the scan guard at line 451, then exactly one branch applies:

| order | line | branch | effect on the command |
|---|---|---|---|
| 1 | 991 | `dynamic_escape_allowed or corridor_commit_active` | full escape / corridor-commit override |
| 2 | 1159 | `self._dynamic_recovery_active` | recovery manoeuvre override |
| 3 | 1362 | `deadline_supervisor_active` | raises `v` to a speed floor |
| 4 | 1375 | `guard_result["emergency_stop"]` | `v = 0` |
| 5 | 1378 | `reason == "front_soft_block"` | `v = min(max(0, v), front_soft_block_max_speed)` |
| 6 | 1389 | `guard_result["should_slow_down"]` | `v *= slow_scale` **only if `v > 0`** |

Branch 5 clamps reverse to zero. Branch 6 deliberately does not — its inline comment
records why: *"Mapping a planner-vetted reverse command through max(0, v) traps the
robot against the same obstacle and defeats static/dynamic candidate certification
upstream."* The same trap remains present in branch 5, which is a smaller latent
issue and is **not** what caused the chapter-1 deadlock (branch 6 was active there).

---

## 3. Principal finding: every unstick mechanism is gated on dynamic escape

During the chapter-1 deadlock, measured over the final 300 steps of three seeds:

| latch | 791101204 | 791101208 | 791101211 |
|---|---|---|---|
| `dynamic_escape_allowed` | 0% | 0% | 0% |
| `dynamic_escape_reactive` | 0% | 0% | 0% |
| `dynamic_escape_corridor_active` | 0% | 0% | 0% |
| `dynamic_escape_held` | 0% | 0% | 0% |
| `dynamic_recovery_active` | 0% | 0% | 0% |
| `dynamic_recovery_mode` | `inactive` | `inactive` | `inactive` |
| `dynamic_deadline_supervisor_active` | 0% | 0% | 0% |
| `safety_reason` | `front_obstacle_slow` | (same) | (same) |
| `safety_override` | 78% | 74% | 75% |

`dynamic_recovery_mode` took the single value `inactive` for the entire deadlock.
**Branches 1–5 never armed. Only branch 6 — the plain front-sector slowdown — was
ever active.**

### Why

`arbiter.py:681` gates recovery on `self._dynamic_escape_seen`, which is set only at
line 670, inside the dynamic-escape path. The deadline supervisor (line 821)
additionally requires `self._dynamic_deadline_conflict_seen`.

Every mechanism the arbiter has for breaking out of a stall is conditioned on a
prior **dynamic-obstacle** interaction. At the deadlock the nearest dynamic obstacle
was 1.4–3.4 m away and `forecast_count` was 0, so no escape ever occurred, so nothing
downstream could arm.

**Structural conclusion: the arbiter has no recovery path for a static-geometry
stall.** A robot crawling against static geometry with no dynamic threat falls
through to branch 6 and remains there indefinitely.

---

## 4. Refinement, not correction, of the earlier diagnosis

`POST_HOC_DIAGNOSIS.md` §3 attributed the deadlock to the speed governor;
`DEADLOCK_ROOT_CAUSE.md` corrected that to static-filter starvation. This read
refines the safety layer's actual contribution:

* the governor is branch 6 and only **scales** an already-small command
  (0.03 → 0.02 m/s); it is not the primary constraint, as previously corrected;
* the safety layer's real contribution is the **absence of any escape hatch** —
  that is why a 2 cm/s crawl persists for 900 steps rather than being interrupted.

Complete causal chain for the chapter-1 timeouts:

1. Planner static clearance is 0.10 m over-conservative (confirmed segment-thickness
   bug) → static-map filter passes 1.67% of candidates.
2. Surviving candidates are near-zero velocity → planner proposes ~0.03 m/s.
3. Branch 6 scales that to ~0.02 m/s.
4. **No recovery mechanism can arm, because all are gated on dynamic escape.**
5. ~900 steps at 2 cm/s → timeout 2.32 m from goal.

The flagged segment-thickness fix addresses step 1 only. **Step 4 is an independent
latent gap and will survive that fix.**

---

## 5. Interaction with the two changes made in this workstream

* **`noise_basis`** — no interaction. The arbiter consumes a `ControlCommand` and
  scan guard output; it is indifferent to how candidates were sampled.
* **`known_static_map_segment_half_thickness`** — no direct interaction. The arbiter
  does not read the known-static map. Indirect only: relieving the static filter
  should raise the planner's proposed velocity, which branch 6 will then scale rather
  than floor. No arbiter change is needed for the A/B to be valid.

Neither change alters any arbiter code path. The frozen safety contract is intact.

---

## 6. What this read did *not* cover

* `__init__` lines 12–418 were skimmed for the parameters in use, not read in full.
* The internals of branches 1–3 (escape geometry, corridor commit, recovery
  manoeuvre construction, deadline speed-floor arithmetic) were read only at the
  level of their entry conditions. They were dormant throughout the episodes under
  study, so their internals could not have affected this diagnosis — but they are
  **not** validated by this read and would matter on any map where dynamic escape
  does fire.
* `evaluation/metrics.py` beyond the clearance path, and most of
  `rl_driven_mppi.py`, remain unread.

---

## 7. Recommended follow-up, not to be acted on yet

A static-stall recovery trigger — arming on `stuck_steps` or on sustained
low-`|v|` with positive static clearance, independent of `_dynamic_escape_seen` —
is the natural fix for §3. It should **not** be built until the segment-thickness
A/B reports, because if that fix restores the feasible set the stall may not recur,
and building a recovery mechanism for a stall that no longer happens repeats the
pattern that produced the earlier amendment series.

Recorded as a hypothesis with a measurement attached, not as work in progress.
