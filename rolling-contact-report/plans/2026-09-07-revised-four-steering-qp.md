# Revised Four-Steering QP — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Adapt the C++ implementation to the revised `\eqref{eq:four-steering-wheel-qp}` in `rolling-contact-qp-corrected.tex`, satisfy the test suite specified in `rolling-contact-qp-tests.tex`, and end with `mc_mujoco -f rolling-contact-report/config/mc_rtc-ranger-mini-v3-keyboard.yaml` driving the Ranger Mini V3 smoothly under keyboard control.

**Architecture:** mc_rtc solves the *whole-body* QP (`eq:rolling-whole-body-qp`) over `alphaD`, not the planar reduction the revised equation is written in. Each correction therefore needs an explicit carry-over judgement — some transfer directly, one probably does not. That judgement is Task 0 and gates everything after it.

**Tech Stack:** C++17, mc_rtc (`mc_rbdyn` / `mc_solver` / `mc_tasks` / `mc_control`), Eigen, RBDyn/SpaceVecAlg, Tasks and TVM backends, Boost.Test + CTest, MuJoCo via `mc_mujoco`.

---

## What actually changed in the theory

Diffed against the currently-implemented `rolling-contact-qp.tex`. Five deltas, in descending order of impact.

### R1. The four lateral rows are softened with an explicit slack (`re:four-steering-lateral-slack`)

This is the substantive change, and the remark is emphatic that it is "not a stylistic choice". Writing
`a_i^T = (l_i^Π)^T H_i = [−sin δ_i, cos δ_i, x_i cos δ_i + y_i sin δ_i]`, the four lateral rows form
`A ξ̇ = c` with `A ∈ R^{4×3}` — **four equations on three unknowns**:

- For steering angles **not** coordinated on a common ICR, `rank(A) = 3`, so the only admissible planar twist is `ξ̇ = 0`. **A hard-row QP freezes the chassis.**
- When the angles **are** ICR-coordinated, `rank(A) = 2` and motion is admissible, but consistency additionally needs `c ∈ range(A)` — a coordination condition on the *measured* steering rates that no controller can guarantee from measurement.

So with hard rows the QP is generically infeasible. The decision vector gains `σ^lat ∈ R^4`, each lateral row becomes `… = σ^lat_i`, and the objective gains `w_σ ‖σ^lat‖²₂` with a **single common weight** across the four wheels (so the least-squares solution does not preferentially skid one wheel), set large relative to the tracking weights.

**This retires `steeringPlanar` / `steeringPlanarWheels` entirely.** That option exists only to dodge this rank problem by keeping 2 of 4 lateral rows, and its own doc comment admits the pair "should name a pair that remains independent throughout the intended maneuvers". It is also the asymmetry I suspected during the earlier `ackermann_right` investigation: `{front_left, rear_left}` is not mirror-symmetric.

### R2. Normal load becomes exogenous in the planar reduction (`sec:planar-normal-load`)

In the reduction, `n_Π^T J_red,i = 0` — every column of the reduced contact Jacobian produces tangent motion, so a purely normal force generates no reduced generalized force. Two consequences the report calls "fatal if ignored":

1. The friction cone is **vacuous**: with a symmetric generator basis `C_i 1 ∝ n_i`, so `λ_i ↦ λ_i + s·1`, `s ≥ 0` preserves every constraint and every objective term while inflating the normal force without bound.
2. The normal *kinematic* row reduces identically to `0 = 0` and must **not** be appended.

Remedy: one scalar row per contact, `n_Π^T C_i λ_i = f^ref_{n,i}` with `f^ref_{n,i} ≥ 0` (`eq:planar-normal-load-row`), plus the load-distribution objective, since a four-wheel chassis is statically indeterminate.

**⚠ This is the delta most likely NOT to carry over — see Task 0.**

### R3. Force objective with Tikhonov term (`eq:planar-force-regularization`)

`f_force(x) = Σ_i ( w_{f,i} ‖C_i λ_i − f^ref_{n,i} n_Π‖²₂ + ε_λ ‖λ_i‖²₂ )`. The generators appear in no motion objective, which leaves the Hessian singular on the whole `λ` block. The first term is a physical load-distribution task; `ε_λ` is a small Tikhonov term that removes the polyhedral-representation redundancy and "must stay small enough not to bias the physical force". Test **QP-02** pins this by asserting `ε_λ = 0` reproduces the singular case.

### R4. Matrix twist weight (`eq:planar-twist-weight`)

`e_ξ^T W_ξ e_ξ` with `W_ξ = diag(w_vx, w_vy, w_ω)`, not a scalar weight, "because its entries carry both" translational and rotational units. This is where unit normalisation is applied. Test **QP-06** asserts unit rescaling leaves the physical solution invariant.

### R5. Standing assumptions A1–A5 (`sec:planar-qp-assumptions`)

- **A1 `as:planar-basis`** — all planar quantities in the **chassis-aligned** basis, so `ρ_i` and `H_i` are constant and `Ḣ_i = 0`.
- **A2 `as:rigid-wheel`** — constant radius; relaxing adds `−ṙ_i θ̇_i` per rolling row.
- **A3 `as:steering-axis`** — steering axis ∥ `n_Π` through the carrier; zero trail/scrub. True for the Ranger URDF (drive joint origin coincident with the steer axis).
- **A4 `as:one-plane`** — all contacts share one plane.
- **A5 `as:normal-load`** — a normal-load estimate is available.

**A1 is violated today.** `mc_rolling_contact_controller.cpp:391` builds the tangent basis from the world `UnitX` projected off the terrain normal — world-fixed, not chassis-aligned. Tests **GEO-01** (chassis-heading invariance), **GEO-02** (orthonormal, right-handed, body-constant), **ROW-04** (basis-convention pin on the rows) and **SMK-10** (basis rotates with the chassis) all target exactly this.

### Verified non-issue

`\norm[two]{·}` expands via `applyNormSymbols` (`configurations.tex:561-571`) to `‖·‖²₂` — **squared**. The slack and force penalties are quadratic and the problem remains a QP. No second-order-cone solver is needed.

---

## Current-implementation deltas (verified)

| Revised theory | Current code | Status |
| --- | --- | --- |
| 4 softened lateral rows + `σ^lat` | hard rows; `steeringPlanar` keeps 2 of 4 | **missing** |
| `n^T C_i λ_i = f^ref_n` | absent (`grep` finds no normal-load row) | **missing, gated on Task 0** |
| `ε_λ ‖λ‖²` Tikhonov | absent | **missing** |
| `W_ξ = diag(w_vx, w_vy, w_ω)` | scalar weights | **missing** |
| chassis-aligned basis | world-fixed `UnitX` projection (`:391`) | **wrong** |
| rate rows `w_θ̇`, `w_δ̇` | present (`rollingRateWeight`, `steeringRateWeight`) | ✅ done |
| affine rate prediction | present (soft rows, `A = dt·S`) | ✅ done |
| rate bounds | `KinematicsConstraint` `vl()/vu()` | ✅ done |
| `steeringWheelReference` inversion | present | ✅ done |

---

## Task 0 — Decide the whole-body carry-over (BLOCKING, do first)

**Do not skip this.** R2's argument is derived *in the planar reduction*, where `n_Π^T J_red,i = 0`. mc_rtc solves the whole-body QP, where the floating base has a vertical DOF and `n^T J_i ≠ 0` — so the normal force *is* observable, gravity fixes it through the equations of motion, and the vacuous-cone ray may not exist. By the same token the normal kinematic row is **not** `0 = 0` in whole-body coordinates; it constrains vertical base motion, so `constrainNormal = true` is probably correct as-is and R2's "must not be appended" does **not** transfer.

Getting this wrong in either direction is expensive: adding a redundant prescribed-normal-load row would fight gravity and over-constrain the QP; omitting a genuinely needed one leaves an unbounded ray.

- [ ] **Step 1: Establish whether the vacuous ray exists in the current whole-body QP**

Write a throwaway diagnostic (not a committed test) that, on a solved four-steering QP, takes the solution `λ*` and forms `λ* + s·1` for `s` in `{0, 1, 10, 100}`. Check whether each remains feasible for every constraint **and** leaves the objective unchanged. Report the objective delta and the first violated constraint.

- [ ] **Step 2: Check the normal row's rank contribution**

Assemble the hard constraint matrix and compute the rank with and without the four normal rows (`axis == normalAxis`). If the rank increases by 4, they carry information and must stay; if it does not increase at all, the report's `0 = 0` argument transfers.

- [ ] **Step 3: Record the decision**

Write the finding into `rolling-contact-report/control-issue.md` with the numbers. Then:

- **If the ray does NOT exist and the normal rows DO add rank** (the expected outcome): skip Task 4's normal-load row, keep `constrainNormal = true`, and implement only the **Tikhonov** part of R3 (which is about Hessian conditioning on the `λ` block and is needed regardless). Note the divergence from the planar equation in the plan and the report.
- **If the ray DOES exist**: implement Task 4 in full.

Everything downstream of Task 4 is unaffected by this decision; Tasks 1–3 and 5–8 proceed either way.

---

## File structure

| File | Responsibility | Change |
| --- | --- | --- |
| `include/mc_rbdyn/RollingContact.h` / `src/mc_rbdyn/RollingContact.cpp` | wheel geometry, planar rows | chassis-aligned basis helper; `Ḣ = 0` assertion |
| `include/mc_solver/RollingContactConstraint.h` / `.cpp` | kinematic rows | lateral slack variables; retire `steeringPlanar` |
| `include/mc_solver/RollingContactDynamicsConstraint.h` / `.cpp` | dynamics, cone | Tikhonov term; normal-load row (gated on Task 0) |
| `src/mc_control/samples/RollingContact/mc_rolling_contact_controller.{h,cpp}` | sample controller | chassis-aligned basis; `W_ξ`; slack weight; config |
| `tests/testRollingContact.cpp` | pure geometry unit tests | Layer A/B testcards |
| `tests/testRollingContactSolver.cpp` | solver unit tests | Layer B/C/D/E/F testcards |
| `src/mc_control/samples/RollingContact/test_controller_lifecycle.cpp` | controller tests | Layer G/H testcards |
| `rolling-contact-report/config/*.yaml` | runtime configs | new weights; remove `steeringPlanarWheels` |

**Build/test commands** (from the repository root):

```sh
cmake --build build -j 8
CUDA_VISIBLE_DEVICES=-1 MC_RTC_DISABLE_CONVEX_GENERATION_PATCH=ON \
  ctest --test-dir build -R 'RollingContact|testRollingContact' --output-on-failure -j1
CUDA_VISIBLE_DEVICES=-1 MC_RTC_DISABLE_CONVEX_GENERATION_PATCH=ON \
  ctest --test-dir build --output-on-failure -j1          # full suite, currently 104/104
```

Always `-j1` — this environment has unrelated SEGFAULT flakiness under parallel ctest. Every CTest runs with an isolated `HOME=build/test-home`, so `~/.config/mc_rtc/mc_rtc.yaml` cannot reach it; when running the ticker or MuJoCo **by hand**, set `HOME` to an empty scratch directory. A stale user config has already invalidated one "clean" verification run in this project.

**Current state that must not regress:** rolling CTest 14/14, full CTest 104/104, MuJoCo 8/8.

---

## Task 1 — Chassis-aligned tangent basis (A1)

Satisfies **GEO-01**, **GEO-02**, **GEO-03**, **ROW-04**, **SMK-10**.

**Files:** `src/mc_control/samples/RollingContact/mc_rolling_contact_controller.cpp:388-396`; `include/mc_rbdyn/RollingContact.h`; `tests/testRollingContactRobot.cpp`.

- [ ] **Step 1: Write the failing tests**

`GEO-01` — planar geometry is chassis-heading invariant. Place the robot at yaw `0`, `π/4`, `π/2`, `−2π/3`; assert each wheel's planar offset `ρ_i` expressed in the chassis basis is **identical** across all four headings to `1e-12`. Today this fails: the basis is world-fixed, so `ρ_i` rotates with heading.

`GEO-02` — the basis is orthonormal, right-handed and body-constant: `‖e_x‖ = ‖e_y‖ = 1`, `e_x · e_y = 0`, `e_x × e_y = n`, and `Ḣ_i = 0` under pure chassis rotation.

`GEO-03` — a degenerate forward axis (chassis heading parallel to the terrain normal) **aborts** rather than normalising garbage. Assert a thrown `std::invalid_argument`, not a silently normalised near-zero vector.

### ⚠ Correction — do NOT make `terrainTangentX_`/`terrainTangentY_` chassis-aligned

An earlier draft of this task said to rebuild `terrainTangentX_` from the chassis forward axis. **That is wrong and would break the controller.** Those two vectors are the **world-fixed yaw reference frame**, not the planar row basis. They are consumed at `:1003`, `:1109`, `:1166`, `:1292`, `:1545` as

```cpp
  measuredYaw = std::atan2(measuredHeading.dot(terrainTangentY_), measuredHeading.dot(terrainTangentX_));
```

and at `:1208-1211` to build the trajectory `heading`/`side` vectors. If the basis rotated with the chassis then `measuredHeading · terrainTangentX_ ≡ 1` and `· terrainTangentY_ ≡ 0`, so **`measuredYaw` would collapse to identically zero** and every piece of yaw bookkeeping — `baseYawTarget_`, the closed-loop keyboard heading, the trajectory frame — would silently break. They must stay world-fixed.

### A1 needs the same carry-over analysis as R2

A1 constrains where `ρ_i` and `H_i` are resolved, so that `Ḣ_i = 0` in the differentiated rows. Both `ρ_i` and `H_i` are objects of the **planar reduction**. mc_rtc's whole-body implementation has no `H_i`: `RollingContactGeometry` builds its rows from the carrier Jacobian and inertial-frame direction vectors, and the carrier Jacobian's time dependence is already carried exactly by `normalAcceleration()` (`Jdot * alpha`). So there is no `Ḣ_i` term to drop.

Moreover `wheelOffsets_` is **already** chassis-frame and constant: `mc_rolling_contact_controller.cpp:250-258` computes `chassis.rotation() * (carrier - chassis)`. That is `ρ_i` in the chassis-aligned basis, exactly as A1 requires, and it is what `steeringWheelReference()` consumes.

- [ ] **Step 2: Establish whether A1 is already satisfied, before changing anything**

Verify the two claims above by reading the code, then decide:

- **If `ρ_i` is already chassis-frame and the whole-body rows carry `Jdot` exactly** (the expected outcome): A1 is already satisfied in substance. Implement **GEO-01** and **GEO-02** as *regression pins* on the existing behaviour — asserting `ρ_i` is heading-invariant and that the wheel offsets stay constant as the chassis rotates — and record in `control-issue.md` that A1 does not require a code change in whole-body form, with the reasoning. Then skip to Task 2.
- **If some planar quantity IS resolved in a world-fixed basis** and does vary with heading, fix *that specific quantity* — not `terrainTangentX_`/`Y_`.

- [ ] **Step 3: Implement GEO-03 regardless**

`GEO-03` (degenerate axis aborts rather than normalising garbage) is worth having either way. The existing construction at `:391-396` silently falls back from `UnitX` to `UnitY` when the terrain normal is near-parallel to `UnitX`; that fallback is reasonable, but a normal that is degenerate against **both** should throw rather than produce a near-zero normalised vector. Add that guard and a test.

- [ ] **Step 4: Verify, then run the full rolling suite**

Existing numbers must be **unchanged** if you took the first branch — this task then adds tests only. If any existing test shifts, you have changed behaviour you did not intend; stop and find out why.

- [ ] **Step 5: Commit**

```bash
git add src/mc_control/samples/RollingContact/ tests/testRollingContactRobot.cpp rolling-contact-report/control-issue.md
git commit -m "test(rolling): pin the chassis-aligned planar geometry invariants"
```

---

## Task 2 — Lateral slack variables (R1)

Satisfies **ROW-08**, **ROW-09**, **GEO-09**, **QP-04**, **QP-05**, **BND-06**, **SMK-06**, **SMK-07**.

This is the core of the revision. It removes a known wart (`steeringPlanar`) rather than adding a workaround.

**Files:** `include/mc_solver/RollingContactConstraint.h`; `src/mc_solver/RollingContactConstraint.cpp`; `tests/testRollingContactSolver.cpp`.

- [ ] **Step 1: Write the failing tests**

`ROW-08` — lateral-row structural rank. Build a four-steering constraint with **non**-ICR-coordinated steering angles and assert the assembled lateral block has `rank(A) = 3`; with ICR-coordinated angles assert `rank(A) = 2`. Use `Eigen::FullPivLU` with an explicit threshold and assert the singular values, not just the rank integer, so the test reports *how* near-singular it is.

`GEO-09` — ICR concurrency is exactly the rank condition: assert the two are equivalent over a sweep of steering configurations.

`ROW-09` — the slacks are exactly the range-space defect. For a known-inconsistent `c ∉ range(A)`, assert `σ* = c − A ξ̇*` and that `σ*` equals the residual of the least-squares projection, to `1e-9`.

`SMK-07` — incompatible four-wheel data surfaces as slack rather than as infeasibility: the solve **succeeds** and `‖σ‖` is non-zero.

**The decisive regression test:** with hard lateral rows and non-coordinated angles, the QP freezes the chassis (`ξ̇ ≈ 0`) or fails; with slacks it produces a sensible twist. Assert both halves so the test demonstrates the difference rather than only the fixed behaviour.

- [ ] **Step 2: Run, verify they fail**

- [ ] **Step 3: Add the slack decision variables**

Extend `RollingContactConstraintOptions`:

```cpp
  /** Soften the lateral rows with one slack per wheel.
   *
   * Four lateral rows act on three planar chassis DOF. For steering angles not
   * coordinated on a common ICR the block has full row rank 3, so hard rows
   * admit only the zero twist and freeze the chassis; when they are
   * coordinated, consistency further needs c in range(A), a condition on the
   * measured steering rates that no controller can guarantee. The slacks make
   * the QP always feasible and report the incompatibility.
   */
  bool softLateralRows = false;
  /** Single common weight across wheels, so the least-squares solution does not
   * preferentially skid one wheel. Set large relative to the tracking weights
   * so the lateral rows behave as a high-priority task. */
  double lateralSlackWeight = 1e5;
```

Implementation route: mc_rtc's Tasks backend has no free slack columns in `alphaD`, so realise the softening the same way the existing soft block does — move the four lateral rows from the hard matrix into the soft objective at `lateralSlackWeight`, which is mathematically identical to an explicit `σ` with a quadratic penalty (minimising `w‖Ax − c‖²` **is** minimising `w‖σ‖²` s.t. `Ax − c = σ`). Expose the realised `σ` through a new `lateralSlack()` accessor computed as `A x* − c` after each solve, so **ROW-09** can assert on it and the controller can log it.

Prefer this over adding real decision columns: it needs no change to the decision-vector layout in either backend, and the existing per-row weight machinery (`addRow(..., weight)`) already supports it. If **ROW-09** cannot be satisfied this way, fall back to explicit columns and say why.

- [ ] **Step 4: Retire `steeringPlanar`**

Once all four lateral rows are soft, the 2-of-4 selection has no purpose. Remove `steeringPlanar` and `steeringPlanarWheels` from the options, the validator, the `buildRowLayout` branch, the ConstraintSet loader, the JSON schema, and every config that sets them (`mc_rtc-four-steering*.yaml` set `steeringPlanarWheels: [front_left, rear_left]`).

Keep `differentialPlanar` — a two-wheel chassis has 2 lateral rows on 3 DOF and is not over-determined, so its single-row selection is still correct.

Update `RollingContactConstraintOptions::validate` and its tests accordingly.

- [ ] **Step 5: Verify and commit**

Both the rolling suite and MuJoCo must stay green. **Watch `four-crab` and `four-ackermann-left` especially** — those are the manoeuvres where non-coordinated steering angles arise, and **SMK-06** calls crab translation "the manoeuvre that separates T1 from T2".

```bash
git commit -m "feat(rolling): soften the four lateral rows with per-wheel slacks"
```

---

## Task 3 — Matrix twist weight `W_ξ` (R4)

Satisfies **QP-06**.

**Files:** controller `.h`/`.cpp`; configs.

- [ ] **Step 1: Write the failing test**

`QP-06` — unit rescaling leaves the physical solution invariant. Solve a reference problem; then rescale the angular component of the twist reference and the corresponding weight consistently (e.g. express `ω` in deg/s and scale `w_ω` by `(π/180)²`) and assert the resulting **physical** twist is unchanged to `1e-9`. With a scalar weight this fails, because the translational and rotational errors are summed in incommensurable units.

- [ ] **Step 2: Implement**

Replace the scalar twist weight with `Eigen::Vector3d` diagonal entries `(w_vx, w_vy, w_ω)`, configurable as `twistWeight: [vx, vy, omega]`. Keep a scalar form accepted for backward compatibility, expanded to a uniform diagonal with a deprecation warning.

Document at the declaration that this is where unit normalisation is applied, and that `w_ω` carries `(rad/s)⁻²` while `w_vx`/`w_vy` carry `(m/s)⁻²`.

- [ ] **Step 3: Verify and commit**

---

## Task 4 — Normal load and force regularisation (R2, R3)

**Gated on Task 0.** The Tikhonov half is unconditional; the normal-load row is conditional.

Satisfies **QP-01**, **QP-02**, **DYN-07**, **FRI-01**, **FRI-05**, **BND-07**.

**Files:** `include/mc_solver/RollingContactDynamicsConstraint.h`; `src/mc_solver/RollingContactDynamicsConstraint.cpp`; `tests/testRollingContactSolver.cpp`.

- [ ] **Step 1 (unconditional): Tikhonov term on the generators**

`QP-01` — assert the Hessian is positive definite under the corrected objective (smallest eigenvalue `> 0` by a stated margin). `QP-02` — assert `ε_λ = 0` reproduces the singular case, i.e. the smallest eigenvalue collapses to ~0. Write both first.

Add `double generatorRegularization = 1e-6;` (name it for what it does), applied as `ε_λ ‖λ‖²`. Document that it exists to remove polyhedral-representation redundancy and "must stay small enough not to bias the physical force" — and assert that bound: **FRI-01** should check the solved contact force with `ε_λ` at its default and at `10×` differ by less than a stated tolerance.

- [ ] **Step 2 (conditional on Task 0): normal-load row**

Only if Task 0 found the vacuous ray. Add `n^T C_i λ_i = f^ref_{n,i}` per contact, with `f^ref_{n,i} ≥ 0` supplied per wheel.

`BND-07` — the row requires a **strictly positive** right-hand side; assert a rejection for `f^ref ≤ 0`. `DYN-07` — the normal-load model closes the three discarded equilibrium rows; assert `Σ_i f^ref_{n,i} = m g cos(slope)` and that the moments about the two horizontal axes balance. `FRI-05` — with the row present the cone **binds**; assert that removing it lets the normal force grow without bound.

Source `f^ref` from a quasi-static load distribution (the four-wheel case is statically indeterminate, so the load-distribution objective selects one solution). For a flat, symmetric chassis this is `m g / 4` per wheel; derive it from the robot's mass and wheel offsets rather than hardcoding.

- [ ] **Step 3: Verify and commit**

---

## Task 5 — Assumption guards and the rejection contract (R5)

Satisfies **QP-10**, **GEO-03**, **GEO-07**, **ROW-11**, **DYN-05**, **DYN-09**, **BND-02**, **BND-06**, **FRI-02**, **FRI-03**.

The suite has a whole class of **negative** testcards. These assert the implementation *refuses* invalid input rather than silently producing plausible garbage — the most valuable tests here, and the easiest to omit.

- [ ] **Step 1: Enumerate every negative testcard and write them all first**

- `GEO-03` degenerate forward axis aborts (done in Task 1)
- `GEO-07` steering angle is **chassis relative** and its rate is the **joint** rate — assert an absolute-angle interpretation is rejected or produces a detectably different result
- `ROW-04` basis-convention pin on the rows
- `ROW-11` T2 rows carry **no** proportional stabilization — assert `Kp` does not appear in them
- `DYN-05` steering torque **cannot** enter the chassis yaw row — assert the coupling coefficient is exactly zero
- `DYN-09` the quasi-static torque bound must not duplicate the wheel row
- `BND-02` a yawing chassis with locked steering consumes **no** steering-rate budget
- `BND-06` the lateral slacks carry **no** box constraint
- `FRI-02` the cone is rebuilt when the wheel steers; `FRI-03` when the terrain normal changes — both are staleness tests, so mutate the state and assert the cone actually changed
- `QP-10` rejection contract — invalid configurations throw with actionable messages

- [ ] **Step 2: Implement whatever guards are missing, then verify and commit**

---

## Task 6 — Oracles and property tests

Satisfies **ORC-01**…**ORC-06**, **ROW-01**, **ROW-03**, **ROW-05**, **ROW-07**, **DYN-01**…**DYN-04**, **DYN-06**, **QP-03**, **QP-07**, **QP-08**.

- [ ] **Step 1: `ORC-01` Richardson finite differences with a convergence-rate assertion**

Not just "the derivative is close" — assert the error decreases at the **expected order** as the step shrinks. A wrong analytic derivative can pass a fixed-tolerance check and fails this.

- [ ] **Step 2: `ORC-02` reduced planar QP against its full-coordinate preimage**

**This is the highest-value test in the suite for this codebase**, because it directly validates the whole-body ↔ planar-reduction relationship that Task 0 reasons about informally. Solve the whole-body QP, project to the planar coordinates, and assert it matches the explicit planar QP's solution.

- [ ] **Step 3: `ORC-03`/`ORC-04` closed-form oracles**

Differential-drive closed form on the solved prediction; known-ICR analytic oracle for the four-steering case.

- [ ] **Step 4: `ORC-05` metamorphic relations, `ORC-06` seeded randomized sweep**

Seeded and replayable — record the seed in the failure message so any failure is reproducible.

- [ ] **Step 5: `QP-03`/`QP-07` solver-independent checks**

Primal uniqueness across warm starts and both backends; KKT residual as an optimality oracle independent of which solver ran.

- [ ] **Step 6: Verify and commit**

---

## Task 7 — Smoke sequence

Satisfies **SMK-01**…**SMK-13**. Several map onto MuJoCo cases that already exist; extend `run-mujoco-suite.py` rather than duplicating.

- [ ] `SMK-02` static hold, flat and ramp — `SMK-03` straight line — `SMK-04` in-place rotation — `SMK-05` constant-radius arc **and the centripetal term** — `SMK-06` crab (separates T1 from T2) — `SMK-07` incompatible data surfaces as slack — `SMK-08` steering-rate-limited slalom — `SMK-09` slope sweep to friction and torque failure (assert it fails at the *predicted* slope) — `SMK-11` odometry cross-check **including where it must disagree** — `SMK-12` terrain transition — `SMK-13` long-horizon closed loop.

- [ ] Wire the P0 subset into CTest; keep P1/P2 behind an opt-in flag so the default suite stays fast.

- [ ] Populate the **Traceability** section of `rolling-contact-qp-tests.tex` (`:1530`) mapping each testcard to its implementing test name, and fill in **Residual gaps** (`:1592`) with anything deliberately not implemented and why.

---

## Task 8 — Acceptance: keyboard driving in mc_mujoco

- [ ] **Step 1: Full regression**

Rolling CTest, full CTest (104/104 baseline), MuJoCo suite 8/8 on both backends.

- [ ] **Step 2: The acceptance command**

```sh
mc_mujoco -f rolling-contact-report/config/mc_rtc-ranger-mini-v3-keyboard.yaml
```

Confirm first that `mc_mujoco` is on `PATH` and that this config selects the Ranger and the `RollingContact` controller with the observer pipeline active — the log must show `RollingContactPipeline: Encoder → BodySensor`. Note the config path is repo-relative, so run from the repository root.

Drive with W/A/S/D and Q/E and verify:

- each key moves the chassis in the commanded direction, with no scale factors anywhere in the path;
- **`w_σ` is large enough that lateral slack stays near zero in coordinated manoeuvres** and grows only during steering transients — log `‖σ^lat‖` and report its value in straight-line, crab and arc;
- steering is smooth through the `+/− π/2` branch and across W→Q→A transitions;
- drive torque stays well under the 35 Nm limit and no wheel detaches.

- [ ] **Step 3: Tune `w_σ` against MuJoCo, not the ticker**

The single free parameter introduced by this plan. The report says "large relative to the tracking weights so the lateral rows behave as a high-priority task". Sweep it and report the knee. **Use MuJoCo as the arbiter** — the kinematic ticker has no contact physics, and tuning against it has already caused one regression in this project (`driveAcceleration` 20→100, which passed ticker metrics while saturating drive torque and detaching wheels in simulation).

- [ ] **Step 4: Promote the corrected theory and update the report**

```sh
git mv rolling-contact-report/rolling-contact-qp.tex rolling-contact-report/rolling-contact-qp-prev.tex
git mv rolling-contact-report/rolling-contact-qp-corrected.tex rolling-contact-report/rolling-contact-qp.tex
```

Fill in its `\section{Validations}`, and record in `README.md` that the lateral rows are softened and why `steeringPlanar` was retired.

---

## Assumptions and open questions

1. **Whole-body vs planar reduction.** The revised equation is the *planar* QP; mc_rtc solves the *whole-body* one. **Two** deltas may not transfer, and each has a measurement gate: R2 (Task 0) and A1 (Task 1 Step 2). R1, R3 and R4 transfer directly — the lateral over-determination is a property of the geometry, not of the coordinate choice, and the Hessian/units arguments are coordinate-independent.

   The general lesson, which applies to any further deltas found later: a correction derived from `n_Π^T J_red,i = 0` or from `Ḣ_i = 0` is a statement about the reduction's coordinates. Check it against the whole-body form before implementing it.
2. **`w_σ` has no prescribed value.** The report gives only a relative instruction. Task 8 Step 3 fixes it empirically against MuJoCo.
3. **`f^ref_{n,i}` source.** For the Ranger, a flat symmetric quasi-static distribution (`m g / 4`) is the natural default, derived from the model. A suspension model or force sensors would be better but neither exists here.
4. **`mc_mujoco` availability.** The acceptance command assumes `mc_mujoco` is installed and on `PATH` with the Ranger mapping. The repo builds its own runner at `/tmp/rolling-contact-cpu-mujoco/` via `build-cpu-mujoco-runner.sh`; if the system `mc_mujoco` is missing or its mappings are stale, install the mappings from `rolling-contact-report/mujoco/` first.
5. **Test-suite scale.** `rolling-contact-qp-tests.tex` specifies roughly 80 testcards. Tasks 5–7 implement the **P0** set first; P1 follows; P2 is opt-in. Attempting all 80 before any MuJoCo validation would defer the acceptance criterion too long — but the P0 negative tests in Task 5 are not optional, since they are what stop the implementation silently accepting invalid input.
