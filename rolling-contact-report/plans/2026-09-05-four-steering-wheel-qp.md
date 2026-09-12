# Four-Steering-Wheel QP Correction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Ranger Mini V3 four-steering chassis track WASD/QE twist commands smoothly by giving each wheel *two* optimized rotating rates — rolling (pitch) and steering (yaw) — inside the QP, as specified by `\eqref{eq:four-steering-wheel-qp}` in `rolling-contact-qp-new.tex`.

**Architecture:** mc_rtc instantiates the report's QP in whole-body form: `G(q) * alphaD = rhs` over `robot.mb().nrDof()` columns, solved by `RollingContactConstraint` (kinematic rows) plus `RollingContactDynamicsConstraint` (dynamics, joint/velocity/torque bounds). The corrected theory adds the rotating-rate block `nu_rot^{4s,+} in R^8` with its affine prediction, its rate-tracking objective terms, and its rate bounds. In whole-body coordinates the prediction is affine in `alphaD` (`theta_dot^+ = theta_dot + dt * S * alphaD`), so the whole correction becomes two extra **soft rows per steering wheel** plus reference generation — no new decision variables and no non-convexity. The sample controller's analytic steering IK (which today drives the steering hinges through the posture task, outside the QP, with empirical scale factors) is replaced by reference generation that feeds those rows.

**Tech Stack:** C++17, mc_rtc (mc_rbdyn / mc_solver / mc_tasks / mc_control), Eigen, RBDyn/SpaceVecAlg, Tasks and TVM QP backends, Boost.Test + CTest, MuJoCo via `mc_mujoco` for physical validation.

---

## Background: verified root cause

Established by reading the sources listed below — each claim here has a file:line anchor so the implementer can confirm it before changing anything.

1. **The Ranger steering axis passes exactly through the wheel centre.**
   `src/mc_robots/rolling_contact_description/urdf/ranger_mini_v3.urdf:40-49` — the chain is
   `chassis --front_left_steer(axis 0 0 1)--> front_left_knuckle --front_left_drive(axis 0 1 0)--> front_left_wheel`,
   and the drive joint carries **no `<origin>`**, so its frame is coincident with the knuckle frame.
   Consequence: a steering rotation produces **zero translation** of the wheel centre. This matches
   `rolling-contact-qp-new.tex` §"Steering-wheel rotational velocity coupling": *"a steering rotation about the
   terrain normal does not translate the wheel center when the steering axis passes through the wheel carrier."*

2. **The rolling rows therefore have an all-zero column for the steering acceleration.**
   `RollingContactGeometry::update` builds only the three translational rows from the carrier *translational*
   Jacobian (`src/mc_rbdyn/RollingContact.cpp:460` takes `fullJacobian.bottomRows<3>()`) and subtracts a single
   drive-rate term (`src/mc_rbdyn/RollingContact.cpp:362`:
   `result_.rollingMatrix.row(0) -= input.radius * input.spinSign * input.wheelSelector`).
   There is exactly **one** selector, for the drive joint (`src/mc_rbdyn/RollingContact.cpp:415-416`).

3. **There is no steering selector at all.** `RollingContactRobotGeometry::Impl` validates that
   `description.steeringJoint` is a 1-DoF revolute joint and then **discards the index**
   (`src/mc_rbdyn/RollingContact.cpp:403-412`). Nothing in `RollingContactKinematics`
   (`include/mc_rbdyn/RollingContact.h:146-161`) can address the steering DoF.

4. **Net effect: the QP has no equation that relates `delta_dot` to anything.** `delta_ddot` is free within its
   joint bounds and contributes nothing to any row, so the optimizer has no reason and no mechanism to steer.
   The rolling rows only see the *current* `delta` (through the directions `t_i`, `l_i`) and the *measured*
   `delta_dot` (through the `Gdot` bias).

5. **So the sample controller steers open-loop, outside the QP.** `mc_rolling_contact_controller.cpp:1254-1540`
   computes steering hinge *positions* analytically and pushes them into the generic posture task
   (`postureTask->target(targets)` at `:1535`). That layer needs a sign mirror
   (`wheelYaw = -(...) * (yaw + correction)`, `:1315-1316`), two empirical gains
   (`keyboardYawScale_ = 1.7`, `keyboardMixedDriveScale_ = 1.15`, config
   `rolling-contact-report/config/mc_rtc-ranger-mini-v3-keyboard.yaml`), `+/-pi/2` branch-cut hysteresis
   (`:1395-1455`), a "do not spin a wheel while it is still being steered" gate (`:1352-1360`) and a bounded
   command-transition grace (`mc_rolling_contact_controller.h:103-107`). These are all symptoms of item 4, and
   they are why QE/WASD respond poorly.

6. **Independently, the acceptance command does not select the Ranger at all.** Verified by running
   `/home/yuquan/.local/bin/mc_rtc_ticker -s robot:=RollingContactRangerMiniV3 --run-for 0.05`: the
   `robot:=...` token is silently ignored, the ticker loads `/usr/local/etc/mc_rtc.yaml` and runs the default
   JVRC1 robot. `utils/mc_rtc_ticker.cpp:16-27` registers no positional options and no `robot` option.

**What is already correct and must not be "fixed":**

- Keeping `t_dot`, `l_dot` evaluated from the *measured* `delta_dot` inside one solve. The report is explicit
  (`rolling-contact-qp-new.tex`, remark under `eq:four-steering-wheel-qp`): *"Keeping the direction fixed inside
  one solve preserves the QP structure."* Making `delta_dot` in the bias an optimization variable would make the
  row bilinear and the problem non-convex.
- Not adding explicit `G^rot_i` identity rows. Because the steering and drive rates *are* generalized
  coordinates here, `G^rot_i` is a kinematic identity; the report says *"its differentiated identity row may be
  omitted [...] Do not add a second copy of an identity row to the same QP."*
- The predicted-rate **bounds** of `eq:four-steering-wheel-qp` already exist:
  `RollingContactDynamicsConstraint : public DynamicsConstraint`
  (`include/mc_solver/RollingContactDynamicsConstraint.h:33`) inherits `KinematicsConstraint`, which applies
  `robot.vl()/vu()` (`src/mc_solver/KinematicsConstraint.cpp:156-157`). The URDF gives `velocity="8"` for the
  steer joints and `velocity="30"` for the drive joints. **No new bound rows are needed.**

**Therefore the implementable delta is exactly three things:**

- **A.** A steering selector so the steering DoF is addressable (Task 1).
- **B.** Two soft rows per steering wheel implementing the rate-tracking objective terms
  `w_theta_dot * (theta_dot^+ - theta_dot^ref)^2` and `w_delta_dot * (delta_dot^+ - delta_dot^ref)^2`
  from `eq:four-steering-wheel-qp` (Tasks 2-3).
- **C.** Reference generation from the commanded chassis twist that replaces the analytic IK and its fudge
  factors (Tasks 4-5).

Tasks 6-8 cover the sign bug, the ticker invocation, and end-to-end validation.

---

## File Structure

| File | Responsibility | Change |
| --- | --- | --- |
| `include/mc_rbdyn/RollingContact.h` | Backend-neutral wheel data model | Modify: add `steeringSelector` to `RollingContactKinematics`; add `measuredRollingRate` / `measuredSteeringRate` to `RollingContactGeometryResult` |
| `src/mc_rbdyn/RollingContact.cpp` | Geometry computation | Modify: resolve the steering DoF, fill the selector, report measured rates |
| `include/mc_solver/RollingContactConstraint.h` | Kinematic rolling rows | Modify: rate-objective options and reference API |
| `src/mc_solver/RollingContactConstraint.cpp` | Row layout and assembly | Modify: two new soft row axes |
| `src/mc_control/samples/RollingContact/mc_rolling_contact_controller.h` | Sample controller state | Modify: replace IK/fudge members with reference members |
| `src/mc_control/samples/RollingContact/mc_rolling_contact_controller.cpp` | Sample controller logic | Modify: reference generation replaces analytic IK |
| `utils/mc_rtc_ticker.cpp` | Ticker CLI | Modify: accept `key:=value` overrides |
| `tests/testRollingContact.cpp` | mc_rbdyn unit tests | Modify: selector and measured-rate tests |
| `tests/testRollingContactSolver.cpp` | Solver unit tests | Modify: rate-row tests, both backends |
| `src/mc_control/samples/RollingContact/test_controller_lifecycle.cpp` | Controller tests | Modify: twist-sign and tracking tests |
| `~/.config/mc_rtc/mc_rtc.yaml` | User runtime config | Create: makes the bare acceptance command select the Ranger |

**Build/test commands used throughout** (run from the repository root, `/home/yuquan/local/mc_rtc_sources/mc_rtc`):

```sh
cmake --build build -j
CUDA_VISIBLE_DEVICES=-1 MC_RTC_DISABLE_CONVEX_GENERATION_PATCH=ON \
  ctest --test-dir build -R 'RollingContact|testRollingContact' --output-on-failure -j1
```

Every task ends with a commit. Do not batch commits.

---

### Task 1: Expose the steering degree of freedom in the wheel geometry

Adds the steering selector `S_delta` and the two measured rotating rates. This is pure plumbing: no row or
solution changes yet, so the existing test suite must stay green.

**Files:**
- Modify: `include/mc_rbdyn/RollingContact.h:146-193`
- Modify: `src/mc_rbdyn/RollingContact.cpp:285-330`, `:380-421`, `:452-476`
- Test: `tests/testRollingContact.cpp`

- [ ] **Step 1: Write the failing test**

Append to `tests/testRollingContact.cpp`, before the final closing of the file:

```cpp
BOOST_AUTO_TEST_CASE(SteeringSelectorAddressesTheSteeringDof)
{
  // rolling_4s exposes <name>_steer / <name>_drive per wheel.
  auto robots = makeFourSteeringRobots();
  const auto & robot = robots->robot();

  mc_rbdyn::RollingContactDescription description;
  description.name = "front_left";
  description.carrierFrame = "front_left_carrier";
  description.wheelBody = "front_left_wheel";
  description.driveJoint = "front_left_drive";
  description.steeringJoint = "front_left_steer";
  description.radius = 0.125;
  description.width = 0.08;

  mc_rbdyn::RollingContactRobotGeometry geometry(robot, description);
  geometry.update(robot, Eigen::Vector3d::UnitZ());

  const auto & kinematics = geometry.kinematics();
  const auto nrDof = robot.mb().nrDof();
  BOOST_REQUIRE_EQUAL(kinematics.steeringSelector.size(), nrDof);

  const auto steeringDof = robot.mb().jointPosInDof(static_cast<int>(robot.jointIndexByName("front_left_steer")));
  const auto driveDof = robot.mb().jointPosInDof(static_cast<int>(robot.jointIndexByName("front_left_drive")));
  BOOST_CHECK_CLOSE(kinematics.steeringSelector(steeringDof), 1.0, 1e-9);
  BOOST_CHECK_SMALL(kinematics.steeringSelector(driveDof), 1e-12);
  BOOST_CHECK_CLOSE(kinematics.steeringSelector.sum(), 1.0, 1e-9);
}

BOOST_AUTO_TEST_CASE(MeasuredRotatingRatesFollowJointVelocities)
{
  auto robots = makeFourSteeringRobots();
  auto & robot = robots->robot();

  mc_rbdyn::RollingContactDescription description;
  description.name = "front_left";
  description.carrierFrame = "front_left_carrier";
  description.wheelBody = "front_left_wheel";
  description.driveJoint = "front_left_drive";
  description.steeringJoint = "front_left_steer";
  description.radius = 0.125;
  description.width = 0.08;

  robot.mbc().alpha[robot.jointIndexByName("front_left_drive")][0] = 2.5;
  robot.mbc().alpha[robot.jointIndexByName("front_left_steer")][0] = -0.75;
  robot.forwardKinematics();
  robot.forwardVelocity();

  mc_rbdyn::RollingContactRobotGeometry geometry(robot, description);
  const auto & result = geometry.update(robot, Eigen::Vector3d::UnitZ());

  BOOST_CHECK_CLOSE(result.measuredRollingRate, 2.5, 1e-9);
  BOOST_CHECK_CLOSE(result.measuredSteeringRate, -0.75, 1e-9);
}

BOOST_AUTO_TEST_CASE(DifferentialWheelHasEmptySteeringSelector)
{
  auto robots = makeDifferentialRobots();
  const auto & robot = robots->robot();

  mc_rbdyn::RollingContactDescription description;
  description.name = "left";
  description.carrierFrame = "left_carrier";
  description.wheelBody = "left_wheel";
  description.driveJoint = "left_drive";
  description.radius = 0.2;
  description.width = 0.08;

  mc_rbdyn::RollingContactRobotGeometry geometry(robot, description);
  const auto & result = geometry.update(robot, Eigen::Vector3d::UnitZ());

  BOOST_CHECK_EQUAL(geometry.kinematics().steeringSelector.size(), 0);
  BOOST_CHECK_SMALL(result.measuredSteeringRate, 1e-12);
}
```

`makeFourSteeringRobots()` / `makeDifferentialRobots()` are the loader helpers already used by
`tests/testRollingContactRobot.cpp`. If `tests/testRollingContact.cpp` does not already have them, copy the
helper from `tests/testRollingContactRobot.cpp:30-62` verbatim rather than inventing a new one, and add the
same `ROLLING_CONTACT_DESCRIPTION_SOURCE_PATH` compile definition and `rolling_contact` dependency that
`tests/CMakeLists.txt:239-250` gives `testRollingContactRobot`.

- [ ] **Step 2: Run test to verify it fails**

```sh
cmake --build build -j --target testRollingContact
```

Expected: **compile error**, `'struct mc_rbdyn::RollingContactKinematics' has no member named 'steeringSelector'`.

- [ ] **Step 3: Add the fields to the header**

In `include/mc_rbdyn/RollingContact.h`, inside `struct RollingContactKinematics` (after `wheelSelector`, line 154):

```cpp
  Eigen::RowVectorXd wheelSelector;
  /** Row selecting the steering (yaw) joint rate, empty for a non-steering wheel.
   *
   * When non-empty it must have the same size as wheelSelector. It addresses the
   * second rotating velocity of a steering wheel: see the rotational coupling
   * omega_{w/c} = theta_dot * l + delta_dot * n of the rolling-contact report.
   */
  Eigen::RowVectorXd steeringSelector;
```

In `struct RollingContactGeometryResult` (after `rightHandedError`, line 192):

```cpp
  double orthonormalError = 0.0;
  double rightHandedError = 0.0;
  /** Measured rolling (pitch) rate, wheelSelector * generalizedVelocity. */
  double measuredRollingRate = 0.0;
  /** Measured steering (yaw) rate, steeringSelector * generalizedVelocity; zero without a steering joint. */
  double measuredSteeringRate = 0.0;
```

- [ ] **Step 4: Validate and populate the selector in `RollingContactGeometry::update`**

In `src/mc_rbdyn/RollingContact.cpp`, in the validation block near line 291, after the existing
`wheelSelector` check, add:

```cpp
  if(input.steeringSelector.size() != 0
     && (input.steeringSelector.rows() != 1 || input.steeringSelector.cols() != nrDof_))
  {
    throw std::invalid_argument("Rolling contact steeringSelector must be empty or 1 x nrDof");
  }
  if(input.steeringSelector.size() != 0) { requireFinite(input.steeringSelector, "steeringSelector"); }
```

Then, immediately after `result_.rollingMatrix.row(0) -= input.radius * input.spinSign * input.wheelSelector;`
(line 362), record the measured rates:

```cpp
  result_.measuredRollingRate = input.wheelSelector.dot(input.generalizedVelocity);
  result_.measuredSteeringRate =
      input.steeringSelector.size() == 0 ? 0.0 : input.steeringSelector.dot(input.generalizedVelocity);
```

**Do not touch `result_.lateralDirectionRate` or `result_.rollingDirectionRate`.** They are derived from
`input.wheelAxleRate`, which already contains the steering contribution because the axle direction is taken
from the wheel body's world pose (`src/mc_rbdyn/RollingContact.cpp:467-471`). `input.steeringRate` is a
*separate* opt-in path for callers that supply a static axle, and `RollingContactRobotGeometry` deliberately
sets it to `0.0` (`:473`). Adding the measured steering rate there would double-count the direction
derivative and silently corrupt every four-steering bias term.

- [ ] **Step 5: Resolve the steering DoF in `RollingContactRobotGeometry::Impl`**

In `src/mc_rbdyn/RollingContact.cpp`, replace the steering validation block at lines 403-412 with a version
that keeps the DoF index:

```cpp
    if(!description.steeringJoint.empty())
    {
      const auto steeringIndex = robot.jointIndexByName(description.steeringJoint);
      const auto & steeringJoint = robot.mb().joint(steeringIndex);
      if(steeringJoint.dof() != 1 || steeringJoint.type() != rbd::Joint::Rev)
      {
        throw std::invalid_argument("Rolling contact steeringJoint must be a one-DoF revolute joint: "
                                    + description.steeringJoint);
      }
      steeringDof = robot.mb().jointPosInDof(static_cast<int>(steeringIndex));
    }
```

Add the member next to `int driveDof;` (line 436):

```cpp
  int driveDof;
  int steeringDof = -1;
```

And in the selector initialisation block (lines 414-417), after `input.wheelSelector(driveDof) = 1.0;`:

```cpp
    if(steeringDof >= 0)
    {
      input.steeringSelector.setZero(robot.mb().nrDof());
      input.steeringSelector(steeringDof) = 1.0;
    }
```

- [ ] **Step 6: Run tests to verify they pass**

```sh
cmake --build build -j --target testRollingContact
CUDA_VISIBLE_DEVICES=-1 MC_RTC_DISABLE_CONVEX_GENERATION_PATCH=ON \
  ctest --test-dir build -R 'testRollingContact$' --output-on-failure
```

Expected: PASS, including the three new cases.

- [ ] **Step 7: Run the whole rolling suite to confirm nothing regressed**

```sh
CUDA_VISIBLE_DEVICES=-1 MC_RTC_DISABLE_CONVEX_GENERATION_PATCH=ON \
  ctest --test-dir build -R 'RollingContact|testRollingContact' --output-on-failure -j1
```

Expected: PASS. This task adds no rows, so every existing residual/parity test must be bit-identical.

- [ ] **Step 8: Commit**

```bash
git add include/mc_rbdyn/RollingContact.h src/mc_rbdyn/RollingContact.cpp tests/testRollingContact.cpp
git commit -m "feat(rolling): expose the steering DoF selector and measured rotating rates"
```

---

### Task 2: Add the rotating-rate objective options and reference API

Introduces the configuration surface for `w_theta_dot`, `w_delta_dot` and the per-wheel references
`theta_dot^ref`, `delta_dot^ref`. Still no rows: this task only stores state, so behaviour is unchanged.

**Files:**
- Modify: `include/mc_solver/RollingContactConstraint.h:38-60`, `:90-105`
- Modify: `src/mc_solver/RollingContactConstraint.cpp:32-65`, `:67-170`, `:503-570`
- Test: `tests/testRollingContactSolver.cpp`

- [ ] **Step 1: Write the failing test**

Append to `tests/testRollingContactSolver.cpp`:

```cpp
BOOST_AUTO_TEST_CASE(RollingRateReferencesRoundTripAndValidate)
{
  auto robots = makeFourSteeringRobots();
  auto wheels = makeFourSteeringWheels();

  mc_solver::RollingContactConstraintOptions options;
  options.steeringPlanar = true;
  options.trackRotatingRates = true;
  BOOST_CHECK_CLOSE(options.rollingRateWeight, 200.0, 1e-9);
  BOOST_CHECK_CLOSE(options.steeringRateWeight, 200.0, 1e-9);

  mc_solver::RollingContactConstraint constraint(*robots, 0, wheels, options);

  constraint.rotatingRateReference("front_left", 3.0, -0.5);
  BOOST_CHECK_CLOSE(constraint.rollingRateReference("front_left"), 3.0, 1e-9);
  BOOST_CHECK_CLOSE(constraint.steeringRateReference("front_left"), -0.5, 1e-9);

  // Unknown wheels are a programming error, not a silent no-op.
  BOOST_CHECK_THROW(constraint.rotatingRateReference("nope", 0.0, 0.0), std::out_of_range);

  // Non-finite references must be rejected before they can poison the QP.
  BOOST_CHECK_THROW(constraint.rotatingRateReference("front_left", std::nan(""), 0.0), std::invalid_argument);

  // Negative weights are invalid.
  mc_solver::RollingContactConstraintOptions bad;
  bad.rollingRateWeight = -1.0;
  BOOST_CHECK_THROW(bad.validate(4), std::invalid_argument);
}
```

`makeFourSteeringRobots()` and `makeFourSteeringWheels()` already exist in
`tests/testRollingContactSolver.cpp` (used by `RollingTasksFourSteeringAckermannTarget`); reuse them.

- [ ] **Step 2: Run test to verify it fails**

```sh
cmake --build build -j --target testRollingContactSolver
```

Expected: **compile error**, `'struct mc_solver::RollingContactConstraintOptions' has no member named 'trackRotatingRates'`.

- [ ] **Step 3: Extend the options struct**

In `include/mc_solver/RollingContactConstraint.h`, inside `RollingContactConstraintOptions`, after
`steeringPlanarWheels` (line 57):

```cpp
  std::vector<std::string> steeringPlanarWheels;
  /** Track the eight rotating velocities of a four-steering chassis.
   *
   * When true, each wheel contributes a soft row tracking its predicted rolling
   * rate, and each steering wheel a second soft row tracking its predicted
   * steering rate. These are the w_thetaDot and w_deltaDot objective terms of
   * the four-steering-wheel QP; the corresponding predicted-rate bounds are
   * already supplied by the joint velocity limits of KinematicsConstraint.
   */
  bool trackRotatingRates = false;
  /** Objective weight on (thetaDot^+ - thetaDot^ref). */
  double rollingRateWeight = 200.0;
  /** Objective weight on (deltaDot^+ - deltaDot^ref). */
  double steeringRateWeight = 200.0;
```

Declare the reference accessors after `rollingWeight()` (line 94):

```cpp
  /** Set the predicted-rate references for one wheel.
   *
   * @throws std::out_of_range if the wheel is unknown.
   * @throws std::invalid_argument if either value is not finite.
   */
  void rotatingRateReference(const std::string & wheel, double rollingRate, double steeringRate);
  double rollingRateReference(const std::string & wheel) const;
  double steeringRateReference(const std::string & wheel) const;
```

- [ ] **Step 4: Validate the new weights**

In `src/mc_solver/RollingContactConstraint.cpp`, inside
`RollingContactConstraintOptions::validate` (starting line 32), before its closing brace:

```cpp
  if(!(rollingRateWeight >= 0.0) || !(steeringRateWeight >= 0.0))
  {
    throw std::invalid_argument("Rolling contact rate weights must be non-negative and finite");
  }
```

`!(x >= 0.0)` rejects NaN as well as negatives; a bare `x < 0.0` would let NaN through.

- [ ] **Step 5: Store the references in the Impl and implement the accessors**

In `struct RollingContactConstraint::Impl` (line 67), next to `activations`:

```cpp
  std::vector<double> rollingRateReferences;
  std::vector<double> steeringRateReferences;
  double dt = 0.0;
```

Size them where `activations` is sized, with `rollingRateReferences.assign(wheels.size(), 0.0);` and the same
for `steeringRateReferences`.

Then add the three public functions next to `RollingContactConstraint::activation` (line 558):

```cpp
void RollingContactConstraint::rotatingRateReference(const std::string & wheel, double rollingRate,
                                                     double steeringRate)
{
  if(!std::isfinite(rollingRate) || !std::isfinite(steeringRate))
  {
    throw std::invalid_argument("Rolling contact rate references must be finite");
  }
  const auto index = impl_->wheelIndex(wheel);
  impl_->rollingRateReferences[index] = rollingRate;
  impl_->steeringRateReferences[index] = steeringRate;
}

double RollingContactConstraint::rollingRateReference(const std::string & wheel) const
{
  return impl_->rollingRateReferences[impl_->wheelIndex(wheel)];
}

double RollingContactConstraint::steeringRateReference(const std::string & wheel) const
{
  return impl_->steeringRateReferences[impl_->wheelIndex(wheel)];
}
```

`Impl::wheelIndex` (line 318) already throws `std::out_of_range` for an unknown name, which is what the test
expects. Add `#include <cmath>` to the translation unit if it is not already present.

- [ ] **Step 6: Run tests to verify they pass**

```sh
cmake --build build -j --target testRollingContactSolver
CUDA_VISIBLE_DEVICES=-1 MC_RTC_DISABLE_CONVEX_GENERATION_PATCH=ON \
  ctest --test-dir build -R 'testRollingContactSolver' --output-on-failure
```

Expected: PASS. `trackRotatingRates` still defaults to `false`, so no existing test changes behaviour.

- [ ] **Step 7: Commit**

```bash
git add include/mc_solver/RollingContactConstraint.h src/mc_solver/RollingContactConstraint.cpp \
        tests/testRollingContactSolver.cpp
git commit -m "feat(rolling): add rotating-rate objective options and per-wheel references"
```

---

### Task 3: Emit the predicted rotating-rate rows

This is the core correction. For each wheel the QP gains the soft row

```
  || dt * S_theta * alphaD - (thetaDot^ref - thetaDot) ||^2   weighted by rollingRateWeight
```

and, for each steering wheel,

```
  || dt * S_delta * alphaD - (deltaDot^ref - deltaDot) ||^2   weighted by steeringRateWeight
```

which is exactly `w_thetaDot (thetaDot^+ - thetaDot^ref)^2 + w_deltaDot (deltaDot^+ - deltaDot^ref)^2` after
substituting the affine prediction `rate^+ = rate + dt * acceleration`.

**Weighting mechanism.** The soft block is a single quadratic objective at `options.rollingWeight`. A row scaled
by `s` contributes `rollingWeight * s^2 * residual^2`, so to realise a per-row weight `w` the row scale must be
`sqrt(w / rollingWeight)`. This composes with the existing activation scale, giving
`sqrt(activation * w / rollingWeight)`.

**Files:**
- Modify: `src/mc_solver/RollingContactConstraint.cpp:160-315`
- Test: `tests/testRollingContactSolver.cpp`

- [ ] **Step 1: Write the failing test**

Append to `tests/testRollingContactSolver.cpp`:

```cpp
BOOST_AUTO_TEST_CASE(RotatingRateRowsMatchTheAffinePrediction)
{
  auto robots = makeFourSteeringRobots();
  auto & robot = robots->robot();
  auto wheels = makeFourSteeringWheels();

  robot.mbc().alpha[robot.jointIndexByName("front_left_drive")][0] = 1.5;
  robot.mbc().alpha[robot.jointIndexByName("front_left_steer")][0] = 0.25;
  robot.forwardKinematics();
  robot.forwardVelocity();

  mc_solver::RollingContactConstraintOptions options;
  options.steeringPlanar = true;
  options.trackRotatingRates = true;
  options.rollingWeight = 1000.0;
  options.rollingRateWeight = 250.0;
  options.steeringRateWeight = 40.0;

  mc_solver::RollingContactConstraint constraint(*robots, 0, wheels, options);
  auto solver = makeTasksSolver(robots);          // existing helper, dt = 0.005
  constraint.addToSolver(*solver);
  constraint.rotatingRateReference("front_left", 4.5, -0.75);
  constraint.update(*solver);

  const auto & labels = constraint.softRowLabels();
  const auto rolling = std::find(labels.begin(), labels.end(), "front_left/rolling-rate");
  const auto steering = std::find(labels.begin(), labels.end(), "front_left/steering-rate");
  BOOST_REQUIRE(rolling != labels.end());
  BOOST_REQUIRE(steering != labels.end());

  const auto rollingRow = static_cast<Eigen::Index>(std::distance(labels.begin(), rolling));
  const auto steeringRow = static_cast<Eigen::Index>(std::distance(labels.begin(), steering));
  const auto & A = constraint.softMatrix();
  const auto & b = constraint.softRhs();

  const double dt = solver->dt();
  const double rollingScale = std::sqrt(250.0 / 1000.0);
  const double steeringScale = std::sqrt(40.0 / 1000.0);
  const auto driveDof = robot.mb().jointPosInDof(static_cast<int>(robot.jointIndexByName("front_left_drive")));
  const auto steerDof = robot.mb().jointPosInDof(static_cast<int>(robot.jointIndexByName("front_left_steer")));

  // Row = scale * dt * selector, rhs = scale * (reference - measured).
  BOOST_CHECK_CLOSE(A(rollingRow, driveDof), rollingScale * dt, 1e-9);
  BOOST_CHECK_CLOSE(b(rollingRow), rollingScale * (4.5 - 1.5), 1e-9);
  BOOST_CHECK_SMALL(A(rollingRow, steerDof), 1e-12);

  BOOST_CHECK_CLOSE(A(steeringRow, steerDof), steeringScale * dt, 1e-9);
  BOOST_CHECK_CLOSE(b(steeringRow), steeringScale * (-0.75 - 0.25), 1e-9);
  BOOST_CHECK_SMALL(A(steeringRow, driveDof), 1e-12);

  constraint.removeFromSolver(*solver);
}

BOOST_AUTO_TEST_CASE(RotatingRateRowsAreAbsentWhenDisabledOrDetached)
{
  auto robots = makeFourSteeringRobots();
  auto wheels = makeFourSteeringWheels();

  mc_solver::RollingContactConstraintOptions options;
  options.steeringPlanar = true;
  options.trackRotatingRates = false;
  mc_solver::RollingContactConstraint disabled(*robots, 0, wheels, options);
  auto solver = makeTasksSolver(robots);
  disabled.addToSolver(*solver);
  disabled.update(*solver);
  for(const auto & label : disabled.softRowLabels())
  {
    BOOST_CHECK(label.find("-rate") == std::string::npos);
  }
  disabled.removeFromSolver(*solver);

  options.trackRotatingRates = true;
  mc_solver::RollingContactConstraint enabled(*robots, 0, wheels, options);
  auto solver2 = makeTasksSolver(robots);
  enabled.addToSolver(*solver2);
  enabled.mode("front_left", mc_rbdyn::RollingContactMode::Detached, 0.0);
  enabled.update(*solver2);
  for(const auto & label : enabled.softRowLabels())
  {
    BOOST_CHECK(label != "front_left/rolling-rate");
    BOOST_CHECK(label != "front_left/steering-rate");
  }
  // The other three wheels keep both of their rate rows.
  const auto & labels = enabled.softRowLabels();
  BOOST_CHECK(std::find(labels.begin(), labels.end(), "rear_right/steering-rate") != labels.end());
  enabled.removeFromSolver(*solver2);
}

BOOST_AUTO_TEST_CASE(RotatingRateRowsSteerTowardsTheReference)
{
  // With a steering-rate reference the QP must produce a steering acceleration
  // of the requested sign. Before this change delta_ddot was unconstrained and
  // the solver returned zero for it.
  auto robots = makeFourSteeringRobots();
  auto wheels = makeFourSteeringWheels();

  mc_solver::RollingContactConstraintOptions options;
  options.steeringPlanar = true;
  options.trackRotatingRates = true;
  mc_solver::RollingContactConstraint constraint(*robots, 0, wheels, options);
  auto solver = makeTasksSolver(robots);
  constraint.addToSolver(*solver);
  for(const auto & wheel : wheels) { constraint.rotatingRateReference(wheel.name, 0.0, 0.6); }

  BOOST_REQUIRE(solver->run());
  const auto & robot = robots->robot();
  const auto steerDof = robot.mb().jointPosInDof(static_cast<int>(robot.jointIndexByName("front_left_steer")));
  const Eigen::VectorXd alphaD = solver->alphaDVector(0);
  BOOST_CHECK_GT(alphaD(steerDof), 1.0);

  constraint.removeFromSolver(*solver);
}
```

If `makeTasksSolver` / `alphaDVector` are not the exact helper names in this file, use whatever
`RollingTasksMatrixPlacementAndHardCases` (`tests/testRollingContactSolver.cpp:332`) uses to build a solver and
read the solution — reuse, do not add a parallel helper.

- [ ] **Step 2: Run test to verify it fails**

```sh
cmake --build build -j --target testRollingContactSolver
CUDA_VISIBLE_DEVICES=-1 MC_RTC_DISABLE_CONVEX_GENERATION_PATCH=ON \
  ctest --test-dir build -R 'testRollingContactSolver' --output-on-failure
```

Expected: FAIL — `BOOST_REQUIRE(rolling != labels.end())` fails because no `front_left/rolling-rate` row exists.

- [ ] **Step 3: Name the two new axes**

In `src/mc_solver/RollingContactConstraint.cpp`, extend `Impl::axisName` (line 170):

```cpp
  static const char * axisName(int axis)
  {
    if(axis == 0) { return "longitudinal"; }
    if(axis == 1) { return "lateral"; }
    if(axis == 2) { return "normal"; }
    if(axis == 4) { return "rolling-rate"; }
    if(axis == 5) { return "steering-rate"; }
    return "drive-lock";
  }
```

Axis 3 remains drive-lock; do not renumber the existing axes, several tests match on those labels.

- [ ] **Step 4: Carry the per-row weight**

Extend the `Row` struct (line 160-168) with a weight, defaulting to the block weight:

```cpp
    bool driveLock;
    double scale;
    double weight;
```

and give `addRow` an optional weight that folds into the scale:

```cpp
  void addRow(size_t wheel, int axis, bool hard, bool fixedLongitudinal = false, bool driveLock = false,
              double weight = -1.0)
  {
    const double blockWeight = options.rollingWeight;
    const double rowWeight = weight < 0.0 ? blockWeight : weight;
    // A soft row scaled by s contributes blockWeight * s^2 * residual^2, so
    // s = sqrt(rowWeight / blockWeight) realises the requested per-row weight.
    const double weightScale = hard || blockWeight <= 0.0 ? 1.0 : std::sqrt(rowWeight / blockWeight);
    const double scale = (hard ? 1.0 : std::sqrt(activations[wheel])) * weightScale;
    auto & rows = hard ? hardRows : softRows;
    auto & labels = hard ? hardLabels : softLabels;
    rows.push_back({wheel, axis, fixedLongitudinal, driveLock, scale, rowWeight});
    labels.push_back(wheels[wheel].name + "/" + axisName(axis));
  }
```

- [ ] **Step 5: Append the rate rows in `buildRowLayout`**

In `buildRowLayout`, immediately before the `hardA.setZero(...)` sizing call (line 273), add:

```cpp
    if(options.trackRotatingRates)
    {
      for(size_t i = 0; i < wheels.size(); ++i)
      {
        if(wheels[i].mode == mc_rbdyn::RollingContactMode::Detached || activations[i] <= 0.0) { continue; }
        addRow(i, 4, false, false, false, options.rollingRateWeight);
        if(!wheels[i].steeringJoint.empty())
        {
          addRow(i, 5, false, false, false, options.steeringRateWeight);
        }
      }
    }
```

The rate rows are always soft: they are objective terms in `eq:four-steering-wheel-qp`, never equalities. A
detached wheel contributes neither.

- [ ] **Step 6: Fill the rate rows**

In `Impl::fillRow` (line 291), handle the new axes before the existing `else` branch:

```cpp
    if(row.driveLock)
    {
      // Existing branch: keep exactly as it is today (lines 293-300).
      A = geometries[row.wheel].kinematics().wheelSelector;
      const double rate = geometries[row.wheel].kinematics().wheelSelector.dot(
          geometries[row.wheel].kinematics().generalizedVelocity);
      rhs(rowIndex) = -options.velocityGain * rate;
    }
    else if(row.axis == 4 || row.axis == 5)
    {
      const auto & kinematics = geometries[row.wheel].kinematics();
      const auto & result = results[row.wheel];
      const bool steering = row.axis == 5;
      // rate^+ = rate + dt * (S * alphaD); track (rate^+ - reference).
      A = dt * (steering ? kinematics.steeringSelector : kinematics.wheelSelector);
      const double measured = steering ? result.measuredSteeringRate : result.measuredRollingRate;
      const double reference =
          steering ? steeringRateReferences[row.wheel] : rollingRateReferences[row.wheel];
      rhs(rowIndex) = reference - measured;
    }
    else
    {
      // Existing branch: keep exactly as it is today (lines 304-309).
      const auto & result = results[row.wheel];
      A = result.rollingMatrix.row(row.axis);
      rhs(rowIndex) = result.rhs(row.axis);
    }
```

The trailing `A *= row.scale; rhs(rowIndex) *= row.scale;` at the end of `fillRow` already applies to the new
branch, which is what produces the `scale * dt * selector` matrix entry and the `scale * (reference - measured)`
right-hand side the test checks.

Note the `row.fixedLongitudinal` block below must not run for axes 4/5; it is already guarded by
`if(row.fixedLongitudinal)` and rate rows are always created with `fixedLongitudinal == false`.

- [ ] **Step 7: Capture the timestep**

In `RollingContactConstraint::update` (line 395), as the first statement of the function body:

```cpp
  impl_->dt = solver.dt();
```

`fillRow` is `const`, so read `dt` from the Impl member. `dt` must be set before the first `fillRow` call in
the same `update`, which this ordering guarantees.

- [ ] **Step 8: Rebuild the layout when the references or weights imply new rows**

`buildRowLayout` is already re-run whenever modes or activations change and bumps `layoutRevision` /
`tvmLayoutDirty` on a label change (line 285-292). Because rate rows depend only on
`trackRotatingRates`, the mode and the activation — never on the reference values — no extra invalidation is
needed. Confirm this by inspection; do not add a rebuild in `rotatingRateReference`, which is called every
control cycle and must stay allocation-free.

- [ ] **Step 9: Run tests to verify they pass**

```sh
cmake --build build -j --target testRollingContactSolver
CUDA_VISIBLE_DEVICES=-1 MC_RTC_DISABLE_CONVEX_GENERATION_PATCH=ON \
  ctest --test-dir build -R 'testRollingContactSolver' --output-on-failure
```

Expected: PASS, including the three new cases. The TVM parity tests
(`RollingTasksTVMStaticSolutionParity`, `RollingTVMSoftRowsInfluenceSolution`) must also pass unchanged — the
TVM path consumes the same `softMatrix()`/`softRhs()`, so it inherits the rows for free. If they fail, the
cause is `tvmLayoutDirty` not being set when the rate rows first appear; fix that rather than special-casing
the backend.

- [ ] **Step 10: Commit**

```bash
git add src/mc_solver/RollingContactConstraint.cpp tests/testRollingContactSolver.cpp
git commit -m "feat(rolling): emit predicted rolling and steering rate rows in the QP"
```

---

### Task 4: Derive steering and rolling references from the commanded twist

Implements the inverse of `eq:steering-expanded-constraints` as a small, testable free function, so the
reference generation can be unit-tested without a solver and without the keyboard.

For wheel `i` at chassis offset `rho_i = (x_i, y_i)` and commanded planar twist `(v_x, v_y, omega)`:

```
  p_i   = (v_x - omega * y_i,  v_y + omega * x_i)     // eq:steering-carrier-velocity
  delta_i^ref  = atan2(p_i.y, p_i.x)                  // eq:steering-expanded-constraints, lateral row = 0
  thetaDot_i^ref = (p_i . [cos delta_i^ref, sin delta_i^ref]) / r_i
```

with two corrections that are geometry, not tuning:

- **Branch selection.** The steering joint is limited to `+/- pi/2`. `delta` and `delta + pi` describe the same
  wheel line with opposite rolling sign, so pick the representation inside the limit that is nearest the
  measured angle and negate `thetaDot^ref` when the flipped one is chosen.
- **Degenerate command.** When `|p_i|` is below a threshold the heading is undefined; hold the measured angle
  and command zero rolling rate.

Then `deltaDot_i^ref = clamp(wrap(delta_i^ref - delta_i) / tau, +/- deltaDot_max)`, where `tau` is a
first-order convergence time constant and `deltaDot_max` comes from the joint velocity limit.

**Files:**
- Modify: `include/mc_rbdyn/RollingContact.h` (after `steeringRollingMatrix`, line 293)
- Modify: `src/mc_rbdyn/RollingContact.cpp` (after `steeringRollingMatrix`)
- Test: `tests/testRollingContact.cpp`

- [ ] **Step 1: Write the failing test**

Append to `tests/testRollingContact.cpp`:

```cpp
BOOST_AUTO_TEST_CASE(SteeringReferenceInvertsTheExpandedConstraints)
{
  mc_rbdyn::PlanarWheel wheel;
  wheel.offset = Eigen::Vector2d(0.247, 0.182);
  wheel.radius = 0.125;
  wheel.steeringAngle = 0.0;

  // Pure forward: no steering, rolling rate = v / r.
  {
    const auto reference = mc_rbdyn::steeringWheelReference(wheel, Eigen::Vector3d(1.0, 0.0, 0.0), 0.0);
    BOOST_CHECK_SMALL(reference.steeringAngle, 1e-9);
    BOOST_CHECK_CLOSE(reference.rollingRate, 1.0 / 0.125, 1e-9);
  }

  // Pure crab (A/D): steer to +90 degrees, rolling rate = v / r.
  {
    const auto reference = mc_rbdyn::steeringWheelReference(wheel, Eigen::Vector3d(0.0, 1.0, 0.0), 0.0);
    BOOST_CHECK_CLOSE(reference.steeringAngle, 0.5 * M_PI, 1e-6);
    BOOST_CHECK_CLOSE(reference.rollingRate, 1.0 / 0.125, 1e-6);
  }

  // Pure yaw (Q/E): the wheel is tangent to the circle through its offset, and
  // a positive omega must give a positive rolling rate on the front-left wheel.
  {
    const auto reference = mc_rbdyn::steeringWheelReference(wheel, Eigen::Vector3d(0.0, 0.0, 1.0), 0.0);
    const double expected = std::atan2(0.247, -0.182);
    // atan2 result lies outside +/- pi/2, so the flipped representation is used.
    BOOST_CHECK_CLOSE(reference.steeringAngle, expected - M_PI, 1e-6);
    BOOST_CHECK_LT(reference.rollingRate, 0.0);
    BOOST_CHECK_CLOSE(std::abs(reference.rollingRate), std::hypot(0.182, 0.247) / 0.125, 1e-6);
  }

  // Reconstructing the twist from the reference reproduces the expanded rows.
  {
    const Eigen::Vector3d twist(0.4, -0.2, 0.3);
    const auto reference = mc_rbdyn::steeringWheelReference(wheel, twist, 0.0);
    const double c = std::cos(reference.steeringAngle);
    const double s = std::sin(reference.steeringAngle);
    const double px = twist.x() - twist.z() * wheel.offset.y();
    const double py = twist.y() + twist.z() * wheel.offset.x();
    BOOST_CHECK_CLOSE(c * px + s * py, wheel.radius * reference.rollingRate, 1e-6);   // rolling row
    BOOST_CHECK_SMALL(-s * px + c * py, 1e-9);                                        // lateral row
  }
}

BOOST_AUTO_TEST_CASE(SteeringReferenceStaysWithinLimitsAndAvoidsBranchChatter)
{
  mc_rbdyn::PlanarWheel wheel;
  wheel.offset = Eigen::Vector2d(0.247, 0.182);
  wheel.radius = 0.125;

  // A crabbed wheel held near -90 degrees must not jump to +90 when the
  // commanded lateral direction is perturbed by numerical noise.
  const auto negative = mc_rbdyn::steeringWheelReference(wheel, Eigen::Vector3d(0.0, -1.0, 0.0), -0.5 * M_PI);
  BOOST_CHECK_CLOSE(negative.steeringAngle, -0.5 * M_PI, 1e-6);
  const auto perturbed = mc_rbdyn::steeringWheelReference(wheel, Eigen::Vector3d(1e-9, -1.0, 0.0), -0.5 * M_PI);
  BOOST_CHECK_SMALL(std::abs(perturbed.steeringAngle - negative.steeringAngle), 1e-3);

  // Every reference respects the +/- pi/2 hinge limit.
  for(double angle = -M_PI; angle <= M_PI; angle += 0.05)
  {
    const Eigen::Vector3d twist(std::cos(angle), std::sin(angle), 0.4);
    const auto reference = mc_rbdyn::steeringWheelReference(wheel, twist, 0.0);
    BOOST_CHECK_LE(std::abs(reference.steeringAngle), 0.5 * M_PI + 1e-9);
    BOOST_CHECK(std::isfinite(reference.rollingRate));
  }
}

BOOST_AUTO_TEST_CASE(SteeringReferenceHoldsPoseForADegenerateCommand)
{
  mc_rbdyn::PlanarWheel wheel;
  wheel.offset = Eigen::Vector2d(0.247, 0.182);
  wheel.radius = 0.125;

  const auto reference = mc_rbdyn::steeringWheelReference(wheel, Eigen::Vector3d::Zero(), 0.3);
  BOOST_CHECK_CLOSE(reference.steeringAngle, 0.3, 1e-9);
  BOOST_CHECK_SMALL(reference.rollingRate, 1e-12);
}
```

The tests above use `M_PI`. It is not guaranteed by the C++ standard, and the rest of this codebase spells the
constant out (for example `mc_rolling_contact_controller.cpp:1387`). Either add
`constexpr double pi = 3.14159265358979323846;` at the top of the test file and use that, or confirm
`<cmath>` on this toolchain provides `M_PI` before keeping it.

- [ ] **Step 2: Run test to verify it fails**

```sh
cmake --build build -j --target testRollingContact
```

Expected: **compile error**, `'steeringWheelReference' is not a member of 'mc_rbdyn'`.

- [ ] **Step 3: Declare the reference type and function**

In `include/mc_rbdyn/RollingContact.h`, after `steeringRollingMatrix` (line 293):

```cpp
/** Reference steering pose and rolling rate for one steering wheel. */
struct MC_RBDYN_DLLAPI SteeringWheelReference
{
  /** Steering angle within +/- pi/2, nearest to the measured angle. */
  double steeringAngle = 0.0;
  /** Rolling rate consistent with steeringAngle, signed for the chosen branch. */
  double rollingRate = 0.0;
  /** False when the command was degenerate and the measured pose was held. */
  bool commanded = false;
};

/** Invert the expanded steering constraints for one wheel.
 *
 * Solves the lateral row of the four-steering expansion for the wheel heading
 * and the rolling row for the wheel rate, given a planar chassis twist
 * [vx, vy, omega]. The returned angle is the representation inside the +/- pi/2
 * hinge range that is nearest @p measuredSteeringAngle; the rolling rate is
 * negated when the flipped representation is selected. A command whose wheel
 * speed falls below @p speedThreshold is degenerate: the measured angle is held
 * and the rate is zero.
 */
MC_RBDYN_DLLAPI SteeringWheelReference steeringWheelReference(const PlanarWheel & wheel,
                                                              const Eigen::Vector3d & planarTwist,
                                                              double measuredSteeringAngle,
                                                              double speedThreshold = 1e-3);
```

- [ ] **Step 4: Implement it**

In `src/mc_rbdyn/RollingContact.cpp`, after `steeringRollingMatrix`:

```cpp
SteeringWheelReference steeringWheelReference(const PlanarWheel & wheel,
                                              const Eigen::Vector3d & planarTwist,
                                              double measuredSteeringAngle,
                                              double speedThreshold)
{
  wheel.validate();
  requireFinite(planarTwist, "planarTwist");
  if(!finite(measuredSteeringAngle) || !finite(speedThreshold) || speedThreshold < 0.0)
  {
    throw std::invalid_argument("steeringWheelReference measuredSteeringAngle and speedThreshold must be finite");
  }

  constexpr double halfPi = 0.5 * 3.14159265358979323846;
  constexpr double pi = 3.14159265358979323846;

  SteeringWheelReference out;
  // eq:steering-carrier-velocity
  const Eigen::Vector2d point(planarTwist.x() - planarTwist.z() * wheel.offset.y(),
                              planarTwist.y() + planarTwist.z() * wheel.offset.x());
  if(point.norm() < speedThreshold)
  {
    out.steeringAngle = measuredSteeringAngle;
    out.rollingRate = 0.0;
    out.commanded = false;
    return out;
  }

  // Lateral row of eq:steering-expanded-constraints is zero exactly when the
  // wheel heading is aligned with the carrier velocity.
  const double raw = std::atan2(point.y(), point.x());
  // The two equivalent poses, separated by pi with opposite rolling sign.
  const double flipped = raw > 0.0 ? raw - pi : raw + pi;
  const bool rawValid = std::abs(raw) <= halfPi;
  const bool flippedValid = std::abs(flipped) <= halfPi;

  double angle = raw;
  double sign = 1.0;
  if(rawValid && flippedValid)
  {
    // Both representations are reachable: keep the one nearest the measurement
    // so noise around the branch cut cannot make the hinge chatter.
    if(std::abs(flipped - measuredSteeringAngle) < std::abs(raw - measuredSteeringAngle))
    {
      angle = flipped;
      sign = -1.0;
    }
  }
  else if(flippedValid)
  {
    angle = flipped;
    sign = -1.0;
  }
  angle = std::clamp(angle, -halfPi, halfPi);

  // Rolling row of eq:steering-expanded-constraints solved for thetaDot.
  out.steeringAngle = angle;
  out.rollingRate = sign * (std::cos(angle) * point.x() + std::sin(angle) * point.y())
                    / (wheel.radius * wheel.spinSign);
  out.commanded = true;
  return out;
}
```

`sign` multiplies the projection because the projection is already computed with the *selected* angle; when the
flipped branch is taken the projection is negative and `sign` restores the physically correct rate sign for the
wheel line. The `wheel.spinSign` division matches the convention of `steeringRollingMatrix`, which subtracts
`radius * spinSign * wheelSelector` from the longitudinal row.

Add `#include <algorithm>` for `std::clamp` if it is not already included.

- [ ] **Step 5: Run tests to verify they pass**

```sh
cmake --build build -j --target testRollingContact
CUDA_VISIBLE_DEVICES=-1 MC_RTC_DISABLE_CONVEX_GENERATION_PATCH=ON \
  ctest --test-dir build -R 'testRollingContact$' --output-on-failure
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add include/mc_rbdyn/RollingContact.h src/mc_rbdyn/RollingContact.cpp tests/testRollingContact.cpp
git commit -m "feat(rolling): invert the expanded steering constraints into wheel references"
```

---

### Task 5: Drive the sample controller from the QP rate references

Replaces the analytic steering IK and its empirical corrections with `steeringWheelReference` plus the QP rate
rows. This is the task that makes WASD/QE work; it is also the largest deletion.

**Files:**
- Modify: `src/mc_control/samples/RollingContact/mc_rolling_contact_controller.h:70-113`
- Modify: `src/mc_control/samples/RollingContact/mc_rolling_contact_controller.cpp:346-400`, `:1254-1540`
- Test: `src/mc_control/samples/RollingContact/test_controller_lifecycle.cpp`

- [ ] **Step 1: Write the failing test**

Append to `src/mc_control/samples/RollingContact/test_controller_lifecycle.cpp`:

```cpp
BOOST_AUTO_TEST_CASE(FourSteeringTracksCommandedTwistSigns)
{
  // Each commanded twist must move the chassis in the commanded direction.
  // Before this change a positive yaw command rotated the chassis clockwise,
  // which the keyboard path patched with an explicit sign mirror.
  struct Case
  {
    const char * name;
    Eigen::Vector3d twist;   // vx, vy, omega in the chassis frame
  };
  const std::vector<Case> cases = {{"forward", {0.3, 0.0, 0.0}},
                                   {"backward", {-0.3, 0.0, 0.0}},
                                   {"crab-left", {0.0, 0.3, 0.0}},
                                   {"crab-right", {0.0, -0.3, 0.0}},
                                   {"yaw-positive", {0.0, 0.0, 0.5}},
                                   {"yaw-negative", {0.0, 0.0, -0.5}}};

  for(const auto & test : cases)
  {
    auto controller = makeRangerController();     // existing helper in this file
    controller->setCommandedTwist(test.twist);    // added in Step 4

    const sva::PTransformd start = controller->robot().posW();
    for(int i = 0; i < 400; ++i) { BOOST_REQUIRE_MESSAGE(controller->run(), test.name); }
    const sva::PTransformd end = controller->robot().posW();

    const Eigen::Vector3d motion = start.rotation() * (end.translation() - start.translation());
    const Eigen::Matrix3d relative = end.rotation() * start.rotation().transpose();
    const double yaw = std::atan2(relative(1, 0), relative(0, 0));

    if(std::abs(test.twist.x()) > 1e-9)
    {
      BOOST_CHECK_MESSAGE(motion.x() * test.twist.x() > 0.0, test.name);
      BOOST_CHECK_MESSAGE(std::abs(motion.x()) > 0.05, test.name);
    }
    if(std::abs(test.twist.y()) > 1e-9)
    {
      BOOST_CHECK_MESSAGE(motion.y() * test.twist.y() > 0.0, test.name);
      BOOST_CHECK_MESSAGE(std::abs(motion.y()) > 0.05, test.name);
    }
    if(std::abs(test.twist.z()) > 1e-9)
    {
      BOOST_CHECK_MESSAGE(yaw * test.twist.z() > 0.0, test.name);
      BOOST_CHECK_MESSAGE(std::abs(yaw) > 0.1, test.name);
    }
  }
}

BOOST_AUTO_TEST_CASE(FourSteeringCommandChangeKeepsResidualsBounded)
{
  // Switching between commands must not need a transition grace period: the QP
  // arbitrates steering and drive together, so no wheel receives drive torque
  // against a stale contact direction.
  auto controller = makeRangerController();
  const std::vector<Eigen::Vector3d> sequence = {{0.3, 0.0, 0.0}, {0.0, 0.0, 0.5}, {0.0, 0.3, 0.0}, {0.3, 0.0, 0.4}};

  for(const auto & twist : sequence)
  {
    controller->setCommandedTwist(twist);
    for(int i = 0; i < 200; ++i)
    {
      BOOST_REQUIRE(controller->run());
      BOOST_CHECK_LT(controller->maxLateralResidual(), 0.05);
    }
  }
}
```

`makeRangerController()` follows the pattern of `RollingContactControllerScenarioMatrix`
(`test_controller_lifecycle.cpp:62`); build it on the `ranger_mini_v3` variant with
`scenario: four_steering_reference` and `closedLoopFeedback: true`. `maxLateralResidual()` needs a public
accessor for the existing `maxLateralResidual_` member — add it in Step 4.

- [ ] **Step 2: Run test to verify it fails**

```sh
cmake --build build -j --target testRollingContactControllerLifecycle
```

Expected: **compile error**, `'class mc_control::MCRollingContactController' has no member named 'setCommandedTwist'`.

- [ ] **Step 3: Enable rate tracking on the constraint**

In `mc_rolling_contact_controller.cpp`, where `rolling_` is constructed (near line 346-374), set the new
options for the four-steering case:

The constructor already builds a `RollingContactConstraintOptions` and fills `terrainNormal`, `longitudinal`,
`velocityGain`, `rollingWeight`, `differentialPlanar`, `steeringPlanar` and `steeringPlanarWheels` from
`settings`. Leave every one of those lines untouched and append:

```cpp
  if(fourSteering_)
  {
    rollingOptions.trackRotatingRates = true;
    rollingOptions.rollingRateWeight = settings("rollingRateWeight", 200.0);
    rollingOptions.steeringRateWeight = settings("steeringRateWeight", 200.0);
  }
```

- [ ] **Step 4: Add the twist command entry point and the residual accessor**

In `mc_rolling_contact_controller.h`, in the public section after `reset`:

```cpp
  bool run() override;
  void reset(const ControllerResetData & data) override;

  /** Set the commanded planar chassis twist [vx, vy, omega] in the chassis frame.
   *
   * This is the single entry point used by the keyboard, the GUI and the
   * scripted scenarios. It only stores the command; the references are rebuilt
   * on the next run().
   */
  void setCommandedTwist(const Eigen::Vector3d & twist);
  const Eigen::Vector3d & commandedTwist() const noexcept { return commandedTwist_; }
  double maxLateralResidual() const noexcept { return maxLateralResidual_; }
```

In the private section, add:

```cpp
  Eigen::Vector3d commandedTwist_ = Eigen::Vector3d::Zero();
  double steeringTimeConstant_ = 0.15;
  double maxSteeringRate_ = 8.0;
  std::vector<double> steeringRateReferences_;
```

and **delete** these members, which exist only to patch the open-loop steering:

```cpp
  double keyboardYawScale_ = 1.0;                  // delete
  double keyboardMixedDriveScale_ = 1.15;          // delete
  double keyboardCommandTransitionGrace_ = 0.0;    // delete
  bool keyboardSteeringReady_ = true;              // delete
  double keyboardWheelPositionLookahead_ = 0.0;    // delete
```

Also delete `updateKeyboardCommandTransition()` from the header and its definition, and remove its call site.

Implement the setter in the `.cpp`:

```cpp
void MCRollingContactController::setCommandedTwist(const Eigen::Vector3d & twist)
{
  if(!twist.allFinite())
  {
    mc_rtc::log::error_and_throw<std::invalid_argument>("RollingContact commanded twist must be finite");
  }
  commandedTwist_ = twist;
}
```

- [ ] **Step 5: Replace the four-steering branch of `updateReference`**

Replace the whole four-steering `else` branch of `updateReference`
(`mc_rolling_contact_controller.cpp:1281-1537`, from `else` through `postureTask->target(targets);`) with:

```cpp
  else
  {
    const double dt = solver().dt();
    const double maxSteeringStep = maxSteeringRate_ * dt;
    for(size_t i = 0; i < wheels_.size(); ++i)
    {
      mc_rbdyn::PlanarWheel planar;
      planar.offset = wheelOffsets_[i];
      planar.radius = wheels_[i].radius;
      planar.spinSign = wheels_[i].spinSign;

      const auto steeringJoint = robot().jointIndexByName(wheels_[i].steeringJoint);
      const double measuredSteering = robot().mbc().q[steeringJoint][0];
      planar.steeringAngle = measuredSteering;
      planar.steeringRate = robot().mbc().alpha[steeringJoint][0];

      const auto reference = mc_rbdyn::steeringWheelReference(planar, commandedTwist_, measuredSteering);

      // First-order convergence to the reference heading, saturated by the
      // hinge velocity limit. This is deltaDot^ref of the four-steering QP.
      const double error = reference.steeringAngle - measuredSteering;
      const double steeringRate =
          std::clamp(error / steeringTimeConstant_, -maxSteeringRate_, maxSteeringRate_);

      steeringTargets_[i] = measuredSteering + std::clamp(steeringRate * dt, -maxSteeringStep, maxSteeringStep);
      steeringRateReferences_[i] = steeringRate;
      wheelReferenceRates_[i] = reference.rollingRate;
      driveTargets_[i] += reference.rollingRate * dt;

      // The QP owns both rates from here on.
      rolling_->rotatingRateReference(wheels_[i].name, reference.rollingRate, steeringRate);
    }
  }
```

Note what is gone and why:

- the `wheelYaw` sign mirror — the sign is now fixed once, in `spinSign` (Task 6);
- `keyboardYawScale_` and `keyboardMixedDriveScale_` — the QP tracks the requested rate directly instead of
  compensating an open-loop position command;
- the `steeringAligned` / `allSteeringAligned` drive gating — the QP now sees the steering rate, so it will not
  drive a wheel against a stale contact direction;
- the branch-cut hysteresis block — that logic now lives once inside `steeringWheelReference`;
- the `keyboardStopped` / `pureLateralKeyboard` special cases — subsumed by the degenerate-command branch.

Keep the posture task only as a weak regulariser: reduce the wheel-joint entries so they cannot fight the rate
rows. In the constructor block at `:388-397`, set the steering weight to a small value
(`settings("steeringPostureWeight", 1.0)`) instead of the current `postureWeight`.

- [ ] **Step 6: Route the keyboard and GUI through the new entry point**

Wherever the keyboard poll produces `forward`, `lateral`, `yaw` and wherever
`guiForwardCommand_` / `guiLateralCommand_` / `guiYawCommand_` are read, replace the direct use of those
scalars with a single call:

```cpp
  setCommandedTwist({forward, lateral, yaw});
```

Keep `keyboardLinearSpeed` / `keyboardAngularSpeed` scaling at the point where the key state becomes a
velocity; only the fudge factors are removed.

- [ ] **Step 7: Size the new vector on reset**

In `MCRollingContactController::reset` and in the constructor where `steeringTargets_` is sized (line 408-410):

```cpp
  steeringRateReferences_.assign(wheels_.size(), 0.0);
  commandedTwist_.setZero();
```

- [ ] **Step 8: Log the new references**

Next to the existing per-wheel log entries (`:721-730`):

```cpp
    logger().addLogEntry("RollingContact_" + wheel.name + "_steeringRateRef",
                         [this, i]() { return steeringRateReferences_[i]; });
    logger().addLogEntry("RollingContact_" + wheel.name + "_rollingRateRef",
                         [this, i]() { return wheelReferenceRates_[i]; });
```

and the commanded twist:

```cpp
  logger().addLogEntry("RollingContact_commandedTwist", [this]() -> const Eigen::Vector3d & { return commandedTwist_; });
```

- [ ] **Step 9: Run tests to verify they pass**

```sh
cmake --build build -j
CUDA_VISIBLE_DEVICES=-1 MC_RTC_DISABLE_CONVEX_GENERATION_PATCH=ON \
  ctest --test-dir build -R 'RollingContact' --output-on-failure -j1
```

Expected: PASS. If `FourSteeringTracksCommandedTwistSigns` fails only on the two yaw cases, stop and do
Task 6 first — that is the sign bug, not a tracking failure.

- [ ] **Step 10: Commit**

```bash
git add src/mc_control/samples/RollingContact/
git commit -m "feat(rolling): drive four-steering references through the QP rate rows"
```

---

### Task 6: Fix the yaw sign at its source

The keyboard path currently negates the yaw used for the wheel IK, with the comment *"a positive wheel
spin/steering solution generated directly from +yaw rotates the chassis clockwise"*
(`mc_rolling_contact_controller.cpp:1309-1316`). Task 5 deletes that mirror, so the underlying convention must
be correct.

**The convention, derived from the model.** The drive axis is `+y` (`ranger_mini_v3.urdf:49`). With the contact
below the wheel centre at `r = (0, 0, -R)` and `omega = thetaDot * y`, rolling without slip gives
`v_centre = -omega x r = +thetaDot * R * x`. So a **positive** drive rate moves the wheel **forward (+x)** and
`spinSign = +1` is correct for this URDF. `steeringRollingMatrix` and `RollingContactGeometry` already encode
that. If the assembled controller still rotates the wrong way, the error is in the wheel **offset ordering** or
in a MuJoCo actuator sign, not in the QP.

**Files:**
- Modify: `src/mc_control/samples/RollingContact/mc_rolling_contact_controller.cpp` (offset construction)
- Modify: `rolling-contact-report/mujoco/ranger_mini_v3.yaml` if the mismatch is in the mc_mujoco mapping
- Test: `tests/testRollingContactRobot.cpp`

- [ ] **Step 1: Write the failing test**

Append to `tests/testRollingContactRobot.cpp`:

```cpp
BOOST_AUTO_TEST_CASE(RangerWheelOffsetsMatchTheUrdfLayout)
{
  auto robots = makeRangerMiniV3Robots();
  const auto & robot = robots->robot();
  const sva::PTransformd X_0_chassis = robot.frame("chassis").position();

  const std::vector<std::pair<std::string, Eigen::Vector2d>> expected = {
      {"front_left", {0.247, 0.182}},
      {"front_right", {0.247, -0.182}},
      {"rear_left", {-0.247, 0.182}},
      {"rear_right", {-0.247, -0.182}}};

  for(const auto & [name, offset] : expected)
  {
    const sva::PTransformd X_0_carrier = robot.frame(name + "_carrier").position();
    const Eigen::Vector3d local = X_0_chassis.rotation() * (X_0_carrier.translation() - X_0_chassis.translation());
    BOOST_CHECK_SMALL(std::abs(local.x() - offset.x()), 1e-6);
    BOOST_CHECK_SMALL(std::abs(local.y() - offset.y()), 1e-6);
  }
}

BOOST_AUTO_TEST_CASE(PositiveYawGivesPositiveRollingRateOnTheFrontLeftWheel)
{
  // With +z yaw the front-left wheel (x>0, y>0) travels along -y in the chassis
  // frame; after branch selection the reference rolling rate must be negative
  // and the front-right wheel positive. This pins the sign convention that the
  // keyboard path used to mirror by hand.
  mc_rbdyn::PlanarWheel frontLeft;
  frontLeft.offset = Eigen::Vector2d(0.247, 0.182);
  frontLeft.radius = 0.125;
  mc_rbdyn::PlanarWheel frontRight = frontLeft;
  frontRight.offset = Eigen::Vector2d(0.247, -0.182);

  const Eigen::Vector3d yawOnly(0.0, 0.0, 1.0);
  const auto left = mc_rbdyn::steeringWheelReference(frontLeft, yawOnly, 0.0);
  const auto right = mc_rbdyn::steeringWheelReference(frontRight, yawOnly, 0.0);
  BOOST_CHECK_LT(left.rollingRate * right.rollingRate, 0.0);
}
```

- [ ] **Step 2: Run test to verify it fails or passes**

```sh
cmake --build build -j --target testRollingContactRobot
CUDA_VISIBLE_DEVICES=-1 MC_RTC_DISABLE_CONVEX_GENERATION_PATCH=ON \
  ctest --test-dir build -R 'testRollingContactRobot' --output-on-failure
```

If both pass, the model conventions are already right and the mirror was compensating a MuJoCo actuator sign —
go to Step 4. If `RangerWheelOffsetsMatchTheUrdfLayout` fails, the controller's `wheelOffsets_` ordering does
not match `makeWheels()`; fix that in Step 3.

- [ ] **Step 3: Correct the offset ordering if the test failed**

`makeWheels()` builds names in the order `front_left, front_right, rear_left, rear_right`
(`mc_rolling_contact_controller.cpp:848-855`). Make `wheelOffsets_` be filled from the actual carrier frame
positions rather than a hand-written table, so the two can never drift apart:

```cpp
  wheelOffsets_.resize(wheels_.size());
  const sva::PTransformd X_0_chassis = robot().frame("chassis").position();
  for(size_t i = 0; i < wheels_.size(); ++i)
  {
    const sva::PTransformd X_0_carrier = robot().frame(wheels_[i].carrierFrame).position();
    const Eigen::Vector3d local =
        X_0_chassis.rotation() * (X_0_carrier.translation() - X_0_chassis.translation());
    wheelOffsets_[i] = local.head<2>();
  }
```

- [ ] **Step 4: Check the MuJoCo actuator signs**

Inspect `rolling-contact-report/mujoco/ranger_mini_v3.yaml` and the model it maps to. If a drive or steer
actuator has a negative gear relative to the URDF axis, correct it **in the mapping**, and record the finding
in `rolling-contact-report/issues.md`. Do not reintroduce a sign flip in the controller: the controller is
shared by the ticker (no physics) and MuJoCo, and a controller-side mirror would break the ticker.

- [ ] **Step 5: Run the full rolling suite**

```sh
CUDA_VISIBLE_DEVICES=-1 MC_RTC_DISABLE_CONVEX_GENERATION_PATCH=ON \
  ctest --test-dir build -R 'RollingContact|testRollingContact' --output-on-failure -j1
```

Expected: PASS, including `FourSteeringTracksCommandedTwistSigns` from Task 5.

- [ ] **Step 6: Commit**

```bash
git add tests/testRollingContactRobot.cpp src/mc_control/samples/RollingContact/ \
        rolling-contact-report/mujoco/ranger_mini_v3.yaml rolling-contact-report/issues.md
git commit -m "fix(rolling): pin the four-steering yaw sign convention to the model"
```

---

### Task 7: Make `mc_rtc_ticker -s robot:=<name>` select the robot

Verified failure: the acceptance command silently ignores `robot:=RollingContactRangerMiniV3` and runs the
default JVRC1 robot from `/usr/local/etc/mc_rtc.yaml`.

**Assumption, stated explicitly:** `robot:=NAME` is roslaunch-style argument syntax, and the intent is
"run the ticker with this robot". This task teaches the binary to accept `key:=value` tokens and map them onto
the global configuration. `robot` and `controller` are supported; anything else is a hard error rather than a
silent no-op, so a typo can never again look like success.

**Files:**
- Modify: `utils/mc_rtc_ticker.cpp`
- Create: `~/.config/mc_rtc/mc_rtc.yaml`

- [ ] **Step 1: Parse the overrides**

In `utils/mc_rtc_ticker.cpp`, register a positional catch-all and translate it. Add to the options description:

```cpp
      ("replay-outputs", po::bool_switch(&replay_outputs), "Enable outputs replay (override controller)")
      ("override", po::value<std::vector<std::string>>()->multitoken(),
       "roslaunch-style overrides, e.g. robot:=JVRC1 controller:=RollingContact");
```

and replace the parser invocation:

```cpp
    po::positional_options_description positional;
    positional.add("override", -1);
    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).options(desc).positional(positional).run(), vm);
    po::notify(vm);
```

- [ ] **Step 2: Turn the overrides into a configuration layer**

After `po::notify(vm)` and before the `Ticker` is constructed:

```cpp
    if(vm.count("override"))
    {
      mc_rtc::Configuration overrides;
      for(const auto & token : vm["override"].as<std::vector<std::string>>())
      {
        const auto separator = token.find(":=");
        if(separator == std::string::npos)
        {
          mc_rtc::log::error_and_throw("Unexpected argument \"{}\": expected key:=value", token);
        }
        const auto key = token.substr(0, separator);
        const auto value = token.substr(separator + 2);
        if(key == "robot") { overrides.add("MainRobot", value); }
        else if(key == "controller")
        {
          overrides.add("Enabled", std::vector<std::string>{value});
          overrides.add("Default", value);
        }
        else { mc_rtc::log::error_and_throw("Unsupported override \"{}\"", key); }
      }
      // MCGlobalController merges every file named by MC_RTC_CONTROLLER_CONFIG on
      // top of the user configuration, so writing one temporary file needs no
      // change to the global controller itself.
      const auto path = (std::filesystem::temp_directory_path()
                         / fmt::format("mc_rtc_ticker_overrides_{}.yaml", getpid())).string();
      overrides.save(path);
      std::string chain = path;
      if(const char * existing = std::getenv("MC_RTC_CONTROLLER_CONFIG"))
      {
        chain += std::string(":") + existing;
      }
      setenv("MC_RTC_CONTROLLER_CONFIG", chain.c_str(), 1);
      mc_rtc::log::info("Applied command-line overrides from {}", path);
    }
```

**Verify the precedence before relying on it.** `src/mc_control/mc_global_controller_configuration.cpp:64-95`
documents the list as "applied from last to first". Read that block and place the temporary file at whichever
end actually wins; the test in Step 3 is what proves it.

- [ ] **Step 3: Verify manually**

```sh
cmake --build build -j --target mc_rtc_ticker
CUDA_VISIBLE_DEVICES=-1 ./build/utils/mc_rtc_ticker -s robot:=JVRC1 --run-for 0.05 2>&1 | grep -i "MainRobot\|robot module"
CUDA_VISIBLE_DEVICES=-1 ./build/utils/mc_rtc_ticker -s bogus:=1 --run-for 0.05 2>&1 | tail -3
```

Expected: the first selects JVRC1; the second exits with `Unsupported override "bogus"` rather than running.

- [ ] **Step 4: Write the user configuration**

The acceptance command passes no `-f`, so it reads `/usr/local/etc/mc_rtc.yaml` plus
`~/.config/mc_rtc/mc_rtc.yaml`. Create the latter so the RollingContact controller is enabled and configured:

```sh
mkdir -p ~/.config/mc_rtc
cat > ~/.config/mc_rtc/mc_rtc.yaml <<'YAML'
MainRobot: RollingContactRangerMiniV3
Enabled: [RollingContact]
Default: RollingContact
Timestep: 0.005
ControllerModulePaths: [/home/yuquan/local/mc_rtc_sources/mc_rtc/build/src/mc_control/samples/RollingContact]
RobotModulePaths: [/home/yuquan/local/mc_rtc_sources/mc_rtc/build/src/mc_robots]
RollingContact:
  scenario: keyboard
  closedLoopFeedback: true
  keyboardLinearSpeed: 0.3
  keyboardAngularSpeed: 0.5
  keyboardPollIntervalMs: 10
  keyboardExitKey: x
  rollingRateWeight: 200.0
  steeringRateWeight: 200.0
  steeringPostureWeight: 1.0
  positionFeedbackGain: 5.0
  velocityGain: 20.0
  longitudinal: hard
  terrainNormal: [0.0, 0.0, 1.0]
YAML
```

The empirical keys are deliberately absent: `keyboardYawScale`, `keyboardMixedDriveScale`,
`keyboardWheelPositionLookahead` and `keyboardSteeringRate` no longer exist. Delete them from
`rolling-contact-report/config/mc_rtc-ranger-mini-v3-keyboard.yaml` too, or that file will fail to load once
the controller rejects unknown keys.

- [ ] **Step 5: Commit**

```bash
git add utils/mc_rtc_ticker.cpp rolling-contact-report/config/mc_rtc-ranger-mini-v3-keyboard.yaml
git commit -m "feat(ticker): accept roslaunch-style robot:= and controller:= overrides"
```

---

### Task 8: End-to-end validation and report update

- [ ] **Step 1: Run the complete repository regression**

```sh
CUDA_VISIBLE_DEVICES=-1 MC_RTC_DISABLE_CONVEX_GENERATION_PATCH=ON \
  ctest --test-dir build --output-on-failure -j1
```

Expected: PASS. Record the exact pass/fail counts; do not proceed on a partial pass.

- [ ] **Step 2: Run the acceptance command**

```sh
/home/yuquan/.local/bin/mc_rtc_ticker -s robot:=RollingContactRangerMiniV3
```

Expected: the log line reports the Ranger robot and the RollingContact controller. Drive with W/A/S/D and Q/E
and confirm, in the GUI or the log:

- W/S move the chassis along `+x` / `-x`, A/D along `+y` / `-y`, Q/E yaw in the `+z` / `-z` sense;
- `RollingContact_commandedTwist` and the measured chassis twist agree in sign and to within ~10% in magnitude,
  with no scale factor applied anywhere;
- `RollingContact_*_steeringRateRef` and the measured steering joint velocities agree;
- `maxLateralResidual` stays below 0.05 through every command change, including W -> Q -> A transitions.

- [ ] **Step 3: Run the MuJoCo suite**

```sh
CUDA_VISIBLE_DEVICES=-1 rolling-contact-report/scripts/build-cpu-mujoco-runner.sh
python3 rolling-contact-report/scripts/run-mujoco-suite.py \
  --runner /tmp/rolling-contact-cpu-mujoco/runner-build/rolling_contact_mujoco_runner \
  --build build --artifact-dir /tmp/rolling-contact-mujoco-results \
  --backend both --repetitions 1 \
  --case four-forward --case four-crab --case four-ackermann-left --case four-pure-yaw
```

Expected: every case passes on both the Tasks and TVM backends. `four-pure-yaw` is the case that exercised the
deleted sign mirror, so treat any failure there as a Task 6 regression.

- [ ] **Step 4: Promote the corrected theory**

`rolling-contact-qp.tex` is the stale document; `rolling-contact-qp-new.tex` is the corrected one and is what
this implementation now matches.

```sh
git mv rolling-contact-report/rolling-contact-qp.tex rolling-contact-report/rolling-contact-qp-old.tex
git mv rolling-contact-report/rolling-contact-qp-new.tex rolling-contact-report/rolling-contact-qp.tex
```

Update the `\section{Validations}` list at the end of the promoted file with the tests that now cover it:
`RotatingRateRowsMatchTheAffinePrediction`, `SteeringReferenceInvertsTheExpandedConstraints`,
`FourSteeringTracksCommandedTwistSigns`, and the four MuJoCo cases.

- [ ] **Step 5: Update the report's own documentation**

In `rolling-contact-report/README.md`, replace the four-steering description with the corrected one: eight
optimized rotating velocities, rate references from `steeringWheelReference`, and no empirical scale factors.
In `rolling-contact-report/control-issue.md`, close out the steering issue with a pointer to this plan.

- [ ] **Step 6: Commit**

```bash
git add rolling-contact-report/
git commit -m "docs(rolling): promote the corrected four-steering QP and refresh validation notes"
```

---

## Open questions and stated assumptions

1. **`robot:=` syntax (Task 7).** Assumed to be roslaunch-style shorthand for "use this robot". Implemented as
   a general `key:=value` override in the ticker. If the intent was instead to launch through
   `roslaunch mc_rtc_ticker control_display.launch robot:=...` from `mc_rtc_ros`, Task 7's code change is
   unnecessary and only the `~/.config/mc_rtc/mc_rtc.yaml` step in Step 4 is needed.
2. **Ticker vs MuJoCo.** `mc_rtc_ticker` integrates the robot open-loop with no contact physics, so "drive
   around smoothly" there validates the QP and the reference generation but not the contact behaviour. Task 8
   Step 3 covers the physical validation; both must pass.
3. **`steeringTimeConstant_ = 0.15 s` and `maxSteeringRate_ = 8 rad/s`.** The rate limit is the URDF hinge
   limit (`velocity="8"`). The time constant is the one genuinely free tuning parameter left; start at 0.15 s
   and reduce it only if `FourSteeringCommandChangeKeepsResidualsBounded` shows sluggish reorientation.
4. **`spinSign`.** Derived above to be `+1` for this URDF. Task 6 Step 2 will confirm it empirically; if a
   MuJoCo actuator disagrees, fix the mapping, not the controller.
