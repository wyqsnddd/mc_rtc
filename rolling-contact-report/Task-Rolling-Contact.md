# CPU rolling-contact support for `mc_rtc`

## Assessment of the original task

The original task identifies the right theory, useful reference projects, a toy-robot direction, a controller goal, and
MuJoCo as the final simulator. It is not yet an implementation task that Codex can execute without inventing important
behavior. In particular, it does not define:

- the exact relationship between the report's decision vectors and the variables that the two `mc_rtc` solver backends
  actually expose;
- whether the first milestone is ideal rolling only or the complete fixed/rolling/sliding mode-aware formulation;
- the carrier-center Jacobian, instantaneous contact-force point, local rolling frame, and wheel-rate conventions that
  must be kept distinct;
- how an RSDF surface becomes a dynamically correct wheel contact rather than a fixed material point on the tire;
- the public API, configuration schema, supported solver backends, ownership, or add/remove lifecycle;
- the numerical oracle, intermediate matrix checks, failure behavior, or acceptance tolerances;
- the contents and packaging of the toy robot, sample controller, automated tests, and `mc_mujoco` experiment;
- reproducible commands and evidence required before the task can be called complete; and
- the required execution device.

The execution-device requirement is now frozen: **the final implementation is CPU-only**. `Task-GPU.md` is a reference
for the quality and completeness of the roadmap, its staged gates, and its evidence discipline. It is not a request to
use a GPU. The rolling-contact calculation, QP solve, controller, state updates, tests, and production validation must
run on a conventional Linux CPU. The required path must not depend on CUDA, a GPU, Torch, device tensors, or a
GPU-resident solver.

The current source also rules out several apparent shortcuts:

- `/home/yuquan/local/mc_rtc_sources/ip_constraints` demonstrates constraints over the existing Tasks decision vector.
  It reads `tasks::qp::SolverData`, `alphaDBegin`, and `lambdaBegin`; it does **not** add arbitrary optimization
  variables to Tasks.
- The installed Tasks `SolverData` contains only robot accelerations followed by contact-force multipliers. Joint
  torques are reconstructed by `MotionConstr`; they are not independent Tasks QP variables.
- TVM has explicit robot acceleration, torque, and point-force variables, but the existing
  `mc_solver::{Equality,Inequality,GenInequality}Constraint*` helpers are Tasks-specific. Backend support must therefore
  be designed, not assumed.
- `mc_rbdyn::CylindricalSurface` exists, but `mc_rbdyn::Contact::taskContact` currently accepts only planar and gripper
  surfaces. A cylindrical RSDF alone cannot enter the Tasks contact problem.
- A two-point collinear `PlanarSurface` can represent a line of force generators, but the ordinary
  `ContactConstraint` still describes fixed or statically freed relative motion. A constant six-axis DoF mask cannot
  impose carrier speed equal to wheel rim speed.
- A contact point rigidly attached to a spinning wheel link is one material point on the tire. It rotates away from the
  terrain contact. The force point and friction frame must represent the instantaneous geometric contact without
  rebuilding the whole contact set every control cycle.

The recommended implementation is therefore a staged CPU extension of the existing whole-body QP. The first production
milestone is a dynamically consistent, ideal-rolling differential-drive robot on flat terrain using the Tasks backend.
The implementation then adds ramps, soft longitudinal rolling, contact modes, four steering wheels, and CPU TVM
parity. No visual demonstration substitutes for the algebraic, dynamics, and solver gates below.

## Goal

Add rolling contacts as a first-class, CPU-executed `mc_rtc` capability. A controller must be able to register wheels,
construct the local rolling frame and instantaneous force geometry, impose normal/lateral/longitudinal rolling
relations in the same CPU whole-body QP as dynamics and contact forces, expose slip and feasibility diagnostics, and
safely transition among fixed, rolling, sliding, and detached modes.

The finished work must demonstrate both QPs described in `rolling-contact-qp.tex`:

1. the differential-drive specialization in `\eqref{eq:differential-drive-qp}`; and
2. the four-steering-wheel specialization in `\eqref{eq:four-steering-wheel-qp}`.

Both are implementations of the common whole-body model in `\eqref{eq:ideal-rolling-whole-body-qp}` and, after the
ideal milestones pass, the mode-aware extension in `\eqref{eq:rolling-whole-body-qp}`. They must not become separate
kinematic solvers that bypass floating-base dynamics, friction, torque limits, or the controller's state estimator.

## CPU execution contract

The following requirements are non-negotiable:

- All production C++ code runs on CPU through the existing Eigen/RBDyn/Tasks/TVM and selected CPU QP-solver stack.
- The build and runtime have no required CUDA toolkit, GPU driver, Torch, GPU array, or device-runtime dependency.
- Unit, integration, ticker, and `mc_mujoco` tests must pass on a CPU-only machine or in an environment where GPUs are
  hidden. A GPU may be physically installed in the workstation, but the tests must not discover or use it.
- The public API contains ordinary host-side C++ objects and Eigen values. It must not expose device pointers or make
  execution conditional on GPU availability.
- Performance is measured with wall-clock and CPU-time statistics after warm-up. Record CPU model, core count,
  frequency policy, compiler, build type, thread count, and solver. Report both single-thread results and the exact
  threading configuration used by the final controller.
- The control-cycle gate is based on the configured CPU control period. Passing because work is asynchronous on another
  device is not allowed.
- CPU numerical behavior is the authoritative result. There is no GPU oracle, GPU migration, or GPU fallback in this
  task.

Add an automated dependency check that rejects accidental required linkage to CUDA, cuBLAS, cuSolver, Torch, or another
GPU runtime in the rolling-contact targets. Optional unrelated features elsewhere in a developer's mc_rtc build do not
invalidate the project, but rolling-contact targets and test commands must work with those optional features disabled.

## Final implementation checkpoint

Date: 2026-09-01 (Asia/Shanghai)

Status: **Complete — all roadmap stages pass on CPU**

The implementation described by this roadmap is present in the commit containing this document. The authoritative
human-readable checkpoint is [`evidence/final-report.md`](evidence/final-report.md), the machine-readable checkpoint is
[`evidence/final-report.json`](evidence/final-report.json), and all 50 Tasks/TVM physical scenario summaries are in
[`evidence/mujoco-summary.json`](evidence/mujoco-summary.json). The report also retains the 10,000-cycle timing data,
dynamic CPU linkage proof, and a reproducible validation plot.

| Stage | Final status | Evidence |
| --- | --- | --- |
| Roadmap/source audit | Pass | This roadmap, `architecture.md`, and the normative QP report |
| 0 — baseline and architecture | Pass | ADR-001, Stage 0 record, final full regression |
| 1 — toy robots | Pass | Differential/four-steering loaders and `testRollingContactRobot` |
| 2 — standalone rolling math | Pass | `testRollingContact`, finite differences, zero-allocation hot loop |
| 3 — minimal solver insertion | Pass | Tasks/TVM hard and soft rows in `testRollingContactSolver` |
| 4 — dynamic contact forces | Pass | Instantaneous point/cone, virtual-work, torque, and layout tests |
| 5 — differential controller | Pass | Headless controller, lifecycle, invalid-input, and mode CTests |
| 6 — differential CPU `mc_mujoco` | Pass | Flat, ramp, slip, disturbance, contact-loss, replay scenarios |
| 7 — contact modes | Pass | Fixed/rolling/sliding/detached transition and recovery corpus |
| 8 — four-steering QP | Pass | Straight, reverse, crab, Ackermann, yaw, steering-rate, ramp corpus |
| 9 — CPU backend parity | Pass | Direct coefficient/solution parity and full Tasks/TVM scenario matrix |
| 10 — regression and handoff | Pass | 102/102 CTest, ASan, schemas, docs, CPU timing/linkage, clean staged checkout |

Key final results are 13/13 rolling CTests, 102/102 complete CTests, 4/4 selected ASan tests, 250 deterministic
MuJoCo processes with zero missed deadlines, and zero heap allocations in 10,000 post-warm-up geometry updates. The
worst physical-suite P99 controller time was 0.346865 ms against a 5 ms period. The linkage audit found no CUDA,
cuBLAS, cuSolver, Torch, HIP, or OpenCL runtime in any of the five rolling targets. The implementation uses only host
C++/Eigen and the existing Tasks/TVM CPU solvers; no GPU path or fallback exists.

The following section is retained as a historical record of the source state when the roadmap was authored. It must
not be read as the current implementation status.

## Historical status at the start of this roadmap

This document is an implementation specification and gate list, not a claim that rolling contacts are implemented.
The following source audit is complete as of 2026-09-01:

| Area | Existing mechanism | Consequence for rolling contacts |
| --- | --- | --- |
| Tasks variables | `tasks::qp::SolverData` lays out `alphaD` then contact `lambda` | Wheel accelerations already belong to `alphaD`; do not add duplicate wheel-acceleration variables |
| Tasks torques | `mc_solver::DynamicsConstraint`/`tasks::qp::MotionConstr` reconstructs torque and applies bounds | The report's explicit `tau` block maps to an eliminated variable in the Tasks backend |
| TVM variables | `mc_tvm::Robot` and `DynamicFunction` use acceleration, torque, and per-point force variables | The report maps more literally, but a TVM rolling function and backend adapter are still required |
| Custom constraints | `mc_solver::EqualityConstraintRobot` and the `ip_constraints` package fill rows in existing variables | Useful pattern for the first Tasks prototype, not an extra-variable facility |
| Ordinary geometry | `mc_solver::ContactConstraint` fixes or frees selected components of relative frame motion | A DoF mask can retain selected rows but cannot express `v_carrier = r * phi_dot` |
| Contact forces | Tasks uses friction-cone multipliers; TVM uses point-force variables | Rolling must reuse the same force solution as dynamics rather than creating a second traction calculation |
| Surfaces | Planar, cylindrical, and gripper RSDF types exist | The current Tasks conversion rejects cylindrical surfaces; a line model needs an explicit supported path |
| Contact points | Existing force points are fixed in a body frame after contact construction | A wheel needs a state-dependent instantaneous geometric point and rolling-frame cone |
| Controller lifecycle | `MCController::addContact`, `removeContact`, `setContacts`, logger, GUI, and loaders already exist | Rolling additions must have equivalent ownership, update, logging, and removal behavior |
| Tests | Boost.Test unit tests and ticker-based controller integration tests are available | Add intermediate math, solver, lifecycle, and controller tests before simulation |
| Simulator | `mc_mujoco` is required but was not found under `/home/yuquan/local` during this audit | Installation/version discovery is a preflight gate; do not fabricate successful simulator results |

The current branch and dependency revisions must be re-audited when implementation starts. If source behavior differs
from this table, update the table and the relevant gate before writing feature code.

### Initial roadmap status

Only the source/task audit used to write this roadmap has been performed. No implementation or validation stage should
be inferred from the existence of the theory PDF or this document.

| Stage | Initial status | Evidence or next action |
| --- | --- | --- |
| Roadmap/source audit | Pass | `rolling-contact-qp.tex`, current mc_rtc sources, installed Tasks headers, and the three local reference projects were inspected |
| 0 — baseline and architecture | In progress | CPU/build/dependency evidence and ADR-001 are recorded; toy variable inventory, `mc_mujoco`, clean full-suite baseline, and dynamic coefficient probe remain |
| 1 — toy robots | Not started | Create tracked differential and four-steering descriptions |
| 2 — standalone rolling math | Not started | Implement the CPU geometry/Jacobian oracle and finite-difference tests |
| 3 — minimal solver insertion | Not started | Prove hard/soft rows in a minimal Tasks CPU QP |
| 4 — dynamic contact forces | Not started | Prove instantaneous force geometry, virtual work, and wheel torque |
| 5 — differential controller | Not started | Add deterministic controller scenarios and CPU deadline tests |
| 6 — differential `mc_mujoco` | Blocked | Locate/install `mc_mujoco` after the analytical/controller gates pass |
| 7 — contact modes | Not started | Implement and test fixed/rolling/sliding/detached transitions |
| 8 — four-steering QP | Not started | Extend the common model and repeat controller/simulator gates |
| 9 — CPU backend parity | Not started | Complete and compare Tasks and TVM implementations |
| 10 — regression and handoff | Not started | Run full regression, CPU performance, documentation, and clean-checkout gates |

## Required outcome

The task is complete only when all of the following artifacts exist and their gates pass:

1. **Core model and API**
   - A backend-neutral rolling-contact description containing wheel/carrier/contact geometry, wheel and optional
     steering joints, radius, width or line endpoints, terrain frame, friction, requested mode, and activation.
   - A deterministic rolling-geometry calculation that returns the orthonormal local frame, carrier and force points,
     Jacobian rows, normal-acceleration terms, velocity residual, acceleration-level right-hand side, and diagnostics.
   - Solver integration for hard normal and lateral rows, hard or soft longitudinal rolling, contact forces, dynamics,
     joint/torque bounds, and contact add/update/remove lifecycle.
   - Fixed, rolling, sliding, and detached modes with explicit transition and failure semantics.

2. **CPU solver backends**
   - A complete Tasks implementation using its native reduced decision vector (`alphaD`, `lambda`) without pretending
     that torque or slack is an existing independent variable.
   - TVM behavior that is either fully equivalent through the gates or fails immediately with a clear unsupported
     error until the TVM stage is complete. A silent no-op is never acceptable.
   - Before final completion, Tasks and TVM must both pass the common ideal-rolling and lifecycle corpus on CPU.

3. **Robot descriptions**
   - A tracked, installable differential-drive toy robot with a floating chassis, two continuous drive joints,
     collision/inertial data, force/contact descriptions, actuator limits, and a MuJoCo-compatible model.
   - A tracked four-steering-wheel variant or configuration with four drive and four steering joints.
   - Machine-checked joint, body, frame, surface, radius, width, axis, sign, initial pose, and actuator mappings.

4. **Controller examples**
   - A full `mc_rtc` controller, structured like the local `BipedalLocomotionController` and repository samples, that
     owns the constraints/tasks, validates configuration, logs all diagnostics, and follows straight,
     constant-curvature, and sinusoidal paths with the differential robot.
   - Four-steering-wheel examples for straight, crab, and turning motion.
   - Safe behavior for infeasible commands, solver failure, invalid geometry, and contact-mode transitions.

5. **Automated validation**
   - CPU unit tests for every equation block used from `rolling-contact-qp.tex`.
   - Matrix and residual tests around the solver boundary, including independent finite-difference checks.
   - Controller smoke and trajectory tests that run without a GUI.
   - Closed-loop `mc_mujoco` experiments on flat terrain and a ramp, including low-friction/slip cases.
   - CPU-only dependency, timing, and allocation reports.
   - Machine-readable reports and retained logs that support every measured claim.

6. **Documentation and handoff**
   - Public API/configuration documentation, a tutorial, exact build/run/test commands, a limitations section, and an
     evidence log.
   - No required source, configuration, robot asset, or test may remain only in an ignored `controllers/` or `robots/`
     development directory or in an absolute user-specific path.

## Scope and non-goals

The required final scope is conventional rigid wheels for differential and four-steering chassis on a plane or a
constant-slope ramp. It includes ideal rolling, soft longitudinal rolling, fixed/rolling/sliding/detached modes, a
polyhedral friction model, drive/steering torque limits, and simulator validation.

The following are not required for the first complete delivery:

- GPU execution, CUDA kernels, Torch models, device tensors, or a CPU-to-GPU migration;
- deformable tires, pneumatic tire models, rolling resistance, or finite tire-patch pressure distributions;
- omni-wheels, mecanum wheels, passive casters, tracks, or spherical wheels;
- complementarity, mixed-integer contact-mode selection, or a nonlinear optimizer;
- online terrain reconstruction beyond a supplied plane pose/normal;
- a learned slip detector or learned controller;
- hardware deployment;
- a general-purpose arbitrary-variable extension to the Tasks library;
- merging the toy robot into the installed set of production robot modules if a tracked companion example package is
  cleaner.

These may be added later only as separately named variants. They must not weaken or replace the rigid-wheel CPU oracle.

## Execution rules for Codex

1. Read all of `rolling-contact-qp.tex`, not only the three top-level QP equations, before implementing math. The local
   frame, carrier/material-point distinction, acceleration-level stabilization, dynamic force model, planar
   specializations, mode logic, and validation table are normative together.
2. Inspect the exact installed Tasks, TVM, RBDyn, mc_rtc, CPU QP solver, and `mc_mujoco` revisions at implementation
   time. Do not infer an API from a different branch or from memory.
3. Preserve unrelated user changes. The roadmap does not authorize cleaning ignored/untracked files or modifying
   `/home/yuquan/local/mc_rtc_sources/ip_constraints`, `jvrc_description`, or `BipedalLocomotionController`.
4. Use those external trees as read-only references. Any required source must be implemented in the selected rolling
   contact package/repository and recorded by a commit.
5. Do not pass a stage from animation, GUI output, or a solver success boolean alone. Save the intermediate quantities
   and assert the algebraic residuals required by the gate.
6. Do not tune around an infeasible model by enlarging torque limits, friction, wheel radius, solver tolerances, or task
   weights without recording the change and rerunning all earlier gates.
7. Do not rebuild the full contact set every control cycle as the production solution. That changes variable layout,
   destroys warm starts, and can hide contact-lifecycle bugs. A rebuild may be used only in an explicitly labeled spike
   that is discarded before the production gate.
8. Do not impose both the generic Jacobian rolling rows and the equivalent explicit chassis rows. Tests must detect
   duplicate or rank-redundant insertion.
9. Do not claim physical rolling if the QP force is applied to a carrier center without the equivalent wheel-axis
   moment. The virtual-work and wheel-torque gates are mandatory.
10. Do not add a GPU dependency, optional GPU fast path, or GPU-only test while satisfying this task. Optimize the CPU
    implementation through preallocation, sparsity, warm starts, and measured solver choices.
11. Every report must distinguish implemented behavior, measured evidence, known failures, and proposed future work.
    Never mark a checkbox complete merely because code was written.

## Mathematical contract to freeze

### State and frame conventions

- `q`, `alpha`, and `alphaD` use RBDyn/mc_rtc ordering. The floating joint, if present, comes first; all drive and
  steering joints are named and resolved, never assumed to be the last entries.
- The wheel spin rate is part of generalized velocity. In the preferred homogeneous form it is selected by `S_phi_i`;
  it is not also supplied as an unrelated command variable.
- `c_i` is the rolling center on the carrier/axle. `p_i = c_i - r_i n_i` is the instantaneous geometric contact point.
  Neither is an arbitrary fixed vertex on the rendered wheel mesh.
- `J_c_i` is the translational Jacobian of the carrier center under the convention in Remark
  `\ref{re:rolling-jacobian-convention}`. It must not include a second rim-speed contribution.
- The inertial-frame local contact basis is `[t_i, l_i, n_i]`, where `l_i = n_i x t_i`. Every cycle must verify unit
  length, mutual orthogonality, right-handedness, and finite values.
- Positive wheel spin, positive rolling direction, steering-angle sign, force sign, torque sign, and SpaceVecAlg
  angular/linear ordering must be demonstrated by named unit tests.

### Velocity and acceleration rows

For each ideal rolling wheel, freeze the homogeneous relation from `\eqref{eq:homogeneous-rolling-block}`:

```text
A_i(q) * alpha = 0

A_i = [ t_i^T J_c_i - r_i S_phi_i
        l_i^T J_c_i
        n_i^T J_c_i ]
```

and the stabilized acceleration-level relation from `\eqref{eq:homogeneous-rolling-acceleration}`:

```text
A_i(q) * alphaD = -A_dot_i(q, alpha) * alpha - Kp * A_i(q) * alpha.
```

The implementation may compute the bias through analytic RBDyn normal-acceleration quantities instead of explicitly
forming all of `A_dot`, but the resulting right-hand side must match an independent finite difference. For steering
wheels, direction derivatives must include `delta_dot`; for ramps, all directions must be expressed in the supplied
terrain tangent basis.

Mode rows are:

| Mode | Hard kinematic rows | Soft/objective rows | Force behavior |
| --- | --- | --- | --- |
| Fixed/sticking | zero relative angular/linear motion selected by the configured fixed-contact contract | optional transition smoothing | admissible contact cone |
| Rolling | normal and lateral | longitudinal is hard for certified ideal cases, otherwise a weighted residual | admissible contact cone in rolling frame |
| Sliding | normal only | no no-slip objective that conflicts with measured sliding | cone plus frozen one-cycle opposing friction direction when enabled |
| Detached | none | none | contact force exactly zero/variables removed according to backend contract |

The direct quadratic longitudinal residual in `\eqref{eq:rolling-task}` is the required Tasks soft formulation. It is
equivalent to eliminating a free slack with a quadratic penalty for this row and avoids inventing an unavailable Tasks
variable. If an explicit slack is later needed for bounds or logging, add it through a separately reviewed solver
extension and prove equivalence on the unconstrained-slack corpus.

### Dynamics and force contract

- Rolling contacts participate in the same equation of motion and force variables as all other contacts. There is no
  post-QP force or torque patch.
- The Tasks implementation maps the report's explicit dynamics/torque equations to `alphaD` and `lambda`; torque is
  reconstructed by `MotionConstr` and checked against the robot's limits.
- The CPU TVM implementation may retain explicit torque and point-force variables but must produce equivalent
  generalized accelerations, contact wrenches, and torques for the common corpus.
- A line contact uses an explicitly documented generator model. The initial conventional wheel model is two endpoints
  across the wheel width with a common rolling-frame friction basis. Record the exact number and ordering of generators.
- The instantaneous force point and cone frame are QP data evaluated at the measured state and frozen during one solve.
  Their generalized-force map must include the moment `(-r_i n_i) x f_i` about the wheel center so the wheel row
  recovers the expected drive-torque/traction relationship.
- The quasi-static relation `|f_t| <= |tau_phi| / r` is a diagnostic consequence, not an extra constraint when the
  complete dynamics row and torque bound are already present.
- A negative normal force, non-finite generator, invalid friction coefficient, zero/negative radius, missing joint,
  duplicate wheel identifier, or degenerate line must be rejected deterministically.

### Differential and four-steering specializations

For the differential robot, the velocity oracle is `\eqref{eq:differential-geometric-constraints}` and its closed-form
diagnostic is `\eqref{eq:differential-linear-velocity}`--`\eqref{eq:differential-angular-velocity}`. Insert only three
independent planar rows: left longitudinal, right longitudinal, and one lateral row. The second identical lateral row
must be detected as redundant and omitted.

For four steering wheels, use `\eqref{eq:steering-expanded-constraints}` and include the known steering-rate terms from
`\eqref{eq:steering-direction-derivatives}` in the acceleration bias. All wheels map to one chassis twist; never solve
four independent chassis velocities and average them.

## Proposed public contract

Exact C++ names may change after the Stage 0 API review, but the information and ownership must remain explicit. A
representative configuration is:

```yaml
type: rollingContact
robot: rolling_contact_toy
terrainFrame: ground
mode: rolling
longitudinal: soft       # hard or soft
kp: 20.0
rollingWeight: 1000.0
transitionTime: 0.2
wheels:
  - name: left
    carrierFrame: left_wheel_center
    forceBody: left_wheel
    driveJoint: left_wheel_joint
    radius: 0.15
    width: 0.06
    friction: 0.7
    spinSign: 1.0
  - name: right
    carrierFrame: right_wheel_center
    forceBody: right_wheel
    driveJoint: right_wheel_joint
    radius: 0.15
    width: 0.06
    friction: 0.7
    spinSign: 1.0
```

A steering wheel additionally names its steering joint and zero-angle rolling direction. Terrain data may be updated at
runtime, but configuration parsing and joint/frame resolution happen once.

Per-wheel diagnostics must be available through typed C++ accessors, logger entries, and a compact GUI panel:

```text
requested_mode, estimated_mode, activation
rolling_direction, lateral_direction, normal_direction
carrier_position, instantaneous_contact_position
rolling_velocity_residual, lateral_velocity_residual, normal_velocity_residual
rolling_acceleration_residual, qp_longitudinal_slack_or_task_residual
normal_force, tangential_force, friction_margin
drive_torque, drive_torque_margin
solver_valid, invalid_reason, transition_dwell_time
```

Diagnostics must be cleared or marked invalid on failure. They must never retain values from the last successful solve
while claiming to describe the current cycle.

## Repository architecture and proposed deliverables

Before implementation, Codex must turn this table into a short architecture decision record with final names and
backend ownership. The paths below are the expected starting point, not permission to scatter code arbitrarily.

| Concern | Existing references | Proposed deliverable |
| --- | --- | --- |
| Geometry/data | `mc_rbdyn::Contact`, `Surface`, `RobotFrame`, RBDyn Jacobians | `mc_rbdyn` rolling description and a small backend-neutral geometry result type |
| Tasks hard rows | `EqualityConstraintRobot`, `utils::UpdateRobot`, `ip_constraints` | solver implementation filling complete Tasks rows in `alphaD`/`lambda` layout |
| Tasks soft row | existing MetaTask/Tasks task wrappers | rolling MetaTask or internal Tasks task over the longitudinal residual |
| TVM hard/soft rows | `mc_tvm::ContactFunction`, `TVMQPSolver::problem()` | CPU rolling TVM function plus constraint/task requirements |
| Dynamic force point | Tasks bilateral/unilateral contact and `mc_tvm::DynamicFunction::ForceContact` | rolling contact-force adapter with updateable point/frame and stable variable identity |
| Configuration | `ConstraintSetLoader`, `MetaTaskLoader`, JSON schemas | loader entries, schema, examples, invalid-config tests |
| Controller | `src/mc_control/samples`, local `BipedalLocomotionController` | tracked sample or companion package with normal controller ownership/lifecycle |
| Robot assets | local `jvrc_description` URDF/RSDF pattern | tracked toy URDF, RSDF/config, primitives, and MuJoCo model/conversion |
| Unit tests | `tests/testConstraintImplementation.cpp`, `testSolverBackend.cpp` | geometry, matrix, dynamics, lifecycle, backend-parity tests |
| Integration tests | `tests/controllers` ticker harness | differential and steering controller tests with machine-checked logs |
| Documentation | `doc/_i18n/en/tutorials` and schemas | rolling-contact tutorial, API docs, configuration reference, evidence report |

Do not place required assets only in top-level `robots/` or `controllers/`: both directories intentionally ignore
external projects. Either add tracked tests/samples to existing source/test trees or create a tracked companion example
package and document how it is built from a clean checkout.

## Workplan and acceptance gates

Do not begin a stage's production implementation until the previous stage's exit gate passes. A bounded exploratory
spike may inform a decision, but spike code and results must be labeled and cannot satisfy a production gate.

### Stage 0 — Freeze baseline, scope, CPU environment, and architecture

1. Record the mc_rtc commit, dirty status, CPU model/topology, compiler, CMake, Eigen, RBDyn, Tasks, TVM, SpaceVecAlg,
   CPU solver, ROS 2, MuJoCo, and `mc_mujoco` revisions.
2. Run the existing CPU build and tests before adding rolling code.
3. Run the same build with GPUs hidden and optional CUDA/GPU packages disabled or absent.
4. Read and map the following source paths:
   - matching declarations and implementations under `include/mc_solver/` and `src/mc_solver/` for `QPSolver`,
     `TasksQPSolver`, `TVMQPSolver`, `ContactConstraint`, and `DynamicsConstraint`;
   - matching declarations and implementations under `include/mc_rbdyn/` and `src/mc_rbdyn/` for `Contact`, `Surface`,
     `PlanarSurface`, and `CylindricalSurface`;
   - matching declarations and implementations under `include/mc_tvm/` and `src/mc_tvm/` for `ContactFunction` and
     `DynamicFunction`;
   - the `MCController` contact lifecycle in `include/mc_control/MCController.h` and `src/mc_control/MCController.cpp`;
   - the installed Tasks `QPSolverData`, contact, motion, task, and constraint headers;
   - `/home/yuquan/local/mc_rtc_sources/ip_constraints` as a read-only constraint-layout example.
5. Write an architecture decision that answers:
   - where rolling geometry lives;
   - how Tasks and TVM share math but own solver objects independently;
   - how the instantaneous force point/cone updates without changing decision layout;
   - how the standard contact's geometric rows are disabled or partitioned to avoid duplicates;
   - how soft rolling is represented in Tasks;
   - how mode changes preserve ownership and warm starts;
   - where tracked robot/controller assets live; and
   - which CPU solver and thread settings are authoritative.
6. Update the progress table in this document with `Not started`, `In progress`, `Pass`, `Fail`, or `Blocked` for every
   stage. Do not pre-mark future work complete.

Baseline commands:

```sh
cmake -S . -B build-rolling -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DBUILD_TESTING=ON -DENABLE_FAST_TESTS=ON
cmake --build build-rolling -j"$(nproc)"
ctest --test-dir build-rolling --output-on-failure
command -v mc_mujoco
```

The final roadmap update must add the actual project options used to disable optional GPU dependencies. Do not invent
an option name before confirming it in the active CMake tree.

**Validation 0A — baseline preservation:** store command lines, exit codes, test counts, and any pre-existing failures.

**Validation 0B — decision-vector proof:** add a tiny diagnostic test that records Tasks `nrVars`, every
`alphaDBegin`, `lambdaBegin`, and contact generator count for the future toy topology. Record the corresponding TVM
variables. The architecture document must map every symbol in the report's ideal decision vector to each backend.

**Validation 0C — CPU-only proof:** inspect linked libraries for every rolling target, run the tests with GPU devices
hidden, and fail if the rolling feature requires a CUDA/Torch/GPU runtime. Record process thread count and CPU affinity
for performance runs.

**Exit:** the baseline is reproducible on CPU; all dependencies are identified; the backend and force-point design is
approved by executable probes; and no unresolved item can change the toy topology or solver formulation.

### Stage 1 — Create and validate the toy robots

Build the differential robot first. Its minimum topology is a floating chassis, left and right wheel links, continuous
revolute drive joints with parallel axle directions, nonzero inertias, collision geometry, realistic effort/velocity
limits, named carrier-center frames, wheel radius/width metadata, and a ground/environment model. The initial state must
place the geometric contact line on the ground without penetration.

The description must keep these concepts separate:

- wheel visual/collision cylinder;
- carrier-center frame used by rolling kinematics;
- wheel body used by the force Jacobian;
- instantaneous force line computed from the wheel center and terrain normal;
- an RSDF/configuration object used to register the rolling contact.

Do not encode the bottom of the tire as one forever-fixed point in the wheel link. If a two-point planar surface is
used to bootstrap force generators, document that it is a line approximation and prove how its state-dependent offset
and frame are updated.

Then add the four-steering variant with four drive joints and four steering joints. Its zero steering angles, joint
axes, corner offsets, joint limits, drive signs, and reference pose must be explicit.

**Validation 1A — load/topology:** load each robot through `RobotLoader`; assert exact joint/body/frame/surface names,
joint types and axes, DoF order, mass/inertia validity, limits, initial pose, wheel radii, widths, and no duplicate names.

**Validation 1B — geometry:** at zero and seeded wheel/steering angles, assert that each carrier center, contact point,
line endpoint, and local basis matches an independent calculation. Spinning a circular wheel must not move the
geometric contact location for an unchanged chassis and terrain plane.

**Validation 1C — packaging:** configure and load both robots from a clean build without any absolute path, ignored
source, installed user asset, or dependency on `jvrc_description`.

**Exit:** both robot variants load deterministically and the geometric-contact invariance test passes before any QP is
allowed to use their contact data.

### Stage 2 — Implement the standalone CPU rolling-math oracle

Implement a solver-independent C++/Eigen kernel that receives a resolved wheel description, robot state, and terrain
plane, and returns all rolling rows and diagnostics. Keep allocations out of its per-cycle update after initialization.
The kernel must be ordinary CPU C++ and deterministic for a fixed input and floating-point environment.

Test in this order:

1. local basis normalization and sign;
2. `p_i = c_i - r_i n_i` and line endpoints;
3. carrier translational Jacobian and wheel selector;
4. `A_i * alpha` against direct carrier/rim relative velocity;
5. analytic or RBDyn-derived acceleration bias;
6. differential geometric matrix and its three independent rows;
7. four-steering matrix including `delta_dot` terms;
8. slip velocity from `\eqref{eq:slip-velocity}`;
9. flat-to-ramp coordinate equivariance;
10. invalid-input containment and recovery on the next valid update.

**Validation 2A — finite differences:** perturb every floating-base coordinate and every drive/steering joint relevant
to each wheel. Compare contact/carrier position and rotation-log differences with independently constructed Jacobian
columns. Use central differences and report the worst wheel, state, row, column, absolute error, and relative error.

**Validation 2B — `A_dot * alpha`:** compare the analytic/normal-acceleration right-hand side with a central difference
of `A(q) * alpha` along the state velocity. Include nonzero base twist, drive rates, steering rates, and a ramp.

**Validation 2C — planar identities:** for at least 100 seeded states, the differential rows must recover the closed-form
linear/yaw velocities from the report. Four-steering compatible commands must recover one common twist; an intentionally
incompatible command must produce a nonzero least-squares residual before solver softening.

**Validation 2D — equivariance:** rotate and translate the complete robot/terrain setup. Scalar rolling/lateral/normal
residuals must remain invariant after mapping both results to the same basis.

**Validation 2E — CPU determinism/allocation:** replay the corpus in fresh processes and compare the retained matrices
and diagnostics field-for-field. After warm-up, an allocation-instrumented update loop must perform no heap allocation.

**Exit:** the CPU kernel passes all deterministic and property tests without loading a QP solver. Matrix snapshots
become the oracle for backend integration.

### Stage 3 — Prove CPU solver insertion on a minimal problem

Implement the smallest possible Tasks constraint/task adapters around the Stage 2 kernel. Do not start with the full
controller.

For Tasks:

- fill full-width equality rows using `SolverData::alphaDBegin(robotIndex)`;
- leave all unrelated robot and `lambda` columns exactly zero;
- use the existing `EqualityConstraintRobot`/`UpdateRobot` pattern where it is sufficient;
- implement the soft longitudinal residual as a quadratic task over the same acceleration row;
- call `updateNrVars` after contact-layout changes and verify every cached index;
- never hard-code floating-base size, wheel indices, lambda offsets, or generator counts.

For TVM, create only an architecture spike in this stage: demonstrate a CPU function with the same value, velocity,
normal-acceleration, and Jacobian convention and show how it enters `LinearizedControlProblem`. Production TVM work is
gated later.

**Validation 3A — matrix placement:** compare every active and structural-zero coefficient against the Stage 2 oracle.
Deliberately reorder robots, joints, wheels, and contacts; name-based resolution must produce the same logical matrix.

**Validation 3B — hard row:** solve feasible straight, in-place turn, and constant-curvature acceleration cases. Require
the declared equality residual and independently reconstructed residual to pass tolerance.

**Validation 3C — soft row:** create a conflict with a fixed contact. The fixed row must remain satisfied, the solver
must remain feasible, and the rolling task residual must equal the independently computed best feasible residual.
Increasing rolling weight must reduce the residual monotonically until it reaches the higher-priority feasibility
limit; it must not violate hard rows.

**Validation 3D — lifecycle:** add, update, remove, and re-add the rolling object around multiple solver runs. Assert
stable object ownership, no duplicate rows/tasks, no stale indices, no use-after-free, and deterministic recovery.

**Validation 3E — CPU-only:** repeat the minimal solver corpus with GPUs hidden and inspect target linkage. Results and
timings must come from the CPU solver selected at Stage 0.

**Exit:** a headless CPU QP proves exact row insertion, hard/soft behavior, reordering safety, and lifecycle before
contact-force dynamics are introduced.

### Stage 4 — Implement dynamically correct rolling contact forces

Connect the rolling kinematics to the same contact-force variables used by the robot dynamics. This is the stage most
likely to require a deliberate extension below the public `mc_rtc` wrappers; treat it as core functionality, not a
controller-local patch.

Required behavior:

1. create the configured two-endpoint line-force representation with a stable variable/generator ordering;
2. update the instantaneous point offset and local friction frame from the current wheel/terrain geometry;
3. keep the decision-variable identity and dimensions stable during ordinary wheel rotation and steering;
4. apply point forces through the correct current point Jacobians so the generalized wheel row contains traction
   torque;
5. expose resultant force/wrench and per-endpoint forces in one declared frame;
6. apply nonnegative generator/friction constraints and normal-force diagnostics;
7. combine with existing dynamics and torque bounds, not a duplicated local wheel equation;
8. support contact removal/detachment without leaving force variables active.

If the installed Tasks library cannot update the required point/cone data safely, choose and document one of these
production solutions:

- add an upstreamable rolling-contact/contact-geometry extension to Tasks and consume it through `mc_rtc`;
- represent the necessary generalized-force columns with existing stable lambda variables through a reviewed custom
  dynamics constraint; or
- revise the primary CPU backend decision with evidence and move the production milestone to TVM.

Do not modify installed headers/libraries in place and then rely on the unrecorded result.

**Validation 4A — virtual work:** for seeded states and arbitrary admissible endpoint forces, compare `J_p^T f` against
finite-difference virtual work for every generalized coordinate. Report the worst row and sample.

**Validation 4B — wheel torque:** on an isolated wheel and on the full differential robot, verify the drive-joint
generalized-force contribution and the quasi-static `tau/r` diagnostic, including both signs and unequal radii.

**Validation 4C — cone frame:** rotate wheel steering and ramp normal independently. Every generator must remain inside
the declared rolling-frame cone; tangential/normal component reconstruction must match direct dot products.

**Validation 4D — dynamics:** compare the assembled equation-of-motion residual before and after solving. Assert torque
bounds from the robot model, normal-force sign, friction margin, and reconstructed contact wrench.

**Validation 4E — no rebuild:** instrument contact/variable creation. After warm-up, ordinary wheel spin, chassis
motion, steering, and terrain-frame updates must not recreate solver variables or change row/column counts.

**Exit:** kinematic rolling, force feasibility, generalized dynamics, and drive torque are one algebraically consistent
CPU QP with stable layout.

### Stage 5 — Complete the differential-drive controller

Create a tracked CPU controller that owns the rolling constraint/task, dynamics constraint, posture/base tasks,
trajectory generator, mode manager, logger, and GUI. Follow normal `MCController`/FSM ownership patterns from the local
`BipedalLocomotionController` and in-tree samples; do not put solver objects in static globals.

The controller must provide deterministic, noninteractive configurations for:

- zero command/hold;
- straight forward and reverse motion;
- in-place left and right turns;
- constant-radius left and right circles;
- a sinusoidal or S-shaped planar path;
- unequal left/right wheel radii as a calibration test;
- infeasible wheel-rate command with soft rolling;
- stop, reset, and controller reload.

At each cycle log reference and measured base pose/twist, wheel position/rate/torque, every rolling residual, force and
friction margin, mode/activation, solver status, solve time, and CPU update time. The log checker must fail on missing
fields, non-finite values, timestamp discontinuity, or stale diagnostics.

**Validation 5A — ticker smoke:** run the controller headlessly for at least 1,000 control cycles in open-loop ticker
mode. Require successful solver status, bounded state, finite logs, and clean shutdown/reload.

```sh
build-rolling/utils/mc_rtc_ticker \
  -f rolling-contact-report/config/mc_rtc-differential.yaml \
  --run-for 5 --no-sync
```

The final path may differ, but the checked-in configuration and exact command must exist.

**Validation 5B — kinematic trajectory:** reconstruct the planar pose independently from wheel motion and compare it
with the controller state for every deterministic path. Report RMS, P95, and maximum position, yaw, velocity, rolling,
and lateral residual errors.

**Validation 5C — failure containment:** inject invalid radius, missing joint, NaN terrain normal, infeasible hard
command, solver failure, and detached contact. Configuration errors must fail before running; runtime failures must
produce the documented safe command and invalid diagnostics without stale output.

**Validation 5D — CPU deadline:** over at least 10,000 post-warm-up no-sync cycles, report median/P95/P99/max rolling
update, QP build, QP solve, and total controller-cycle time. The P99 total must remain below the configured controller
period with the frozen thread settings, or this stage fails.

**Exit:** every differential scenario has an automated CPU test and machine-readable report. Animation or a
hand-operated GUI is optional evidence only.

### Stage 6 — Validate differential rolling in CPU `mc_mujoco`

Install or locate `mc_mujoco` from its authoritative repository, record its exact revision and MuJoCo version, and add
the toy model through its supported robot-description path. Run both `mc_rtc` and MuJoCo on CPU. Do not weaken wheel
friction/contact settings merely to match the QP; document the mapping between QP and simulator parameters.

The simulator adapter must map:

- free-base pose and twist conventions;
- drive joint names, order, signs, positions, velocities, and torques;
- wheel collision radius/width and ground contact parameters;
- actuator effort/velocity limits;
- contact forces or sensors used for diagnostics;
- simulation timestep versus controller timestep;
- terrain pose for flat and constant-slope cases.

Required closed-loop scenarios are straight, reverse, in-place turn, constant-radius circle, sinusoidal path, ramp
ascent/descent, low friction, a lateral impulse, and loss/recovery of one wheel contact where physically meaningful.

**Validation 6A — static parity:** before stepping, compare model topology, initial transforms, masses/inertias, limits,
wheel geometry, terrain plane, and joint ordering between RBDyn and MuJoCo.

**Validation 6B — no-slip trajectories:** on nominal friction, compute tracking and slip from simulator state rather
than controller predictions. Require finite bounded behavior and report pose/twist tracking, wheel-rate consistency,
longitudinal/lateral/normal slip, normal force, torque margin, and friction margin.

**Validation 6C — ramp:** express residuals in the ramp tangent frame. Gravity must change force/torque margins while
the geometric velocity identities remain valid in that basis.

**Validation 6D — low friction and disturbance:** the simulator must produce measurable slip when traction is
insufficient. The controller must not report certified ideal rolling indefinitely; hysteresis and mode/fallback behavior
must match the documented thresholds.

**Validation 6E — replay:** repeat every scenario at least five times from a reset state. Configuration, random seeds,
and references must be retained. Report variation rather than selecting only the best run.

**Validation 6F — CPU-only simulation:** run the full headless suite with GPU devices hidden. Record total CPU usage,
real-time factor, missed controller deadlines, and peak resident memory. Rendering is not part of the acceptance path.

**Exit:** the differential milestone passes CPU physics-based metrics with retained logs and exact commands. If
`mc_mujoco` cannot expose a required sensor or contact quantity, add an independent MuJoCo-side CPU calculation; do not
substitute a controller-internal value and label it measured.

### Stage 7 — Add mode-aware rolling and transitions

Implement the requested and estimated contact modes from `\eqref{eq:slip-hysteresis}` and the mode-aware QP. Mode
selection remains outside the convex QP. It combines schedule, filtered slip, rolling-task residual, normal force,
friction margin, torque margin, separate entry/exit thresholds, and minimum dwell time.

Transitions must vary task weight, hard-row relaxation, or activation continuously without duplicating rows. Define
the exact policy for transitions that would otherwise switch a row from hard to absent. Detached force variables must
be zero or removed according to the backend contract.

**Validation 7A — state machine:** unit-test every allowed and forbidden transition, hysteresis boundary, dwell timer,
reset, invalid measurement, and recovery path with a deterministic clock.

**Validation 7B — fixed/rolling conflict:** reproduce `\eqref{eq:fixed-rolling-conflict}`. Fixed rows remain satisfied;
the incompatible longitudinal command appears as the measured soft residual; the solver does not become spuriously
infeasible.

**Validation 7C — sliding direction:** for established sliding, force opposes the measured tangential slip direction
within the polyhedral approximation. Zero/near-zero slip uses the documented epsilon policy and must not create a NaN
or arbitrary discontinuous direction.

**Validation 7D — continuity:** log generalized acceleration, torque, force, and task residual across every transition.
Assert configured bounds on step-to-step discontinuities and report maxima.

**Exit:** flat/ramp MuJoCo tests trigger and recover from sliding/detachment predictably, and transition tests pass
without stale forces, duplicate constraints, or solver-layout churn.

### Stage 8 — Implement the four-steering-wheel QP

Extend the same CPU data model and solver objects; do not fork a second rolling implementation. Each wheel adds a drive
joint and steering joint in the robot model, while the Tasks decision still uses the robot's existing generalized
accelerations and contact multipliers.

Required command families are:

- all steering angles zero with straight drive;
- parallel nonzero steering with crab motion;
- compatible constant-curvature/Ackermann-like steering;
- pure yaw where allowed by the selected steering geometry;
- nonzero steering rates to exercise direction-derivative terms;
- intentionally incompatible steering/wheel rates to exercise soft residuals;
- the same cases on a constant-slope ramp.

**Validation 8A — matrix parity:** compare every wheel's longitudinal/lateral coefficient and acceleration bias with
the Stage 2 oracle. Include nonzero `delta_dot`; a test that omits this term must fail.

**Validation 8B — common twist:** compatible wheel commands reconstruct one chassis twist within tolerance. No code may
average four independently inferred twists as the controller solution.

**Validation 8C — incompatible command:** hard normal/lateral safety remains satisfied while the longitudinal task
residual identifies the incompatible wheels by name.

**Validation 8D — controller and MuJoCo:** repeat the headless ticker, topology, no-slip, ramp, low-friction, transition,
CPU deadline, and replay gates for the steering robot.

**Exit:** both explicit QP specializations from the report are implemented through the same CPU rolling model and pass
equivalent evidence gates.

### Stage 9 — Complete CPU backend parity

Finish the TVM implementation using the Stage 2 CPU oracle and the already validated Tasks behavior. Share only
immutable descriptions and solver-independent calculations; do not share backend object pointers or rely on casts
across backends.

**Validation 9A — coefficient/function parity:** compare TVM value, velocity, normal acceleration, Jacobian, and
contact-force generalized columns with the common oracle and Tasks logical rows at the same state.

**Validation 9B — solution parity:** for feasible deterministic cases, compare generalized acceleration, wheel torque,
resultant contact wrench, and rolling residual. Because force distributions can be non-unique, also compare objective
value, dynamics residual, bounds, and resultant wrench rather than requiring identical individual cone generators.

**Validation 9C — lifecycle and modes:** repeat add/remove/re-add, reset, controller reload, fixed/rolling/sliding/
detached transitions, and invalid-input recovery on TVM.

**Validation 9D — controller matrix:** run every differential and steering ticker scenario on both CPU backends. Run a
representative flat, ramp, and low-friction CPU MuJoCo scenario per backend.

**Exit:** both CPU backends pass the declared common contract. If a deliberate backend-specific numerical difference
remains, it is bounded, tested, and documented; no backend silently drops rolling rows or forces.

### Stage 10 — Regressions, CPU performance, documentation, and handoff

1. Run a fresh full build and complete CTest suite, not only rolling tests.
2. Run formatting, schema, documentation, static-analysis, and sanitizer checks applicable to changed code.
3. Measure CPU solve/build time and allocations before and after adding rolling contacts for zero, two, and four wheels.
4. Verify no per-cycle heap allocation in the geometry kernel after warm-up and no unbounded logger/GUI growth.
5. Verify the rolling targets do not require or link GPU runtimes.
6. Add the tutorial, API docs, configuration schema/examples, troubleshooting, sign/frame diagram, and limitations.
7. Provide a clean-checkout command sequence for build, unit tests, ticker tests, robot loading, `mc_mujoco`, report
   generation, and log analysis.
8. Update this task's status/evidence sections with measured results and unresolved failures.

**Validation 10A — full regression:** all pre-existing tests that passed at Stage 0 still pass. Any changed baseline is
explained by a reviewed intentional behavior change.

**Validation 10B — CPU real-time behavior:** report median/P95/P99/max solve-and-build and rolling update time for both
backends and both robots over at least 10,000 post-warm-up cycles. Report one-thread and production-thread settings.
P99 total controller time must remain below the controller period; otherwise the task remains incomplete.

**Validation 10C — clean CPU checkout:** a fresh clone on a CPU-only environment can build, load both robots, run the
deterministic rolling CTests/controller smokes, and locate or clearly skip simulator-only tests when `mc_mujoco` is
unavailable. The target CPU machine must run the full simulator suite before release.

**Exit:** all required stages pass on CPU, reports are retained, documentation is complete, and no required file depends
on the developer's home-directory references.

## Required validation corpus

The automated corpus must cover at least the following named dimensions. Use a compact pairwise corpus plus the named
boundary cases; an undocumented random-only test is insufficient.

| Dimension | Required values |
| --- | --- |
| Chassis | differential; four steering |
| CPU backend | Tasks; TVM |
| Terrain | flat; positive ramp; negative ramp; globally translated/yaw-rotated plane |
| Motion | hold; forward/reverse; left/right turn; circle; sinusoid; crab; nonzero steering rate |
| Radius | equal nominal; unequal; invalid zero; invalid negative |
| Friction | high nominal; boundary; deliberately insufficient; invalid negative/NaN |
| Contact mode | fixed; hard rolling; soft rolling; sliding; detached; transition/recovery |
| Contact set | all wheels; one detached; zero active; remove/re-add |
| State | zero velocity; nonzero base twist; nonzero wheel rates; nonzero steering rates; seeded valid states |
| Command feasibility | compatible; rank-redundant; fixed-contact conflict; mutually incompatible wheel commands |
| Lifecycle | construct; add; update; reset; remove; re-add; controller reload; backend switch by configuration |

Every corpus case records its name, seed, robot/config hashes, backend, CPU/thread settings, timestep, terrain, wheel
metadata, state, command, active modes, expected validity, and exact assertion tolerances.

## Numerical acceptance policy

Freeze concrete tolerances after Stage 0 using double precision on CPU and record the rationale. The initial targets
below are the default; changing them requires a written numerical analysis and rerunning all prior gates.

| Quantity | Initial target |
| --- | ---: |
| Basis norm/orthogonality/right-handedness | `1e-12` absolute |
| Static geometry and planar closed-form identities | `1e-10` absolute |
| Matrix coefficient parity in double precision | `1e-10` absolute, `1e-9` relative |
| Jacobian central finite difference | `1e-6` absolute/relative after step-size sweep |
| `A_dot * alpha` finite difference | `1e-5` absolute/relative after step-size sweep |
| Hard rolling/contact equality residual | `1e-8` scaled infinity norm |
| Dynamics equality residual | `1e-8` scaled infinity norm |
| Inequality/torque/friction violation | `1e-9` plus solver feasibility tolerance |
| Virtual-work/generalized-force parity | `1e-9` absolute/relative |
| Tasks/TVM state and resultant-wrench parity | `1e-7` scaled absolute/relative |

Simulation thresholds must be frozen from physically meaningful nominal CPU runs before tuning the controller, and then
applied to untouched validation runs. At minimum define RMS, P95, and maximum limits for position/yaw tracking,
longitudinal/lateral/normal slip, torque violation, friction violation, solver-failure fraction, real-time factor,
deadline misses, and transition discontinuity. Do not define success only as “the robot did not fall.”

## Failure and safety semantics

- Invalid static configuration throws during load with the wheel name and offending field.
- Invalid runtime terrain/state data marks the rolling update invalid, removes or relaxes rows according to the
  documented safe policy, commands a bounded stop/hold behavior, and logs one structured reason.
- A failed QP returns the controller's existing failure signal and must not integrate an invalid result.
- Hard rolling infeasibility is never silently converted to a larger solver tolerance. The controller may switch to the
  explicitly configured soft/sliding policy and must log that transition.
- Zero active contacts is valid only for a declared flight/detached controller mode; contact forces and rolling rows
  are absent/zero and diagnostics say why.
- A reactivated contact begins from current measured geometry and resets hysteresis/transition state deterministically.
- NaN/Inf in any row, bound, force, torque, residual, or solver output fails the current cycle and never enters the
  robot integrator.
- Logs and GUI values carry a current-cycle validity flag. Stale last-good values may be displayed for debugging only
  if labeled separately as historical.
- Absence of a GPU is normal and must never be reported as a rolling-contact error.

## Evidence report required at completion

Update this document with a dated implementation checkpoint similar in rigor to `Task-GPU.md`. It must contain:

- commit hashes and dirty state for mc_rtc and every companion/upstream dependency change;
- final class/file/config/asset inventory;
- architecture decisions and any deviation from this roadmap;
- stage table with Pass/Fail/Blocked and direct evidence paths;
- CPU model/topology, compiler, build type, dependency/simulator versions, solver, CPU affinity, and thread settings;
- explicit proof that rolling targets run without required GPU libraries or devices;
- exact commands and exit codes;
- CTest names/counts and full regression result;
- worst-case math, matrix, dynamics, solver, and backend-parity errors with case/row/column identities;
- controller trajectory metrics for every named scenario;
- `mc_mujoco` topology, trajectory, ramp, slip, disturbance, transition, replay, and real-time-factor results;
- CPU solve/update timing, deadline, allocation, and peak-memory statistics;
- known limitations and failed gates;
- representative plots or recordings, plus the machine-readable data from which they were produced.

Generated binaries, large logs, videos, and simulator caches should normally remain outside Git. Compact schemas,
golden fixtures, JSON summaries, plotting scripts, and documentation needed to reproduce them must be tracked.

## Definition of done

Rolling contact is **not complete** when the toy robot merely moves, when only a DoF mask is added, when a controller
tracks wheel rates without traction torque, when a line surface is visually present, when one backend/demo passes, or
when the required path runs only with GPU software/hardware.

It is complete when:

1. the carrier/rim kinematics and acceleration rows match the report and independent finite differences;
2. the instantaneous line-force geometry produces the correct generalized force and wheel torque;
3. hard and soft rolling coexist with whole-body dynamics, bounds, and ordinary contacts in one CPU QP;
4. fixed, rolling, sliding, and detached modes have tested deterministic transitions and failures;
5. differential and four-steering robot/controller examples pass headless CPU unit and integration tests;
6. Tasks and TVM pass the common CPU contract without a silent unsupported path;
7. flat/ramp/low-friction CPU `mc_mujoco` experiments pass frozen metrics with retained evidence;
8. CPU P99 controller time satisfies the configured period with documented thread settings;
9. the complete pre-existing test suite remains green; and
10. a clean CPU-only checkout can reproduce the results without the three absolute-path reference projects.

If any gate remains failed, the final report must say that the implementation is partial and name the next concrete
work item. A detailed partial implementation is acceptable evidence; relabeling it complete is not.
