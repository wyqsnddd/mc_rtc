# ADR-001: CPU rolling-contact architecture

- Status: Accepted and implemented
- Date: 2026-09-01
- Scope: differential-drive and four-steering rigid wheels on a plane or constant-slope ramp
- Normative model: `rolling-contact-qp.tex`

## Decision

Rolling contact is implemented as one backend-neutral geometry/state layer in `mc_rbdyn` and two backend-owned solver
adapters in `mc_solver`/`mc_tvm`. All production calculations use ordinary C++17 and Eigen on the host CPU. The
feature has no CUDA, Torch, GPU-array, device-pointer, or asynchronous accelerator path.

The geometry layer deliberately exposes two distinct objects for every wheel:

1. the carrier center `c`, whose translational Jacobian appears in the rolling rows; and
2. the instantaneous contact point `p = c - r n`, whose point Jacobian maps the optimized contact force into the
   equations of motion.

The local frame is `[t, l, n]`, with `l = n x t`. A resolved wheel description provides the drive joint, optional
steering joint, carrier frame, wheel body, radius, width, spin sign, axle direction, terrain plane, friction, requested
mode, and activation. A per-cycle result owns the normalized basis, center, force point, line endpoints, carrier and
point Jacobians, homogeneous rolling matrix, acceleration bias/right-hand side, slip, and validation diagnostics.

## Ownership and source layout

| Concern | Owner | Selected source location |
| --- | --- | --- |
| Description, modes, geometry, planar oracle | `mc_rbdyn` | `include/mc_rbdyn/RollingContact.h`, `src/mc_rbdyn/RollingContact.cpp` |
| Tasks hard rolling rows | `mc_solver` | `RollingContactConstraint` with a Tasks implementation over the existing `alphaD` block |
| Tasks soft longitudinal row | `mc_tasks`/Tasks adapter | one quadratic residual task over `alphaD`; no invented slack decision variable |
| Tasks dynamic point/cone | `RollingContactDynamicsConstraint::RollingMotionConstr` | stable contact IDs and lambda blocks with per-cycle generalized-force columns |
| TVM rolling rows | `mc_tvm` | a TVM function with the same backend-neutral geometry values and derivatives |
| TVM dynamic point/cone | TVM dynamic contact | stable point-force variables with updated point/frame coefficients |
| Loaders/schema | `mc_solver` and `mc_tasks` | existing constraint/task loader registries and JSON schemas |
| Example robots/controllers | tracked repository modules/samples | `src/mc_robots/rolling_contact_description` and `src/mc_control/samples/RollingContact` |
| Unit/integration evidence | CTest and report fixtures | `tests/testRollingContact*.cpp`, `rolling-contact-report/evidence` |

Required assets are not placed only in the ignored top-level `robots/` or `controllers/` directories.

## Solver-variable mapping

The symbols in the report describe a mathematical QP. Each backend retains its native variable model.

| Report symbol | Tasks backend | TVM backend |
| --- | --- | --- |
| generalized acceleration `alphaD` | native robot `alphaD` segment selected by `SolverData::alphaDBegin(r)` | `mc_tvm::Robot::alphaD()` variable |
| torque `tau` | eliminated; reconstructed and bounded by `tasks::qp::MotionConstr` | an independent actuated-only variable; floating-base effort is excluded from the decision vector |
| force generators `lambda` | native contact multipliers beginning at `SolverData::lambdaBegin(contact)` | native point-force variables and TVM contact requirements |
| longitudinal slack `sigma` | represented by a quadratic task residual; no extra Tasks variable is claimed | quadratic residual task (an explicit slack is unnecessary for the selected least-squares form) |
| contact mode/activation | controller-owned fixed-size state and coefficients, never a QP integer variable | same |

The Tasks production path never appends arbitrary variables. A diagnostic test records `nrVars`, all acceleration and
lambda offsets, and generator counts for the toy topology. TVM gets an equivalent variable inventory test. For a
floating robot, `mc_tvm::DynamicFunction` places `-I` only in the actuated rows of its torque Jacobian. The complete
floating-base dynamics equality is still enforced, but base support must come from contact force. After a successful
solve, `TVMQPSolver` maps the actuated result into the robot's full effort vector and explicitly zeros the first six
entries. This prevents a non-physical optimized base wrench from satisfying gravity when rolling contact is active.

## Dynamic contact-force geometry

The number and order of force variables are frozen when a wheel contact is registered. Every cycle updates only the
numerical point and friction-generator coefficients:

- transform the inertial instantaneous line endpoints `p +/- width/2 l` into the current wheel-body frame;
- express the rolling-frame cone generators in that same body frame;
- refresh the point Jacobian/generator blocks consumed by dynamics and torque bounds; and
- leave the contact ID, point count, generator count, lambda offsets, and warm-start vector unchanged.

The Tasks API copies point and cone data into `SolverData` and again into `MotionConstr` during `updateNrVars`. Rather
than changing the external Tasks project, the implementation derives a narrowly scoped `RollingMotionConstr`. It
registers ordinary, fixed-size `UnilateralContact` objects once, resolves their stable lambda offsets in
`updateNrVars`, calls the base `MotionConstr::update`, and then replaces exactly those rolling-contact generalized-force
columns with the instantaneous `-J_p^T G` coefficients. Torque reconstruction and bounds therefore continue to use the
same matrix, without changing the decision layout or rebuilding the contact set.

TVM creates two persistent body-frame point-force variables per wheel, plus persistent force-cone and mode functions.
Add/remove operations remove and restore their requirements but retain the variables and functions so TVM's assignment
mapping remains valid across recovery and lifecycle re-entry. Removed task graphs enter a solver-owned retirement queue
until the next solve refreshes TVM's weighted-least-squares cache; if no solve follows, the queue outlives the TVM
problem. This prevents stale graph-node destruction without allocating in steady-state control cycles. Each update
changes the point pose, rolling frame, cone,
and mode coefficients only.

A carrier-frame proxy that omits the moment `(-r n) x f`, or rebuilding the full contact list every cycle, is rejected.

## Geometric-row partition

The ordinary contact owns normal maintenance and force feasibility. It does not also impose fixed longitudinal or
lateral motion for a rolling wheel. Rolling kinematics are partitioned by mode:

| Mode | Rolling kinematic policy | Dynamic force policy | Variable layout |
| --- | --- | --- | --- |
| fixed | hard zero carrier-longitudinal row, lateral/normal rows, and hard drive lock | full friction pyramid | retained |
| rolling | lateral/normal hard; longitudinal hard or quadratic according to configuration and activation | full friction pyramid | retained |
| sliding | removes that wheel's longitudinal no-slip row; shared planar rows use remaining attached wheels | two endpoint forces restricted to the frozen pyramid generator opposing established slip | retained |
| detached | removes that wheel's contact rows | both endpoint forces fixed to zero | retained |

Differential and four-steering planar chassis use the explicit common-base geometric tangential rows. The equivalent
generic carrier-Jacobian tangential rows are not inserted simultaneously. A rank test detects accidental duplicates.

## Mode transitions and failure behavior

The controller owns requested mode, estimated mode, activation, hysteresis thresholds, filtered observations, minimum
dwell time, and transition state. Transition coefficients change continuously while force-variable identity remains
fixed. A partially activated row is soft; promotion to a hard row is delayed until its measured acceleration residual
is below the recovery threshold. The controller temporarily raises the soft rolling weight during recovery. Invalid
or stale external contact measurements request the safe detached state, invalid geometry/configuration throws before
registration, and solver failure invokes `safeStop`; none can silently disable a constraint.

Tasks and TVM are both production-supported. They share the geometry oracle and public constraint objects but retain
backend-native solver graph objects. Direct tests compare coefficients, acceleration, actuated torque, resultant
contact wrench, residuals, contact-mode behavior, and repeated add/remove lifecycle. The default TVM least-squares
solver's redundant static-hold acceleration result is accepted at `1e-6` scaled tolerance (measured error below
`4e-7`); coefficient parity remains at `1e-10` and wrench/torque parity uses `1e-7` scaled tolerance.

## Authoritative CPU configuration

The reference build is `RelWithDebInfo`, GCC 13, C++17, Eigen 3.4, and the repository's existing CPU solver stack.
Tasks uses its QLD path and TVM uses its default least-squares solver. Correctness tests run with
`CUDA_VISIBLE_DEVICES=-1`. Timing evidence records CPU affinity and the controller's non-threaded logging/no-sync
configuration after warm-up. No GPU-dependent rolling option exists in this tree, so a CTest linkage audit rejects
CUDA, cuBLAS, cuSolver, Torch, HIP, and OpenCL runtime dependencies in all rolling targets.

## Consequences and mandatory probes

- Geometry can be tested independently of either optimizer and shared without sharing solver objects.
- Tasks needs only the repository-local `RollingMotionConstr` override, not an external Tasks patch or a general
  arbitrary-variable extension.
- Soft rolling has the same least-squares semantics in both backends even though their native decision vectors differ.
- Dynamic force correctness is accepted only after point-Jacobian, virtual-work, wheel-torque, friction-frame, and
  stable-layout tests pass.
- `mc_mujoco` is a downstream validation gate. It does not define controller geometry or replace analytical tests.
