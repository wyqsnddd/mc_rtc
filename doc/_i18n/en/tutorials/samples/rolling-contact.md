The `RollingContact` sample demonstrates dynamically consistent rigid-wheel contact on a conventional CPU. It includes
the `rolling_diff` differential-drive robot, the `rolling_4s` four-steering-wheel robot, the Tasks and TVM solver
backends, contact-mode transitions, and a headless MuJoCo validation suite. CUDA, Torch, and a GPU are neither used nor
required.

## What the rolling constraint represents

Each wheel is described by a carrier-center frame, the wheel body that receives contact forces, a drive joint, an
optional steering joint, a radius, a width, friction, and a spin sign. The runtime geometry uses the terrain normal to
construct the instantaneous line contact and the right-handed world-frame basis `[t, l, n]`:

```text
                         wheel axle / lateral direction l
                lineStart o-------------o lineEnd
                          \      c      /   c: carrier center
                           \     | r   /
                            \    v    /
terrain plane --------------- p --------------------------> t
                              ^ n

l = n x t             p = c - r n             [angular; linear] wrench order
```

The carrier Jacobian contains only carrier motion. Wheel-rim speed enters once, through the named drive-joint selector.
For a wheel with generalized velocity `alpha`, the ideal rows are:

```text
(t^T J_c - r S_drive) alpha = 0   longitudinal rolling
 l^T J_c              alpha = 0   lateral no-slip
 n^T J_c              alpha = 0   normal no-separation
```

At acceleration level, the implementation includes the finite-difference-validated `A_dot * alpha` bias and velocity
stabilization. The two line endpoints each own a four-generator friction pyramid. Their forces enter floating-base
dynamics at the instantaneous wheel geometry, so drive torque, contact wrench, and virtual work remain consistent.

## Public C++ API

Use {% doxygen mc_rbdyn::RollingContactDescription %} for configuration-independent wheel data,
{% doxygen mc_solver::RollingContactConstraint %} for the kinematic rows, and
{% doxygen mc_solver::RollingContactDynamicsConstraint %} for forces, dynamics, and torque bounds. Both constraint
objects support Tasks and TVM through the same API.

```cpp
std::vector<mc_rbdyn::RollingContactDescription> wheels;
mc_rbdyn::RollingContactDescription left;
left.name = "left";
left.carrierFrame = "left_carrier";
left.wheelBody = "left_wheel";
left.driveJoint = "left_drive";
left.radius = 0.2;
left.width = 0.08;
left.friction = 0.8;
left.validate();
wheels.push_back(left);

mc_solver::RollingContactConstraintOptions options;
options.longitudinal = mc_solver::RollingContactLongitudinal::Hard;
options.velocityGain = 20.0;
options.differentialPlanar = true;

auto dynamics = std::make_unique<mc_solver::RollingContactDynamicsConstraint>(
    robots(), robotIndex, solver().dt(), wheels);
auto rolling = std::make_unique<mc_solver::RollingContactConstraint>(
    robots(), robotIndex, wheels, options);

solver().addConstraintSet(*dynamics);
solver().addConstraintSet(*rolling);
```

Keep both objects alive while they are registered. Remove `rolling` before `dynamics`, and remove tasks that reference
them before destroying the solver. The sample controller's destructor is an executable lifecycle example. Adding an
already-added object and removing an already-removed object are safe no-ops.

The constraint loader provides the same configuration path:

```yaml
constraints:
  - type: rollingContactDynamics
    robot: rolling_diff
    terrainNormal: [0.0, 0.0, 1.0]
    wheels: &wheels
      - name: left
        carrierFrame: left_carrier
        forceBody: left_wheel
        driveJoint: left_drive
        radius: 0.2
        width: 0.08
        friction: 0.8
      - name: right
        carrierFrame: right_carrier
        forceBody: right_wheel
        driveJoint: right_drive
        radius: 0.2
        width: 0.08
        friction: 0.8
  - type: rollingContact
    robot: rolling_diff
    longitudinal: hard
    velocityGain: 20.0
    differentialPlanar: true
    wheels: *wheels
```

For `rolling_4s`, set `steeringJoint` on every wheel, set `steeringPlanar: true`, and name two independent wheels in
`steeringPlanarWheels` (the sample uses `front_left` and `rear_left`). Do not enable both planar specializations, and do
not add duplicate generic chassis rows.

## Build and run without a GPU

From the mc_rtc source root on Ubuntu 24.04:

```sh
cmake -S . -B build -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build --parallel 2
CUDA_VISIBLE_DEVICES=-1 MC_RTC_DISABLE_CONVEX_GENERATION_PATCH=ON \
  ctest --test-dir build -R 'RollingContact|testRollingContact' --output-on-failure
```

The tracked ticker configurations disable GUI, plugins, and threaded logging. Run a five-second differential Tasks
example with:

```sh
CUDA_VISIBLE_DEVICES=-1 MC_RTC_DISABLE_CONVEX_GENERATION_PATCH=ON \
  build/utils/mc_rtc_ticker \
  -f rolling-contact-report/config/mc_rtc-differential.yaml \
  --run-for 5 --no-sync
```

Select one of these combinations:

| Robot | Tasks configuration | TVM configuration | Representative scenarios |
| --- | --- | --- | --- |
| Differential | `mc_rtc-differential.yaml` | `mc_rtc-differential-tvm.yaml` | hold, forward/reverse, turn, circle, sinusoid, unequal radii, mode cycle |
| Four steering | `mc_rtc-four-steering.yaml` | `mc_rtc-four-steering-tvm.yaml` | forward/reverse, crab, Ackermann turns, pure yaw, steering rate, incompatible command, mode cycle |

Change `RollingContact.scenario` in a copied configuration. Differential-only scenario names are rejected for the
four-steering robot and vice versa. All command values must be finite; periods and weights must be positive.

## Hard, soft, and mode-aware behavior

`longitudinal: hard` makes a fully activated rolling row an equality. `longitudinal: soft` retains it as a weighted
objective. During recovery, activation ramps from zero to one and the controller uses `recoveryRollingWeight`; it
promotes the row to a hard equality only after the acceleration residual is small enough. This avoids an infeasible
jump after contact returns.

| Mode | Kinematic policy | Force policy |
| --- | --- | --- |
| `fixed` | longitudinal, lateral, and normal rows | full friction pyramid |
| `rolling` | rolling longitudinal plus lateral/normal rows | full friction pyramid |
| `sliding` | removes the slipping no-slip row | freezes the closest generator opposing established slip |
| `detached` | removes contact rows | keeps the variable layout stable but fixes all wheel forces to zero |

The mode manager applies enter/exit hysteresis, minimum dwell time, filtering, and a transition ramp. Invalid or stale
external contact measurements fail safe to `detached`. A simulator can publish observations through
`RollingContact::SetMeasuredContact`; without them, the sample marks the fallback explicitly instead of presenting
model-derived quantities as measured contact.

Useful log fields include solver success, backend, build-and-solve time, total controller time, dynamics residual,
floating-base effort, per-wheel rolling/lateral/normal residuals, mode, activation, normal and tangential QP forces,
friction margin, drive torque, torque margin, and external-measurement validity. The CPU Tasks backend reconstructs
actuator torque after eliminating it from its decision vector. TVM optimizes an actuated-only torque variable; the six
floating-base effort entries are always zero and are never available as a fictitious support wrench.

## Headless CPU MuJoCo validation

The validation runner pins `mc_mujoco` and MuJoCo versions, installs them under `/tmp`, applies an isolated user-model
destination patch, and audits every target for CUDA/Torch/GPU runtime linkage:

```sh
rolling-contact-report/scripts/build-cpu-mujoco-runner.sh

python3 rolling-contact-report/scripts/run-mujoco-suite.py \
  --runner /tmp/rolling-contact-cpu-mujoco/runner-build/rolling_contact_mujoco_runner \
  --artifact-dir /tmp/rolling-contact-mujoco-suite \
  --backend both --cycles 1000 --warmup-cycles 100 --repetitions 5
```

The suite covers flat ground, ramps, low friction, impulses, contact loss, mode cycling, differential trajectories, and
four-steering maneuvers. It compares RBDyn and MuJoCo topology before stepping, consumes MuJoCo contact forces in the
mode manager, checks deterministic replay, and emits JSON summaries plus per-run CSV files. Rendering is not part of
the acceptance path.

## Troubleshooting and limitations

- `Unknown frame/body/joint`: every configured name is resolved when the constraint is constructed. Check the robot
  module variant and use its carrier frame, force body, and drive/steering joint names exactly.
- `terrainNormal must be finite and nonzero`: supply a world-frame plane normal; online terrain reconstruction is not
  implemented.
- Immediate detach or persistent sliding: inspect measurement validity, normal force, friction/torque margins, and the
  enter/exit thresholds before changing QP weights.
- Failure during hard-row recovery: use soft longitudinal recovery, keep `recoveryRollingWeight >= rollingWeight`, and
  inspect the hard-promotion residual.
- Unexpected TVM base support: it is a bug if `floating_base_effort_norm` is nonzero. TVM uses only actuated torque in
  dynamics.
- MuJoCo model lookup failure: rebuild with the provided script; the mappings are installed into the isolated
  `/tmp/rolling-contact-cpu-mujoco/mc_mujoco-user` destination.

This implementation models conventional rigid wheels on a plane or constant-slope ramp with a polyhedral friction
cone. It does not model tire deformation, rolling resistance, arbitrary height fields, complementarity, casters,
omni/mecanum wheels, tracks, learned slip, or hardware-specific sensing. Sliding selects one discrete pyramid generator
rather than solving a nonlinear Coulomb cone. These boundaries are deliberate and are shared by both CPU backends.
