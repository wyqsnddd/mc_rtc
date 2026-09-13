# CPU rolling-contact and MuJoCo run guide

This directory contains the reproducible CPU validation path for the rolling-contact controller. The
four-steering configuration uses the self-contained Ranger Mini V3 model (`ranger_mini_v3`), with four
independent steering and drive joints. Robot selection follows the installed mc_rtc convention: `MainRobot` is a
single alias argument (for example, `MainRobot: RangerMiniV3Robot`).

## How four-steering commands reach the QP

The QP optimizes **eight rotating velocities** for the Ranger: one rolling rate and one steering rate per
wheel. This matters because the steering axis passes through the wheel centre, so steering produces no
translation of that centre; a rolling row built only from the carrier *translational* Jacobian therefore has
an identically zero column for the steering joint, and the QP cannot plan steering at all. The rotating-rate
rows of the four-steering QP supply that column. `RotatingRateRowsSteerTowardsTheReference` measures the
difference directly: for a `+0.6 rad/s` steering-rate reference the optimal steering acceleration is exactly
`0` without the rate block and `0.597 rad/s^2` (Tasks) / `1.481 rad/s^2` (TVM) with it, each matching its own
closed form.

The per-wheel references come from `mc_rbdyn::steeringWheelReference`, which inverts the expanded
four-steering rolling and zero-lateral-slip equalities for the commanded projected twist, including the
Ranger's +/-90 degree hinge limits. There are **no empirical scale factors** left in this path: the yaw sign
mirror, `keyboardYawScale = 1.7`, `keyboardMixedDriveScale = 1.15`, the branch-cut hysteresis, the
"don't spin while steering" gate and the command-transition grace period all existed to compensate an
open-loop steering command and are removed. `setCommandedTwist()` is the single entry point shared by the
keyboard, the GUI number inputs and the scripted scenarios.

### Softened lateral rows and `lateralSlackWeight`

The four zero-lateral-slip rows are **softened**, each with its own slip slack, rather than hard equalities.
This is not a YAML option on the four-steering robot: `mc_rolling_contact_controller.cpp` sets
`options.softLateralRows = fourSteering_` unconditionally, because the reason to soften is a property of
having four independently steered wheels, not a tuning choice. Four lateral rows constrain only the
chassis's three planar degrees of freedom; whenever the steering angles are not exactly coordinated on a
common instantaneous center of rotation - which measured steering angles generally are not - the only twist
that satisfies all four hard rows is zero, i.e. a frozen chassis. `RollingContactConstraint.cpp` names this
directly in a comment on the row it removed: "the frozen-chassis failure `softLateralRows` exists to
prevent."

`steeringPlanar` and `steeringPlanarWheels` are gone for the same reason. That earlier device kept only two
of the four lateral rows hard, on a pair that had to be chosen by hand, and left the other two unconstrained
- a partial workaround for the same over-determination. It has no remaining purpose now that all four rows
are softened together, and `RollingContactConstraint`'s config parsing treats either key as a hard error
("`steeringPlanar` and `steeringPlanarWheels` were removed; the four lateral rows are now softened together,
so use `softLateralRows` and `lateralSlackWeight` instead") rather than silently ignoring it.
`differentialPlanar` is untouched: a two-wheel chassis has two lateral rows on three planar degrees of
freedom, which is not over-determined, so keeping that single row hard is still correct.

`lateralSlackWeight` (default `1e7`, both in the controller and pinned explicitly in
`config/mc_rtc-four-steering*.yaml` and `config/mc_rtc-ranger-mini-v3-keyboard.yaml`) trades off against two
different failure modes on either side, only one of which a purely kinematic ticker run can see. Below
roughly `5e6` the worst lateral slip over a scripted commanded-twist sequence exceeds its `0.15` m/s bound
(`1.85` m/s at `1e5`); above roughly `1e9` the closed-loop `four-pure-yaw` yaw-tracking error in `mc_mujoco`
exceeds its `0.15` rad bound (`0.171` rad). The shipped `1e7` sits inside that window, roughly a factor of 3
above the lower edge and a factor of 2 below the upper one.

### `dt^2` rate-weight scaling

A rate row's coefficients carry the prediction step `dt`, so `rollingRateWeight` and `steeringRateWeight`
enter the objective multiplied by `dt^2`. At the 5 ms timestep the checked-in `4.0e7` is an equivalent
acceleration-task weight of `4e7 * 0.005^2 = 1000`, twice the chassis task weights. The controller default is
`1000 / dt^2`, i.e. the same effective 1000; writing a naive `200` in a configuration yields an effective
`0.005` and makes the rate rows inert. `RotatingRateRowsFollowTheBlockWeight` pins the realised weight and
pins that it is rebuilt when the rolling block weight changes.

### Selecting the robot from the ticker command line

`mc_rtc_ticker` accepts roslaunch-style overrides, so a profile does not have to be edited to swap the robot
or the controller:

```sh
mc_rtc_ticker -f rolling-contact-report/config/mc_rtc-four-steering.yaml \
  robot:=RangerMiniV3Robot controller:=RollingContact --run-for 5 --no-sync
```

`robot:=` sets `MainRobot` and `controller:=` sets `Default` plus `Enabled`; both are applied after the `-f`
file is merged.

## Status: known open failures

Both of the failures previously listed here were the same defect and are **closed**: the contact-mode
estimator was consuming the QP's own contact multiplier as if it were a contact-presence measurement. Four
coplanar wheel contacts leave the normal-force distribution with a one-dimensional null space - the diagonal
mode `(+1, -1, -1, +1)`, which produces no net force and no net moment - and nothing in this QP's objective
penalises it, so the solver may return zero on one wheel while the chassis is level and every wheel is loaded.
`desiredMode()` turns that zero into an *immediate* `detached` verdict, and the verdict is self-confirming
because a detached wheel's multipliers are identically zero.

- **Right-hand Ackermann in the ticker**: now `rolling` for 1000/1000 cycles on all four wheels and the
  chassis travels 0.656 m in 5 s (was: `detached` from cycle 2 for 781 cycles, 0.042 m).
- **`four-crab` / `four-ackermann-left` in closed-loop MuJoCo**: now all four wheels `rolling` for a 25 s
  run with no QP failure. Crab tracks 0.1977 m/s of a commanded 0.2 m/s and settles at 177/181/187/190 N,
  against 183/0/205/341 N with `front_right` detached for 4990 of 4995 cycles before.

Still open:

- **Four-steering yaw tracking** - see the dedicated section below. This is why
  `four-ackermann-acceptance` and `four-pure-yaw-acceptance` are `expected_failure` in the suite.
- **`four-steering-rate` and `four-low-friction`** in the CPU suite; see the divergence section below.
- **`sliding` latches.** The same category error survives one level down. `desiredMode()`'s `recovered`
  test requires `frictionMargin > frictionMarginEnter` (1e-3 N), and that margin is `mu * N - F_t` computed
  from the QP's own multipliers. The friction cone is a *constraint* of that QP, so the margin sits at
  exactly 0 whenever the solution uses all the friction available - which is normal under an aggressive
  command - and `recovered` can then never become true. In a 60 s keyboard session
  (`W`, `W+E`, `A`, `A+Q`, `x`) all four wheels enter `sliding` at t = 27.21 s, on the 90-degree re-steer
  from a turn into a pure crab, and stay there. It is not currently harmful: tracking is unchanged across
  the boundary (|v| 0.2875 rolling vs 0.2871 sliding; yaw 0.424 vs 0.418 rad/s of a commanded 0.5) and the
  QP never fails. Before the normal-force fix the same instant drove the multiplier to 0 and the wheels
  latched `detached` instead, which *was* harmful.

No unit test covered a mixed forward-plus-*negative*-yaw command, which is why the first was not caught.
`QpContactMultiplierAloneNeverDetachesAWheel` in `test_controller_lifecycle.cpp` now pins the cause directly.

## What the CPU MuJoCo suite does and does not cover

`scripts/run-mujoco-suite.py` drives `mujoco/rolling_contact_mujoco_runner.cpp`, which links the real
`mc_mujoco::MjSim`. It is not a physics proxy - the physics is MuJoCo 3.3.6 - but it is **not** the same
regime as the acceptance command

```sh
mc_mujoco -f rolling-contact-report/config/mc_rtc-ranger-mini-v3-keyboard.yaml
```

and the differences are large enough that the suite passed manoeuvres real `mc_mujoco` failed. Each is now an
explicit per-case switch (`--torque-control`, `--measured-contacts`, `--preset-steering`), recorded in every
replay's JSON and printed next to each case name, instead of being hard-coded inside the runner.

| divergence | suite (historical cases) | `mc_mujoco -f ...` | why it matters |
| --- | --- | --- | --- |
| actuation | `torque_control = true` | `false` (mc_mujoco's own default; the acceptance command passes no `--torque-control`) | with `false` the `<motor>` actuators are driven by mc_mujoco's joint PD on the controller's `q`/`alpha` output. Under PD, `four-crab` detached `front_right` at t = 0.025 s; under torque control the same config and binary ran 25 s clean. |
| contact sensing | injects MuJoCo's true normal/tangential force and slips through `RollingContact::SetMeasuredContact` every substep | nothing - `RollingContact_*_simulator_measurement_valid` is 0 for the whole run | the contact-mode estimator otherwise falls back to the QP's own contact multiplier. This is what structurally hid the `front_right` detachment: with ground truth the estimator never reads the multiplier at all. |
| initial state | four-steering crab/ackermann/pure-yaw reset with the hinges **already** at the manoeuvre's steady-state angle | hinges at zero | skips the steering transient, which is where the detachment happens. |
| contact margin | `geom_margin >= 1e-4` added to the ground and wheel geoms | model default (0) | deliberate: a coplanar four-wheel reset can otherwise select only one axle on the first substep. Kept, and it does not mask any failure found here. |
| terrain | ground friction overwritten to `[f, 0.01, 0.001]`, wheel friction to `f` | model values | only matters for the ramp/low-friction cases. |
| command amplitude | `yaw_rate = 0.05` on `four-pure-yaw` / `four-ackermann-*` | `keyboardAngularSpeed: 0.5` | 7-10x below what an operator can command. |
| duration | 1000 cycles = 5 s | unbounded | the heading-target windup below needs 9.7 s at 0.35 rad/s to fail; the suite could never reach it. |
| bounds | `tracking_yaw_error_max_rad <= 0.15` | - | at 0.05 rad/s over 5 s the whole commanded rotation is 0.25 rad, so the bound is 60% of the commanded motion. `four-pure-yaw` passes it while turning only 0.1615 rad of the 0.25 commanded (65%). |

`steeringWeight: 10.0` in the generated configuration is also a dead key - the controller reads
`steeringPostureWeight`. It has never had any effect.

The `*-acceptance` cases at the end of `CASES` run with all three switches off, at the speeds the acceptance
configuration commands, and long enough to reach the windup regime. `four-ackermann-acceptance` and
`four-pure-yaw-acceptance` are marked `expected_failure`: they reproduce the four-steering yaw-tracking defect
that is characterised but not fixed (see below). The suite fails if an `expected_failure` case ever starts
passing, so the expectation cannot go stale.

Two pre-existing case failures are **not** caused by the fixes on this branch - their reports are bit-identical
with the controller reverted to `1f136644ad`:

- `four-steering-rate` requires `max_rolling_slip_mps >= 1e-2` and a `sliding`/`detached` estimate. The
  archived Sep-1 evidence has 0.0703 m/s and `['detached', 'rolling']`; today the sweep produces 2.36e-4 m/s
  and stays rolling throughout, so the case's premise no longer holds.
- `four-low-friction` requires `max_rolling_slip_mps >= 2e-2` on a mu = 0.005 surface. Archived: 1.296 m/s;
  today: 1.59e-4 m/s - the robot no longer slips on ice.

Both regressed somewhere in the four-steering rate-row rewrite and are open.

### ORC-08: the independent cross-check against the real `mc_mujoco`

Because the CPU suite cannot observe the divergences tabulated above,
`scripts/cross-check-mc-mujoco.py` drives the installed `mc_mujoco` binary headless
(`--without-visualization --without-mc-rtc-gui`), asserts that
`<wheel>_simulator_measurement_valid` is false for the whole run - so the contact-mode estimator is
genuinely reading the QP's own multiplier and no ground truth is being injected - and only then checks
the trajectory. It is opt-in because it needs `mc_mujoco` installed and writes hundreds of megabytes of
log per run:

```sh
cmake -S . -B build -DROLLING_CONTACT_MC_MUJOCO_CROSS_CHECK=ON
cmake --build build -j
CUDA_VISIBLE_DEVICES=-1 MC_RTC_DISABLE_CONVEX_GENERATION_PATCH=ON \
  ctest --test-dir build -j1 -L rolling-contact-mujoco --output-on-failure
```

Two tests are registered, one per actuation regime, because the two disagree by a factor of twelve.
On `mc_rtc-four-steering.yaml` (crab at 0.15 m/s) over a 5 s window:

| regime | commanded | control robot | `mc_mujoco` | tracked | controller/simulator discrepancy |
| --- | --- | --- | --- | --- | --- |
| joint PD (mc_mujoco default, the acceptance command) | 0.7500 m | 0.7439 m | 0.7398 m | 98.6% | 0.0234 m |
| `--torque-control` (the CPU suite's regime) | 0.7500 m | 0.7439 m | 0.5094 m | 67.9% | 0.2953 m |

The control robot tracks its own command to within 1% in both, so the QP is doing what it was asked.
**Open:** under `--torque-control` the simulator delivers about two thirds of it, and the gap does not
converge as the timestep shrinks - 0.2953 m, 0.2151 m, 0.2914 m at `Timestep` 5 ms, 2.5 ms and 1 ms.
The testcard's own criterion is that convergence order is what separates a defect from a
discretization difference, so this is a real controller/simulator disagreement in the torque-control
path. No wheel detached and drive torque peaked at 6.5 Nm against the 35 Nm limit in either regime, so
it is neither a contact-loss nor a saturation failure. The historical joint-PD `front_right`
detachment tabulated above does not reproduce on this config with the current code.

## Open: four-steering yaw tracking

Commanded pure yaw is tracked at a fraction of the command, in every actuation mode, in real `mc_mujoco`:

| command | achieved | note |
| --- | --- | --- |
| `pure_yaw` 0.35 rad/s, PD | 0.0235 rad/s (6.7%) | steering hinges park at -1.011 / +0.862 rad against a correct +/-0.9358 rad reference |
| `pure_yaw` 0.35 rad/s, `--torque-control` | 0.107 rad/s (31%) | hinges park at -1.085 / +0.734 rad |
| keyboard `E` at `keyboardAngularSpeed: 0.5` | 0.16 rad/s (31%) | hinges park at -0.845 / +1.030 rad |

The reference itself is correct and mirror-symmetric (recovered from the logged rate reference and measured
angle as -0.9358 / +0.9358 rad); the hinges simply do not reach it. All four are displaced by the same
~0.09-0.135 rad in the same rotational sense, which is exactly the steering error a rigid chassis cannot
absorb, and it pins `RollingContact_lateral_slack_norm` at 1.5-1.9.

Localised by a single-variable experiment: narrowing the wheel collision cylinder's half-width from 0.04 m to
0.004 m in `ranger_mini_v3.xml` (10x less contact line, everything else identical) raises yaw tracking from
31% to 89%, drops the hinge offset from 0.135 to 0.05 rad and bounds the lateral slack at 0.15-0.32. The QP
places its contact **forces** on a line of the wheel's full 0.08 m width, but imposes its rolling/lateral/normal
**constraints** at the single centre contact point only, so it has no term that says re-steering a loaded wheel
scrubs the patch. mc_mujoco does: with `condim=4` and `friction="0.8 0.01 0.001"` at 184 N the resisting yaw
moment is roughly `0.8 * 184 * 0.04 + 0.01 * 184 = 7.7 N.m`, against the 1.3-2.6 N.m of steering torque the QP
asks for. Fixing this means constraining wheel spin about the contact normal in
`mc_solver::RollingContactConstraint`, which is a formulation change and is not attempted here.

## Build and run the tests

Configure and build mc_rtc from the repository root:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_TESTING=ON
cmake --build build --parallel 2
```

Run the focused rolling-contact tests. Every registered CTest now runs with `HOME` pinned to
`build/test-home`, so `~/.config/mc_rtc/mc_rtc.yaml` and `~/.config/mc_rtc/controllers/` cannot leak into a
fixture. This is not optional hygiene: mc_rtc merges the user global configuration *before* the `-f` file, and
`LoadUserConfiguration` is read before that file is merged, so a fixture cannot opt out from its own YAML. A
developer profile that sets `MainRobot`/`Enabled` fails 55 of the 104 tests without the pin. The fixture
configs additionally pin `closedLoopFeedback: false` explicitly, which is worth keeping on its own merits.

```sh
CUDA_VISIBLE_DEVICES=-1 MC_RTC_DISABLE_CONVEX_GENERATION_PATCH=ON \
  ctest --test-dir build -R 'RollingContact|testRollingContact' --output-on-failure -j1
```

Run the complete repository regression when changing shared mc_rtc code:

```sh
CUDA_VISIBLE_DEVICES=-1 MC_RTC_DISABLE_CONVEX_GENERATION_PATCH=ON \
  ctest --test-dir build --output-on-failure -j1
```

The rolling test families cover geometry and finite differences, one-argument robot aliases, Ranger topology and
wheel dimensions, Tasks/TVM solver parity, controller lifecycle, contact-mode transitions, and all deterministic
four-steering command modes.

## Build the CPU MuJoCo runner

The pinned MuJoCo 3.3.6 build and runner are isolated under `/tmp`; this keeps the reproducibility artifacts separate
from the user installation. The interactive commands below use the mc_mujoco binary already installed at
`~/local/bin/mc_mujoco` and the mappings installed at `~/local/share/mc_mujoco`. The helper also installs the
`rolling_diff` and `ranger_mini_v3` mappings into its own isolated mc_mujoco user directory:

```sh
CUDA_VISIBLE_DEVICES=-1 rolling-contact-report/scripts/build-cpu-mujoco-runner.sh
```

The resulting runner is:

```text
/tmp/rolling-contact-cpu-mujoco/runner-build/rolling_contact_mujoco_runner
```

Run a small deterministic smoke matrix for both CPU backends:

```sh
python3 rolling-contact-report/scripts/run-mujoco-suite.py \
  --runner /tmp/rolling-contact-cpu-mujoco/runner-build/rolling_contact_mujoco_runner \
  --build build --artifact-dir /tmp/rolling-contact-mujoco-results \
  --backend both --repetitions 1 \
  --case four-forward --case four-crab --case four-ackermann-left --case four-pure-yaw
```

The complete suite is selected by omitting `--case` (25 scenarios × Tasks/TVM × the requested repetitions). The
runner is headless and CPU-only; it validates static MuJoCo/RBDyn parity, contact state, slip, force/torque bounds,
odometry, tracking, determinism, and disturbance recovery.

## Run a differential-drive robot in mc_mujoco

Use the Tasks or TVM configuration from this directory. The configuration already selects the one-argument
`RangerMiniV3Differential` alias and uses the source-tree controller/model paths:

```sh
source /opt/ros/jazzy/setup.bash
export MC_RTC_DISABLE_CONVEX_GENERATION_PATCH=ON
export CUDA_VISIBLE_DEVICES=-1
export LD_LIBRARY_PATH="$PWD/build/src:$PWD/build/plugins/ROS:$PWD/build/deps/tasks-system-install/lib:/usr/local/lib:/opt/ros/jazzy/lib"
/home/yuquan/local/bin/mc_mujoco -f rolling-contact-report/config/mc_rtc-differential.yaml
```

For TVM, use `mc_rtc-differential-tvm.yaml`; for the fixed/rolling/sliding/detached transition demonstration, use
`mc_rtc-differential-modes.yaml`. Add `-s` to synchronize simulation time with wall time. Close the MuJoCo window or
press Ctrl-C to stop.

## Run the four-steer Ranger Mini V3 robot in mc_mujoco

The visible model is primitive/self-contained so it does not depend on unshipped V2 mesh files. The model has a
0.494 m wheelbase, 0.364 m track, 0.125 m wheel radius, and four steer/drive pairs. Run the deterministic crab
example with:

```sh
/home/yuquan/local/bin/mc_mujoco -f rolling-contact-report/config/mc_rtc-four-steering.yaml
```

Use `mc_rtc-four-steering-tvm.yaml` to select the TVM backend. Ensure the `ranger_mini_v3` MuJoCo mapping is installed
at `~/local/share/mc_mujoco/ranger_mini_v3.yaml` (the completed install in this workspace has it there; the CPU build
helper installs a separate copy under its isolated `/tmp/rolling-contact-cpu-mujoco/mc_mujoco-user` directory).

## Interactive keyboard control

`mc_rtc-ranger-mini-v3-keyboard.yaml` enables the optional RoboticsUtils `KeyboardCapture` adapter. RoboticsUtils
must be installed (the controller CMake probe looks for `RoboticsUtils::Keyboard` and links its GEOS dependency).
Keyboard capture requires a real terminal on standard input, so start mc_mujoco from the terminal rather than from a
pipe or a background service:

```sh
cd /home/yuquan/local/mc_rtc_sources/mc_rtc
source /opt/ros/jazzy/setup.bash
export MC_RTC_DISABLE_CONVEX_GENERATION_PATCH=ON
export CUDA_VISIBLE_DEVICES=-1
export LD_LIBRARY_PATH="$PWD/build/src:$PWD/build/plugins/ROS:$PWD/build/deps/tasks-system-install/lib:/usr/local/lib:/opt/ros/jazzy/lib"
KEYBOARD_CFG="$PWD/rolling-contact-report/config/mc_rtc-ranger-mini-v3-keyboard.yaml"
grep -qx 'MainRobot: RangerMiniV3Robot' "$KEYBOARD_CFG"
/home/yuquan/local/bin/mc_mujoco -s \
  -f "$KEYBOARD_CFG"
```

The repository root is required because the keyboard YAML's controller, robot, and observer module paths are
relative to it. If the startup output lists `JVRC1` instead of `ranger_mini_v3`, stop the process and correct the
working directory before sending keyboard commands.

The profile keeps the standard IPC endpoint used by mc_mujoco's built-in visualizer. `ControllerServer` probes and
retries stale socket pathnames left by an interrupted run, but it will deliberately refuse to replace a live
controller. If the bind error names `/tmp/mc_rtc_pub.ipc`, check `ss -xlpn | grep mc_rtc` and stop the older
`mc_mujoco`/ticker instance before launching this one.

The startup banner confirms that capture is active. Commands are interpreted in the robot frame:

| Key | Command |
| --- | --- |
| `W` / `S` | forward / reverse |
| `A` / `D` | translate left / right (four-wheel crab motion) |
| `Q` / `E` | rotate left / right |
| `space` | clear all latched velocity axes while keeping keyboard capture running |
| `x` | stop keyboard capture and return to zero command |

The controller converts RoboticsUtils' lateral/yaw signs to mc_rtc's +Y-left/+Z-counter-clockwise convention and
publishes the result through `MCRollingContactController::setCommandedTwist`, the single entry point shared by the
keyboard, the GUI and the scripted scenarios. `mc_rbdyn::steeringWheelReference` then inverts the expanded
four-steering rolling rows per wheel, including the ±90° Ranger steering limits, and the QP's rotating-rate rows
track both the rolling and the steering rate. There is no analytic hinge IK, no drive-hold barrier and no
command-transition grace period left in the controller: the QP arbitrates steering and drive in the same solve.
Terminal input supplies key-down bytes rather than continuous release state, so commands are latched explicitly, and
every axis a key does not rewrite is indistinguishable from one the operator is still holding down. Each of W/S and
A/D therefore commands *one* translation axis and clears the other two: `W` is "drive forward", not "drive forward
and keep whatever else was running". This makes Q/E followed by W a true straight command instead of an unintended
circle, and A followed by W a true straight command instead of a permanent diagonal. That second half was missing
until the `A`/`D` and `W`/`S` branches cleared each other's axis: because both translation axes are commanded at the
same `keyboardLinearSpeed`, a latched pair is always `(v, v)` and the robot travelled at exactly 45 degrees to the
direction it pointed for the rest of the session, with only `space` or `x` able to clear it.
`KeyboardForwardAfterACrabAndAYawTravelsAlongTheChassisHeading` in `test_controller_lifecycle.cpp` pins this by
typing the keys into a pseudo-terminal, so the production key handler runs unmodified. Q/E still add yaw to the
current translation, so W+Q/W+E provides mixed-radius motion. `space` clears every axis.
The Tasks backend supplies chassis linear/angular and wheel-rate
feed-forward, and synchronizes MuJoCo encoder/body-sensor measurements into the controller MBC before each cycle. It
then emits measured-state-relative wheel `q`/`alpha` references, matching the position/velocity signals consumed by
mc_mujoco. This keeps the chassis target and actual motion coupled instead of integrating an untracked absolute target.
`keyboardYawFeedbackGain` closes the remaining heading error while a pure Q/E command is held. The two empirical
wheel-rate scale factors this profile used to need (`keyboardYawScale`, `keyboardMixedDriveScale`) are gone: they
compensated an open-loop position command, whereas the QP now tracks the requested rate directly. Their weights are
`rollingRateWeight` and `steeringRateWeight`; a rate row's coefficient carries the prediction step, so its weight
enters the objective multiplied by `dt^2` and the checked-in 4e7 is an equivalent acceleration-task weight of 1000.
The controller logs `RollingContact_commandedTwist`, `RollingContact_<wheel>_rollingRateRef`,
`RollingContact_<wheel>_steeringRateRef`, `RollingContact_keyboard_yaw_error` and
`RollingContact_keyboard_yaw_correction`; use them together with the physical `FloatingBase_orientation_*`
quaternion rather than treating a changing yaw reference as proof that the chassis rotated. The pure-yaw checker is:

```sh
build/utils/mc_bin_to_log KEYBOARD_LOG /tmp/keyboard.csv
python3 rolling-contact-report/scripts/check-keyboard-yaw.py /tmp/keyboard.csv \
  --direction q --expected-rate 0.5 --rate-tolerance 0.1
```

Run it once with `--direction q` and once with `--direction e`; it verifies the measured +/-0.5 rad/s sign and rate,
settled yaw error, solver/diagnostic status, residuals, and zero error after the `x` stop.
For mixed-radius validation, convert a run containing W+Q and W+E intervals and run
`python3 rolling-contact-report/scripts/check-keyboard-radius.py /tmp/keyboard-qe-radius.csv`; it checks physical
body-frame forward speed, opposite Q/E yaw signs, the approximately 0.6 m measured radius, rolling modes, and the
absence of contact fallback. Body-sensor linear velocities are converted from MuJoCo's inertial frame to RBDyn's
body-frame free-joint convention before residuals and radius are evaluated.
The Ranger model keeps the URDF chassis link, chassis geometry, inertial origin,
MuJoCo free-joint, and `FloatingBase` sensor at one coincident origin. The
`RollingContact_floating_base_chassis_error` log entry is a runtime check of
that contract and should remain at numerical zero; MuJoCo exposes a
visual-only `chassis_frame` site at the same point.
When the helper reports that capture has stopped, the controller forces all three references to zero, so `x` is a safe
stop even though the helper retains its last cached command internally. Close the MuJoCo window or press Ctrl-C to
terminate the simulator itself. The `RollingContact::GetKeyboardStatus` datastore call and the
`RollingContact_reference_*_speed`, base-target, task-reference, and tracking-error log entries expose the full command
handoff for GUI/log integration.

The `x` stop is completed by the next controller poll, which joins the RoboticsUtils worker from the controller thread
and restores the terminal safely (the callback itself never attempts to join its own worker thread).

For a headless terminal smoke test, use `--without-visualization --without-mc-rtc-gui -s`, type `w`, `a`, `q`, `x`,
then press Ctrl-C. A recorded log should show nonzero `RollingContact_front_left_target` and steering targets while
the keys are active.

## Ramp terrain

`src/mc_robots/ranger_mini_v3_description/mujoco/ramp_terrain.xml` is flat ground with four driveable ramp lanes -
5, 10, 15 and 20 degrees at `y` = 0, 4, 8 and 12, plus a flat reference lane at `y` = -4. Every lane has its toe at
`x` = 0.6 m, its crest at `x` = 2.1 m, its plateau end at `x` = 3.1 m and its toe-out at `x` = 4.6 m.

It is a `ground` model, not part of the robot: `mc_control::MCController` always loads `env/ground`
(`MCController.cpp:96`) and mc_mujoco resolves one MuJoCo model per mc_rtc robot from `<module>.yaml`
(`mj_sim.cpp:110-127`), so `ground.yaml` is the one hook that changes the world without touching the robot
description or the controller. A "scene XML that includes the robot" is **not** possible here: mc_mujoco expands
MuJoCo `<include>` only inside `asset`, `contact`, `actuator` and `sensor`
(`mj_utils_merge_mujoco_models.cpp:344-363`), never inside `worldbody`.

Install the mapping once, then select the terrain per configuration:

```sh
cp rolling-contact-report/mujoco/ground.yaml ~/local/share/mc_mujoco/
/home/yuquan/local/bin/mc_mujoco -s -f rolling-contact-report/config/mc_rtc-ranger-ramps-mujoco.yaml
```

The selection is one line. mc_mujoco prefers a section of `ground.yaml` keyed by the first element of `MainRobot`
over the top-level `xmlModelPath` (`mj_sim.cpp:233-248`), so `MainRobot: RangerMiniV3Robot` keeps the stock flat
plane while `MainRobot: RangerMiniV3Ramps` - an alias for exactly the same robot - selects the ramps. Installing
the mapping therefore changes nothing for the existing configurations, the suite, or any other robot.

`scripts/run-mujoco-suite.py` does not rely on that: it writes and **removes** `ground.yaml` in mc_mujoco's model
folder around every case, from `Case.terrain`, and restores the flat mapping in a `finally`, so a ramp case cannot
leak into the next run.

## Terrain validation

```sh
python3 rolling-contact-report/scripts/run-ramp-validation.py \
  --runner /tmp/rolling-contact-cpu-mujoco/runner-build/rolling_contact_mujoco_runner \
  --build build --artifact-dir /tmp/rolling-contact-ramp-validation
```

Two sections, both driven through the deterministic runner:

- **`slope`** - a uniformly tilted world at 0, 5, 10, 15, 20, 25 and 30 degrees with the controller told the matching
  normal, so assumption A4 (`as:one-plane`) holds exactly, crossed with ten commands covering pure forward, pure
  lateral (`crab` at a steering angle of pi/2), pure yaw and four forward-by-yaw combinations.
- **`transition`** - the ramp terrain, forward at 0.3 m/s for 20 s, which is long enough to cover toe, crest, plateau
  and toe-out. Each angle is run twice: once with the controller told the flat normal (what it can actually know) and
  once with the incline's normal pre-loaded, which separates "the constant is wrong" from "the terrain is not one
  plane".

Both actuation regimes are reported. `torque` drives the MuJoCo motors from the QP's joint torques; `pd` is
mc_mujoco's own default, where mc_mujoco PD-tracks the controller's `q`/`alpha` output through
`ranger_mini_v3-gains.txt`, and it is the regime `mc_mujoco -f <config>` uses.

`scripts/run-trajectory-terrain.py` does the same for the `RangerTrajectory` FSM sample, which the deterministic
runner cannot drive (it requires fifteen `RollingContact::Get*` datastore calls that the FSM sample does not
register), so it runs mc_mujoco headless and measures the three curves from the binary log. It reports the deviation
of the **simulated** chassis (`ff_real`) as well as of the control robot (`ChassisCurve_surfacePose`): the sample's
own configuration declares no `robots:` block and `fsm::Controller` does not add `env/ground`
(`fsm/Controller.cpp:29` uses the robot-vector overload), so without the `robots:` block that
`config/mc_rtc-ranger-trajectory-{flat,ramps}-mujoco.yaml` add at the top level the chassis is in free fall for the
whole run while the control robot still reports millimetric tracking.

## Relevant files

- `config/mc_rtc-differential*.yaml`: differential-drive Tasks/TVM/mode examples;
- `config/mc_rtc-four-steering*.yaml`: Ranger Mini V3 deterministic Tasks/TVM examples;
- `config/mc_rtc-ranger-mini-v3-keyboard.yaml`: interactive keyboard example;
- `config/mc_rtc-ranger-ramps-mujoco.yaml`: the same keyboard profile on the ramp terrain;
- `config/mc_rtc-ranger-trajectory-{flat,ramps}-mujoco.yaml`: the RangerTrajectory FSM on flat ground and on the ramps;
- `mujoco/ranger_mini_v3.yaml`, `mujoco/ground.yaml`: mc_mujoco model mappings;
- `scripts/build-cpu-mujoco-runner.sh`: pinned CPU build/install helper;
- `scripts/run-mujoco-suite.py`: deterministic headless scenario driver and report checker;
- `scripts/run-ramp-validation.py`: slope and slope-transition sweep; and
- `scripts/run-trajectory-terrain.py`: RangerTrajectory curve tracking under mc_mujoco.
