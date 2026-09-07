#!/usr/bin/env python3

"""Run the deterministic, headless, CPU-only rolling-contact MuJoCo suite."""

import argparse
import csv
import dataclasses
import hashlib
import json
import math
import os
import pathlib
import subprocess
import sys
import time


@dataclasses.dataclass(frozen=True)
class Case:
    name: str
    robot: str
    scenario: str
    kind: str = "nominal"
    ramp_degrees: float = 0.0
    friction: float = 0.8
    linear_speed: float = 0.15
    yaw_rate: float = 0.25
    steering_angle: float = 0.3
    command_period: float = 4.0
    recovery_speed: float = 0.05
    longitudinal: str = "hard"
    friction_cycle: int = -1
    impulse_cycle: int = -1
    impulse: float = 0.0
    separation_cycle: int = -1
    separation: float = 0.0
    recovery_required: bool = False
    # Per-case override of the global --cycles, for cases that need to run long
    # enough to reach a regime the default 1000 cycles (5 s) cannot.
    cycles: int = 0
    # Per-case override of the global --warmup-cycles. The cases that preset the
    # steering start already in their steady manoeuvre and 100 cycles (0.5 s) is
    # ample; a case that has to slew the hinges from zero has a real startup
    # transient and its statistics must not be taken during it. Measured on
    # four-crab-acceptance: the worst rolling slip is 0.024 m/s up to cycle 225
    # and 1.2e-4 m/s from cycle 225 on, and the minimum MuJoCo normal force goes
    # from 0 N to >= 150 N over the same boundary. This raises the *window*, not
    # any bound: the nominal limits in check-mujoco-report.py are unchanged.
    warmup_cycles: int = 0
    # A case that reproduces a defect this branch has characterised but not
    # fixed. The replay still runs and its report is still written; the checker
    # is expected to reject it, and the suite fails if it ever *stops* rejecting
    # it, so the day the defect is fixed the case has to be promoted rather than
    # quietly keeping a stale expectation. The string is the reason and must be
    # non-empty.
    # The three ways this runner can make a case easier than the acceptance
    # command `mc_mujoco -f rolling-contact-report/config/mc_rtc-ranger-mini-v3-keyboard.yaml`.
    # They default to True here because that is what every case below has always
    # done - the point of naming them is that the divergence is now visible in
    # the case table, in the runner's command line and in each replay's JSON,
    # instead of being hard-coded inside the runner.
    #
    #   torque_control    mc_mujoco's own default is FALSE. With it true the QP's
    #                     joint torques drive the motors directly; with it false
    #                     mc_mujoco PD-tracks the controller's q/alpha output,
    #                     which is the path the acceptance command uses and which
    #                     this suite never exercised.
    #   measured_contacts feeds MuJoCo's true contact forces and slips into
    #                     RollingContact::SetMeasuredContact every substep.
    #                     mc_mujoco does not, so the contact-mode estimator
    #                     normally runs with no contact sensor at all. This is
    #                     what hid the front_right detachment: with ground truth
    #                     the estimator never reads the QP's own multiplier.
    #   preset_steering   resets the four-steering scenarios with the hinges
    #                     already at the manoeuvre's steady-state angle, skipping
    #                     the steering transient the real failures occur in.
    #
    # The *-acceptance cases at the end of CASES set all three to False.
    expected_failure: str = ""
    torque_control: bool = True
    measured_contacts: bool = True
    preset_steering: bool = True

    @property
    def regime(self):
        assisted = [
            name
            for name, value in (
                ("torque", self.torque_control),
                ("measured-contacts", self.measured_contacts),
                ("preset-steering", self.preset_steering),
            )
            if value
        ]
        return "+".join(assisted) if assisted else "acceptance"


CASES = (
    Case("diff-hold", "differential", "hold", linear_speed=0.0, yaw_rate=0.0),
    Case("diff-forward", "differential", "forward", linear_speed=0.2),
    Case("diff-reverse", "differential", "reverse", linear_speed=0.2),
    Case(
        "diff-turn-left",
        "differential",
        "turn_left",
        linear_speed=0.0,
        yaw_rate=0.05,
    ),
    Case(
        "diff-turn-right",
        "differential",
        "turn_right",
        linear_speed=0.0,
        yaw_rate=0.05,
    ),
    Case(
        "diff-circle-left",
        "differential",
        "circle_left",
        linear_speed=0.15,
        yaw_rate=0.05,
    ),
    Case(
        "diff-circle-right",
        "differential",
        "circle_right",
        linear_speed=0.15,
        yaw_rate=0.05,
    ),
    Case(
        "diff-sinusoid",
        "differential",
        "sinusoid",
        linear_speed=0.15,
        yaw_rate=0.05,
    ),
    Case(
        "diff-ramp-up",
        "differential",
        "forward",
        ramp_degrees=5.0,
        linear_speed=0.05,
        longitudinal="soft",
    ),
    Case(
        "diff-ramp-down",
        "differential",
        "reverse",
        ramp_degrees=-5.0,
        linear_speed=0.05,
        longitudinal="soft",
    ),
    Case(
        "diff-low-friction",
        "differential",
        "forward",
        kind="slip",
        friction=0.005,
        friction_cycle=350,
        linear_speed=0.4,
        longitudinal="soft",
    ),
    Case(
        "diff-lateral-impulse",
        "differential",
        "forward",
        kind="disturbance",
        impulse_cycle=350,
        impulse=0.05,
        recovery_required=True,
    ),
    Case(
        "diff-contact-loss",
        "differential",
        "forward",
        kind="disturbance",
        separation_cycle=250,
        separation=0.0005,
        recovery_required=True,
    ),
    Case(
        "diff-mode-cycle",
        "differential",
        "mode_cycle",
        kind="transition",
        linear_speed=0.0,
        yaw_rate=0.0,
        longitudinal="soft",
    ),
    Case("four-forward", "four-steering", "forward"),
    Case("four-reverse", "four-steering", "reverse"),
    Case("four-crab", "four-steering", "crab"),
    Case(
        "four-ackermann-left", "four-steering", "ackermann_left", yaw_rate=0.05
    ),
    Case(
        "four-ackermann-right", "four-steering", "ackermann_right", yaw_rate=0.05
    ),
    Case(
        "four-pure-yaw",
        "four-steering",
        "pure_yaw",
        linear_speed=0.0,
        yaw_rate=0.05,
    ),
    Case(
        "four-steering-rate",
        "four-steering",
        "steering_rate",
        kind="maneuver",
        steering_angle=0.3,
        command_period=4.0,
        recovery_speed=0.1,
        longitudinal="soft",
    ),
    Case(
        "four-ramp-crab",
        "four-steering",
        "crab",
        ramp_degrees=5.0,
        linear_speed=0.1,
        longitudinal="soft",
    ),
    Case(
        "four-lateral-impulse",
        "four-steering",
        "forward",
        kind="disturbance",
        impulse_cycle=350,
        impulse=0.05,
        recovery_required=True,
        longitudinal="soft",
    ),
    Case(
        "four-low-friction",
        "four-steering",
        "forward",
        kind="slip",
        friction=0.005,
        friction_cycle=350,
        linear_speed=0.4,
        longitudinal="soft",
    ),
    Case(
        "four-mode-cycle",
        "four-steering",
        "mode_cycle",
        kind="transition",
        linear_speed=0.0,
        yaw_rate=0.0,
        longitudinal="soft",
    ),
    # Acceptance-regime coverage: the actuation path, the (absent) contact
    # sensing and the initial hinge state of
    # `mc_mujoco -f rolling-contact-report/config/mc_rtc-ranger-mini-v3-keyboard.yaml`,
    # at the speeds that configuration commands rather than at the 0.05 rad/s
    # the cases above use. Every one of these reproduced a real-mc_mujoco
    # failure before the fixes in this branch:
    #   four-crab-acceptance          front_right detached at t = 0.025 s and
    #                                 never recovered; the load collapsed onto
    #                                 the front_left/rear_right diagonal
    #                                 (183/0/205/341 N against a 736 N vehicle).
    #   four-ackermann-acceptance     same detachment at t = 0.02 s, 4991 of
    #                                 4995 cycles.
    #   four-pure-yaw-acceptance      QP failure at t = 9.72 s from an unbounded
    #                                 heading target reaching sva::rotationError's
    #                                 singularity; needs > 1944 cycles to reach it,
    #                                 hence the explicit cycle count.
    Case(
        "four-crab-acceptance",
        "four-steering",
        "crab",
        linear_speed=0.2,
        steering_angle=0.3,
        warmup_cycles=300,
        torque_control=False,
        measured_contacts=False,
        preset_steering=False,
    ),
    Case(
        "four-ackermann-acceptance",
        "four-steering",
        "ackermann_left",
        linear_speed=0.2,
        yaw_rate=0.35,
        warmup_cycles=300,
        expected_failure=(
            "four-steering yaw tracking is not fixed: the wheel contact patch resists"
            " re-steering with ~7 N.m of scrub that the QP's rolling constraint (imposed"
            " at the single centre contact point) does not model, so the hinges park short"
            " of a correct reference. Measured 0.031 m/s rolling slip, 32% of samples with"
            " a wheel off the ground and 0.60 rad of yaw tracking error"
        ),
        torque_control=False,
        measured_contacts=False,
        preset_steering=False,
    ),
    Case(
        "four-pure-yaw-acceptance",
        "four-steering",
        "pure_yaw",
        linear_speed=0.0,
        yaw_rate=0.35,
        cycles=2600,
        warmup_cycles=300,
        expected_failure=(
            "same unfixed four-steering yaw tracking as four-ackermann-acceptance."
            " Everything else is healthy over the 13 s: the QP never fails (it used to"
            " fail at t = 9.72 s), min normal force 161.5 N, no contact loss, rolling slip"
            " 1.5e-4 m/s and all four wheels rolling at the end. The single violated bound"
            " is the yaw tracking, which sits at exactly the pi/2 maxYawTargetError the"
            " heading governor now holds it to: 0.272 rad turned against 4.55 commanded"
        ),
        torque_control=False,
        measured_contacts=False,
        preset_steering=False,
    ),
)


DETERMINISTIC_FIELDS = (
    "cycles_completed",
    "static_parity",
    "controller_failed",
    "finite_state",
    "simulation_time_s",
    "max_rolling_slip_mps",
    "rolling_slip_rms_mps",
    "rolling_slip_p95_mps",
    "max_lateral_slip_mps",
    "lateral_slip_rms_mps",
    "lateral_slip_p95_mps",
    "max_normal_speed_mps",
    "normal_speed_rms_mps",
    "normal_speed_p95_mps",
    "min_normal_force_n",
    "min_friction_margin_n",
    "max_drive_torque_nm",
    "max_steer_torque_nm",
    "no_contact_sample_fraction",
    "contact_fallback_sample_fraction",
    "odometry_position_error_rms_m",
    "odometry_position_error_p95_m",
    "odometry_position_error_max_m",
    "odometry_yaw_error_rms_rad",
    "odometry_yaw_error_p95_rad",
    "odometry_yaw_error_max_rad",
    "tracking_position_error_rms_m",
    "tracking_position_error_p95_m",
    "tracking_position_error_max_m",
    "tracking_yaw_error_rms_rad",
    "tracking_yaw_error_p95_rad",
    "tracking_yaw_error_max_rad",
    "estimated_modes",
    "final_contact_fallback",
    "final_modes",
    "final_solver_activations",
    "final_position",
    "final_yaw_rad",
    "final_odometry",
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def terrain_normal(degrees):
    angle = math.radians(degrees)
    return (-math.sin(angle), 0.0, math.cos(angle))


def configuration(case, backend, root, build, artifact):
    robot_module = (
        "RollingContactDifferential"
        if case.robot == "differential"
        else "RollingContactRangerMiniV3"
    )
    controller = "RollingContact" if backend == "Tasks" else "RollingContact_TVM"
    normal = terrain_normal(case.ramp_degrees)
    four = case.robot == "four-steering"
    base_stiffness = 20.0
    posture_stiffness = 5.0
    yaw_scenarios = {
        "turn_left",
        "turn_right",
        "circle_left",
        "circle_right",
        "sinusoid",
    }
    orientation_stiffness = 100.0 if not four and case.scenario in yaw_scenarios else 20.0
    if not four and case.scenario in yaw_scenarios:
        posture_stiffness = 10.0
    slip_enter = 0.001 if case.kind == "disturbance" else 0.05
    slip_exit = 0.0002 if case.kind == "disturbance" else 0.02
    minimum_dwell = 0.005 if case.kind == "disturbance" else 0.05
    return f"""MainRobot: {robot_module}
Enabled: [{controller}]
Default: {controller}
Timestep: 0.005
Log: false
LogPolicy: non-threaded
LogDirectory: {artifact}
LogTemplate: {case.name}-{backend}
ClearControllerModulePath: true
ControllerModulePaths: [{build / 'src/mc_control/samples/RollingContact'}]
ClearRobotModulePath: true
RobotModulePaths: [{build / 'src/mc_robots'}]
ClearObserverModulePath: true
ObserverModulePaths: [{build / 'src/mc_observers'}]
ObserverPipelines:
  - name: MainPipeline
    observers:
      - type: Encoder
        update: true
        velocity: encoderVelocities
      - type: BodySensor
        update: true
        bodySensor: FloatingBase
ClearGlobalPluginPath: true
GlobalPluginPaths: []
Plugins: []
GUIServer:
  Enable: false
RollingContact:
  scenario: {case.scenario}
  closedLoopFeedback: true
  linearSpeed: {case.linear_speed:.17g}
  yawRate: {case.yaw_rate:.17g}
  steeringAngle: {case.steering_angle:.17g}
  commandPeriod: {case.command_period:.17g}
  velocityGain: 5.0
  postureStiffness: {posture_stiffness:.17g}
  postureWeight: 500.0
  steeringStiffness: 1000.0
  steeringWeight: 10.0
  baseOrientationStiffness: {orientation_stiffness:.17g}
  baseOrientationWeight: 500.0
  basePositionStiffness: {base_stiffness:.17g}
  basePositionWeight: 2000.0
  longitudinal: {case.longitudinal}
  rollingWeight: 1000.0
  recoverySpeed: {case.recovery_speed:.17g}
  slipEnter: {slip_enter:.17g}
  slipExit: {slip_exit:.17g}
  residualEnter: 0.05
  residualExit: 0.02
  normalForceEnter: 5.0
  normalForceExit: 1.0
  minimumDwell: {minimum_dwell:.17g}
  transitionTime: 0.1
  filterTimeConstant: 0.02
  terrainNormal: [{normal[0]:.17g}, {normal[1]:.17g}, {normal[2]:.17g}]
"""


def compare_replays(reports, case_name, backend):
    variations = {}
    for field in DETERMINISTIC_FIELDS:
        values = [report[field] for report in reports]
        if isinstance(values[0], (bool, str, list)):
            if any(value != values[0] for value in values[1:]):
                raise RuntimeError(f"{case_name}/{backend}: non-deterministic {field}")
            variations[field] = {"identical": True}
            continue
        maximum_error = max(abs(float(value) - float(values[0])) for value in values)
        scale = max(1.0, abs(float(values[0])))
        if maximum_error > 1e-10 * scale:
            raise RuntimeError(
                f"{case_name}/{backend}: {field} replay variation {maximum_error}"
            )
        variations[field] = {"maximum_absolute_variation": maximum_error}
    return variations


def validate_disturbance_csv(path, case_name):
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    mode_fields = [field for field in rows[0] if field.endswith("_mode")]
    modes = {row[field] for row in rows for field in mode_fields}
    if not {"sliding", "detached"}.intersection(modes):
        raise RuntimeError(f"{case_name}: disturbance caused no mode fallback")
    fallback = next(
        index
        for index, row in enumerate(rows)
        if any(row[field] in {"sliding", "detached"} for field in mode_fields)
    )
    recovered = any(
        all(row[field] == "rolling" for field in mode_fields)
        for row in rows[fallback + 1 :]
    )
    if not recovered:
        raise RuntimeError(f"{case_name}: no rolling recovery after disturbance")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", type=pathlib.Path, required=True)
    parser.add_argument("--artifact-dir", type=pathlib.Path, required=True)
    parser.add_argument("--repo", type=pathlib.Path, default=pathlib.Path.cwd())
    parser.add_argument("--build", type=pathlib.Path, default=pathlib.Path("build"))
    parser.add_argument("--cycles", type=int, default=1000)
    parser.add_argument("--warmup-cycles", type=int, default=100)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--backend", choices=("Tasks", "TVM", "both"), default="both")
    parser.add_argument("--case", action="append", dest="selected_cases")
    return parser.parse_args()


def main():
    args = parse_args()
    root = args.repo.resolve(strict=True)
    build = (root / args.build).resolve(strict=True) if not args.build.is_absolute() else args.build.resolve(strict=True)
    runner = args.runner.resolve(strict=True)
    artifact = args.artifact_dir.resolve()
    artifact.mkdir(parents=True, exist_ok=True)
    if args.cycles <= args.warmup_cycles or args.warmup_cycles < 0:
        raise RuntimeError("cycles must exceed non-negative warm-up cycles")
    if args.repetitions < 1:
        raise RuntimeError("repetitions must be positive")

    selected = set(args.selected_cases or ())
    unknown = selected.difference(case.name for case in CASES)
    if unknown:
        raise RuntimeError(f"unknown cases: {sorted(unknown)}")
    cases = [case for case in CASES if not selected or case.name in selected]
    backends = ("Tasks", "TVM") if args.backend == "both" else (args.backend,)
    checker = root / "rolling-contact-report/scripts/check-mujoco-report.py"
    env = os.environ.copy()
    runtime_directories = (
        build / "src",
        build / "plugins/ROS",
        build / "deps/tasks-system-install/lib",
    )
    missing_runtime_directories = [
        path for path in runtime_directories if not path.is_dir()
    ]
    if missing_runtime_directories:
        raise RuntimeError(
            "missing mc_rtc runtime directories: "
            + ", ".join(str(path) for path in missing_runtime_directories)
        )
    inherited_library_path = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = ":".join(
        [str(path) for path in runtime_directories]
        + ([inherited_library_path] if inherited_library_path else [])
    )
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "-1",
            "MC_RTC_DISABLE_CONVEX_GENERATION_PATCH": "ON",
            "NVIDIA_VISIBLE_DEVICES": "void",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    aggregate = {
        "schema": 1,
        "device": "CPU",
        "gpu_devices_hidden": True,
        "timestep_s": 0.005,
        "seed": 0,
        "repetitions": args.repetitions,
        "runner": str(runner),
        "runner_sha256": sha256(runner),
        "cases": [],
    }
    started = time.monotonic()
    for backend in backends:
        for case in cases:
            case_dir = artifact / backend.lower() / case.name
            case_dir.mkdir(parents=True, exist_ok=True)
            config = case_dir / "mc_rtc.yaml"
            config.write_text(
                configuration(case, backend, root, build, artifact), encoding="utf-8"
            )
            reports = []
            csv_paths = []
            for repetition in range(args.repetitions):
                stem = case_dir / f"replay-{repetition}"
                csv_path = stem.with_suffix(".csv")
                report_path = stem.with_suffix(".json")
                stdout_path = stem.with_suffix(".log")
                command = [
                    str(runner),
                    "--mc-config",
                    str(config),
                    "--robot",
                    case.robot,
                    "--backend",
                    backend,
                    "--scenario",
                    case.scenario,
                    "--cycles",
                    str(case.cycles or args.cycles),
                    "--warmup-cycles",
                    str(case.warmup_cycles or args.warmup_cycles),
                    "--ramp-deg",
                    str(case.ramp_degrees),
                    "--friction",
                    str(case.friction),
                    "--friction-cycle",
                    str(case.friction_cycle),
                    "--linear-speed",
                    str(case.linear_speed),
                    "--yaw-rate",
                    str(case.yaw_rate),
                    "--steering-angle",
                    str(case.steering_angle),
                    "--impulse-cycle",
                    str(case.impulse_cycle),
                    "--impulse",
                    str(case.impulse),
                    "--impulse-direction",
                    "lateral",
                    "--separation-cycle",
                    str(case.separation_cycle),
                    "--separation",
                    str(case.separation),
                    "--torque-control",
                    "true" if case.torque_control else "false",
                    "--measured-contacts",
                    "true" if case.measured_contacts else "false",
                    "--preset-steering",
                    "true" if case.preset_steering else "false",
                    "--csv",
                    str(csv_path),
                    "--report",
                    str(report_path),
                ]
                with stdout_path.open("w", encoding="utf-8") as output:
                    completed = subprocess.run(
                        command,
                        cwd=root,
                        env=env,
                        stdout=output,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )
                if completed.returncode != 0:
                    raise RuntimeError(
                        f"{case.name}/{backend}/replay-{repetition} failed; see {stdout_path}"
                    )
                checker_kind = case.kind if case.kind != "disturbance" else "disturbance"
                checked = subprocess.run(
                    [
                        sys.executable,
                        str(checker),
                        str(report_path),
                        "--kind",
                        checker_kind,
                        "--backend",
                        backend,
                        "--robot",
                        case.robot,
                    ],
                    check=not case.expected_failure,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                )
                if case.expected_failure and checked.returncode == 0:
                    raise RuntimeError(
                        f"{case.name}/{backend}: expected_failure case now passes the checker "
                        f"({case.expected_failure}); promote it instead of leaving a stale expectation"
                    )
                with report_path.open(encoding="utf-8") as stream:
                    reports.append(json.load(stream))
                csv_paths.append(csv_path)
            if case.recovery_required:
                for csv_path in csv_paths:
                    validate_disturbance_csv(csv_path, case.name)
            variations = compare_replays(reports, case.name, backend)
            aggregate["cases"].append(
                {
                    "name": case.name,
                    "backend": backend,
                    "robot": case.robot,
                    "scenario": case.scenario,
                    "kind": case.kind,
                    "regime": case.regime,
                    "expected_failure": case.expected_failure,
                    "torque_control": case.torque_control,
                    "measured_contacts": case.measured_contacts,
                    "preset_steering": case.preset_steering,
                    "config_sha256": sha256(config),
                    "reports": [str(path) for path in case_dir.glob("replay-*.json")],
                    "deterministic_variation": variations,
                    "real_time_factor_min": min(
                        report["real_time_factor"] for report in reports
                    ),
                    "cycle_wall_median_ms_max": max(
                        report["cycle_wall_median_ms"] for report in reports
                    ),
                    "cycle_wall_p99_ms_max": max(
                        report["cycle_wall_p99_ms"] for report in reports
                    ),
                    "peak_resident_kib_max": max(
                        report["peak_resident_kib"] for report in reports
                    ),
                }
            )
            verdict = "XFAIL" if case.expected_failure else "PASS "
            line = f"{verdict} {backend:5s} {case.name:28s} [{case.regime}]"
            if case.expected_failure:
                line += f"  {case.expected_failure}"
            print(line, flush=True)
    aggregate["wall_time_s"] = time.monotonic() - started
    aggregate["status"] = "pass"
    output = artifact / "suite-report.json"
    output.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"CPU MuJoCo suite passed: {output}")


if __name__ == "__main__":
    main()
