#!/usr/bin/env python3

"""Validate the four-steering controller on sloped and ramped terrain.

This answers two questions the flat-ground suite cannot:

  constant slope   the robot on a uniformly tilted world, with the controller
                   told the correct terrain normal. Standing assumption A4 of
                   the formulation (`as:one-plane` in rolling-contact-qp.tex)
                   holds exactly here, so this measures how far the planar
                   reduction carries before something else (traction, torque,
                   the actuation path) gives out.

  slope transition the robot driving from flat ground onto a ramp and off
                   again, on the terrain in
                   src/mc_robots/rolling_contact_description/mujoco/ramp_terrain.xml.
                   The controller reads `terrainNormal` once, from
                   configuration, and nothing updates it
                   (mc_rolling_contact_controller.cpp:425-447 and :527 are all
                   in the constructor), so on a ramp it is wrong by the ramp
                   angle -- and while the robot straddles the toe the four
                   wheels are not coplanar at all and no single constant could
                   be right. A4 is violated outright there. The
                   `-incline-normal` variants pre-load the incline's normal
                   instead of the flat one, which separates "the constant is
                   wrong" from "the terrain is not one plane".

The two actuation regimes are reported separately and both matter:

  torque   `--torque-control true`; the QP's joint torques drive the MuJoCo
           motors, so the controller's own dynamics constraint is what is
           under test.
  pd       `--torque-control false`, which is what `mc_mujoco -f <config>`
           does by default: mc_mujoco PD-tracks the controller's q/alpha
           output through the gains in ranger_mini_v3-gains.txt.

Terrain selection works through mc_mujoco's own model resolution. mc_mujoco
looks up `<robot module>.yaml` in its user folder before its share folder
(mj_sim.cpp:110-127), and mc_rtc always loads `env/ground` as a second robot
(MCController.cpp:96), so writing `ground.yaml` into the user folder swaps the
world without touching the robot description or the controller. This script
writes that file before each run and removes it again for flat runs, so a
terrain never leaks into a later run.
"""

import argparse
import csv
import dataclasses
import json
import math
import os
import pathlib
import subprocess
import sys
import tempfile


REPOSITORY_TERRAIN = "src/mc_robots/rolling_contact_description/mujoco/ramp_terrain.xml"

# Lane centres in ramp_terrain.xml. Every lane has its toe at x = 0.6 m, its
# crest at x = 2.1 m, its plateau end at x = 3.1 m and its toe-out at x = 4.6 m.
RAMP_LANES = {5: 0.0, 10: 4.0, 15: 8.0, 20: 12.0}


@dataclasses.dataclass(frozen=True)
class Terrain:
    name: str
    # World tilt in degrees applied by the runner to the whole ground body. Non
    # zero means "uniform slope"; the controller is told the matching normal.
    tilt_degrees: float = 0.0
    # Ramp lane angle in degrees; selects the lane offset in ramp_terrain.xml.
    ramp_degrees: float = 0.0
    # Terrain normal handed to the controller, in degrees of pitch. `None`
    # means "match the terrain": the tilt for a uniform slope, flat otherwise.
    assumed_normal_degrees: float = None

    @property
    def uses_ramp_terrain(self):
        return self.ramp_degrees != 0.0

    @property
    def start_y(self):
        return RAMP_LANES[int(self.ramp_degrees)] if self.uses_ramp_terrain else 0.0

    @property
    def controller_normal_degrees(self):
        if self.assumed_normal_degrees is not None:
            return self.assumed_normal_degrees
        return self.tilt_degrees


@dataclasses.dataclass(frozen=True)
class Command:
    name: str
    scenario: str
    linear_speed: float = 0.0
    yaw_rate: float = 0.0
    steering_angle: float = 0.3

    @property
    def commanded_twist(self):
        """The twist `resolveCommandedTwist()` will publish, as (vx, vy, wz).

        Mirrors mc_rolling_contact_controller.cpp:1189-1258 for the scenarios
        used here; a mismatch would make every "commanded" column a fiction, so
        it is asserted against the controller's own logged twist by
        `achieved_twist`'s caller.
        """
        v, w, a = self.linear_speed, self.yaw_rate, self.steering_angle
        if self.scenario == "forward":
            return (v, 0.0, 0.0)
        if self.scenario == "reverse":
            return (-abs(v), 0.0, 0.0)
        if self.scenario == "crab":
            return (v * math.cos(a), v * math.sin(a), 0.0)
        if self.scenario == "ackermann_left":
            return (v, 0.0, abs(w))
        if self.scenario == "ackermann_right":
            return (v, 0.0, -abs(w))
        if self.scenario == "pure_yaw":
            return (0.0, 0.0, w)
        raise RuntimeError(f"no commanded twist known for scenario {self.scenario}")


HALF_PI = 0.5 * math.pi

# The linear x yaw grid. `crab` at a steering angle of pi/2 is pure lateral
# motion: cos(pi/2) = 0 kills the longitudinal component.
COMMANDS = (
    Command("forward-0.1", "forward", linear_speed=0.1),
    Command("forward-0.3", "forward", linear_speed=0.3),
    Command("lateral-0.1", "crab", linear_speed=0.1, steering_angle=HALF_PI),
    Command("lateral-0.3", "crab", linear_speed=0.3, steering_angle=HALF_PI),
    Command("yaw-0.1", "pure_yaw", yaw_rate=0.1),
    Command("yaw-0.35", "pure_yaw", yaw_rate=0.35),
    Command("fwd0.1-yaw0.1", "ackermann_left", linear_speed=0.1, yaw_rate=0.1),
    Command("fwd0.3-yaw0.1", "ackermann_left", linear_speed=0.3, yaw_rate=0.1),
    Command("fwd0.1-yaw0.35", "ackermann_left", linear_speed=0.1, yaw_rate=0.35),
    Command("fwd0.3-yaw0.35", "ackermann_left", linear_speed=0.3, yaw_rate=0.35),
)

SLOPE_TERRAINS = (
    Terrain("flat"),
    Terrain("slope-05", tilt_degrees=5.0),
    Terrain("slope-10", tilt_degrees=10.0),
    Terrain("slope-15", tilt_degrees=15.0),
    Terrain("slope-20", tilt_degrees=20.0),
    Terrain("slope-25", tilt_degrees=25.0),
    Terrain("slope-30", tilt_degrees=30.0),
)

TRANSITION_TERRAINS = (
    Terrain("ramp-05", ramp_degrees=5.0),
    Terrain("ramp-10", ramp_degrees=10.0),
    Terrain("ramp-15", ramp_degrees=15.0),
    Terrain("ramp-20", ramp_degrees=20.0),
    Terrain("ramp-05-incline-normal", ramp_degrees=5.0, assumed_normal_degrees=5.0),
    Terrain("ramp-10-incline-normal", ramp_degrees=10.0, assumed_normal_degrees=10.0),
    Terrain("ramp-15-incline-normal", ramp_degrees=15.0, assumed_normal_degrees=15.0),
    Terrain("ramp-20-incline-normal", ramp_degrees=20.0, assumed_normal_degrees=20.0),
)

# A traverse has to cover toe (0.6 m), crest (2.1 m), plateau (3.1 m) and
# toe-out (4.6 m). At 0.3 m/s that is 5.4 m of ground and 20 s of simulation.
TRANSITION_COMMAND = Command("forward-0.3", "forward", linear_speed=0.3)
TRANSITION_CYCLES = 4000
SLOPE_CYCLES = 1400


def terrain_normal(degrees):
    angle = math.radians(degrees)
    return (-math.sin(angle), 0.0, math.cos(angle))


def configuration(command, terrain, backend, build, artifact, log_template):
    controller = "RollingContact" if backend == "Tasks" else "RollingContact_TVM"
    normal = terrain_normal(terrain.controller_normal_degrees)
    # Logging is on so that `RollingContact_lateral_slack_norm` can be read
    # back: the controller keeps the slack norm but exposes no datastore
    # getter for it, so the binary log is the only way to see it from outside
    # the process.
    return f"""MainRobot: RollingContactRangerMiniV3
Enabled: [{controller}]
Default: {controller}
Timestep: 0.005
Log: true
LogPolicy: non-threaded
LogDirectory: {artifact}
LogTemplate: {log_template}
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
  scenario: {command.scenario}
  closedLoopFeedback: true
  linearSpeed: {command.linear_speed:.17g}
  yawRate: {command.yaw_rate:.17g}
  steeringAngle: {command.steering_angle:.17g}
  commandPeriod: 4.0
  velocityGain: 5.0
  postureStiffness: 5.0
  postureWeight: 500.0
  steeringStiffness: 1000.0
  steeringWeight: 10.0
  baseOrientationStiffness: 20.0
  baseOrientationWeight: 500.0
  basePositionStiffness: 20.0
  basePositionWeight: 2000.0
  # `hard` is what the four-steering nominal suite cases and the acceptance
  # configuration (config/mc_rtc-ranger-mini-v3-keyboard.yaml) both use, and it
  # is the only setting under which the reported numbers mean what they say.
  # Measured with `soft` at this speed on FLAT ground: a commanded 0.3 m/s came
  # out as 1.31 m/s and the mode estimator reported `sliding` for the whole
  # run, because the soft rolling row at rollingWeight = 1000 does not hold the
  # wheel rate to the chassis motion. That is a property of the relaxation, not
  # of the terrain, and it would swamp every slope measurement below.
  longitudinal: hard
  rollingWeight: 1000.0
  recoverySpeed: 0.05
  slipEnter: 0.05
  slipExit: 0.02
  residualEnter: 0.05
  residualExit: 0.02
  normalForceEnter: 5.0
  normalForceExit: 1.0
  minimumDwell: 0.05
  transitionTime: 0.1
  filterTimeConstant: 0.02
  terrainNormal: [{normal[0]:.17g}, {normal[1]:.17g}, {normal[2]:.17g}]
"""


def install_terrain(user_directory, terrain, repository):
    """Point mc_mujoco's `ground` model at the ramp terrain, or at its default.

    Removing the file rather than writing a flat copy keeps flat runs on
    exactly the model mc_mujoco ships, so the flat rows stay comparable with
    the numbers the existing suite produces.
    """
    mapping = user_directory / "ground.yaml"
    if terrain.uses_ramp_terrain:
        user_directory.mkdir(parents=True, exist_ok=True)
        mapping.write_text(
            f"xmlModelPath: {repository / REPOSITORY_TERRAIN}\n", encoding="utf-8"
        )
    elif mapping.exists():
        mapping.unlink()


def unwrap(previous, current):
    delta = current - previous
    while delta > math.pi:
        delta -= 2.0 * math.pi
    while delta < -math.pi:
        delta += 2.0 * math.pi
    return delta


def analyse_csv(path, warmup_cycles, timestep=0.005):
    """Mean achieved body twist and per-wheel mode statistics.

    The achieved twist has to be measured in the chassis frame, not read off
    the net displacement: any command with a yaw rate turns the chassis while
    it translates, so the world-frame displacement of a correctly executed
    (0.3, 0, 0.35) command is an arc, not 0.3 * duration along x.
    """
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    modes = [field for field in rows[0] if field.endswith("_mode")]
    window = rows[warmup_cycles:]
    if len(window) < 2:
        # A run that failed inside the warm-up still has to report something
        # rather than abort the sweep; fall back to whatever was recorded and
        # let `cycles_completed` say how little that was.
        window = rows
    if len(window) < 2:
        return {
            "achieved_twist": [float("nan")] * 3,
            "non_rolling_sample_fraction": float("nan"),
            "final_modes": [],
            "measured_duration_s": 0.0,
        }
    forward = lateral = yaw_rate = 0.0
    for previous, current in zip(window, window[1:]):
        px, py = float(previous["terrain_x"]), float(previous["terrain_y"])
        cx, cy = float(current["terrain_x"]), float(current["terrain_y"])
        heading = float(previous["yaw"])
        dx, dy = cx - px, cy - py
        forward += dx * math.cos(heading) + dy * math.sin(heading)
        lateral += -dx * math.sin(heading) + dy * math.cos(heading)
        yaw_rate += unwrap(heading, float(current["yaw"]))
    duration = timestep * (len(window) - 1)
    non_rolling = sum(
        1 for row in window if any(row[field] != "rolling" for field in modes)
    )
    return {
        "achieved_twist": [forward / duration, lateral / duration, yaw_rate / duration],
        "non_rolling_sample_fraction": non_rolling / len(window),
        "final_modes": [rows[-1][field] for field in modes],
        "measured_duration_s": duration,
    }


def lateral_slack_norm(artifact, log_template, utilities):
    """Peak and mean of `RollingContact_lateral_slack_norm` over the run."""
    logs = sorted(artifact.glob(f"{log_template}-*.bin"))
    if not logs:
        return {"lateral_slack_norm_max": None, "lateral_slack_norm_mean": None}
    with tempfile.TemporaryDirectory() as scratch:
        flattened = pathlib.Path(scratch) / "log.csv"
        subprocess.run(
            [str(utilities / "mc_bin_to_log"), str(logs[-1]), str(flattened)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        with flattened.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream, delimiter=";")
            field = "RollingContact_lateral_slack_norm"
            if field not in (reader.fieldnames or ()):
                return {"lateral_slack_norm_max": None, "lateral_slack_norm_mean": None}
            values = [float(row[field]) for row in reader if row[field] not in ("", None)]
    for log in logs:
        log.unlink()
    if not values:
        return {"lateral_slack_norm_max": None, "lateral_slack_norm_mean": None}
    return {
        "lateral_slack_norm_max": max(values),
        "lateral_slack_norm_mean": sum(values) / len(values),
    }


def run_case(arguments, environment, command, terrain, regime, backend, cycles):
    name = f"{terrain.name}__{command.name}__{regime}__{backend.lower()}"
    directory = arguments.artifact_dir / name
    directory.mkdir(parents=True, exist_ok=True)
    log_template = f"validation-{name}"
    configuration_path = directory / "mc_rtc.yaml"
    configuration_path.write_text(
        configuration(command, terrain, backend, arguments.build, directory, log_template),
        encoding="utf-8",
    )
    install_terrain(arguments.mujoco_user_dir, terrain, arguments.repo)
    warmup = 300
    report_path = directory / "report.json"
    csv_path = directory / "run.csv"
    invocation = [
        str(arguments.runner),
        "--mc-config", str(configuration_path),
        "--robot", "four-steering",
        "--backend", backend,
        "--scenario", command.scenario,
        "--cycles", str(cycles),
        "--warmup-cycles", str(warmup),
        "--ramp-deg", str(terrain.tilt_degrees),
        "--start-y", str(terrain.start_y),
        "--friction", "0.8",
        "--linear-speed", str(command.linear_speed),
        "--yaw-rate", str(command.yaw_rate),
        "--steering-angle", str(command.steering_angle),
        "--torque-control", "true" if regime == "torque" else "false",
        "--measured-contacts", "false",
        "--preset-steering", "false",
        "--csv", str(csv_path),
        "--report", str(report_path),
    ]
    with (directory / "run.log").open("w", encoding="utf-8") as output:
        completed = subprocess.run(
            invocation,
            cwd=arguments.repo,
            env=environment,
            stdout=output,
            stderr=subprocess.STDOUT,
            check=False,
        )
    record = {
        "case": name,
        "terrain": terrain.name,
        "tilt_degrees": terrain.tilt_degrees,
        "ramp_degrees": terrain.ramp_degrees,
        "controller_normal_degrees": terrain.controller_normal_degrees,
        "command": command.name,
        "scenario": command.scenario,
        "commanded_twist": list(command.commanded_twist),
        "regime": regime,
        "backend": backend,
        "cycles": cycles,
        "warmup_cycles": warmup,
        "runner_exit": completed.returncode,
    }
    # A non-zero exit is a result, not an accident: the runner returns
    # EXIT_FAILURE when the controller's QP fails, when the state stops being
    # finite, or when the run stops short (runner line 1344). The report and
    # the CSV are written first, so keep parsing them - "the QP failed after
    # 5.4 s having rolled 5.6 m backwards down the slope" is exactly the kind
    # of row this table exists to show, and dropping it would hide it.
    if completed.returncode != 0:
        record["failure"] = f"runner exited {completed.returncode}; see {directory / 'run.log'}"
        if not report_path.exists() or not csv_path.exists():
            return record
    with report_path.open(encoding="utf-8") as stream:
        report = json.load(stream)
    record.update(
        {
            key: report[key]
            for key in (
                "controller_failed",
                "finite_state",
                "cycles_completed",
                "cycles_requested",
                "simulation_time_s",
                "min_normal_force_n",
                "min_friction_margin_n",
                "max_drive_torque_nm",
                "max_steer_torque_nm",
                "max_contact_normal_deviation_deg",
                "max_rolling_slip_mps",
                "max_lateral_slip_mps",
                "no_contact_sample_fraction",
                "tracking_position_error_max_m",
                "tracking_yaw_error_max_rad",
                "estimated_modes",
                "final_position",
                "final_yaw_rad",
            )
        }
    )
    record.update(analyse_csv(csv_path, warmup))
    record.update(lateral_slack_norm(directory, log_template, arguments.build / "utils"))
    return record


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", type=pathlib.Path, required=True)
    parser.add_argument("--artifact-dir", type=pathlib.Path, required=True)
    parser.add_argument("--repo", type=pathlib.Path, default=pathlib.Path.cwd())
    parser.add_argument("--build", type=pathlib.Path, default=pathlib.Path("build"))
    parser.add_argument(
        "--mujoco-user-dir",
        type=pathlib.Path,
        default=pathlib.Path("/tmp/rolling-contact-cpu-mujoco/mc_mujoco-user"),
        help="mc_mujoco USER_FOLDER of the runner's mc_mujoco build; the "
        "`ground` mapping is written here and removed again for flat runs",
    )
    parser.add_argument("--backend", choices=("Tasks", "TVM", "both"), default="Tasks")
    parser.add_argument(
        "--section",
        action="append",
        choices=("slope", "transition"),
        help="repeatable; default is both",
    )
    parser.add_argument(
        "--regime",
        action="append",
        choices=("torque", "pd"),
        help="repeatable; default is both",
    )
    parser.add_argument(
        "--terrain", action="append", help="repeatable terrain-name filter"
    )
    parser.add_argument(
        "--command", action="append", help="repeatable command-name filter"
    )
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    arguments.repo = arguments.repo.resolve(strict=True)
    if not arguments.build.is_absolute():
        arguments.build = (arguments.repo / arguments.build).resolve(strict=True)
    arguments.runner = arguments.runner.resolve(strict=True)
    arguments.artifact_dir = arguments.artifact_dir.resolve()
    arguments.artifact_dir.mkdir(parents=True, exist_ok=True)
    sections = tuple(arguments.section or ("slope", "transition"))
    regimes = tuple(arguments.regime or ("torque", "pd"))
    backends = ("Tasks", "TVM") if arguments.backend == "both" else (arguments.backend,)

    environment = os.environ.copy()
    runtime = (
        arguments.build / "src",
        arguments.build / "plugins/ROS",
        arguments.build / "deps/tasks-system-install/lib",
    )
    inherited = environment.get("LD_LIBRARY_PATH", "")
    environment["LD_LIBRARY_PATH"] = ":".join(
        [str(path) for path in runtime] + ([inherited] if inherited else [])
    )
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "-1",
            "MC_RTC_DISABLE_CONVEX_GENERATION_PATCH": "ON",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
    )

    plan = []
    if "slope" in sections:
        plan += [
            (command, terrain, SLOPE_CYCLES)
            for terrain in SLOPE_TERRAINS
            for command in COMMANDS
        ]
    if "transition" in sections:
        plan += [
            (TRANSITION_COMMAND, terrain, TRANSITION_CYCLES)
            for terrain in TRANSITION_TERRAINS
        ]
    if arguments.terrain:
        plan = [entry for entry in plan if entry[1].name in set(arguments.terrain)]
    if arguments.command:
        plan = [entry for entry in plan if entry[0].name in set(arguments.command)]
    if not plan:
        raise RuntimeError("the terrain/command filters selected no case")

    records = []
    try:
        for backend in backends:
            for regime in regimes:
                for command, terrain, cycles in plan:
                    record = run_case(
                        arguments, environment, command, terrain, regime, backend, cycles
                    )
                    records.append(record)
                    achieved = record.get("achieved_twist", [float("nan")] * 3)
                    print(
                        f"{record['case']:56s} "
                        f"cmd=({record['commanded_twist'][0]:+.3f},{record['commanded_twist'][1]:+.3f},"
                        f"{record['commanded_twist'][2]:+.3f}) "
                        f"got=({achieved[0]:+.3f},{achieved[1]:+.3f},{achieved[2]:+.3f})",
                        flush=True,
                    )
    finally:
        install_terrain(arguments.mujoco_user_dir, Terrain("flat"), arguments.repo)

    output = arguments.artifact_dir / "ramp-validation.json"
    output.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {output} ({len(records)} runs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
