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
                   src/mc_robots/ranger_mini_v3_description/mujoco/ramp_terrain.xml.
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


REPOSITORY_TERRAIN = "src/mc_robots/ranger_mini_v3_description/mujoco/ramp_terrain.xml"

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

# Longitudinal stations of every lane in ramp_terrain.xml, and the Ranger's
# wheelbase. The two concave breaks - where the surface pitches UP under the
# chassis - are the toe and the toe-out; the crest and the plateau end are
# convex, and a rigid chassis provably high-centres there.
LANE_TOE = 0.600
LANE_CREST = 2.100
LANE_PLATEAU_END = 3.100
LANE_TOE_OUT = 4.600
CONCAVE_BREAKS = (LANE_TOE, LANE_TOE_OUT)
CONVEX_BREAKS = (LANE_CREST, LANE_PLATEAU_END)
WHEELBASE = 0.494


def terrain_normal(degrees):
    angle = math.radians(degrees)
    return (-math.sin(angle), 0.0, math.cos(angle))


def lane_height(x, ramp_degrees):
    """Height profile h(x) of one lane of ramp_terrain.xml."""
    slope = math.tan(math.radians(ramp_degrees))
    if x <= LANE_TOE:
        return 0.0
    if x <= LANE_CREST:
        return (x - LANE_TOE) * slope
    crest = (LANE_CREST - LANE_TOE) * slope
    if x <= LANE_PLATEAU_END:
        return crest
    if x <= LANE_TOE_OUT:
        return crest - (x - LANE_PLATEAU_END) * slope
    return 0.0


def bridging_pitch(x, ramp_degrees):
    """Pitch of the chord between the two axles, in degrees.

    This - not `terrain_normal()` - is the ground truth for a chassis attitude
    on a ramp lane. The surface pitch is a step function at every break, and a
    0.494 m wheelbase cannot follow a step: over the 0.494 m before the toe the
    front axle is already climbing while the rear is still flat, and the chassis
    rides the chord between them. Comparing an attitude estimate against the
    surface pitch would report that whole interval as a failure of the estimator
    when it is a property of the vehicle.
    """
    half = 0.5 * WHEELBASE
    rise = lane_height(x + half, ramp_degrees) - lane_height(x - half, ramp_degrees)
    return math.degrees(math.atan2(rise, WHEELBASE))


def quaternion_rows(qw, qx, qy, qz):
    """Third row and third column of the rotation matrix of (qw, qx, qy, qz).

    mc_rtc logs an sva::PTransformd's rotation as the quaternion of E_0_b, the
    WORLD-TO-BODY map, so the body +z axis in world coordinates is E_0_b^T e_z,
    i.e. the third ROW. The third column is returned alongside purely so the
    caller can pin that convention against a run whose answer is known - see
    `sign_pin` in the record. The two differ exactly in the sign of x, which is
    the sign this whole comparison turns on.
    """
    row2 = (2.0 * (qx * qz - qw * qy), 2.0 * (qy * qz + qw * qx), 1.0 - 2.0 * (qx * qx + qy * qy))
    col2 = (2.0 * (qx * qz + qw * qy), 2.0 * (qy * qz - qw * qx), 1.0 - 2.0 * (qx * qx + qy * qy))
    return row2, col2


def tilt_of(normal):
    """Unsigned tilt from the world vertical and signed pitch, in degrees."""
    nx, _, nz = normal
    return (
        math.degrees(math.acos(max(-1.0, min(1.0, nz)))),
        math.degrees(math.atan2(-nx, nz)),
    )


def configuration(command, terrain, backend, build, artifact, log_template, tilt):
    controller = "RollingContact" if backend == "Tasks" else "RollingContact_TVM"
    normal = terrain_normal(terrain.controller_normal_degrees)
    # Logging is on so that `RollingContact_lateral_slack_norm` can be read
    # back: the controller keeps the slack norm but exposes no datastore
    # getter for it, so the binary log is the only way to see it from outside
    # the process.
    return f"""MainRobot: RangerMiniV3Robot
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
      # Runs last and reads realRobot().posW() for the yaw it cannot estimate,
      # so BodySensor has to have written it. `updateRobot: false` keeps that
      # same pose intact as the ground truth this estimate is scored against -
      # in the same run, which is the only way the comparison is honest.
      - type: VelocityAidedTilt
        update: true
        updateRobot: false
        imuBodySensor: ChassisIMU
        alpha: {tilt[0]:.17g}
        beta: {tilt[1]:.17g}
        gamma: {tilt[2]:.17g}
        minimumActivation: 1.0
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


EMPTY_LOG_METRICS = {
    "lateral_slack_norm_max": None,
    "lateral_slack_norm_mean": None,
    "tilt": None,
}


def log_metrics(artifact, log_template, utilities, observer="VelocityAidedTilt", pipeline="MainPipeline"):
    """Everything that only the mc_rtc binary log carries.

    Two things live here and nowhere else: `RollingContact_lateral_slack_norm`
    (the controller keeps it but publishes no datastore getter), and the tilt
    estimate together with the floating-base pose it has to be scored against.
    Reading both in one pass matters because the ground truth and the estimate
    MUST come from the same run - a comparison against a separately recorded
    reference would be comparing two different trajectories.
    """
    logs = sorted(artifact.glob(f"{log_template}-*.bin"))
    if not logs:
        return dict(EMPTY_LOG_METRICS)
    prefix = f"Observers_{pipeline}_{observer}"
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
            fields = set(reader.fieldnames or ())
            slack_field = "RollingContact_lateral_slack_norm"
            has_slack = slack_field in fields
            estimate_fields = [f"{prefix}_normal_{axis}" for axis in "xyz"]
            truth_fields = [f"RollingContact_base_pose_q{axis}" for axis in "wxyz"]
            has_tilt = fields.issuperset(estimate_fields) and fields.issuperset(truth_fields)
            slack = []
            tilt = []
            for row in reader:
                if has_slack and row[slack_field] not in ("", None):
                    slack.append(float(row[slack_field]))
                if not has_tilt:
                    continue
                estimate = tuple(float(row[field]) for field in estimate_fields)
                row2, col2 = quaternion_rows(*(float(row[field]) for field in truth_fields))
                estimate_tilt, estimate_pitch = tilt_of(estimate)
                truth_tilt, truth_pitch = tilt_of(row2)
                tilt.append(
                    {
                        "estimate_tilt_deg": estimate_tilt,
                        "estimate_pitch_deg": estimate_pitch,
                        "truth_tilt_deg": truth_tilt,
                        "truth_pitch_deg": truth_pitch,
                        # The two candidate conventions for "body +z in world",
                        # kept so the sign can be pinned against a run whose
                        # answer is known instead of assumed.
                        "truth_row2_x": row2[0],
                        "truth_col2_x": col2[0],
                    }
                )
    for log in logs:
        log.unlink()
    metrics = dict(EMPTY_LOG_METRICS)
    if slack:
        metrics["lateral_slack_norm_max"] = max(slack)
        metrics["lateral_slack_norm_mean"] = sum(slack) / len(slack)
    if tilt:
        metrics["tilt"] = tilt
    return metrics


def tilt_criteria(tilt, csv_path, terrain, warmup):
    """PC-1, PC-2 and PC-3 of the tilt-estimation plan.

    The tilt series and the runner CSV both carry one sample per controller
    cycle, so they are aligned by index and truncated to the shorter of the two;
    `aligned_samples` reports how many that left.
    """
    if not tilt:
        return {}
    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    wheels = sorted(
        field[: -len("_contact_normal_deviation")]
        for field in (rows[0] if rows else {})
        if field.endswith("_contact_normal_deviation")
    )
    count = min(len(tilt), len(rows))
    window = range(min(warmup, count), count)

    # PC-2: steady state, measured where the estimate is not being asked to
    # track a transition - the whole run on flat ground and constant slopes, and
    # the plateau plus the two constant-slope stretches on a ramp lane.
    def steady(index):
        if not terrain.uses_ramp_terrain:
            return True
        x = float(rows[index]["base_x"])
        margin = 0.5 * WHEELBASE
        return any(
            low + margin <= x <= high - margin
            for low, high in ((LANE_TOE, LANE_CREST), (LANE_CREST, LANE_PLATEAU_END),
                              (LANE_PLATEAU_END, LANE_TOE_OUT))
        ) or x < LANE_TOE - margin

    errors = [tilt[i]["estimate_tilt_deg"] - tilt[i]["truth_tilt_deg"] for i in window]
    steady_errors = [tilt[i]["estimate_tilt_deg"] - tilt[i]["truth_tilt_deg"] for i in window if steady(i)]
    criteria = {
        "aligned_samples": count,
        "tilt_error_max_deg": max((abs(error) for error in errors), default=None),
        "tilt_error_rms_deg": (
            math.sqrt(sum(error * error for error in errors) / len(errors)) if errors else None
        ),
        "tilt_error_steady_rms_deg": (
            math.sqrt(sum(error * error for error in steady_errors) / len(steady_errors))
            if steady_errors
            else None
        ),
        "tilt_error_steady_max_deg": max((abs(error) for error in steady_errors), default=None),
        # The sign pin: on a constant slope the runner tilts the world by
        # AngleAxisd(-theta, UnitY()), so the ground normal leans towards -x and
        # the body +z that rests on it must have a NEGATIVE x. Reported rather
        # than assumed, because the unit tests elsewhere in this tree use the
        # opposite sign and reading it off them would be wrong here.
        "sign_pin_row2_x": tilt[-1]["truth_row2_x"],
        "sign_pin_col2_x": tilt[-1]["truth_col2_x"],
    }

    # PC-1: a model-free envelope. Where all four wheels report contact the
    # chassis attitude must lie between the shallowest and the steepest surface
    # the four wheels are touching. Needs no geometric model and covers the
    # bridging interval by construction.
    #
    # Only meaningful where the runner's own reference normal IS the world
    # vertical, i.e. on terrains it does not tilt: `<wheel>_contact_normal_
    # deviation` is measured against `--ramp-deg`'s normal, so on a uniformly
    # tilted world it reports ~0 for a chassis the estimator correctly places at
    # the tilt angle, and the comparison would be between two different
    # references rather than between an estimate and an envelope.
    if wheels and terrain.tilt_degrees == 0.0:
        excursions = []
        covered = 0
        for index in window:
            row = rows[index]
            if any(int(float(row[f"{wheel}_contacts"])) == 0 for wheel in wheels):
                continue
            deviations = [
                math.degrees(float(row[f"{wheel}_contact_normal_deviation"])) for wheel in wheels
            ]
            estimate = tilt[index]["estimate_tilt_deg"]
            excursions.append(max(0.0, min(deviations) - estimate, estimate - max(deviations)))
            covered += 1
        criteria["pc1_samples"] = covered
        criteria["pc1_max_excursion_deg"] = max(excursions, default=None)

    # How far the estimate lags the measured attitude, as a pure time shift:
    # the shift that minimises the residual between the two series. Measured
    # against the LOGGED pose rather than against the bridging model, so that it
    # is a property of the filter alone and carries no terrain modelling. The
    # linearisation predicts alpha / (beta g) seconds.
    best = None
    for shift in range(0, 201):
        residual = 0.0
        samples = 0
        for i in window:
            if i - shift < 0:
                continue
            delta = tilt[i]["estimate_pitch_deg"] - tilt[i - shift]["truth_pitch_deg"]
            residual += delta * delta
            samples += 1
        if samples and (best is None or residual / samples < best[1]):
            best = (shift, residual / samples)
    if best is not None:
        criteria["tilt_lag_s"] = best[0] * 0.005
        criteria["tilt_lag_residual_rms_deg"] = math.sqrt(best[1])

    if terrain.uses_ramp_terrain:
        margin = 0.5 * WHEELBASE

        def stretch_error(low, high):
            values = [
                abs(tilt[i]["estimate_tilt_deg"] - tilt[i]["truth_tilt_deg"])
                for i in window
                if low <= float(rows[i]["base_x"]) <= high
            ]
            if not values:
                return None, None, 0
            return (
                max(values),
                math.sqrt(sum(value * value for value in values) / len(values)),
                len(values),
            )

        toe_peak, _, toe_samples = stretch_error(LANE_TOE - margin, LANE_TOE + margin)
        _, incline_rms, incline_samples = stretch_error(LANE_TOE + margin, LANE_CREST - margin)
        criteria["toe_peak_deg"] = toe_peak
        criteria["toe_samples"] = toe_samples
        criteria["incline_rms_deg"] = incline_rms
        criteria["incline_samples"] = incline_samples

    # PC-3: the transitions, against the bridging chord rather than the surface.
    # Only the concave breaks are asserted; a rigid chassis provably high-centres
    # at a convex one, so those are reported and not scored.
    if terrain.uses_ramp_terrain:
        for label, stations, key in (
            ("concave", CONCAVE_BREAKS, "pc3_concave_max_deg"),
            ("convex", CONVEX_BREAKS, "pc3_convex_max_deg"),
        ):
            worst = None
            for index in window:
                x = float(rows[index]["base_x"])
                if not any(abs(x - station) <= 0.5 * WHEELBASE for station in stations):
                    continue
                error = abs(
                    tilt[index]["estimate_pitch_deg"] - bridging_pitch(x, terrain.ramp_degrees)
                )
                worst = error if worst is None else max(worst, error)
            criteria[key] = worst
            del label
    return criteria


def run_case(arguments, environment, command, terrain, regime, backend, cycles):
    name = f"{terrain.name}__{command.name}__{regime}__{backend.lower()}"
    directory = arguments.artifact_dir / name
    directory.mkdir(parents=True, exist_ok=True)
    log_template = f"validation-{name}"
    configuration_path = directory / "mc_rtc.yaml"
    tilt_gains = (arguments.tilt_alpha, arguments.tilt_beta, arguments.tilt_gamma)
    configuration_path.write_text(
        configuration(command, terrain, backend, arguments.build, directory, log_template, tilt_gains),
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
        # Which terrain normal the QP's constraints were built from. The tilt
        # observer runs in every configuration, but nothing consumes it yet, so
        # this is "config" for the whole sweep; it is recorded as an axis so a
        # later closed-loop run is comparable against these rows rather than
        # replacing them.
        "terrain_normal_source": "config",
        "tilt_gains": list(tilt_gains),
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
    metrics = log_metrics(directory, log_template, arguments.build / "utils")
    tilt = metrics.pop("tilt")
    record.update(metrics)
    record.update(tilt_criteria(tilt, csv_path, terrain, warmup))
    if tilt:
        # The per-cycle series, kept next to the run it came from: every tilt
        # number in the summary is a reduction of this, and a reduction alone
        # cannot say WHERE an excursion happened.
        with (directory / "tilt.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=sorted(tilt[0]))
            writer.writeheader()
            writer.writerows(tilt)
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
    parser.add_argument("--tilt-alpha", type=float, default=10.0)
    parser.add_argument("--tilt-beta", type=float, default=4.0)
    parser.add_argument("--tilt-gamma", type=float, default=10.0)
    parser.add_argument(
        "--output-name",
        default="ramp-validation.json",
        help="name of the summary file written into --artifact-dir",
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

                    def show(key):
                        value = record.get(key)
                        return "   n/a" if value is None else f"{value:6.3f}"

                    print(
                        f"{record['case']:56s} "
                        f"cmd=({record['commanded_twist'][0]:+.3f},{record['commanded_twist'][1]:+.3f},"
                        f"{record['commanded_twist'][2]:+.3f}) "
                        f"got=({achieved[0]:+.3f},{achieved[1]:+.3f},{achieved[2]:+.3f}) "
                        f"tilt(rms/max/steady)=({show('tilt_error_rms_deg')},{show('tilt_error_max_deg')},"
                        f"{show('tilt_error_steady_rms_deg')}) deg "
                        f"pc1={show('pc1_max_excursion_deg')} pc3={show('pc3_concave_max_deg')}",
                        flush=True,
                    )
    finally:
        install_terrain(arguments.mujoco_user_dir, Terrain("flat"), arguments.repo)

    output = arguments.artifact_dir / arguments.output_name
    output.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {output} ({len(records)} runs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
