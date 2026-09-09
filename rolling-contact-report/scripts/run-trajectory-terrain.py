#!/usr/bin/env python3

"""Measure RangerTrajectory curve tracking under mc_mujoco, per FSM state.

The `RangerTrajectory` sample drives a circle, a straight line and a Lissajous
curve from YAML. This runs it headless in mc_mujoco and reports, for each of
the three curve states, how far the chassis stayed from the reference the QP
was handed. Run it once on flat ground and once on the ramp terrain and the
difference is the terrain's contribution.

Why mc_mujoco and not the deterministic runner: the runner
(rolling-contact-report/mujoco/rolling_contact_mujoco_runner.cpp) reads the
controller through fifteen `RollingContact::Get*` datastore calls and refuses to
start without them (`requireControllerBackend`). `RangerTrajectoryController` is
a bare `mc_control::fsm::Controller` subclass that registers none of them, so
the runner cannot drive it and the binary log is the measurement channel.

mc_mujoco has no `--run-for`, so the run is bounded by wall time and the log is
trimmed to a simulated window afterwards, the same way
scripts/cross-check-mc-mujoco.py does it.

The error is computed against `ChassisCurve_refPose`, the pose the trajectory
task handed the QP, not against the analytic spline: `SplineTrajectoryTask`
logs that pose and the spline itself is only reachable from inside the process.
"""

import argparse
import csv
import json
import math
import os
import pathlib
import signal
import subprocess
import sys
import tempfile
import time


# 21 s circle + 6.75 s line + 51 s Lissajous, from the durations in
# src/mc_control/samples/RangerTrajectory/etc/RangerTrajectory.in.yaml.
SEQUENCE_SECONDS = 78.75

CURVE_STATES = ("Ranger::Circle", "Ranger::Line", "Ranger::Lissajous")

# Commanded path lengths, from the generator header of the same file.
COMMANDED_LENGTH = {
    "Ranger::Circle": 6.283,
    "Ranger::Line": 2.000,
    "Ranger::Lissajous": 15.209,
}


def headless_configuration(source, destination, log_directory, template):
    """Copy the configuration with logging redirected and the GUI server off.

    The shipped configurations keep `GUIServer.Enable: true` so that
    mc_mujoco's visualizer can attach. A headless run must not bind
    /tmp/mc_rtc_pub.ipc, or a second run collides with the first.
    """
    lines = []
    in_gui_server = False
    for line in source.read_text().splitlines():
        if line.startswith("GUIServer:"):
            in_gui_server = True
            lines.append(line)
            continue
        if in_gui_server:
            if not line.startswith((" ", "\t", "#")) and line.strip():
                in_gui_server = False
            elif line.strip().startswith("Enable:"):
                lines.append("  Enable: false")
                continue
        if line.startswith("LogDirectory:"):
            lines.append(f"LogDirectory: {log_directory}")
        elif line.startswith("LogTemplate:"):
            lines.append(f"LogTemplate: {template}")
        else:
            lines.append(line)
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_headless(mc_mujoco, configuration, log_directory, template, wall_budget, home, repo):
    command = [
        str(mc_mujoco),
        "--without-visualization",
        "--without-mc-rtc-gui",
        "-f",
        str(configuration),
    ]
    environment = dict(os.environ)
    environment.update(
        {
            "HOME": str(home),
            "CUDA_VISIBLE_DEVICES": "-1",
            "MC_RTC_DISABLE_CONVEX_GENERATION_PATCH": "ON",
        }
    )
    stdout = log_directory / f"{template}.stdout"
    with stdout.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(
            command, stdout=stream, stderr=subprocess.STDOUT, env=environment, cwd=repo
        )
        deadline = time.monotonic() + wall_budget
        while time.monotonic() < deadline and process.poll() is None:
            time.sleep(0.5)
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=30)
    candidates = [p for p in log_directory.glob(f"{template}-*.bin") if not p.is_symlink()]
    if not candidates:
        raise RuntimeError(f"mc_mujoco produced no log; see {stdout}")
    return max(candidates, key=lambda path: path.stat().st_size)


def flatten(utilities, log, window, destination):
    with tempfile.TemporaryDirectory(prefix="trajectory-terrain-") as scratch:
        trimmed = pathlib.Path(scratch) / "window"
        subprocess.run(
            [
                str(utilities / "mc_bin_utils"), "extract", str(log), str(trimmed),
                "--from", "0", "--to", str(window),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        parts = sorted(pathlib.Path(scratch).glob("window*.bin"))
        if not parts:
            raise RuntimeError(f"mc_bin_utils extract produced nothing from {log}")
        subprocess.run(
            [str(utilities / "mc_bin_to_log"), str(parts[0]), str(destination)],
            check=True,
            stdout=subprocess.DEVNULL,
        )
    return destination


def analyse(flattened):
    with flattened.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream, delimiter=";"))
    if not rows:
        raise RuntimeError(f"{flattened} is empty")
    # `ChassisCurve_surfacePose` is the CONTROL robot's chassis, i.e. what the
    # QP thinks it achieved; `ff_real` is the estimated pose of the simulated
    # robot, from the observer pipeline. Reporting only the first is how a run
    # in which the chassis was in free fall for 78.75 s can still look like
    # millimetric tracking, so both are measured here and the second is the
    # physical answer.
    required = (
        "t",
        "Executor_Main",
        "ChassisCurve_surfacePose_tx",
        "ChassisCurve_surfacePose_ty",
        "ChassisCurve_surfacePose_tz",
        "ChassisCurve_refPose_tx",
        "ChassisCurve_refPose_ty",
        "ff_real_tx",
        "ff_real_ty",
        "ff_real_tz",
    )
    missing = [field for field in required if field not in rows[0]]
    if missing:
        raise RuntimeError(f"{flattened}: missing log fields {missing}")

    per_state = {}
    for state in CURVE_STATES:
        window = [row for row in rows if row["Executor_Main"] == state]
        # The task is created by the state, so the first cycle of a state can
        # still carry the previous task's columns; drop empty cells rather than
        # letting them poison the statistics.
        samples = []
        for row in window:
            try:
                sample = (
                    float(row["t"]),
                    float(row["ChassisCurve_surfacePose_tx"]),
                    float(row["ChassisCurve_surfacePose_ty"]),
                    float(row["ChassisCurve_refPose_tx"]),
                    float(row["ChassisCurve_refPose_ty"]),
                    float(row["ff_real_tx"]),
                    float(row["ff_real_ty"]),
                    float(row["ff_real_tz"]),
                )
            except (TypeError, ValueError):
                continue
            samples.append(sample)
        if len(samples) < 2:
            per_state[state] = {"samples": len(samples)}
            continue

        def statistics_for(achieved_x, achieved_y):
            errors = [
                math.hypot(s[achieved_x] - s[3], s[achieved_y] - s[4]) for s in samples
            ]
            length = sum(
                math.hypot(b[achieved_x] - a[achieved_x], b[achieved_y] - a[achieved_y])
                for a, b in zip(samples, samples[1:])
            )
            return {
                "max_deviation_m": max(errors),
                "mean_deviation_m": sum(errors) / len(errors),
                "rms_deviation_m": math.sqrt(sum(e * e for e in errors) / len(errors)),
                "travelled_m": length,
            }

        heights = [s[7] for s in samples]
        per_state[state] = {
            "samples": len(samples),
            "duration_s": samples[-1][0] - samples[0][0],
            "commanded_m": COMMANDED_LENGTH[state],
            "control": statistics_for(1, 2),
            "simulated": statistics_for(5, 6),
            "chassis_height_min_m": min(heights),
            "chassis_height_max_m": max(heights),
        }
    return {
        "rows": len(rows),
        "simulated_span_s": float(rows[-1]["t"]) - float(rows[0]["t"]),
        "states_seen": sorted({row["Executor_Main"] for row in rows}),
        "per_state": per_state,
    }


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mc-mujoco", type=pathlib.Path, default=pathlib.Path("/home/yuquan/local/bin/mc_mujoco"))
    parser.add_argument("--config", type=pathlib.Path, action="append", required=True)
    parser.add_argument("--artifact-dir", type=pathlib.Path, required=True)
    parser.add_argument("--repo", type=pathlib.Path, default=pathlib.Path.cwd())
    parser.add_argument("--build", type=pathlib.Path, default=pathlib.Path("build"))
    parser.add_argument("--window", type=float, default=SEQUENCE_SECONDS)
    parser.add_argument(
        "--wall-budget",
        type=float,
        default=420.0,
        help="seconds of wall clock per run; mc_mujoco has no --run-for",
    )
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    repo = arguments.repo.resolve(strict=True)
    build = arguments.build if arguments.build.is_absolute() else repo / arguments.build
    build = build.resolve(strict=True)
    artifact = arguments.artifact_dir.resolve()
    artifact.mkdir(parents=True, exist_ok=True)
    home = artifact / "scratch-home"
    home.mkdir(exist_ok=True)

    results = []
    for configuration in arguments.config:
        source = (repo / configuration).resolve(strict=True) if not configuration.is_absolute() else configuration.resolve(strict=True)
        template = source.stem
        directory = artifact / template
        directory.mkdir(parents=True, exist_ok=True)
        tuned = directory / "mc_rtc.yaml"
        headless_configuration(source, tuned, directory, template)
        log = run_headless(
            arguments.mc_mujoco, tuned, directory, template, arguments.wall_budget, home, repo
        )
        flattened = flatten(build / "utils", log, arguments.window, directory / "run.csv")
        record = {"config": str(source), "log": str(log)}
        record.update(analyse(flattened))
        results.append(record)
        print(f"\n== {source.name}  ({record['simulated_span_s']:.2f} s of log)")
        for state in CURVE_STATES:
            statistics = record["per_state"][state]
            if statistics.get("samples", 0) < 2:
                print(f"   {state:20s} NOT REACHED ({statistics.get('samples', 0)} samples)")
                continue
            control, simulated = statistics["control"], statistics["simulated"]
            print(
                f"   {state:20s} "
                f"sim max {simulated['max_deviation_m']:6.3f} mean {simulated['mean_deviation_m']:6.3f} m | "
                f"ctl max {control['max_deviation_m']:6.3f} mean {control['mean_deviation_m']:6.3f} m | "
                f"len {simulated['travelled_m']:7.3f} / {statistics['commanded_m']:.3f} m | "
                f"z {statistics['chassis_height_min_m']:.3f}..{statistics['chassis_height_max_m']:.3f}"
            )
    output = artifact / "trajectory-terrain.json"
    output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\nwrote {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
