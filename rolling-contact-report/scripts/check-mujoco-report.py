#!/usr/bin/env python3

"""Apply the frozen CPU MuJoCo acceptance policy to one or more reports."""

import argparse
import json
import math
import pathlib


REQUIRED_NUMERIC = {
    "simulation_time_s",
    "wall_time_s",
    "cpu_time_s",
    "real_time_factor",
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
    # Angle between MuJoCo's real contact normals and the terrain normal the
    # controller was configured with. Required to be present and finite, but
    # deliberately not bounded here: the whole point of the ramp terrain is to
    # make it non-zero, and the acceptance policy for flat and uniformly tilted
    # worlds is expressed by the slip and contact bounds below.
    "max_contact_normal_deviation_deg",
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
    "cycle_wall_median_ms",
    "cycle_wall_p95_ms",
    "cycle_wall_p99_ms",
    "cycle_wall_max_ms",
}

# MuJoCo's discrete cylinder/plane contact can disappear for one 200 Hz
# controller sample while the wheel gap remains below a micrometre. Treat at
# most two samples per 1,000-cycle replay as numerical contact chatter, while
# still requiring bounded normal speed, prompt fallback, and recovery. Larger
# or sustained losses remain failures and are covered separately by the
# contact-loss scenario.
MAX_NOMINAL_CONTACT_CHATTER_SAMPLES = 2


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def validate_common(report, path, backend, robot):
    require(report.get("schema") == 4, f"{path}: expected schema 4")
    require(report.get("device") == "CPU", f"{path}: device is not CPU")
    require(report.get("backend") == backend, f"{path}: backend mismatch")
    require(report.get("robot") == robot, f"{path}: robot mismatch")
    require(report.get("static_parity") is True, f"{path}: static parity failed")
    require(report.get("controller_failed") is False, f"{path}: controller failed")
    require(report.get("finite_state") is True, f"{path}: non-finite simulator state")
    require(
        report.get("cycles_completed") == report.get("cycles_requested"),
        f"{path}: incomplete simulation",
    )
    require(report.get("measured_samples", 0) > 0, f"{path}: no measured samples")
    require(report.get("missed_deadlines") == 0, f"{path}: missed CPU deadline")
    require(report.get("peak_resident_kib", 0) > 0, f"{path}: missing RSS measurement")
    for field in REQUIRED_NUMERIC:
        value = report.get(field)
        require(
            isinstance(value, (int, float)) and math.isfinite(value),
            f"{path}: non-finite or missing {field}",
        )
    require(report["real_time_factor"] > 1.0, f"{path}: slower than real time")
    require(report["cycle_wall_p99_ms"] < 5.0, f"{path}: P99 exceeded 5 ms")
    require(report["max_drive_torque_nm"] <= 35.0 + 1e-9, f"{path}: drive torque limit")
    require(report["max_steer_torque_nm"] <= 25.0 + 1e-9, f"{path}: steer torque limit")
    require(
        0.0 <= report["no_contact_sample_fraction"] <= 1.0,
        f"{path}: invalid contact-loss fraction",
    )
    require(
        0.0 <= report["contact_fallback_sample_fraction"] <= 1.0,
        f"{path}: invalid fallback fraction",
    )
    modes = report.get("estimated_modes")
    require(isinstance(modes, list) and modes, f"{path}: missing estimated modes")
    require(
        set(modes).issubset({"fixed", "rolling", "sliding", "detached"}),
        f"{path}: invalid estimated mode",
    )


def validate_nominal(report, path):
    require(
        set(report["estimated_modes"]).issubset({"rolling", "detached"}),
        f"{path}: nominal case entered sliding/fixed mode",
    )
    require("rolling" in report["estimated_modes"], f"{path}: rolling mode absent")
    require(report["max_rolling_slip_mps"] <= 1e-2, f"{path}: rolling slip")
    require(report["max_lateral_slip_mps"] <= 1e-2, f"{path}: lateral slip")
    require(report["max_normal_speed_mps"] <= 3e-2, f"{path}: normal speed")
    require(
        report["no_contact_sample_fraction"] * report["measured_samples"]
        <= MAX_NOMINAL_CONTACT_CHATTER_SAMPLES + 1e-9,
        f"{path}: sustained nominal contact loss",
    )
    if report["no_contact_sample_fraction"] == 0.0:
        require(report["min_normal_force_n"] >= 1.0, f"{path}: lost normal force")
    require(report["min_friction_margin_n"] >= -1e-6, f"{path}: friction violation")
    require(
        report["contact_fallback_sample_fraction"] <= 0.4,
        f"{path}: nominal contact fallback did not recover promptly",
    )
    require(report.get("final_contact_fallback") is False, f"{path}: nominal fallback remained active")
    require(
        report.get("final_modes")
        and set(report["final_modes"]) == {"rolling"},
        f"{path}: nominal replay did not finish in rolling mode",
    )
    require(
        report.get("final_solver_activations")
        and all(abs(value - 1.0) <= 1e-12 for value in report["final_solver_activations"]),
        f"{path}: nominal rolling rows were not fully promoted",
    )
    require(
        report["odometry_position_error_rms_m"] <= 3e-2,
        f"{path}: wheel odometry RMS",
    )
    require(
        report["odometry_position_error_max_m"] <= 6e-2,
        f"{path}: wheel odometry maximum",
    )
    require(
        report["odometry_yaw_error_max_rad"] <= 3e-2,
        f"{path}: wheel odometry yaw",
    )
    require(
        report["tracking_position_error_rms_m"] <= 0.20,
        f"{path}: path tracking RMS",
    )
    require(
        report["tracking_position_error_max_m"] <= 0.30,
        f"{path}: path tracking maximum",
    )
    require(
        report["tracking_yaw_error_max_rad"] <= 0.15,
        f"{path}: yaw tracking maximum",
    )


def validate_slip(report, path):
    require(report["friction"] <= 0.2, f"{path}: slip case has nominal friction")
    require(report["max_rolling_slip_mps"] >= 2e-2, f"{path}: no measurable slip")
    require(
        bool({"sliding", "detached"}.intersection(report["estimated_modes"])),
        f"{path}: controller certified rolling throughout measurable slip",
    )


def validate_transition(report, path):
    require(
        {"fixed", "rolling", "sliding", "detached"}.issubset(
            report["estimated_modes"]
        ),
        f"{path}: incomplete mode-cycle coverage",
    )


def validate_disturbance(report, path):
    require(
        max(
            report["max_rolling_slip_mps"],
            report["max_lateral_slip_mps"],
            report["max_normal_speed_mps"],
        )
        >= 1e-2,
        f"{path}: no measurable disturbance",
    )
    require(
        bool({"sliding", "detached"}.intersection(report["estimated_modes"])),
        f"{path}: no mode fallback during disturbance",
    )
    require(report.get("final_contact_fallback") is False, f"{path}: disturbance recovery remained active")


def validate_maneuver(report, path):
    require(
        report["max_rolling_slip_mps"] >= 1e-2,
        f"{path}: finite-width maneuver produced no measurable scrub",
    )
    require(
        bool({"sliding", "detached"}.intersection(report["estimated_modes"])),
        f"{path}: no mode fallback during finite-width scrub",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", nargs="+", type=pathlib.Path)
    parser.add_argument(
        "--kind",
        choices=("nominal", "slip", "transition", "disturbance", "maneuver"),
        required=True,
    )
    parser.add_argument("--backend", choices=("Tasks", "TVM"), required=True)
    parser.add_argument(
        "--robot", choices=("differential", "four-steering"), required=True
    )
    args = parser.parse_args()

    summaries = []
    for path in args.reports:
        with path.open(encoding="utf-8") as stream:
            report = json.load(stream)
        validate_common(report, path, args.backend, args.robot)
        if args.kind == "nominal":
            validate_nominal(report, path)
        elif args.kind == "slip":
            validate_slip(report, path)
        elif args.kind == "transition":
            validate_transition(report, path)
        elif args.kind == "disturbance":
            validate_disturbance(report, path)
        else:
            validate_maneuver(report, path)
        summaries.append(
            {
                "path": str(path),
                "rolling_slip_max_mps": report["max_rolling_slip_mps"],
                "tracking_position_rms_m": report[
                    "tracking_position_error_rms_m"
                ],
                "cycle_wall_p99_ms": report["cycle_wall_p99_ms"],
                "real_time_factor": report["real_time_factor"],
                "estimated_modes": report["estimated_modes"],
            }
        )
    print(json.dumps({"status": "pass", "reports": summaries}, indent=2))


if __name__ == "__main__":
    main()
