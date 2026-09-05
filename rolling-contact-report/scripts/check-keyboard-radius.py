#!/usr/bin/env python3

"""Validate physical turning radii for mixed keyboard translation+yaw runs.

The command/reference fields alone only prove that the keyboard adapter saw a
key.  This check reconstructs the measured chassis yaw and body-frame forward
velocity from the MuJoCo floating-base sensors, then verifies that W+Q and W+E
produce opposite signed yaw rates at the requested radius without contact
fallback or a persistent rolling residual.
"""

import argparse
import csv
import json
import math
import pathlib
import statistics


TIMESTEP = 0.005


def yaw_from_quaternion(row):
    w = float(row["FloatingBase_orientation_w"])
    x = float(row["FloatingBase_orientation_x"])
    y = float(row["FloatingBase_orientation_y"])
    z = float(row["FloatingBase_orientation_z"])
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def body_forward_velocity(row):
    """Convert MuJoCo world linear velocity to the logged body convention."""
    w = float(row["FloatingBase_orientation_w"])
    x = float(row["FloatingBase_orientation_x"])
    y = float(row["FloatingBase_orientation_y"])
    z = float(row["FloatingBase_orientation_z"])
    vx = float(row["FloatingBase_linearVelocity_x"])
    vy = float(row["FloatingBase_linearVelocity_y"])
    # The logged FloatingBase quaternion is the inertial-to-body transform.
    r00 = 1.0 - 2.0 * (y * y + z * z)
    r01 = 2.0 * (x * y - w * z)
    return r00 * vx + r01 * vy


def unwrap(values):
    result = []
    offset = 0.0
    previous = values[0]
    for value in values:
        delta = value - previous
        if delta > math.pi:
            offset -= 2.0 * math.pi
        elif delta < -math.pi:
            offset += 2.0 * math.pi
        result.append(value + offset)
        previous = value
    return result


def slope(values, start, end):
    count = end - start
    mean_t = 0.5 * (start + end - 1) * TIMESTEP
    mean_y = sum(values[start:end]) / count
    numerator = sum(
        (index * TIMESTEP - mean_t) * (values[index] - mean_y)
        for index in range(start, end)
    )
    denominator = sum(
        (index * TIMESTEP - mean_t) ** 2 for index in range(start, end)
    )
    return numerator / denominator


def active_interval(rows, sign):
    indices = [
        index
        for index, row in enumerate(rows)
        if float(row["RollingContact_reference_linear_speed"]) > 0.1
        and sign * float(row["RollingContact_reference_yaw_rate"]) > 0.1
    ]
    if not indices:
        raise RuntimeError("no mixed forward+yaw interval for requested direction")
    first, last = min(indices), max(indices)
    if (last - first) * TIMESTEP < 2.0:
        raise RuntimeError("mixed interval is shorter than 2 s")
    return first, last


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=pathlib.Path)
    parser.add_argument("--expected-radius", type=float, default=0.6)
    parser.add_argument("--radius-tolerance", type=float, default=0.15)
    parser.add_argument("--rate-tolerance", type=float, default=0.1)
    parser.add_argument("--expected-rate", type=float, default=0.5)
    parser.add_argument("--residual-limit", type=float, default=0.35)
    parser.add_argument("--steady-residual-limit", type=float, default=0.05)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()

    with args.csv.open(newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter=";"))
    if not rows:
        raise RuntimeError("empty controller log")

    required = {
        "FloatingBase_orientation_w",
        "FloatingBase_orientation_x",
        "FloatingBase_orientation_y",
        "FloatingBase_orientation_z",
        "FloatingBase_linearVelocity_x",
        "FloatingBase_linearVelocity_y",
        "RollingContact_reference_linear_speed",
        "RollingContact_reference_yaw_rate",
        "RollingContact_solver_success",
        "RollingContact_diagnostics_valid",
        "RollingContact_contact_fallback",
        "RollingContact_max_longitudinal_residual",
        "RollingContact_max_lateral_residual",
        "RollingContact_front_left_mode",
        "RollingContact_front_right_mode",
        "RollingContact_rear_left_mode",
        "RollingContact_rear_right_mode",
    }
    missing = required.difference(rows[0])
    if missing:
        raise RuntimeError(f"missing mixed-radius fields: {sorted(missing)}")

    yaw = unwrap([yaw_from_quaternion(row) for row in rows])
    body_vx = [body_forward_velocity(row) for row in rows]
    bad_solver = [
        index
        for index, row in enumerate(rows)
        if row["RollingContact_solver_success"] != "1"
        or row["RollingContact_diagnostics_valid"] != "1"
    ]
    if bad_solver:
        raise RuntimeError(f"solver/diagnostics failed at cycles {bad_solver[:5]}")

    reports = []
    for direction, sign in (("q", 1.0), ("e", -1.0)):
        first, last = active_interval(rows, sign)
        fit_start = min(last - 1, first + int(1.0 / TIMESTEP))
        fit_end = last + 1
        measured_rate = slope(yaw, fit_start, fit_end)
        if abs(sign * measured_rate - args.expected_rate) > args.rate_tolerance:
            raise RuntimeError(
                f"{direction.upper()} measured yaw rate {measured_rate:.4f} rad/s "
                f"is outside +/-{args.expected_rate:.4f} +/- {args.rate_tolerance:.4f}"
            )
        forward_speed = statistics.median(body_vx[fit_start:fit_end])
        measured_radius = abs(forward_speed / measured_rate)
        if abs(measured_radius - args.expected_radius) > args.radius_tolerance:
            raise RuntimeError(
                f"{direction.upper()} measured radius {measured_radius:.4f} m "
                f"is outside {args.expected_radius:.4f} +/- {args.radius_tolerance:.4f}"
            )
        active_rows = rows[first : last + 1]
        if any(row["RollingContact_contact_fallback"] != "0" for row in active_rows):
            raise RuntimeError(f"{direction.upper()} entered contact fallback")
        if any(
            row[field] != "rolling"
            for row in active_rows
            for field in (
                "RollingContact_front_left_mode",
                "RollingContact_front_right_mode",
                "RollingContact_rear_left_mode",
                "RollingContact_rear_right_mode",
            )
        ):
            raise RuntimeError(f"{direction.upper()} left rolling contact mode")
        max_residual = max(
            max(
                abs(float(row["RollingContact_max_longitudinal_residual"])),
                abs(float(row["RollingContact_max_lateral_residual"])),
            )
            for row in active_rows
        )
        if max_residual > args.residual_limit:
            raise RuntimeError(
                f"{direction.upper()} residual {max_residual:.4f} m/s exceeds "
                f"{args.residual_limit:.4f} m/s"
            )
        steady_rows = rows[fit_start : last + 1]
        max_steady_residual = max(
            max(
                abs(float(row["RollingContact_max_longitudinal_residual"])),
                abs(float(row["RollingContact_max_lateral_residual"])),
            )
            for row in steady_rows
        )
        if max_steady_residual > args.steady_residual_limit:
            raise RuntimeError(
                f"{direction.upper()} steady residual {max_steady_residual:.4f} m/s exceeds "
                f"{args.steady_residual_limit:.4f} m/s"
            )
        reports.append(
            {
                "direction": direction,
                "active_seconds": (last - first) * TIMESTEP,
                "measured_yaw_rate_rad_s": measured_rate,
                "measured_forward_speed_m_s": forward_speed,
                "measured_radius_m": measured_radius,
                "max_active_residual_m_s": max_residual,
                "max_steady_residual_m_s": max_steady_residual,
            }
        )

    report = {
        "cycles": len(rows),
        "directions": reports,
        "max_residual_m_s": max(
            max(
                abs(float(row["RollingContact_max_longitudinal_residual"])),
                abs(float(row["RollingContact_max_lateral_residual"])),
            )
            for row in rows
        ),
    }
    serialized = json.dumps(report, indent=2, sort_keys=True)
    print(serialized)
    if args.output:
        args.output.write_text(serialized + "\n")


if __name__ == "__main__":
    main()
