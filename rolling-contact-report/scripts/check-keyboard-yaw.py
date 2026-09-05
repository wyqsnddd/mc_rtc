#!/usr/bin/env python3

"""Validate the physical Q/E yaw response in a converted mc_rtc log.

The controller reference alone is not sufficient: this check reconstructs the
MuJoCo chassis yaw from the logged floating-base quaternion, verifies the sign
of the measured motion, estimates its steady-state rate, and checks that the
closed-loop yaw error settles before the keyboard stop.
"""

import argparse
import csv
import json
import math
import pathlib


def measured_yaw(row):
    w = float(row["FloatingBase_orientation_w"])
    x = float(row["FloatingBase_orientation_x"])
    y = float(row["FloatingBase_orientation_y"])
    z = float(row["FloatingBase_orientation_z"])
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


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


def slope(values, start, end, timestep):
    """Least-squares slope over [start, end)."""
    count = end - start
    mean_t = 0.5 * (start + end - 1) * timestep
    mean_y = sum(values[start:end]) / count
    numerator = 0.0
    denominator = 0.0
    for index in range(start, end):
        dt = index * timestep - mean_t
        numerator += dt * (values[index] - mean_y)
        denominator += dt * dt
    return numerator / denominator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=pathlib.Path)
    parser.add_argument("--direction", choices=("q", "e"), required=True)
    parser.add_argument("--expected-rate", type=float, default=0.5)
    parser.add_argument("--rate-tolerance", type=float, default=0.1)
    parser.add_argument("--min-active-seconds", type=float, default=2.0)
    parser.add_argument("--settle-error", type=float, default=0.15)
    parser.add_argument("--residual-limit", type=float, default=0.2)
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
        "RollingContact_reference_yaw_rate",
        "RollingContact_base_yaw_target",
        "RollingContact_keyboard_yaw_error",
        "RollingContact_keyboard_running",
        "RollingContact_keyboard_status",
        "RollingContact_solver_success",
        "RollingContact_diagnostics_valid",
        "RollingContact_max_longitudinal_residual",
        "RollingContact_base_position_tracking_error",
    }
    missing = required.difference(rows[0])
    if missing:
        raise RuntimeError(f"missing keyboard-yaw fields: {sorted(missing)}")

    timestep = 0.005
    yaw = unwrap([measured_yaw(row) for row in rows])
    sign = 1.0 if args.direction == "q" else -1.0
    active = [
        index
        for index, row in enumerate(rows)
        if sign * float(row["RollingContact_reference_yaw_rate"]) > 1e-6
    ]
    if not active:
        raise RuntimeError(f"no active {args.direction.upper()} yaw interval")
    first, last = min(active), max(active)
    active_seconds = (last - first) * timestep
    if active_seconds < args.min_active_seconds:
        raise RuntimeError(f"{args.direction.upper()} interval is only {active_seconds:.3f}s")

    # Exclude the steering-alignment transient from the rate estimate.
    fit_start = min(last, first + int(1.0 / timestep))
    fit_end = max(fit_start + 1, last + 1)
    measured_rate = slope(yaw, fit_start, fit_end, timestep)
    signed_error = sign * measured_rate - args.expected_rate

    settled_start = min(last, first + int(1.0 / timestep))
    settled_errors = [
        abs(float(rows[index]["RollingContact_keyboard_yaw_error"]))
        for index in range(settled_start, last + 1)
    ]
    max_residual = max(
        abs(float(row["RollingContact_max_longitudinal_residual"])) for row in rows
    )
    bad_solver = [
        index
        for index, row in enumerate(rows)
        if row["RollingContact_solver_success"] != "1"
        or row["RollingContact_diagnostics_valid"] != "1"
    ]
    tail = rows[-min(200, len(rows)) :]
    tail_error = max(abs(float(row["RollingContact_keyboard_yaw_error"])) for row in tail)
    tail_position_error = max(
        float(row["RollingContact_base_position_tracking_error"]) for row in tail
    )
    if bad_solver:
        raise RuntimeError(f"solver/diagnostics failed at cycles {bad_solver[:5]}")
    if abs(signed_error) > args.rate_tolerance:
        raise RuntimeError(
            f"measured {args.direction.upper()} yaw rate {measured_rate:.4f} rad/s "
            f"is outside {args.expected_rate:.4f} +/- {args.rate_tolerance:.4f}"
        )
    if max(settled_errors) > args.settle_error:
        raise RuntimeError(f"settled yaw error exceeded {args.settle_error} rad")
    if max_residual > args.residual_limit:
        raise RuntimeError(f"rolling residual exceeded {args.residual_limit} m/s")
    if rows[-1]["RollingContact_keyboard_running"] != "0":
        raise RuntimeError("keyboard capture did not stop")
    if rows[-1]["RollingContact_keyboard_status"] != "W:0 S:0 A:0 D:0 Q:0 E:0":
        raise RuntimeError("keyboard axes were not cleared")
    if tail_error > 1e-9:
        raise RuntimeError(f"stop yaw error did not converge to zero: {tail_error}")

    report = {
        "direction": args.direction,
        "cycles": len(rows),
        "active_seconds": active_seconds,
        "measured_rate_rad_s": measured_rate,
        "expected_rate_rad_s": sign * args.expected_rate,
        "rate_error_rad_s": measured_rate - sign * args.expected_rate,
        "active_yaw_error_max_rad": max(settled_errors),
        "stop_yaw_error_max_rad": tail_error,
        "stop_position_error_max_m": tail_position_error,
        "max_longitudinal_residual_m_s": max_residual,
    }
    serialized = json.dumps(report, indent=2, sort_keys=True)
    print(serialized)
    if args.output:
        args.output.write_text(serialized + "\n")


if __name__ == "__main__":
    main()
