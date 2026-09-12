#!/usr/bin/env python3

"""Validate deterministic rolling-contact controller logs without a GUI."""

import argparse
import csv
import json
import math
import pathlib
import statistics
import subprocess
import tempfile


def percentile(values, probability):
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(probability * len(ordered)) - 1)
    return ordered[index]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=pathlib.Path)
    parser.add_argument("--cycles", type=int, required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--backend", choices=("Tasks", "TVM"), required=True)
    parser.add_argument("--utils", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--residual-limit", type=float, default=5e-3)
    parser.add_argument("--allow-mode-transitions", action="store_true")
    parser.add_argument(
        "--tilt-error-limit",
        type=float,
        help="degrees; require a tilt estimate on every cycle after --tilt-settle "
        "and bound its error against the reference attitude. Off when unset.",
    )
    parser.add_argument(
        "--tilt-excursion-min",
        type=float,
        default=5.0,
        help="degrees; the reference roll/pitch must sweep at least this far while "
        "--tilt-error-limit is in force. A level chassis compared against a level "
        "estimate passes any error bound; this is what stops that.",
    )
    parser.add_argument(
        "--tilt-settle",
        type=float,
        default=1.0,
        help="seconds of run discarded before the tilt bound applies, for the "
        "filter's own convergence transient.",
    )
    args = parser.parse_args()

    log = args.log.resolve(strict=True)
    with tempfile.TemporaryDirectory(prefix="rolling-contact-log-") as directory:
        flat = pathlib.Path(directory) / "controller.log"
        subprocess.run([args.utils / "mc_bin_to_log", log, flat], check=True)
        with flat.open(newline="") as stream:
            rows = list(csv.DictReader(stream, delimiter=";"))

    if len(rows) != args.cycles:
        raise RuntimeError(f"expected {args.cycles} cycles, got {len(rows)}")
    required = {
        "t",
        "RollingContact_scenario",
        "RollingContact_backend",
        "RollingContact_solver_success",
        "RollingContact_update_ms",
        "RollingContact_solve_and_build_ms",
        "RollingContact_total_ms",
        "RollingContact_max_longitudinal_residual",
        "RollingContact_max_lateral_residual",
        "RollingContact_min_friction_margin",
        "RollingContact_diagnostics_valid",
        "RollingContact_invalid_reason",
        "RollingContact_reference_linear_speed",
        "RollingContact_reference_yaw_rate",
        "RollingContact_base_pose_tx",
        "RollingContact_base_pose_ty",
        "RollingContact_base_pose_tz",
    }
    missing = required.difference(rows[0])
    if missing:
        raise RuntimeError(f"missing log fields: {sorted(missing)}")

    wheel_prefixes = sorted(
        field[: -len("_requested_mode")]
        for field in rows[0]
        if field.startswith("RollingContact_") and field.endswith("_requested_mode")
    )
    if not wheel_prefixes:
        raise RuntimeError("no per-wheel rolling-contact fields")
    wheel_suffixes = {
        "position",
        "rate",
        "target",
        "requested_mode",
        "mode",
        "activation",
        "filtered_slip",
        "dwell_time",
        "measurement_valid",
        "invalid_reason",
        "rolling_residual",
        "lateral_residual",
        "normal_residual",
        "acceleration_residual",
        "normal_force",
        "tangential_force",
        "friction_margin",
        "drive_torque",
        "drive_torque_margin",
    }
    missing_wheel_fields = {
        f"{prefix}_{suffix}"
        for prefix in wheel_prefixes
        for suffix in wheel_suffixes
        if f"{prefix}_{suffix}" not in rows[0]
    }
    if missing_wheel_fields:
        raise RuntimeError(f"missing per-wheel fields: {sorted(missing_wheel_fields)}")

    numeric_fields = required.difference(
        {
            "RollingContact_scenario",
            "RollingContact_backend",
            "RollingContact_invalid_reason",
        }
    )
    wheel_string_suffixes = {"requested_mode", "mode", "invalid_reason"}
    for prefix in wheel_prefixes:
        for suffix in wheel_suffixes.difference(wheel_string_suffixes):
            numeric_fields.add(f"{prefix}_{suffix}")
    numeric = {field: [] for field in numeric_fields}
    requested_modes = {prefix: set() for prefix in wheel_prefixes}
    estimated_modes = {prefix: set() for prefix in wheel_prefixes}
    for index, row in enumerate(rows):
        if row["RollingContact_scenario"] != args.scenario:
            raise RuntimeError(f"scenario changed at cycle {index}")
        if row["RollingContact_backend"] != args.backend:
            raise RuntimeError(f"backend changed at cycle {index}")
        if row["RollingContact_solver_success"] != "1":
            raise RuntimeError(f"solver failed at cycle {index}")
        if row["RollingContact_diagnostics_valid"] != "1":
            raise RuntimeError(f"invalid diagnostics at cycle {index}")
        if row["RollingContact_invalid_reason"]:
            raise RuntimeError(f"controller diagnostic reason at cycle {index}")
        for prefix in wheel_prefixes:
            requested = row[f"{prefix}_requested_mode"]
            estimated = row[f"{prefix}_mode"]
            requested_modes[prefix].add(requested)
            estimated_modes[prefix].add(estimated)
            if requested not in {"fixed", "rolling", "sliding", "detached"}:
                raise RuntimeError(f"invalid requested mode at cycle {index}: {requested}")
            if estimated not in {"fixed", "rolling", "sliding", "detached"}:
                raise RuntimeError(f"invalid estimated mode at cycle {index}: {estimated}")
            if not args.allow_mode_transitions and (requested != "rolling" or estimated != "rolling"):
                raise RuntimeError(f"unexpected contact mode at cycle {index}: {requested}/{estimated}")
            if index > 0 and row[f"{prefix}_measurement_valid"] != "1":
                raise RuntimeError(f"invalid wheel measurement at cycle {index}")
            if row[f"{prefix}_invalid_reason"]:
                raise RuntimeError(f"wheel diagnostic reason at cycle {index}")
        for field in numeric_fields:
            value = float(row[field])
            if not math.isfinite(value):
                raise RuntimeError(f"non-finite {field} at cycle {index}")
            numeric[field].append(value)

    if args.allow_mode_transitions:
        expected_modes = {"fixed", "rolling", "sliding", "detached"}
        for prefix in wheel_prefixes:
            if requested_modes[prefix] != expected_modes:
                raise RuntimeError(
                    f"incomplete requested mode coverage for {prefix}: "
                    f"{sorted(requested_modes[prefix])}"
                )
            if not expected_modes.issubset(estimated_modes[prefix]):
                raise RuntimeError(
                    f"incomplete estimated mode coverage for {prefix}: "
                    f"{sorted(estimated_modes[prefix])}"
                )

    timestep = 0.005
    for index, value in enumerate(numeric["t"]):
        if abs(value - index * timestep) > 1e-9:
            raise RuntimeError(f"timestamp discontinuity at cycle {index}: {value}")
    longitudinal = numeric["RollingContact_max_longitudinal_residual"]
    lateral = numeric["RollingContact_max_lateral_residual"]
    if args.allow_mode_transitions:
        constrained_longitudinal = []
        constrained_lateral = []
        for index, row in enumerate(rows):
            for prefix in wheel_prefixes:
                mode = row[f"{prefix}_mode"]
                activation = numeric[f"{prefix}_activation"][index]
                if mode in {"fixed", "rolling"} and activation >= 1.0 - 1e-12:
                    constrained_longitudinal.append(
                        abs(numeric[f"{prefix}_rolling_residual"][index])
                    )
                    constrained_lateral.append(
                        abs(numeric[f"{prefix}_lateral_residual"][index])
                    )
        if not constrained_longitudinal or not constrained_lateral:
            raise RuntimeError("no fully active constrained rolling samples")
        if (
            max(constrained_longitudinal) > args.residual_limit
            or max(constrained_lateral) > args.residual_limit
        ):
            raise RuntimeError(
                f"active rolling residual exceeded {args.residual_limit} m/s"
            )
    else:
        constrained_longitudinal = [abs(value) for value in longitudinal]
        constrained_lateral = [abs(value) for value in lateral]
        if max(constrained_longitudinal) > args.residual_limit or max(
            constrained_lateral
        ) > args.residual_limit:
            raise RuntimeError(f"rolling residual exceeded {args.residual_limit} m/s")
    for prefix in wheel_prefixes:
        activation = numeric[f"{prefix}_activation"]
        if min(activation) < -1e-12 or max(activation) > 1.0 + 1e-12:
            raise RuntimeError(f"activation outside [0, 1] for {prefix}")
        if min(numeric[f"{prefix}_normal_force"]) < -1e-7:
            raise RuntimeError(f"negative normal force for {prefix}")
        if min(numeric[f"{prefix}_drive_torque_margin"]) < -1e-7:
            raise RuntimeError(f"drive torque limit violated for {prefix}")
    tilt = None
    if args.tilt_error_limit is not None:
        tilt_fields = {
            "RollingContact_attitude_valid",
            "RollingContact_attitude_error",
            "RollingContact_attitude_estimated_rp_x",
            "RollingContact_attitude_estimated_rp_y",
            "RollingContact_attitude_reference_rp_x",
            "RollingContact_attitude_reference_rp_y",
        }
        missing_tilt = tilt_fields.difference(rows[0])
        if missing_tilt:
            raise RuntimeError(f"missing tilt fields: {sorted(missing_tilt)}")
        first = math.ceil(args.tilt_settle / timestep)
        if first >= len(rows):
            raise RuntimeError("--tilt-settle discards the whole run")
        settled = rows[first:]
        for index, row in enumerate(settled, start=first):
            if row["RollingContact_attitude_valid"] != "1":
                raise RuntimeError(f"no tilt estimate at cycle {index}")
        degrees = [math.degrees(float(row["RollingContact_attitude_error"])) for row in settled]
        if not all(math.isfinite(value) for value in degrees):
            raise RuntimeError("non-finite tilt error")
        # The reference roll and pitch have to have gone somewhere. Both are
        # functions of the reference tilt alone, so their peak-to-peak spread is
        # the attitude excursion the estimator was actually asked to follow;
        # bounding an error against a chassis that never moved would pass with
        # an estimator stuck at its initial value.
        excursions = []
        for axis in ("x", "y"):
            values = [
                math.degrees(float(row[f"RollingContact_attitude_reference_rp_{axis}"]))
                for row in settled
            ]
            excursions.append(max(values) - min(values))
        excursion = max(excursions)
        if excursion < args.tilt_excursion_min:
            raise RuntimeError(
                f"reference attitude swept only {excursion:.3f} deg, "
                f"below the {args.tilt_excursion_min} deg this check needs to mean anything"
            )
        if max(degrees) > args.tilt_error_limit:
            raise RuntimeError(
                f"tilt estimate error {max(degrees):.3f} deg exceeded {args.tilt_error_limit} deg"
            )
        tilt = {
            "settle_s": args.tilt_settle,
            "error_deg_max": max(degrees),
            "error_deg_rms": math.sqrt(sum(v * v for v in degrees) / len(degrees)),
            "reference_excursion_deg": excursion,
        }

    update = numeric["RollingContact_update_ms"]
    solve = numeric["RollingContact_solve_and_build_ms"]
    total = numeric["RollingContact_total_ms"]
    if percentile(total[10:], 0.99) >= timestep * 1000.0:
        raise RuntimeError("P99 controller time exceeded the 5 ms period")

    report = {
        "schema": 2,
        "backend": args.backend,
        "device": "CPU",
        "scenario": args.scenario,
        "cycles": len(rows),
        "timestep_s": timestep,
        "longitudinal_residual_max": max(longitudinal),
        "lateral_residual_max": max(lateral),
        "active_longitudinal_residual_max": max(constrained_longitudinal),
        "active_lateral_residual_max": max(constrained_lateral),
        "friction_margin_min": min(numeric["RollingContact_min_friction_margin"]),
        "wheels": {
            prefix.removeprefix("RollingContact_"): {
                "rolling_residual_max": max(abs(v) for v in numeric[f"{prefix}_rolling_residual"]),
                "lateral_residual_max": max(abs(v) for v in numeric[f"{prefix}_lateral_residual"]),
                "acceleration_residual_max": max(
                    abs(v) for v in numeric[f"{prefix}_acceleration_residual"]
                ),
                "normal_force_min": min(numeric[f"{prefix}_normal_force"]),
                "friction_margin_min": min(numeric[f"{prefix}_friction_margin"]),
                "drive_torque_margin_min": min(numeric[f"{prefix}_drive_torque_margin"]),
            }
            for prefix in wheel_prefixes
        },
        "update_ms": {
            "median": statistics.median(update[10:]),
            "p95": percentile(update[10:], 0.95),
            "p99": percentile(update[10:], 0.99),
            "max": max(update[10:]),
        },
        "solve_ms": {
            "median": statistics.median(solve[10:]),
            "p95": percentile(solve[10:], 0.95),
            "p99": percentile(solve[10:], 0.99),
            "max": max(solve[10:]),
        },
        "total_ms": {
            "median": statistics.median(total[10:]),
            "p95": percentile(total[10:], 0.95),
            "p99": percentile(total[10:], 0.99),
            "max": max(total[10:]),
        },
    }
    if tilt is not None:
        report["tilt"] = tilt
    serialized = json.dumps(report, indent=2, sort_keys=True)
    print(serialized)
    if args.output:
        args.output.write_text(serialized + "\n")


if __name__ == "__main__":
    main()
