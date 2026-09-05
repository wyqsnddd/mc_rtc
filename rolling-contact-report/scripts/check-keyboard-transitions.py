#!/usr/bin/env python3

"""Validate yaw-to-translation keyboard handoffs in a converted mc_rtc log.

The terminal keyboard adapter latches key-down events.  A translation key is
therefore expected to clear a previously latched Q/E yaw, while preserving the
bounded steering transition and the zero-error stop behavior.
"""

import argparse
import csv
import json
import math
import pathlib


TRANSLATION = {"W", "S", "A", "D"}
YAW = {"Q", "E"}
WHEELS = ("front_left", "front_right", "rear_left", "rear_right")


def axes(status):
    return {token.split(":", 1)[0] for token in status.split() if token.endswith(":1")}


def finite(row, name):
    value = float(row[name])
    if not math.isfinite(value):
        raise RuntimeError(f"non-finite {name}")
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--grace-limit", type=float, default=0.5)
    parser.add_argument("--speed-limit", type=float, default=0.6)
    parser.add_argument("--tail-samples", type=int, default=200)
    args = parser.parse_args()

    with args.csv.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream, delimiter=";"))
    if not rows:
        raise RuntimeError("empty controller log")

    required = {
        "RollingContact_keyboard_status",
        "RollingContact_keyboard_transition_grace",
        "RollingContact_keyboard_steering_ready",
        "RollingContact_reference_yaw_rate",
        "FloatingBase_linearVelocity_x",
        "FloatingBase_linearVelocity_y",
        "FloatingBase_angularVelocity_z",
        "RollingContact_base_position_tracking_error",
        "RollingContact_solver_success",
        "RollingContact_diagnostics_valid",
        "RollingContact_contact_fallback",
    }
    missing = required.difference(rows[0])
    if missing:
        raise RuntimeError(f"missing transition fields: {sorted(missing)}")

    for index, row in enumerate(rows):
        if row["RollingContact_solver_success"] != "1":
            raise RuntimeError(f"solver failure at row {index}")
        if row["RollingContact_diagnostics_valid"] != "1":
            raise RuntimeError(f"invalid diagnostics at row {index}")
        if row["RollingContact_contact_fallback"] != "0":
            raise RuntimeError(f"contact fallback at row {index}")
        for wheel in WHEELS:
            field = f"RollingContact_{wheel}_mode"
            if field in row and row[field] != "rolling":
                raise RuntimeError(f"{field} is {row[field]} at row {index}")

    transitions = []
    for index in range(1, len(rows)):
        previous = axes(rows[index - 1]["RollingContact_keyboard_status"])
        current = axes(rows[index]["RollingContact_keyboard_status"])
        # A pure-Q/E command followed by W/S/A/D is the safety-critical
        # handoff.  Q/E may also be intentionally combined with a translation
        # key (for example A+E) to command a curved path; those mixed-command
        # rows are not a yaw-to-translation transition and must not be
        # mistaken for a failed yaw-clear event.
        if (
            previous & YAW
            and not previous & TRANSLATION
            and current & TRANSLATION
            and not current & YAW
        ):
            if abs(finite(rows[index], "RollingContact_reference_yaw_rate")) > 1e-12:
                raise RuntimeError(f"non-zero yaw reference at transition row {index}")
            end = min(len(rows), index + 100)
            grace = max(
                finite(row, "RollingContact_keyboard_transition_grace")
                for row in rows[index:end]
            )
            if grace > args.grace_limit + 1e-9:
                raise RuntimeError(
                    f"transition grace {grace} exceeds {args.grace_limit}"
                )
            # The steering target is slewed independently for each hinge. The
            # controller must keep every drive wheel stopped until the last
            # hinge is aligned, otherwise a Q/E -> A/D handoff can launch two
            # wheels in the old direction while the other two are still
            # turning.
            steering_ready = next(
                (
                    j
                    for j in range(index, len(rows))
                    if rows[j]["RollingContact_keyboard_steering_ready"] == "1"
                ),
                None,
            )
            if steering_ready is None:
                raise RuntimeError(
                    f"steering never became ready after transition row {index}"
                )
            drive_fields = [f"RollingContact_{wheel}_rate" for wheel in WHEELS]
            for row_number in range(index, steering_ready):
                if any(
                    abs(finite(rows[row_number], field)) > 1e-12
                    for field in drive_fields
                ):
                    raise RuntimeError(
                        "drive rate was non-zero before all steering aligned "
                        f"at row {row_number}"
                    )
            speed = max(
                math.hypot(finite(row, "FloatingBase_linearVelocity_x"),
                           finite(row, "FloatingBase_linearVelocity_y"))
                for row in rows[index:end]
            )
            angular_speed = max(
                abs(finite(row, "FloatingBase_angularVelocity_z"))
                for row in rows[index:end]
            )
            if speed >= args.speed_limit or angular_speed >= args.speed_limit:
                raise RuntimeError(
                    f"transition speed linear={speed}, angular={angular_speed} "
                    f"exceeds {args.speed_limit}"
                )
            transitions.append(
                {
                    "row": index,
                    "time": finite(rows[index], "t"),
                    "speed": speed,
                    "angular_speed": angular_speed,
                }
            )

    if not transitions:
        raise RuntimeError("no Q/E-to-translation transition found")

    tail = rows[-args.tail_samples :]
    tail_position_error = max(
        finite(row, "RollingContact_base_position_tracking_error") for row in tail
    )
    tail_linear_speed = max(
        math.hypot(finite(row, "FloatingBase_linearVelocity_x"),
                   finite(row, "FloatingBase_linearVelocity_y"))
        for row in tail
    )
    tail_yaw_speed = max(
        abs(finite(row, "FloatingBase_angularVelocity_z")) for row in tail
    )
    report = {
        "rows": len(rows),
        "transitions": transitions,
        "tail_position_error": tail_position_error,
        "tail_linear_speed": tail_linear_speed,
        "tail_yaw_speed": tail_yaw_speed,
    }
    if tail_position_error > 1e-3 or tail_linear_speed > 1e-3 or tail_yaw_speed > 1e-3:
        raise RuntimeError(f"non-converged stopped tail: {report}")
    if args.output:
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
