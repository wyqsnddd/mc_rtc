#!/usr/bin/env python3
"""Validate the Ranger Mini V3 floating-base/chassis frame contract."""

from __future__ import annotations

import argparse
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path


TOLERANCE = 1e-12


def vector(element: ET.Element, attribute: str) -> tuple[float, ...]:
    return tuple(float(value) for value in element.attrib.get(attribute, "").split())


def check_vector(
    actual: tuple[float, ...], expected: tuple[float, ...], label: str
) -> None:
    if len(actual) != len(expected) or any(
        abs(a - b) > TOLERANCE for a, b in zip(actual, expected)
    ):
        raise ValueError(f"{label}: expected {expected}, got {actual}")


def validate_urdf(path: Path) -> None:
    root = ET.parse(path).getroot()
    chassis = root.find("./link[@name='chassis']")
    if chassis is None:
        raise ValueError("URDF has no chassis link")
    inertial = chassis.find("./inertial/origin")
    visual = chassis.find("./visual/origin")
    collision = chassis.find("./collision/origin")
    if inertial is None or visual is None or collision is None:
        raise ValueError("URDF chassis is missing an origin")
    for element, label in (
        (inertial, "URDF inertial"),
        (visual, "URDF visual"),
        (collision, "URDF collision"),
    ):
        check_vector(vector(element, "xyz"), (0.0, 0.0, 0.0), label)
        check_vector(vector(element, "rpy"), (0.0, 0.0, 0.0), f"{label} rotation")


def validate_mujoco(path: Path) -> None:
    root = ET.parse(path).getroot()
    chassis = root.find("./worldbody/body[@name='chassis']")
    if chassis is None:
        raise ValueError("MuJoCo has no chassis body")
    check_vector(vector(chassis, "pos"), (0.0, 0.0, 0.16), "MuJoCo chassis body")
    if chassis.find("./freejoint[@name='root']") is None:
        raise ValueError("MuJoCo chassis has no root free joint")
    inertial = chassis.find("./inertial")
    marker = chassis.find("./site[@name='chassis_frame']")
    shell = chassis.find("./geom[@name='chassis_shell_visual']")
    collision = chassis.find("./geom[@name='chassis_collision']")
    for element, label in (
        (inertial, "MuJoCo inertial"),
        (marker, "MuJoCo chassis marker"),
        (shell, "MuJoCo shell"),
        (collision, "MuJoCo collision"),
    ):
        if element is None:
            raise ValueError(f"missing {label}")
        check_vector(vector(element, "pos"), (0.0, 0.0, 0.0), label)
    # ElementTree has no XPath contains() support; filter the direct children.
    knuckles = [
        body
        for body in chassis.findall("./body")
        if body.attrib.get("name", "").endswith("_knuckle")
    ]
    if len(knuckles) != 4:
        raise ValueError(f"expected four steering knuckles, got {len(knuckles)}")
    for knuckle in knuckles:
        position = vector(knuckle, "pos")
        if len(position) != 3 or not math.isclose(
            position[2], -0.035, abs_tol=TOLERANCE
        ):
            raise ValueError(
                f"{knuckle.attrib['name']} is not on the axle plane: {position}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--mujoco", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    validate_urdf(args.urdf)
    validate_mujoco(args.mujoco)
    result = {
        "urdf": str(args.urdf),
        "mujoco": str(args.mujoco),
        "floating_base_origin": [0.0, 0.0, 0.0],
        "default_height": 0.16,
        "wheel_axle_offset": [0.0, 0.0, -0.035],
        "status": "pass",
    }
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
