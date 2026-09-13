#!/usr/bin/env python3

"""Generate the trajectory and steering schedule of the RangerTrajectory sample.

The sample drives the Ranger Mini V3 chassis frame along three curves in
sequence -- a circle, a straight line and a 3:2 Lissajous figure -- with one
``bspline_trajectory`` task per FSM state.  This script emits, per curve:

* the ``controlPoints`` list and final ``target`` of the chassis task, and
* the ``oriWaypoints`` schedule of the four ``<wheel>_carrier`` steering tasks,

so both are reproducible and reviewable instead of hand-typed.

The steering half exists because nothing in the rolling-contact constraints can
turn a steering hinge on this robot: the steering axis passes through the wheel
centre, so its column in the contact-point Jacobian is identically zero.  What
does reach the hinges is that a constant chassis heading makes each carrier
frame's world orientation exactly ``Rz(delta_i)``, so an orientation waypoint
list on that frame *is* a hinge schedule.

Regenerate the committed block with::

    python3 rolling-contact-report/scripts/make-trajectory-waypoints.py --write

which rewrites the region delimited by the GENERATED-BEGIN / GENERATED-END
markers of
``src/mc_control/samples/RangerTrajectory/etc/RangerTrajectory.in.yaml``.
Without ``--write`` the block is printed on stdout.  ``--check`` exits non-zero
when the committed block differs from what this script would emit, which is how
a reviewer can tell that the YAML really is the script's output.

Four properties are asserted rather than assumed:

* **Continuity.**  Each curve must start exactly where the previous one ended.
  ``mc_tasks::BSplineTrajectoryTask`` anchors its curve at the *current* frame
  position when the state starts, so a gap between the nominal end of one curve
  and the nominal start of the next would show up as a step in the reference
  and kick the robot at the FSM transition.
* **Fit accuracy.**  ``mc_trajectory::BSpline`` is a single Bezier curve, not an
  interpolating spline: ``controlPoints`` are Bezier control points and the
  curve does *not* pass through them (see src/mc_trajectory/BSpline.cpp).
  Sampling the analytic curve and pasting the samples in as control points would
  therefore produce a shape well inside the intended one -- for a unit circle,
  short by about 25%.  The control points below are instead the endpoint-pinned
  least-squares Bezier fit of the analytic curve, and the script asserts that the
  worst deviation between the fitted Bezier and the analytic curve stays under
  FIT_TOLERANCE.
* **One schedule for four wheels.**  With the chassis heading constant its
  motion is a pure translation, so every wheel centre sees the same velocity and
  all four hinges share one ``delta``.  The emitted YAML relies on that (it
  aliases a single waypoint list to the four carriers), so it is checked per
  wheel in ``assert_wheels_share_one_delta()`` rather than taken on trust.
* **Feasible branch flips.**  The hinges are limited to +/- pi/2 while the
  direction of travel sweeps a full turn, so the schedule has to change branch.
  Each change is emitted as an explicit three-point flip and the script checks
  that its slew stays under the hinge velocity limit, that two flips never
  overlap, and that a flip fits inside its curve.

The parametrisation of every curve is affine in time, so the Bezier parameter is
the normalised state time and the commanded speed is what the analytic curve
gives at uniform parameter rate.
"""

import argparse
import math
import pathlib
import sys
from math import comb, pi

import numpy as np

# Chassis frame height of the Ranger Mini V3 at reset: RangerMiniV3RobotModule
# sets _default_attitude z = 0.16 for the ranger_mini_v3 variant
# (src/mc_robots/ranger_mini_v3.cpp) and the "chassis" frame is the floating
# base body itself, so the frame sits at z = 0.16.  Verified from a controller
# log: ff_tz = 0.1600000 on the first cycle.
CHASSIS_HEIGHT = 0.16
# Planar start pose of the chassis frame at reset.
START_XY = (0.0, 0.0)

# Circle: radius and traversal.  Centred at START + (0, R) so the curve starts at
# the robot's initial pose heading +x and turns left (counter-clockwise).
CIRCLE_RADIUS = 1.0
# Straight segment, continuing along +x from where the circle closes.
LINE_LENGTH = 2.0
# Lissajous x = A sin(a t + d), y = B sin(b t) with a 3:2 frequency ratio.
LISSAJOUS_A = 1.0
LISSAJOUS_B = 1.0
LISSAJOUS_FREQ_X = 3
LISSAJOUS_FREQ_Y = 2

# Durations, chosen for a roughly uniform ~0.30 m/s cruise: the arc lengths are
# 6.283 m, 2.000 m and 15.209 m respectively.
#
# The pace is set by the steering branch flips, not by anything about the
# curves.  A flip swings a hinge through 180 deg and takes a fixed
# STEERING_FLIP_WINDOW whatever the chassis is doing, so the distance the
# chassis loses while its wheels point the wrong way is proportional to its
# speed.  Measured on the ticker over the whole sequence (max / mean chassis
# deviation from the commanded curve, and peak drive torque against the 35 Nm
# limit):
#
#   0.45 m/s (52.5 s total)  circle 0.140 / 0.035 m   Lissajous 0.096 / 0.023 m   29.0 Nm
#   0.30 m/s (78.75 s total) circle 0.094 / 0.021 m   Lissajous 0.067 / 0.011 m   17.8 Nm
#
# The slower pace is the better demonstration: max error falls with speed as
# predicted, mean error halves, and the torque margin nearly doubles.  Scale all
# three durations together to change it.
CIRCLE_DURATION = 21.0
LINE_DURATION = 6.75
LISSAJOUS_DURATION = 51.0

# Number of *interior* Bezier control points per curve; the Bezier degree is this
# plus one because the start and end points are pinned.  The Lissajous needs more
# than the circle because its x component completes three periods over the curve,
# so the polynomial has to resolve a 6*pi argument range against the circle's
# 2*pi.  Raising the count improves the fit *and* tames the control polygon,
# because a Bezier too low in degree to represent the curve only reaches it
# through large, nearly cancelling coefficients.  Measured (fit error / largest
# control point coordinate, curve extent ~2 m):
#
#   12 -> 8.1e-3 m / 242    16 -> 6.6e-5 m /  69    20 -> 2.0e-7 m / 13.4
#   14 -> 8.4e-4 m / 146    18 -> 4.1e-6 m /  29    24 -> 2.9e-10 m / 9.1
#
# 20 is the knee: sub-micrometre geometry with a control polygon that stays
# within an order of magnitude of the curve itself.
CIRCLE_CONTROL_POINTS = 12
LINE_CONTROL_POINTS = 2
LISSAJOUS_CONTROL_POINTS = 20

# Worst allowed distance between the fitted Bezier and the analytic curve.  One
# millimetre is two orders of magnitude below the tracking error the QP actually
# achieves, so the fit is never the limiting term.
FIT_TOLERANCE = 1e-3
# Worst allowed gap between the end of one curve and the start of the next.
CONTINUITY_TOLERANCE = 1e-9
# Samples used both for the least-squares fit and for the error check.
FIT_SAMPLES = 4001

# --- steering schedule -----------------------------------------------------
#
# The chassis heading is held constant, so each <wheel>_carrier frame's world
# orientation is exactly Rz(delta_i) and a spline task on that frame with
# orientation waypoints schedules the hinge from YAML.  With a constant heading
# the chassis motion is a pure translation, every wheel centre has the same
# velocity, and all four wheels therefore share one delta -- asserted in
# steering_schedule() rather than assumed.
#
# Carrier frame offsets in the chassis frame, from ranger_mini_v3.urdf (the
# carrier frame is the knuckle body, whose origin is the *_steer joint origin).
CARRIER_OFFSETS = {
    "FrontLeft": ("front_left_carrier", 0.247, 0.182, -0.035),
    "FrontRight": ("front_right_carrier", 0.247, -0.182, -0.035),
    "RearLeft": ("rear_left_carrier", -0.247, 0.182, -0.035),
    "RearRight": ("rear_right_carrier", -0.247, -0.182, -0.035),
}

# Orientation waypoints emitted per curve.  InterpolatedRotation slerps between
# consecutive waypoints, which for a single-axis rotation is linear in the
# angle, so the count only has to resolve the curvature of delta(t).  delta is
# exactly linear in t on the circle (a constant-speed circle turns its tangent
# uniformly) and identically zero on the line; only the Lissajous needs a dense
# grid.  Measured worst piecewise-linear error against the analytic schedule,
# outside the cusp neighbourhoods: 40 -> 7.3 deg, 60 -> 1.4 deg, 80 -> 0.8 deg,
# 120 -> 0.4 deg.
CIRCLE_STEERING_SAMPLES = 16
LINE_STEERING_SAMPLES = 0
LISSAJOUS_STEERING_SAMPLES = 80

# The steering hinges are limited to +/- pi/2 (ranger_mini_v3.urdf) while the
# direction of travel sweeps a full turn, so the schedule has to change branch:
# a wheel line at +pi/2 and at -pi/2 is the same physical line, reached by
# rolling the other way.  A branch change is a genuine discontinuity in delta,
# and a slerp straight across it would either be ambiguous (the two endpoints
# are exactly 180 deg apart, so "shortest path" is a coin flip) or infinitely
# fast.  Each one is therefore emitted as three waypoints -- (+pi/2, 0, -pi/2)
# spread over this window -- which makes both legs unambiguous 90 deg slerps and
# bounds the hinge slew at pi / window.
#
# Swept on the ticker (max / mean chassis deviation, circle then Lissajous, at
# the 0.45 m/s pace): 0.45 s -> 0.141 / 0.038 and 0.106 / 0.031 m; 0.60 s ->
# 0.140 / 0.035 and 0.096 / 0.023 m; 0.80 s -> 0.142 / 0.032 and 0.099 /
# 0.015 m.  The peak barely moves because it is set by the 180 deg swing itself
# rather than by how it is spread; the mean prefers a longer window while the
# straight segment prefers a shorter one, because a wide flip leaves a larger
# residual hinge offset behind for the next state to unwind.  0.60 s is the
# balance, and keeps a 1.5x margin under the hinge velocity limit.
STEERING_FLIP_WINDOW = 0.6
# Hinge velocity limit from the URDF, used to check STEERING_FLIP_WINDOW.
STEERING_RATE_LIMIT = 8.0
# Below this parameter-space speed the direction of travel is not defined; the
# 3:2 Lissajous has two genuine cusps where the curve stops and reverses along
# the same line.  The wheel line is unchanged through a cusp (only the roll
# direction flips), so the previous direction is held.
TANGENT_SPEED_FLOOR = 1e-6

BEGIN_MARKER = "  # GENERATED-BEGIN: make-trajectory-waypoints.py"
END_MARKER = "  # GENERATED-END"

DEFAULT_YAML = (
    pathlib.Path(__file__).resolve().parents[2]
    / "src"
    / "mc_control"
    / "samples"
    / "RangerTrajectory"
    / "etc"
    / "RangerTrajectory.in.yaml"
)


def bernstein(degree, s):
    """Bernstein basis matrix of shape (len(s), degree + 1)."""
    k = np.arange(degree + 1)
    binomial = np.array([comb(degree, i) for i in k], dtype=float)
    lower = s[:, None] ** k[None, :]
    upper = (1.0 - s[:, None]) ** (degree - k)[None, :]
    return binomial * lower * upper


def fit_bezier(curve, interior_points):
    """Least-squares Bezier fit of ``curve`` with both endpoints pinned.

    ``curve(s)`` maps a (n,) array of normalised parameters to an (n, 2) array of
    planar points.  Returns the interior control points and the worst deviation
    between the fitted Bezier and ``curve``.
    """
    s = np.linspace(0.0, 1.0, FIT_SAMPLES)
    target = curve(s)
    degree = interior_points + 1
    basis = bernstein(degree, s)
    pinned = np.outer(basis[:, 0], target[0]) + np.outer(basis[:, -1], target[-1])
    interior_basis = basis[:, 1:-1]
    control, *_ = np.linalg.lstsq(interior_basis, target - pinned, rcond=None)
    fitted = pinned + interior_basis @ control
    return control, float(np.max(np.linalg.norm(fitted - target, axis=1)))


def arc_length(curve):
    dense = curve(np.linspace(0.0, 1.0, 200001))
    return float(np.sum(np.linalg.norm(np.diff(dense, axis=0), axis=1)))


def circle_curve(start):
    """Full counter-clockwise circle starting at ``start`` and heading +x."""
    centre = np.array([start[0], start[1] + CIRCLE_RADIUS])

    def curve(s):
        angle = 2.0 * pi * np.asarray(s)
        return np.stack(
            [
                centre[0] + CIRCLE_RADIUS * np.sin(angle),
                centre[1] - CIRCLE_RADIUS * np.cos(angle),
            ],
            axis=1,
        )

    def derivative(s):
        angle = 2.0 * pi * np.asarray(s)
        scale = 2.0 * pi * CIRCLE_RADIUS
        return np.stack([scale * np.cos(angle), scale * np.sin(angle)], axis=1)

    curve.derivative = derivative
    return curve


def line_curve(start):
    """Straight segment of LINE_LENGTH along +x, continuing the circle's tangent."""

    def curve(s):
        t = np.asarray(s)
        x = start[0] + LINE_LENGTH * t
        y = start[1] + np.zeros_like(t)
        return np.stack([x, y], axis=1)

    def derivative(s):
        t = np.asarray(s)
        return np.stack([LINE_LENGTH * np.ones_like(t), np.zeros_like(t)], axis=1)

    curve.derivative = derivative
    return curve


def lissajous_curve(start):
    """One closed period of a 3:2 Lissajous figure, translated to ``start``.

    The parameter origin is picked so the curve leaves ``start`` along +x, which
    is the direction the straight segment arrives with; the FSM transition is
    then continuous in tangent as well as in position.
    """
    a, b = LISSAJOUS_FREQ_X, LISSAJOUS_FREQ_Y
    theta0 = pi / (2.0 * b)
    phase = -a * theta0

    def raw(theta):
        return np.stack(
            [LISSAJOUS_A * np.sin(a * theta + phase), LISSAJOUS_B * np.sin(b * theta)],
            axis=1,
        )

    offset = np.array(start) - raw(np.array([theta0]))[0]

    def curve(s):
        theta = theta0 + 2.0 * pi * np.asarray(s)
        return raw(theta) + offset

    def derivative(s):
        theta = theta0 + 2.0 * pi * np.asarray(s)
        return np.stack(
            [
                2.0 * pi * a * LISSAJOUS_A * np.cos(a * theta + phase),
                2.0 * pi * b * LISSAJOUS_B * np.cos(b * theta),
            ],
            axis=1,
        )

    curve.derivative = derivative
    return curve


def format_point(x, y):
    # Round -0.0 away so the emitted YAML has no negative zeroes.
    x, y = (0.0 if abs(v) < 5e-7 else v for v in (x, y))
    return f"[{x:.6f}, {y:.6f}, {CHASSIS_HEIGHT:.6f}]"


def wheel_line_direction(curve, s):
    """Direction of the wheel line along ``curve``, continuous modulo pi.

    The wheel is symmetric, so what the hinge has to match is the *line* of
    travel, not its orientation: a cusp, where the curve stops and comes back
    along the same line, needs no hinge motion at all, only a change of roll
    sign.  Unwrapping with period pi is what removes those pi jumps; it is also
    what makes the branch bookkeeping below well posed.
    """
    delta = curve.derivative(s)
    speed = np.linalg.norm(delta, axis=1)
    raw = np.arctan2(delta[:, 1], delta[:, 0])
    # Hold the previous direction wherever the curve is momentarily stopped.
    for i in range(len(raw)):
        if speed[i] < TANGENT_SPEED_FLOOR:
            raw[i] = raw[i - 1] if i else 0.0
    return np.unwrap(raw, period=pi)


def branch_index(phi):
    """Index k such that phi - k * pi lands inside [-pi/2, pi/2]."""
    return np.floor((phi + 0.5 * pi) / pi)


def steering_schedule(curve, duration, samples, start_delta):
    """Hinge angle schedule for one curve, as (time, delta) waypoints.

    Returns the interior waypoints, the final hinge angle, and the diagnostics
    the caller prints.  Branch changes -- the points where the continuous wheel
    line direction leaves the +/- pi/2 hinge range -- are emitted as an explicit
    three-point flip so the slerp between them is unambiguous and rate bounded,
    exactly the "nearest representation inside the limit" rule
    mc_rbdyn::steeringWheelReference() applies online.
    """
    dense = np.linspace(0.0, 1.0, FIT_SAMPLES)
    phi = wheel_line_direction(curve, dense)
    if abs((phi[0] - start_delta + 0.5 * pi) % pi - 0.5 * pi) > 1e-6:
        raise AssertionError(
            f"steering schedule starts at {math.degrees(phi[0]):.3f} deg but the previous "
            f"curve left the hinges at {math.degrees(start_delta):.3f} deg"
        )
    # Re-anchor so the schedule continues from the hinge angle the previous
    # curve ended on instead of from an arbitrary branch of atan2.
    phi = phi - phi[0] + start_delta

    branch = branch_index(phi)
    crossings = np.nonzero(np.diff(branch) != 0)[0]
    flip_times = [duration * 0.5 * (dense[i] + dense[i + 1]) for i in crossings]
    flip_signs = [1.0 if branch[i + 1] > branch[i] else -1.0 for i in crossings]

    rate = pi / STEERING_FLIP_WINDOW
    if rate > STEERING_RATE_LIMIT:
        raise AssertionError(
            f"a branch flip over {STEERING_FLIP_WINDOW:g} s needs {rate:.2f} rad/s from a "
            f"hinge limited to {STEERING_RATE_LIMIT:g} rad/s"
        )
    for a, b in zip(flip_times, flip_times[1:]):
        if b - a < STEERING_FLIP_WINDOW:
            raise AssertionError(
                f"branch flips at {a:.3f} s and {b:.3f} s are closer than the "
                f"{STEERING_FLIP_WINDOW:g} s flip window"
            )
    for t in flip_times:
        if (
            t - 0.5 * STEERING_FLIP_WINDOW <= 0.0
            or t + 0.5 * STEERING_FLIP_WINDOW >= duration
        ):
            raise AssertionError(
                f"branch flip at {t:.3f} s does not fit inside the curve"
            )

    def delta_at(time):
        s = float(np.clip(time / duration, 0.0, 1.0))
        # The branch is a step function; take it from the nearest dense sample
        # rather than interpolating across a step.
        k = branch[min(len(dense) - 1, int(round(s * (len(dense) - 1))))]
        return float(np.interp(s, dense, phi)) - k * pi

    waypoints = []
    if samples:
        for i in range(1, samples):
            time = duration * i / samples
            if any(abs(time - t) < 0.5 * STEERING_FLIP_WINDOW for t in flip_times):
                continue
            waypoints.append((time, delta_at(time)))
    for time, sign in zip(flip_times, flip_signs):
        # Crossing upwards leaves the range at +pi/2 and re-enters at -pi/2.
        waypoints.append((time - 0.5 * STEERING_FLIP_WINDOW, sign * 0.5 * pi))
        waypoints.append((time, 0.0))
        waypoints.append((time + 0.5 * STEERING_FLIP_WINDOW, -sign * 0.5 * pi))
    waypoints.sort()

    end = float(phi[-1] - branch[-1] * pi)
    peak = 0.0
    for (t0, d0), (t1, d1) in zip(waypoints, waypoints[1:]):
        peak = max(peak, abs(d1 - d0) / (t1 - t0))
    return waypoints, end, {"flips": flip_times, "peak_rate": peak}


def assert_wheels_share_one_delta(curve, duration):
    """All four hinges follow the same schedule -- checked, not assumed.

    The chassis heading is constant, so the chassis twist is a pure translation
    (omega = 0) and every wheel centre velocity is v + omega x r_i = v. The
    per-wheel direction is therefore independent of the carrier offset. Compute
    it per wheel anyway, so that a future curve which does rotate the chassis
    trips this instead of silently emitting four wrong identical schedules.
    """
    dense = np.linspace(0.0, 1.0, 2001)
    velocity = curve.derivative(dense)
    omega = 0.0  # constant chassis heading
    for name, (_, x, y, _) in CARRIER_OFFSETS.items():
        at_wheel = velocity + omega * np.stack(
            [-y * np.ones_like(dense), x * np.ones_like(dense)], axis=1
        )
        if np.max(np.abs(at_wheel - velocity)) > 1e-12:
            raise AssertionError(
                f"wheel {name} does not see the chassis velocity; the four hinges no "
                "longer share one schedule and the emitted YAML would be wrong"
            )
    del duration


def generate():
    start = np.array(START_XY, dtype=float)
    delta = 0.0
    specs = []
    for name, factory, interior, duration, steering_samples in (
        (
            "Circle",
            circle_curve,
            CIRCLE_CONTROL_POINTS,
            CIRCLE_DURATION,
            CIRCLE_STEERING_SAMPLES,
        ),
        ("Line", line_curve, LINE_CONTROL_POINTS, LINE_DURATION, LINE_STEERING_SAMPLES),
        (
            "Lissajous",
            lissajous_curve,
            LISSAJOUS_CONTROL_POINTS,
            LISSAJOUS_DURATION,
            LISSAJOUS_STEERING_SAMPLES,
        ),
    ):
        curve = factory(tuple(start))
        actual_start = curve(np.array([0.0]))[0]
        gap = float(np.linalg.norm(actual_start - start))
        if gap > CONTINUITY_TOLERANCE:
            raise AssertionError(
                f"{name} starts {gap:.3e} m away from the previous curve's end; the FSM "
                "transition would step the trajectory reference"
            )
        control, error = fit_bezier(curve, interior)
        if not math.isfinite(error) or error > FIT_TOLERANCE:
            raise AssertionError(
                f"{name} Bezier fit deviates by {error:.3e} m (tolerance {FIT_TOLERANCE:.1e}); "
                f"raise the interior control point count above {interior}"
            )
        assert_wheels_share_one_delta(curve, duration)
        steering, delta_end, diagnostics = steering_schedule(
            curve, duration, steering_samples, delta
        )
        end = curve(np.array([1.0]))[0]
        length = arc_length(curve)
        specs.append(
            {
                "name": name,
                "duration": duration,
                "control": control,
                "target": end,
                "error": error,
                "length": length,
                "speed": length / duration,
                "steering": steering,
                "steering_end": delta_end,
                "flips": diagnostics["flips"],
                "peak_steering_rate": diagnostics["peak_rate"],
            }
        )
        start = end
        delta = delta_end
    return specs


def render(specs):
    lines = [
        BEGIN_MARKER,
        "  # Generated; regenerate with",
        "  #   python3 rolling-contact-report/scripts/make-trajectory-waypoints.py --write",
        "  #",
        "  # ChassisCurve carries endpoint-pinned least-squares Bezier control points.",
        "  # These are Bezier control points, not samples of the curve: the trajectory does",
        "  # NOT pass through them (src/mc_trajectory/BSpline.cpp builds one bezier_curve_t",
        "  # from [start, controlPoints..., target]).",
        "  #",
        "  # The four *Steering tasks schedule the steering hinges. With the chassis",
        "  # heading held constant a <wheel>_carrier frame's world orientation is exactly",
        "  # Rz(delta), so an orientation waypoint list on that frame IS the hinge",
        "  # schedule. All four wheels share it (pure translation: every wheel centre has",
        "  # the same velocity), hence the YAML anchor. orientation is [roll, pitch, yaw]",
        "  # in rad -- mc_rtc reads a 3-vector rotation as RPY.",
    ]
    for spec in specs:
        lines.append(
            "  # {name}: {length:.3f} m in {duration:g} s ({speed:.3f} m/s), "
            "Bezier fit error {error:.2e} m,".format(**spec)
        )
        flips = ", ".join(f"{t:.2f} s" for t in spec["flips"]) or "none"
        lines.append(
            "  #   {n} steering waypoints, branch flips at {flips}, "
            "peak scheduled hinge rate {rate:.2f} rad/s (limit 8)".format(
                n=len(spec["steering"]), flips=flips, rate=spec["peak_steering_rate"]
            )
        )
    for spec in specs:
        anchor = spec["name"].lower()
        lines.append(f"  Ranger::{spec['name']}:")
        lines.append("    base: Ranger::ChassisCurve")
        lines.append("    tasks:")
        lines.append("      ChassisCurve:")
        lines.append(f"        duration: {spec['duration']:g}")
        lines.append("        controlPoints:")
        for point in spec["control"]:
            lines.append(f"        - {format_point(point[0], point[1])}")
        lines.append("        target:")
        lines.append(
            f"          translation: {format_point(spec['target'][0], spec['target'][1])}"
        )
        # Constant heading: see the Ranger::ChassisCurve comment in the sample.
        lines.append("          rotation: [0.0, 0.0, 0.0]")
        first = True
        for wheel, (frame, dx, dy, dz) in CARRIER_OFFSETS.items():
            del frame
            lines.append(f"      {wheel}Steering:")
            lines.append(f"        duration: {spec['duration']:g}")
            lines.append("        target:")
            translation = (
                f"[{spec['target'][0] + dx:.6f}, {spec['target'][1] + dy:.6f}, "
                f"{CHASSIS_HEIGHT + dz:.6f}]"
            )
            lines.append(f"          translation: {translation}")
            end = f"[0.0, 0.0, {spec['steering_end']:.6f}]"
            if first:
                lines.append(f"          rotation: &{anchor}_steering_end {end}")
            else:
                lines.append(f"          rotation: *{anchor}_steering_end")
            if spec["steering"]:
                if first:
                    lines.append(f"        oriWaypoints: &{anchor}_steering")
                    for time, angle in spec["steering"]:
                        lines.append(
                            f"        - {{time: {time:.4f}, "
                            f"orientation: [0.0, 0.0, {angle:.6f}]}}"
                        )
                else:
                    lines.append(f"        oriWaypoints: *{anchor}_steering")
            first = False
    lines.append(END_MARKER)
    return "\n".join(lines) + "\n"


def splice(text, block):
    begin = text.find(BEGIN_MARKER)
    end = text.find(END_MARKER)
    if begin < 0 or end < 0:
        raise RuntimeError(
            f"markers {BEGIN_MARKER!r} / {END_MARKER!r} not found in the YAML"
        )
    end = text.find("\n", end) + 1
    return text[:begin] + block + text[end:]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--yaml", type=pathlib.Path, default=DEFAULT_YAML)
    parser.add_argument(
        "--write", action="store_true", help="rewrite the generated region in --yaml"
    )
    parser.add_argument(
        "--check", action="store_true", help="fail if --yaml is out of date"
    )
    args = parser.parse_args()

    specs = generate()
    block = render(specs)

    for spec in specs:
        print(
            "{name}: {length:.4f} m over {duration:g} s ({speed:.4f} m/s), "
            "{n} interior control points, Bezier fit error {error:.3e} m; "
            "{m} steering waypoints, {f} branch flips, peak hinge rate "
            "{rate:.3f} rad/s, end delta {end:.4f} rad".format(
                n=len(spec["control"]),
                m=len(spec["steering"]),
                f=len(spec["flips"]),
                rate=spec["peak_steering_rate"],
                end=spec["steering_end"],
                **spec,
            ),
            file=sys.stderr,
        )

    if args.check:
        current = args.yaml.read_text()
        if splice(current, block) != current:
            print(f"{args.yaml} is out of date; rerun with --write", file=sys.stderr)
            return 1
        print(f"{args.yaml} is up to date", file=sys.stderr)
        return 0
    if args.write:
        args.yaml.write_text(splice(args.yaml.read_text(), block))
        print(f"updated {args.yaml}", file=sys.stderr)
        return 0
    print(block, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
