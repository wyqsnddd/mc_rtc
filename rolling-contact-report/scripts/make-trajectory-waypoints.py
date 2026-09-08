#!/usr/bin/env python3

"""Generate the Bezier control points of the RangerTrajectory FSM sample.

The sample drives the Ranger Mini V3 chassis frame along three curves in
sequence -- a circle, a straight line and a 3:2 Lissajous figure -- with one
``bspline_trajectory`` task per FSM state.  This script emits the
``controlPoints`` list and the final ``target`` of each of those tasks so the
geometry is reproducible and reviewable instead of hand-typed.

Regenerate the committed block with::

    python3 rolling-contact-report/scripts/make-trajectory-waypoints.py --write

which rewrites the region delimited by the GENERATED-BEGIN / GENERATED-END
markers of
``src/mc_control/samples/RangerTrajectory/etc/RangerTrajectory.in.yaml``.
Without ``--write`` the block is printed on stdout.  ``--check`` exits non-zero
when the committed block differs from what this script would emit, which is how
a reviewer can tell that the YAML really is the script's output.

Two properties are asserted rather than assumed:

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

# Chassis frame height of the Ranger Mini V3 at reset: RollingContactRobotModule
# sets _default_attitude z = 0.16 for the ranger_mini_v3 variant
# (src/mc_robots/rolling_contact.cpp) and the "chassis" frame is the floating
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

# Durations, chosen for a roughly uniform ~0.45 m/s cruise: the arc lengths are
# 6.283 m, 2.000 m and 15.209 m respectively.
CIRCLE_DURATION = 14.0
LINE_DURATION = 4.5
LISSAJOUS_DURATION = 34.0

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

    return curve


def line_curve(start):
    """Straight segment of LINE_LENGTH along +x, continuing the circle's tangent."""

    def curve(s):
        t = np.asarray(s)
        x = start[0] + LINE_LENGTH * t
        y = start[1] + np.zeros_like(t)
        return np.stack([x, y], axis=1)

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

    return curve


def format_point(x, y):
    # Round -0.0 away so the emitted YAML has no negative zeroes.
    x, y = (0.0 if abs(v) < 5e-7 else v for v in (x, y))
    return f"[{x:.6f}, {y:.6f}, {CHASSIS_HEIGHT:.6f}]"


def generate():
    start = np.array(START_XY, dtype=float)
    specs = []
    for name, factory, interior, duration in (
        ("Circle", circle_curve, CIRCLE_CONTROL_POINTS, CIRCLE_DURATION),
        ("Line", line_curve, LINE_CONTROL_POINTS, LINE_DURATION),
        ("Lissajous", lissajous_curve, LISSAJOUS_CONTROL_POINTS, LISSAJOUS_DURATION),
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
            }
        )
        start = end
    return specs


def render(specs):
    lines = [
        BEGIN_MARKER,
        "  # Endpoint-pinned least-squares Bezier control points; regenerate with",
        "  #   python3 rolling-contact-report/scripts/make-trajectory-waypoints.py --write",
        "  # These are Bezier control points, not samples of the curve: the trajectory does",
        "  # NOT pass through them (src/mc_trajectory/BSpline.cpp builds one bezier_curve_t",
        "  # from [start, controlPoints..., target]).",
    ]
    for spec in specs:
        lines.append(
            "  # {name}: {length:.3f} m in {duration:g} s ({speed:.3f} m/s), "
            "Bezier fit error {error:.2e} m".format(**spec)
        )
    for spec in specs:
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
            "{n} interior control points, Bezier fit error {error:.3e} m".format(
                n=len(spec["control"]), **spec
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
