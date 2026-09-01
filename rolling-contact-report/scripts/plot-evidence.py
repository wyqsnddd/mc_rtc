#!/usr/bin/env python3

"""Render the retained rolling-contact evidence as a dependency-free SVG."""

import argparse
import html
import json
import pathlib


WIDTH = 1280
HEIGHT = 1180
LEFT = 90
RIGHT = 30
PLOT_WIDTH = WIDTH - LEFT - RIGHT
COLORS = {"Tasks": "#1976d2", "TVM": "#ef6c00"}


def text(x, y, value, size=14, anchor="start", weight="normal", rotate=None):
    transform = f' transform="rotate({rotate} {x} {y})"' if rotate else ""
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" '
        f'text-anchor="{anchor}" font-weight="{weight}"{transform}>'
        f"{html.escape(str(value))}</text>"
    )


def line(x1, y1, x2, y2, color="#9e9e9e", width=1, dash=None):
    dashed = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" '
        f'y2="{y2:.2f}" stroke="{color}" stroke-width="{width}"{dashed}/>'
    )


def axes(elements, top, height, maximum, ticks, title, unit):
    bottom = top + height
    elements.append(text(LEFT, top - 22, title, 18, weight="bold"))
    elements.append(line(LEFT, top, LEFT, bottom, "#424242", 1.5))
    elements.append(line(LEFT, bottom, WIDTH - RIGHT, bottom, "#424242", 1.5))
    for tick in range(ticks + 1):
        value = maximum * tick / ticks
        y = bottom - height * tick / ticks
        elements.append(line(LEFT, y, WIDTH - RIGHT, y, "#e0e0e0"))
        elements.append(text(LEFT - 10, y + 5, f"{value:.3g}", 12, anchor="end"))
    elements.append(text(20, top + height / 2, unit, 13, anchor="middle", rotate=-90))
    return bottom


def grouped_bars(elements, values, labels, top, height, maximum):
    count = len(labels)
    group = PLOT_WIDTH / count
    bar_width = min(15, group * 0.32)
    bottom = top + height
    for index, label in enumerate(labels):
        center = LEFT + group * (index + 0.5)
        for offset, backend in ((-0.55, "Tasks"), (0.55, "TVM")):
            value = values[backend][index]
            bar_height = height * value / maximum if maximum else 0
            x = center + offset * bar_width - bar_width / 2
            y = bottom - bar_height
            elements.append(
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width:.2f}" '
                f'height="{bar_height:.2f}" fill="{COLORS[backend]}"/>'
            )
        short = label.replace("diff-", "D:", 1).replace("four-", "4S:", 1)
        elements.append(text(center, bottom + 12, short, 10, anchor="end", rotate=-50))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mujoco", type=pathlib.Path, required=True)
    parser.add_argument("--timing", type=pathlib.Path, required=True)
    args = parser.parse_args()

    mujoco = json.loads(args.mujoco.read_text())
    timing = json.loads(args.timing.read_text())
    cases = mujoco["cases"]
    labels = sorted({case["name"] for case in cases})
    by_key = {(case["backend"], case["name"]): case for case in cases}

    tracking = {
        backend: [by_key[(backend, name)]["tracking"]["position_max_m"] for name in labels]
        for backend in COLORS
    }
    slip = {
        backend: [by_key[(backend, name)]["contact"]["rolling_slip_max_mps"] for name in labels]
        for backend in COLORS
    }

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
        f'viewBox="0 0 {WIDTH} {HEIGHT}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<g font-family="DejaVu Sans, sans-serif" fill="#212121">',
        text(WIDTH / 2, 38, "CPU rolling-contact validation summary", 24, anchor="middle", weight="bold"),
        text(WIDTH / 2, 62, "Ubuntu 24.04 · Ryzen 9 9950X · Tasks and TVM · GPU devices hidden", 14, anchor="middle"),
    ]

    top = 110
    height = 180
    maximum = 0.25
    bottom = axes(elements, top, height, maximum, 5, "10,000-cycle controller timing", "P99 wall time (ms)")
    timing_cases = sorted(timing["cases"], key=lambda case: (case["backend"], case["name"]))
    slot = PLOT_WIDTH / len(timing_cases)
    for index, case in enumerate(timing_cases):
        value = case["cycle_wall_ms"]["p99"]
        x = LEFT + slot * index + slot * 0.2
        width = slot * 0.6
        bar_height = height * value / maximum
        y = bottom - bar_height
        elements.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{width:.2f}" height="{bar_height:.2f}" '
            f'fill="{COLORS[case["backend"]]}"/>'
        )
        elements.append(text(x + width / 2, y - 7, f"{value:.3f}", 12, anchor="middle"))
        elements.append(
            text(
                x + width / 2,
                bottom + 18,
                f'{case["backend"]} / {case["name"]}',
                11,
                anchor="end",
                rotate=-20,
            )
        )
    elements.append(text(WIDTH - RIGHT, top - 22, "5 ms deadline (outside plotted range)", 12, anchor="end"))

    top = 455
    height = 230
    tracking_max = max(max(values) for values in tracking.values()) * 1.05
    axes(elements, top, height, tracking_max, 5, "Physical CPU suite: maximum position tracking error", "metres")
    grouped_bars(elements, tracking, labels, top, height, tracking_max)

    top = 840
    height = 230
    slip_max = max(max(values) for values in slip.values()) * 1.05
    axes(elements, top, height, slip_max, 5, "Physical CPU suite: maximum rolling slip", "m/s")
    grouped_bars(elements, slip, labels, top, height, slip_max)

    legend_y = HEIGHT - 24
    legend_x = WIDTH / 2 - 100
    for index, backend in enumerate(("Tasks", "TVM")):
        x = legend_x + index * 140
        elements.append(f'<rect x="{x:.2f}" y="{legend_y - 12}" width="22" height="12" fill="{COLORS[backend]}"/>')
        elements.append(text(x + 30, legend_y, backend, 13))
    elements.extend(["</g>", "</svg>"])
    print("\n".join(elements))


if __name__ == "__main__":
    main()
