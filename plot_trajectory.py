#!/usr/bin/env python3
"""
Plot the trajectory produced by the point_mass_obstacle executable.

Reads trajectory.csv (written next to the executable when it runs) and shows
the XY path together with the obstacle keep-out circle.

Usage:
    python3 plot_trajectory.py [path/to/trajectory.csv]

The obstacle centre/radius default to the values hard-coded in src/main.cpp
(centre (1,1), radius 0.5). Pass --cx/--cy/--r to override.
"""
import argparse
import csv

import matplotlib.pyplot as plt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", default="trajectory.csv")
    ap.add_argument("--cx", type=float, default=1.0)
    ap.add_argument("--cy", type=float, default=1.0)
    ap.add_argument("--r", type=float, default=0.5)
    args = ap.parse_args()

    px, py = [], []
    with open(args.csv) as f:
        for row in csv.DictReader(f):
            px.append(float(row["px"]))
            py.append(float(row["py"]))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(px, py, "-o", ms=3, label="trajectory")
    ax.plot(px[0], py[0], "gs", ms=10, label="start")
    ax.plot(px[-1], py[-1], "r*", ms=14, label="goal")
    ax.add_patch(plt.Circle((args.cx, args.cy), args.r, color="k", alpha=0.25,
                            label="obstacle"))
    ax.set_aspect("equal")
    ax.grid(True)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Point-mass point-to-point with obstacle avoidance")
    ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
