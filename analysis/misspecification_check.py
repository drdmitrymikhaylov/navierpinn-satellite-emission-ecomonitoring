"""Localisation when the estimator's dispersion class or wind speed is wrong.

The 2026-09-28 reference calculation (plume_localisation.py) handed the
estimator the true stability class and the true wind speed. This script re-runs
two of its cells - 1 km and 2 km tower spacing, sigma = 1 ug/m3, class D,
u = 3 m/s, 10 kg/h, 200 trials - with the same seeds, so the correctly specified
rows reproduce the stored numbers trial for trial, and then lets the estimator
assume a neighbouring class (C or E, one step; B or F, two steps) or a wind
speed 20 % off (2.4 or 3.6 m/s) while the truth stays at D and 3 m/s. The
source positions and noise draws are identical across the rows of a cell, so
every difference between rows is paired. A last row tries the obvious repair:
localise under each of the six classes and keep the one with the smallest
residual (the wind speed cannot be fitted the same way - it is degenerate with
the rate, since the plume depends on Q/u only).

Writes results/misspecification_check.json.
Run:  python3 analysis/misspecification_check.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from plume_localisation import localise, plume_unit, rotate, sensor_grid  # noqa: E402

OUT = ROOT / "results" / "misspecification_check.json"

TRUE_CLS, TRUE_U, Q_TRUE, SIGMA_N, N_TRIALS = "D", 3.0, 10.0, 1.0, 200
ASSUMPTIONS = {
    "D 3.0": ("D", 3.0),  # correctly specified; must reproduce plume_localisation.json
    "C 3.0": ("C", 3.0),
    "E 3.0": ("E", 3.0),
    "B 3.0": ("B", 3.0),
    "F 3.0": ("F", 3.0),
    "D 2.4": ("D", 2.4),
    "D 3.6": ("D", 3.6),
}
# (spacing, seed) - the seeds plume_localisation.main() used for these two cells.
CELLS = [(1000.0, 0), (2000.0, 3)]
FIT_CLASSES = ["A", "B", "C", "D", "E", "F"]


def q(a, p):
    return round(float(np.percentile(a, p)), 3)


def run_cell(spacing: float, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    sx, sy_ = sensor_grid(spacing, 12000.0)
    coarse_res = spacing / 20.0
    fine_res = max(5.0, spacing / 200.0)
    pos = {k: [] for k in ASSUMPTIONS}
    rate = {k: [] for k in ASSUMPTIONS}
    detected = []
    fit_cls, fit_pos, fit_rate = [], [], []
    for _ in range(N_TRIALS):
        x0 = rng.uniform(-spacing / 2, spacing / 2)
        y0 = rng.uniform(-spacing / 2, spacing / 2)
        theta = rng.uniform(0, 2 * np.pi)
        dx, dy = rotate(sx - x0, sy_ - y0, theta)
        clean = Q_TRUE * plume_unit(dx, dy, TRUE_U, TRUE_CLS)
        readings = clean + rng.normal(0.0, SIGMA_N, size=clean.shape)
        detected.append(bool(clean.max() >= 3.0 * SIGMA_N))
        for k, (cls_est, u_est) in ASSUMPTIONS.items():
            xh, yh, qh = localise(readings, sx, sy_, theta, u_est, cls_est, spacing, coarse_res, fine_res)
            pos[k].append(float(np.hypot(xh - x0, yh - y0)))
            rate[k].append(float(qh / Q_TRUE))
        # The repair: fit the class as well - localise under each of A-F at the true
        # wind speed and keep the class with the smallest residual sum of squares.
        best = (np.inf, None, None, None, None)
        for cls_est in FIT_CLASSES:
            xh, yh, qh = localise(readings, sx, sy_, theta, TRUE_U, cls_est, spacing, coarse_res, fine_res)
            dxh, dyh = rotate(sx - xh, sy_ - yh, theta)
            ss = float(np.sum((readings - qh * plume_unit(dxh, dyh, TRUE_U, cls_est)) ** 2))
            if ss < best[0]:
                best = (ss, cls_est, xh, yh, qh)
        fit_cls.append(best[1])
        fit_pos.append(float(np.hypot(best[2] - x0, best[3] - y0)))
        fit_rate.append(float(best[4] / Q_TRUE))
    det = np.array(detected)
    base_pos = np.array(pos["D 3.0"])
    base_rate = np.array(rate["D 3.0"])
    rows = {}
    for k, (cls_est, u_est) in ASSUMPTIONS.items():
        p = np.array(pos[k])
        r = np.array(rate[k])
        d = p - base_pos
        row = {
            "cls_assumed": cls_est,
            "u_assumed_m_s": u_est,
            "pos_err_median_m": round(float(np.median(p)), 1),
            "pos_err_p90_m": round(float(np.percentile(p, 90)), 1),
            "pos_err_within_100m_frac": round(float(np.mean(p <= 100.0)), 3),
            "pos_err_within_500m_frac": round(float(np.mean(p <= 500.0)), 3),
            "detected_pos_err_median_m": round(float(np.median(p[det])), 1),
            "detected_pos_err_p90_m": round(float(np.percentile(p[det], 90)), 1),
            "detected_rate_ratio_median": q(r[det], 50),
            "detected_rate_ratio_p10": q(r[det], 10),
            "detected_rate_ratio_p90": q(r[det], 90),
            "paired_vs_correct": {
                "worse_by_more_than_1m_frac": round(float(np.mean(d > 1.0)), 3),
                "better_by_more_than_1m_frac": round(float(np.mean(d < -1.0)), 3),
                "identical_within_1m_frac": round(float(np.mean(np.abs(d) <= 1.0)), 3),
                "pos_diff_median_m": round(float(np.median(d)), 1),
                "detected_pos_diff_median_m": round(float(np.median(d[det])), 1),
                "max_abs_pos_diff_m": round(float(np.max(np.abs(d))), 3),
                "rate_over_correct_median": q(r[det] / base_rate[det], 50),
                "rate_over_correct_max_abs_dev_from_median": round(
                    float(np.max(np.abs(r[det] / base_rate[det] - np.median(r[det] / base_rate[det])))), 6),
            },
        }
        rows[k] = row
    fp, fr = np.array(fit_pos), np.array(fit_rate)
    fc = np.array(fit_cls)
    dfit = fp - base_pos
    rows["fit class"] = {
        "cls_assumed": "fitted (A-F, smallest residual)",
        "u_assumed_m_s": TRUE_U,
        "pos_err_median_m": round(float(np.median(fp)), 1),
        "pos_err_p90_m": round(float(np.percentile(fp, 90)), 1),
        "pos_err_within_100m_frac": round(float(np.mean(fp <= 100.0)), 3),
        "pos_err_within_500m_frac": round(float(np.mean(fp <= 500.0)), 3),
        "detected_pos_err_median_m": round(float(np.median(fp[det])), 1),
        "detected_pos_err_p90_m": round(float(np.percentile(fp[det], 90)), 1),
        "detected_rate_ratio_median": q(fr[det], 50),
        "detected_rate_ratio_p10": q(fr[det], 10),
        "detected_rate_ratio_p90": q(fr[det], 90),
        "class_chosen_counts": {c: int(np.sum(fc == c)) for c in FIT_CLASSES},
        "class_chosen_counts_detected": {c: int(np.sum(fc[det] == c)) for c in FIT_CLASSES},
        "correct_class_frac": round(float(np.mean(fc == TRUE_CLS)), 3),
        "correct_class_frac_detected": round(float(np.mean(fc[det] == TRUE_CLS)), 3),
        "paired_vs_correct": {
            "worse_by_more_than_1m_frac": round(float(np.mean(dfit > 1.0)), 3),
            "better_by_more_than_1m_frac": round(float(np.mean(dfit < -1.0)), 3),
            "identical_within_1m_frac": round(float(np.mean(np.abs(dfit) <= 1.0)), 3),
            "pos_diff_median_m": round(float(np.median(dfit)), 1),
            "detected_pos_diff_median_m": round(float(np.median(dfit[det])), 1),
            "max_abs_pos_diff_m": round(float(np.max(np.abs(dfit))), 3),
            "rate_over_correct_median": q(fr[det] / base_rate[det], 50),
        },
    }
    return {
        "spacing_m": spacing,
        "seed": seed,
        "n_sensors": int(sx.size),
        "n_trials": N_TRIALS,
        "n_detected": int(det.sum()),
        "undetected_frac": round(1.0 - float(det.mean()), 3),
        "rows": rows,
    }


def unit_centreline_ratios() -> dict:
    """True-class (D) centreline concentration over the assumed class's, per kg/h, u = 3."""
    dists = np.array([500.0, 1000.0, 2000.0, 5000.0])
    zero = np.zeros_like(dists)
    c_true = plume_unit(dists, zero, TRUE_U, TRUE_CLS)
    out = {"distances_m": [int(v) for v in dists]}
    for cls in ["B", "C", "D", "E", "F"]:
        c = plume_unit(dists, zero, TRUE_U, cls)
        out[cls] = [round(float(v), 3) for v in c_true / c]
    return out


def main() -> None:
    t0 = time.time()
    out = {
        "truth": {"stability": TRUE_CLS, "u_m_s": TRUE_U, "q_true_kg_h": Q_TRUE,
                  "sigma_n_ug_m3": SIGMA_N, "n_trials": N_TRIALS,
                  "note": "same seeds as plume_localisation.py for these cells; rows are paired"},
        "assumptions": {k: {"cls": c, "u_m_s": u} for k, (c, u) in ASSUMPTIONS.items()},
        "centreline_ratio_true_over_assumed": unit_centreline_ratios(),
        "cells": [],
    }
    for spacing, seed in CELLS:
        cell = run_cell(spacing, seed)
        out["cells"].append(cell)
        for k, row in cell["rows"].items():
            pv = row["paired_vs_correct"]
            print(f"d={spacing:5.0f} {k}: med {row['pos_err_median_m']:7.1f} m  p90 {row['pos_err_p90_m']:7.1f}"
                  f"  det-med {row['detected_pos_err_median_m']:7.1f}  rate {row['detected_rate_ratio_median']:.3f}"
                  f" [{row['detected_rate_ratio_p10']:.3f},{row['detected_rate_ratio_p90']:.3f}]"
                  f"  worse {pv['worse_by_more_than_1m_frac']:.2f} better {pv['better_by_more_than_1m_frac']:.2f}"
                  f"  maxdiff {pv['max_abs_pos_diff_m']:.3f}  ({time.time()-t0:.0f}s)", flush=True)
    out["elapsed_s"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
