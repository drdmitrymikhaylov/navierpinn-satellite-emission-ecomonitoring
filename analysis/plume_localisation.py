"""What a tower-mounted sensor network can resolve: a Gaussian-plume reference calculation.

The project page says a sparse network of gas sensors on telecom towers, plus
satellite columns, "resolves where the emission actually came from". Before any
network is trained, the steady Gaussian plume (the closed-form solution of the
advection-diffusion equation with constant wind and power-law spreading) says
how much a sensor grid can resolve at all. This script computes that on
published dispersion coefficients (Briggs rural, open-country sigma_y and
sigma_z as a function of downwind distance, Pasquill-Gifford classes A-F).

Model (ground-level source, ground-level receptors, no plume rise, flat terrain):

    C(x, y) = Q / (pi * u * sigma_y(x) * sigma_z(x)) * exp(-y^2 / (2 sigma_y(x)^2))

Part 1 - forward: centreline concentration per kg/h and plume half-width by
stability class and distance.

Part 2 - inverse, Monte Carlo: sensors on a square grid of spacing d; a source
of rate Q sits at a random point of the central cell; wind speed u and
direction known (or perturbed by a fixed angular error); each sensor reads the
plume plus Gaussian noise of sd sigma_n. The estimator is maximum likelihood
over a position grid (coarse pass over the cell, fine pass around the
optimum), with the rate solved in closed form at every candidate (least
squares, non-negative). Reported: median and 90th-percentile
localisation error, rate error, and the fraction of trials in which no sensor
reads above 3 sigma_n (nothing to localise).

Pure numpy. Writes results/plume_localisation.json.
Run:  python3 analysis/plume_localisation.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "plume_localisation.json"

# Briggs (1973) rural / open-country coefficients, x in metres.
BRIGGS = {
    "A": ((0.22, 0.0001, -0.5), (0.20, 0.0, 0.0)),
    "B": ((0.16, 0.0001, -0.5), (0.12, 0.0, 0.0)),
    "C": ((0.11, 0.0001, -0.5), (0.08, 0.0002, -0.5)),
    "D": ((0.08, 0.0001, -0.5), (0.06, 0.0015, -0.5)),
    "E": ((0.06, 0.0001, -0.5), (0.03, 0.0003, -1.0)),
    "F": ((0.04, 0.0001, -0.5), (0.016, 0.0003, -1.0)),
}

KG_PER_H_TO_G_PER_S = 1000.0 / 3600.0
G_PER_M3_TO_UG_PER_M3 = 1e6


def sigmas(x: np.ndarray, cls: str) -> tuple[np.ndarray, np.ndarray]:
    (ay, by, py), (az, bz, pz) = BRIGGS[cls]
    x = np.maximum(x, 1.0)
    sy = ay * x * (1.0 + by * x) ** py
    sz = az * x * (1.0 + bz * x) ** pz
    return sy, sz


def plume_unit(dx: np.ndarray, dy: np.ndarray, u: float, cls: str) -> np.ndarray:
    """Concentration in ug/m3 for Q = 1 kg/h. dx downwind, dy crosswind (m)."""
    sy, sz = sigmas(dx, cls)
    q = KG_PER_H_TO_G_PER_S
    c = q / (np.pi * u * sy * sz) * np.exp(-0.5 * (dy / sy) ** 2)
    c = np.where(dx > 0, c, 0.0)
    return c * G_PER_M3_TO_UG_PER_M3


def forward_table(u: float) -> dict:
    dists = [500, 1000, 2000, 5000, 10000]
    out = {"u_m_s": u, "distances_m": dists, "classes": {}}
    for cls in BRIGGS:
        x = np.array(dists, dtype=float)
        sy, sz = sigmas(x, cls)
        c0 = plume_unit(x, np.zeros_like(x), u, cls)
        half_width = np.sqrt(2.0 * np.log(2.0)) * sy  # crosswind distance to C/2
        out["classes"][cls] = {
            "centreline_ug_m3_per_kg_h": [round(float(v), 4) for v in c0],
            "sigma_y_m": [round(float(v), 1) for v in sy],
            "sigma_z_m": [round(float(v), 1) for v in sz],
            "half_width_m": [round(float(v), 1) for v in half_width],
        }
    return out


def rotate(x: np.ndarray, y: np.ndarray, theta: float) -> tuple[np.ndarray, np.ndarray]:
    """Coordinates in a frame whose x axis points downwind (wind direction theta)."""
    c, s = np.cos(theta), np.sin(theta)
    return c * x + s * y, -s * x + c * y


def sensor_grid(spacing: float, half_extent: float) -> tuple[np.ndarray, np.ndarray]:
    n = int(np.floor(half_extent / spacing))
    coords = np.arange(-n, n + 1) * spacing
    gx, gy = np.meshgrid(coords, coords, indexing="ij")
    return gx.ravel(), gy.ravel()


def _ml_on_grid(readings, sx, sy_, theta_assumed, u, cls, cx, cy):
    best = (np.inf, 0.0, 0.0, 0.0)
    chunk = 2000
    for i in range(0, cx.size, chunk):
        ccx = cx[i : i + chunk][:, None]
        ccy = cy[i : i + chunk][:, None]
        dx, dy = rotate(sx[None, :] - ccx, sy_[None, :] - ccy, theta_assumed)
        m = plume_unit(dx, dy, u, cls)  # (ncand, nsens), per kg/h
        mm = np.sum(m * m, axis=1)
        mc = np.sum(m * readings[None, :], axis=1)
        qhat = np.where(mm > 0, np.maximum(mc, 0.0) / np.maximum(mm, 1e-30), 0.0)
        ss = np.sum((readings[None, :] - qhat[:, None] * m) ** 2, axis=1)
        j = int(np.argmin(ss))
        if ss[j] < best[0]:
            best = (float(ss[j]), float(ccx[j, 0]), float(ccy[j, 0]), float(qhat[j]))
    return best


def localise(readings, sx, sy_, theta_assumed, u, cls, spacing, coarse_res, fine_res):
    """ML over candidate positions, coarse-to-fine; rate in closed form at every candidate.

    Coarse pass: the central cell plus a half-cell margin (a 2d x 2d square) at
    coarse_res. Fine pass: +/- 2 coarse steps around the coarse optimum at fine_res.
    """
    lim = spacing
    g = np.arange(-lim, lim + coarse_res / 2, coarse_res)
    cx, cy = np.meshgrid(g, g, indexing="ij")
    _, x1, y1, _ = _ml_on_grid(readings, sx, sy_, theta_assumed, u, cls, cx.ravel(), cy.ravel())
    w = 2 * coarse_res
    gx = np.arange(x1 - w, x1 + w + fine_res / 2, fine_res)
    gy = np.arange(y1 - w, y1 + w + fine_res / 2, fine_res)
    fx, fy = np.meshgrid(gx, gy, indexing="ij")
    _, x2, y2, q2 = _ml_on_grid(readings, sx, sy_, theta_assumed, u, cls, fx.ravel(), fy.ravel())
    return x2, y2, q2


def inverse_mc(
    spacing: float,
    sigma_n: float,
    q_true: float,
    u: float,
    cls: str,
    wind_err_deg: float,
    n_trials: int,
    seed: int,
    half_extent: float = 12000.0,
) -> dict:
    rng = np.random.default_rng(seed)
    sx, sy_ = sensor_grid(spacing, half_extent)
    coarse_res = spacing / 20.0
    fine_res = max(5.0, spacing / 200.0)

    pos_err, rate_err, detected = [], [], []
    max_snr = []
    for _ in range(n_trials):
        x0 = rng.uniform(-spacing / 2, spacing / 2)
        y0 = rng.uniform(-spacing / 2, spacing / 2)
        theta = rng.uniform(0, 2 * np.pi)
        dx, dy = rotate(sx - x0, sy_ - y0, theta)
        clean = q_true * plume_unit(dx, dy, u, cls)
        readings = clean + rng.normal(0.0, sigma_n, size=clean.shape)
        max_snr.append(float(clean.max() / sigma_n))
        detected.append(bool(clean.max() >= 3.0 * sigma_n))
        theta_assumed = theta + np.deg2rad(rng.normal(0.0, wind_err_deg)) if wind_err_deg > 0 else theta
        xh, yh, qh = localise(readings, sx, sy_, theta_assumed, u, cls, spacing, coarse_res, fine_res)
        pos_err.append(float(np.hypot(xh - x0, yh - y0)))
        rate_err.append(float(qh / q_true))

    pos_err = np.array(pos_err)
    rate_err = np.array(rate_err)
    det = np.array(detected)
    n_det = int(det.sum())
    if n_det:
        det_stats = {
            "n_detected": n_det,
            "pos_err_median_m": round(float(np.median(pos_err[det])), 1),
            "pos_err_p90_m": round(float(np.percentile(pos_err[det], 90)), 1),
            "rate_ratio_median": round(float(np.median(rate_err[det])), 3),
            "rate_ratio_p10": round(float(np.percentile(rate_err[det], 10)), 3),
            "rate_ratio_p90": round(float(np.percentile(rate_err[det], 90)), 3),
        }
    else:
        det_stats = {"n_detected": 0}
    return {
        "spacing_m": spacing,
        "sigma_n_ug_m3": sigma_n,
        "stability": cls,
        "wind_err_deg_sd": wind_err_deg,
        "n_sensors": int(sx.size),
        "n_trials": n_trials,
        "coarse_res_m": coarse_res,
        "fine_res_m": fine_res,
        "pos_err_median_m": round(float(np.median(pos_err)), 1),
        "pos_err_p90_m": round(float(np.percentile(pos_err, 90)), 1),
        "pos_err_within_100m_frac": round(float(np.mean(pos_err <= 100.0)), 3),
        "pos_err_within_500m_frac": round(float(np.mean(pos_err <= 500.0)), 3),
        "undetected_frac": round(1.0 - n_det / n_trials, 3),
        "max_snr_median": round(float(np.median(max_snr)), 2),
        "detected_only": det_stats,
    }


def report(label: str, r: dict, t0: float) -> None:
    d = r["detected_only"]
    extra = ""
    if d["n_detected"]:
        extra = (f"  detected-only: med {d['pos_err_median_m']:.0f} m p90 {d['pos_err_p90_m']:.0f} "
                 f"rate {d['rate_ratio_median']:.2f} [{d['rate_ratio_p10']:.2f},{d['rate_ratio_p90']:.2f}]")
    print(f"{label}: med {r['pos_err_median_m']:.0f} m  p90 {r['pos_err_p90_m']:.0f}  "
          f"undetected {r['undetected_frac']:.2f}  snr {r['max_snr_median']:.1f}{extra}  ({time.time()-t0:.0f}s)",
          flush=True)


def main() -> None:
    t0 = time.time()
    u, cls, q_true, n_trials = 3.0, "D", 10.0, 200
    out = {
        "model": "Gaussian plume, ground-level source and receptors, Briggs rural coefficients",
        "config": {"u_m_s": u, "stability": cls, "q_true_kg_h": q_true, "n_trials": n_trials,
                   "grid_half_extent_m": 12000.0, "estimator": "ML coarse-to-fine grid search, rate closed-form"},
        "forward": forward_table(u),
        "inverse_spacing_noise": [],
        "inverse_wind_error": [],
        "inverse_stability": [],
    }
    seed = 0
    for spacing in [1000.0, 2000.0, 5000.0, 10000.0]:
        for sigma_n in [1.0, 5.0, 20.0]:
            r = inverse_mc(spacing, sigma_n, q_true, u, cls, 0.0, n_trials, seed)
            seed += 1
            out["inverse_spacing_noise"].append(r)
            report(f"d={spacing:6.0f} sn={sigma_n:4.1f}", r, t0)
    for werr in [0.0, 5.0, 15.0]:
        r = inverse_mc(1000.0, 1.0, q_true, u, cls, werr, n_trials, 100 + int(werr))
        out["inverse_wind_error"].append(r)
        report(f"wind err {werr:4.1f} deg", r, t0)
    for c in ["B", "D", "F"]:
        r = inverse_mc(2000.0, 1.0, q_true, u, c, 0.0, n_trials, 200 + ord(c))
        out["inverse_stability"].append(r)
        report(f"class {c} d=2000 sn=1", r, t0)
    out["elapsed_s"] = round(time.time() - t0, 1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
