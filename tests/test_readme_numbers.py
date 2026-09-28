"""Pin the numbers quoted in README.md to results/plume_localisation.json.

Run:  python3 -m pytest tests/
"""

import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text()
RES = json.loads((ROOT / "results" / "plume_localisation.json").read_text())


def r0(v):
    """Round half up (Python's round() is half-to-even: round(384.5) == 384)."""
    return int(math.floor(v + 0.5))


def entry(section, **kw):
    hits = [r for r in RES[section] if all(r[k] == v for k, v in kw.items())]
    assert len(hits) == 1, (section, kw, len(hits))
    return hits[0]


def test_forward_table_matches_briggs_formula():
    """The forward numbers in the README are recomputed here from the Briggs coefficients."""
    u = RES["forward"]["u_m_s"]
    d = RES["forward"]["classes"]["D"]
    dists = RES["forward"]["distances_m"]
    i5 = dists.index(5000)
    x = 5000.0
    sy = 0.08 * x * (1 + 0.0001 * x) ** -0.5
    sz = 0.06 * x * (1 + 0.0015 * x) ** -0.5
    c = (1000 / 3600) / (math.pi * u * sy * sz) * 1e6
    assert abs(c - d["centreline_ug_m3_per_kg_h"][i5]) < 1e-3
    assert abs(d["centreline_ug_m3_per_kg_h"][i5] - 0.877) < 5e-4
    assert abs(math.sqrt(2 * math.log(2)) * sy - d["half_width_m"][i5]) < 0.1
    assert r0(d["half_width_m"][i5]) == 385
    i1 = dists.index(1000)
    assert abs(d["centreline_ug_m3_per_kg_h"][i1] - 10.18) < 5e-3
    f = RES["forward"]["classes"]["F"]
    assert abs(f["centreline_ug_m3_per_kg_h"][i1] - 62.8) < 0.05
    b = RES["forward"]["classes"]["B"]
    assert r0(b["half_width_m"][dists.index(10000)]) == 1332


def test_readme_quotes_forward_numbers():
    for s in ["10.18 µg/m³ · 90 m", "0.877 · 385 m", "62.8 µg/m³ · 45 m", "0.022 · 1332 m", "8.8 µg/m³"]:
        assert s in README, s


def test_inverse_1km_sigma1():
    r = entry("inverse_spacing_noise", spacing_m=1000.0, sigma_n_ug_m3=1.0)
    assert r["n_sensors"] == 625
    assert r["pos_err_median_m"] == 113.0
    assert r["pos_err_p90_m"] == 385.0
    assert r["undetected_frac"] == 0.0
    d = r["detected_only"]
    assert (round(d["rate_ratio_median"], 2), round(d["rate_ratio_p10"], 2), round(d["rate_ratio_p90"], 2)) == (1.01, 0.90, 1.09)
    assert "**113 m** / 385 m, 0 % undetected, rate 1.01 [0.90-1.09]" in README


def test_inverse_table_rows():
    checks = [
        (1000.0, 5.0, 428, 1139, "31"),
        (1000.0, 20.0, 880, 1518, "84"),
        (2000.0, 1.0, 719, 1936, "10.5"),
        (2000.0, 5.0, 1592, 2914, "79"),
        (2000.0, 20.0, 1877, 3112, "97"),
        (5000.0, 1.0, 3578, 7299, "69"),
        (5000.0, 5.0, 4628, 7938, "95.5"),
        (10000.0, 1.0, 8349, 13929, "92.5"),
    ]
    for sp, sn, med, p90, undet in checks:
        r = entry("inverse_spacing_noise", spacing_m=sp, sigma_n_ug_m3=sn)
        assert r0(r["pos_err_median_m"]) == med, (sp, sn, r["pos_err_median_m"])
        assert r0(r["pos_err_p90_m"]) == p90, (sp, sn, r["pos_err_p90_m"])
        assert abs(100 * r["undetected_frac"] - float(undet)) < 1e-6, (sp, sn, r["undetected_frac"])
        assert f"{med} m / {p90} m, {undet} % undetected" in README.replace("**", ""), (sp, sn)
    r = entry("inverse_spacing_noise", spacing_m=5000.0, sigma_n_ug_m3=20.0)
    assert r["undetected_frac"] == 0.995
    r = entry("inverse_spacing_noise", spacing_m=10000.0, sigma_n_ug_m3=20.0)
    assert r["undetected_frac"] == 1.0


def test_5km_detected_only_median():
    r = entry("inverse_spacing_noise", spacing_m=5000.0, sigma_n_ug_m3=1.0)
    d = r["detected_only"]
    assert d["n_detected"] == 62
    assert round(d["pos_err_median_m"] / 1000, 1) == 2.6
    assert "62 detected trials" in README


def test_factor_30_and_69_points():
    a = entry("inverse_spacing_noise", spacing_m=1000.0, sigma_n_ug_m3=1.0)
    b = entry("inverse_spacing_noise", spacing_m=5000.0, sigma_n_ug_m3=1.0)
    ratio = b["pos_err_median_m"] / a["pos_err_median_m"]
    assert 28 <= ratio <= 33, ratio
    assert abs(100 * (b["undetected_frac"] - a["undetected_frac"]) - 69) < 1e-6


def test_wind_direction_sweep():
    meds, p90s, rates, p10s = [], [], [], []
    for w in [0.0, 5.0, 15.0]:
        r = entry("inverse_wind_error", wind_err_deg_sd=w)
        assert r["spacing_m"] == 1000.0 and r["sigma_n_ug_m3"] == 1.0 and r["undetected_frac"] == 0.0
        meds.append(r0(r["pos_err_median_m"]))
        p90s.append(r0(r["pos_err_p90_m"]))
        rates.append(round(r["detected_only"]["rate_ratio_median"], 2))
        p10s.append(round(r["detected_only"]["rate_ratio_p10"], 2))
    assert meds == [133, 387, 753]
    assert p90s == [395, 1004, 1359]
    assert rates == [0.99, 0.92, 0.70]
    assert p10s == [0.87, 0.63, 0.05]
    assert "**133 m → 387 m → 753 m**" in README
    assert "395 → 1004 → 1359 m" in README


def test_stability_sweep():
    exp = {"B": ("74.5", 1788, (0.72, 0.00, 2.57)), "D": ("11", 699, (1.00, 0.78, 1.31)), "F": ("3", 371, (0.99, 0.91, 1.12))}
    for cls, (undet, med, rate) in exp.items():
        r = entry("inverse_stability", stability=cls)
        assert r["spacing_m"] == 2000.0 and r["sigma_n_ug_m3"] == 1.0
        assert abs(100 * r["undetected_frac"] - float(undet)) < 1e-6, (cls, r["undetected_frac"])
        assert f"**{undet} % undetected**" in README or f"{undet} % undetected" in README, cls
        assert r0(r["pos_err_median_m"]) == med, (cls, r["pos_err_median_m"])
        d = r["detected_only"]
        got = (round(d["rate_ratio_median"], 2), round(d["rate_ratio_p10"], 2), round(d["rate_ratio_p90"], 2))
        assert got == rate, (cls, got)
        assert f"rate {rate[0]:.2f} [{rate[1]:.2f}-{rate[2]:.2f}]" in README, cls


def test_config_as_stated():
    c = RES["config"]
    assert c["u_m_s"] == 3.0 and c["stability"] == "D" and c["q_true_kg_h"] == 10.0 and c["n_trials"] == 200
    assert "200 trials" in README and "10 kg/h source" in README and "wind 3 m/s" in README
    # the section is dated and the file is referenced
    assert re.search(r"\*Added 2026-09-28\.\*", README)
    assert "results/plume_localisation.json" in README
