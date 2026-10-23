"""Pin the README section "When the estimator is told the wrong weather" to
results/misspecification_check.json, and check that file against
results/plume_localisation.json (the correctly specified rows must reproduce
the stored cells) and against the closed form (a wind-speed error must be a
pure rate error).

Run:  python3 -m pytest tests/
Print the README table rows:  python3 tests/test_misspecification.py
"""

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = " ".join((ROOT / "README.md").read_text().split())
MIS = json.loads((ROOT / "results" / "misspecification_check.json").read_text())
BASE = json.loads((ROOT / "results" / "plume_localisation.json").read_text())

LABELS = {
    "D 3.0": "D, 3 m/s (correct)",
    "C 3.0": "C, 3 m/s (one class less stable)",
    "E 3.0": "E, 3 m/s (one class more stable)",
    "B 3.0": "B, 3 m/s (two classes less stable)",
    "F 3.0": "F, 3 m/s (two classes more stable)",
    "D 2.4": "D, 2.4 m/s (wind 20 % too slow)",
    "D 3.6": "D, 3.6 m/s (wind 20 % too fast)",
    "fit class": "class fitted from the readings, 3 m/s",
}


def r0(v):
    """Round half up to an integer (Python's round() is half-to-even)."""
    return int(math.floor(v + 0.5))


def pct(v):
    """Fraction -> percent with one decimal only when it is a half (0.865 -> 86.5, 0.93 -> 93)."""
    p = v * 100.0
    return f"{p:.1f}" if abs(p - r0(p)) > 0.01 else f"{r0(p)}"


def row(cell, key):
    r = cell["rows"][key]
    pv = r["paired_vs_correct"]
    worse = "-" if key == "D 3.0" else f"{pct(pv['worse_by_more_than_1m_frac'])} %"
    return (f"| {LABELS[key]} | {r0(r['pos_err_median_m'])} m | {r0(r['pos_err_p90_m'])} m | "
            f"{pct(r['pos_err_within_100m_frac'])} % | {worse} | "
            f"{r['detected_rate_ratio_median']:.2f} [{r['detected_rate_ratio_p10']:.2f}-{r['detected_rate_ratio_p90']:.2f}] |")


def cell(spacing):
    hits = [c for c in MIS["cells"] if c["spacing_m"] == spacing]
    assert len(hits) == 1
    return hits[0]


def base_cell(spacing):
    hits = [r for r in BASE["inverse_spacing_noise"] if r["spacing_m"] == spacing and r["sigma_n_ug_m3"] == 1.0]
    assert len(hits) == 1
    return hits[0]


def test_correct_rows_reproduce_the_stored_cells():
    for spacing in (1000.0, 2000.0):
        c, b = cell(spacing), base_cell(spacing)
        r = c["rows"]["D 3.0"]
        assert r["pos_err_median_m"] == b["pos_err_median_m"]
        assert r["pos_err_p90_m"] == b["pos_err_p90_m"]
        assert c["undetected_frac"] == b["undetected_frac"]
        assert c["n_detected"] == b["detected_only"]["n_detected"]
        assert r["detected_rate_ratio_median"] == b["detected_only"]["rate_ratio_median"]
        assert r["detected_rate_ratio_p10"] == b["detected_only"]["rate_ratio_p10"]
        assert r["detected_rate_ratio_p90"] == b["detected_only"]["rate_ratio_p90"]


def test_wind_speed_error_is_a_pure_rate_error():
    """The plume depends on Q/u only: the position fit cannot see u, the rate absorbs it exactly."""
    for spacing in (1000.0, 2000.0):
        c = cell(spacing)
        for key, factor in (("D 2.4", 0.8), ("D 3.6", 1.2)):
            pv = c["rows"][key]["paired_vs_correct"]
            assert pv["max_abs_pos_diff_m"] == 0.0
            assert pv["identical_within_1m_frac"] == 1.0
            assert pv["rate_over_correct_median"] == factor
            assert pv["rate_over_correct_max_abs_dev_from_median"] == 0.0
            assert c["rows"][key]["pos_err_median_m"] == c["rows"]["D 3.0"]["pos_err_median_m"]


def test_class_bias_is_the_centreline_ratio():
    """Rate bias under a wrong class tracks the true/assumed centreline ratio at the tower spacing."""
    ratios = MIS["centreline_ratio_true_over_assumed"]
    for spacing in (1000.0, 2000.0):
        i = ratios["distances_m"].index(int(spacing))
        for key in ("C 3.0", "E 3.0"):
            med = cell(spacing)["rows"][key]["detected_rate_ratio_median"]
            assert abs(med / ratios[key[0]][i] - 1.0) < 0.15, (spacing, key, med, ratios[key[0]][i])
    assert f"{ratios['C'][1]:.2f}" == "2.65" and f"{ratios['E'][1]:.2f}" == "0.46"
    assert "2.65" in README and "0.46" in README


def test_readme_tables_match_the_json():
    for spacing in (1000.0, 2000.0):
        c = cell(spacing)
        for key in LABELS:
            assert " ".join(row(c, key).split()) in README, (spacing, key, row(c, key))


def test_readme_prose_numbers():
    c1, c2 = cell(1000.0), cell(2000.0)
    f1, f2 = c1["rows"]["fit class"], c2["rows"]["fit class"]
    assert f1["class_chosen_counts"]["D"] == 177 and "177 of 200" in README
    assert f2["class_chosen_counts"]["D"] == 80 and "80 of 200" in README
    assert f2["class_chosen_counts"]["C"] == 48 and f2["class_chosen_counts"]["E"] == 46
    assert "48 trials pick C and 46 pick E" in README
    assert c1["rows"]["C 3.0"]["pos_err_median_m"] / c1["rows"]["D 3.0"]["pos_err_median_m"] > 3.9
    assert c1["rows"]["E 3.0"]["pos_err_median_m"] / c1["rows"]["D 3.0"]["pos_err_median_m"] > 2.5
    assert "four times" in README and "two and a half" in README
    assert c1["rows"]["C 3.0"]["pos_err_within_500m_frac"] == 0.57 and "97 % to 57 %" in README
    # the 2 km fit: position worse than the known class on 42 % of the trials, better on 18 %
    pv = f2["paired_vs_correct"]
    assert pct(pv["worse_by_more_than_1m_frac"]) == "42" and pct(pv["better_by_more_than_1m_frac"]) == "18"
    assert "42 % of the trials" in README and "better on 18 %" in README
    assert r0(BASE["inverse_wind_error"][1]["pos_err_median_m"]) == 387 and "387 m" in README


if __name__ == "__main__":
    for spacing in (1000.0, 2000.0):
        print(f"\n{spacing:.0f} m spacing")
        for key in LABELS:
            print(row(cell(spacing), key))
    r = MIS["centreline_ratio_true_over_assumed"]
    print("\ncentreline ratios D/assumed at 1 km, 2 km:", {k: (r[k][1], r[k][2]) for k in "BCEF"})
