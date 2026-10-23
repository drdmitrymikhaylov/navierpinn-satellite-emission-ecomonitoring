# Changelog

## 2026-10-23

- The localisation estimator had been handed the true stability class and wind speed.
  `analysis/misspecification_check.py` re-runs the 1 km and 2 km cells (σ = 1 µg/m³,
  class D, 3 m/s) with the same seeds and a wrong assumption: a wind speed 20 % off is a
  pure rate error (positions identical in all 400 trials, rate exactly ×0.8 / ×1.2 - the
  plume depends on Q/u only); one stability class off at 1 km spacing takes the median
  position error from 113 m to 451 m (D read as C) or 289 m (as E) and puts a factor of
  ×2.70 or ×0.52 on the rate, the ratio of the two classes' centreline concentrations.
  Fitting the class from the readings costs 113 → 125 m at 1 km (true class picked in
  177 of 200 trials) but fails at 2 km (80 of 200; rate 10th-90th percentile 0.40-3.90).
  New README section "When the estimator is told the wrong weather",
  `results/misspecification_check.json`, `tests/test_misspecification.py` (5 tests).

## 2026-10-19

- LICENSE now holds the standard CC BY 4.0 legal text, so GitHub recognises the licence (it showed "Other" before). The terms are unchanged; the note on what each licence covers moved to `NOTICE.md`.

## 2026-09-28

- First content beyond the concept page: a Gaussian-plume reference calculation of
  what a tower grid can resolve (`analysis/plume_localisation.py`, pure numpy, Briggs
  rural coefficients). At 5-10 km tower spacing a 10 kg/h ground source goes
  undetected in 69-92.5 % of wind directions even with 1 µg/m³ sensors, because the
  plume at 5 km is 0.8 km wide and passes between towers; 1 km spacing and σ ≤ 1 µg/m³
  gives a median position error of 113 m and the rate to ±10 %. A 5° wind-direction
  error costs as much as 5× the noise; a convective afternoon (class B) hides 74.5 % of
  sources that a stable night (class F) shows 97 % of. New README section with a
  "what this does not show" paragraph, `results/plume_localisation.json`,
  `tests/test_readme_numbers.py` (9 tests).
