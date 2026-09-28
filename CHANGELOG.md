# Changelog

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
