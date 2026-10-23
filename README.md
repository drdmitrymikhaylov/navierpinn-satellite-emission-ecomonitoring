# Navier PINN - Satellite and Tower-Sensor Emission Detection for Ecological Monitoring

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23529563.svg)](https://doi.org/10.5281/zenodo.23529563)

## Where the idea comes from

I have worked on carbon measurement since 2022, when I published on what AI and IoT sensing mean for carbon market regulation (*BIO Web of Conferences*, 2022). A carbon market is only as good as the measurements under it. A 2026 paper in *Stochastic Environmental Research and Risk Assessment* took the next step with machine learning for CO2 exchange, and granted US patent 12,602,659 covers emission compliance monitoring at sea. Towers are the land version of the same idea: put the sensors where power and connectivity already exist, and let the transport equation do the rest.

## Idea

Emission monitoring fails at the same two points everywhere. Satellites see a column, not a source, and ground sensors see a source, not an area. The physics that connects them is plume transport - advection and diffusion of a scalar in a turbulent boundary layer.

This project puts that transport equation into the model. A sparse network of gas sensors mounted on telecom towers that already exist, combined with satellite column measurements, then resolves where the emission actually came from.

### Why towers

Telecom towers are already powered, already connected, already maintained, and already distributed across the terrain in a pattern designed for coverage. Emission monitoring built on them costs the sensors and nothing else.

## Planned method

Advection-diffusion with a source term, constrained by wind field and stability class. The inverse problem is recovering source location and rate from downwind concentrations. That is exactly the case where a physics-informed network earns its place - sparse, noisy sensors, and a governing equation that is known.

## What a tower grid can resolve before any network is trained

*Added 2026-09-28.* The page above says the tower network "resolves where the emission actually came from". Whether it can depends on tower spacing, sensor noise and the weather, and the steady Gaussian plume - the closed-form solution of the advection-diffusion equation with constant wind and power-law spreading - says by how much. `analysis/plume_localisation.py` (pure numpy, published dispersion coefficients: Briggs rural, Pasquill-Gifford classes A-F) computes it; `results/plume_localisation.json` holds the numbers; `tests/test_readme_numbers.py` pins this section to the file.

**Forward.** Ground-level source, ground-level sensors, wind 3 m/s, flat terrain. Centreline concentration per kg/h of emitted gas, and the crosswind half-width of the plume (distance from the centreline at which the concentration halves):

| stability class | 1 km | 2 km | 5 km | 10 km |
|---|---|---|---|---|
| B (convective, sunny day) | 1.61 µg/m³ · half-width 180 m | 0.42 · 344 m | 0.075 · 769 m | 0.022 · 1332 m |
| D (neutral, overcast or windy) | 10.18 µg/m³ · 90 m | 3.36 · 172 m | 0.877 · 385 m | 0.347 · 666 m |
| F (stable, clear night) | 62.8 µg/m³ · 45 m | 20.2 · 86 m | 5.64 · 192 m | 2.61 · 333 m |

A 10 kg/h source under class D gives 8.8 µg/m³ on the centreline at 5 km, in a plume 0.8 km wide at half maximum. For methane 1 µg/m³ is 1.5 ppb, an analyser-grade number; 20 µg/m³ (30 ppb) is the class of the better mid-cost sensors, and low-cost metal-oxide sensors are worse still.

**Inverse.** Sensors on a square grid of spacing *d*; a 10 kg/h source at a random point of the central cell; wind direction random and known to the estimator; each sensor reads the plume plus Gaussian noise of sd σ. The estimator is maximum likelihood over source position with the rate solved in closed form - the best any method can do with these readings and the true forward model. 200 trials per cell; "undetected" means no sensor read above 3σ, so there was nothing to localise. Position error is the median and 90th percentile over all trials; the rate ratio is the median [10th-90th percentile] of estimated over true rate on the detected trials only. Class D.

| spacing | σ = 1 µg/m³ | σ = 5 µg/m³ | σ = 20 µg/m³ |
|---|---|---|---|
| 1 km (625 towers in 24 × 24 km) | **113 m** / 385 m, 0 % undetected, rate 1.01 [0.90-1.09] | 428 m / 1139 m, 31 % undetected | 880 m / 1518 m, 84 % undetected |
| 2 km (169 towers) | **719 m** / 1936 m, 10.5 % undetected, rate 1.01 [0.71-1.27] | 1592 m / 2914 m, 79 % undetected | 1877 m / 3112 m, 97 % undetected |
| 5 km (25 towers) | 3578 m / 7299 m, **69 % undetected** | 4628 m / 7938 m, 95.5 % undetected | 99.5 % undetected |
| 10 km (9 towers) | 8349 m / 13929 m, **92.5 % undetected** | 99.5 % undetected | 100 % undetected |

The reading is not subtle. At the spacing of a rural telecom grid - 5 to 10 km - a 10 kg/h source is not seen at all in 69-92.5 % of wind directions even with analyser-grade sensors, because the plume at 5 km is 0.8 km wide and passes between the towers; the median position error when it is seen is still 2.6 km (5 km grid, σ = 1, 62 detected trials). Localisation to better than a few hundred metres needs 1 km spacing *and* σ ≤ 1 µg/m³: at 1 km and σ = 5 the median error is already 428 m and a third of the sources go undetected. Rate is recovered to ±10 % (10th-90th percentile) only in the 1 km / σ = 1 cell. Since the plume is linear in the rate, a 100 kg/h source at σ = 10 reads the same as this table's 10 kg/h at σ = 1.

**Wind direction.** Same 1 km / σ = 1 grid, but the estimator's wind direction is off by a Gaussian error of sd 0°, 5° or 15°: median position error **133 m → 387 m → 753 m**, 90th percentile 395 → 1004 → 1359 m, median rate ratio 0.99 → 0.92 → 0.70 (10th percentile 0.87 → 0.63 → 0.05). A 5° direction error costs as much as raising the noise from 1 to 5 µg/m³. (The 0° row is a second seed of the 113 m cell above; 113 vs 133 m is the Monte Carlo scatter at 200 trials.)

**Stability.** 2 km grid, σ = 1, three classes: B (convective) **74.5 % undetected**, median error 1788 m, rate 0.72 [0.00-2.57]; D 11 % undetected, 699 m, rate 1.00 [0.78-1.31]; F (stable night) **3 % undetected**, 371 m, rate 0.99 [0.91-1.12]. The network sees a source best on a clear night, when the plume is narrow and concentrated, and hardly at all on a convective afternoon, when it is mixed through the boundary layer before it reaches the next tower.

**What this does not show.** Steady wind of known speed, flat terrain, one ground-level source, a regular grid (real towers cluster along roads), Gaussian sensor noise with no background drift, the true dispersion class and wind speed handed to the estimator (relaxed in the section below), and no satellite column at all. Every one of these favours the estimator, so the table is an upper bound on what any inverse method - a physics-informed network included - can get from those sensors and that geometry. What a trained model could add is the part the Gaussian plume leaves out: an unsteady wind field, terrain, and the column measurement that constrains the total the towers miss. Whether that closes the gap between a 5 km grid and a 1 km grid is the question the project has to answer, and this calculation says it is a gap of a factor of 30 in median position error and 69 percentage points in detection.

## When the estimator is told the wrong weather

*Added 2026-10-23.* Every row above hands the estimator the true stability class and the true wind speed. Neither is known in the field: the class is read off wind speed, cloud and time of day and is routinely one step uncertain, and a wind speed from the nearest met station is good to perhaps 20 %. `analysis/misspecification_check.py` re-runs the 1 km and 2 km cells (σ = 1 µg/m³, class D, 3 m/s, 10 kg/h, 200 trials) with the same seeds - the "correct" rows below reproduce the table above to the stored decimal - and lets the estimator assume a neighbouring class or a wind speed 20 % off while the truth stays at D and 3 m/s. Same source positions and noise draws in every row, so each difference is paired. The last row tries the obvious repair: fit the class from the readings (localise under each of A-F, keep the smallest residual). `results/misspecification_check.json` holds the numbers; `tests/test_misspecification.py` pins this section to it.

Columns: median position error over all 200 trials, its 90th percentile, share of trials within 100 m, share of trials where the position is worse than under the correct assumption (by more than 1 m), and estimated over true rate, median [10th-90th percentile] on the detected trials.

**1 km spacing (625 towers, every source detected)**

| estimator assumes | median error | 90th pct | within 100 m | worse than correct | rate ratio |
|---|---|---|---|---|---|
| D, 3 m/s (correct) | 113 m | 385 m | 45 % | - | 1.01 [0.90-1.09] |
| C, 3 m/s (one class less stable) | 451 m | 955 m | 11 % | 86.5 % | 2.70 [2.38-3.13] |
| E, 3 m/s (one class more stable) | 289 m | 829 m | 21.5 % | 81.5 % | 0.52 [0.44-0.59] |
| B, 3 m/s (two classes less stable) | 759 m | 1278 m | 3 % | 93 % | 4.91 [3.93-6.13] |
| F, 3 m/s (two classes more stable) | 687 m | 1225 m | 8.5 % | 89 % | 0.23 [0.16-0.29] |
| D, 2.4 m/s (wind 20 % too slow) | 113 m | 385 m | 45 % | 0 % | 0.81 [0.72-0.87] |
| D, 3.6 m/s (wind 20 % too fast) | 113 m | 385 m | 45 % | 0 % | 1.21 [1.08-1.31] |
| class fitted from the readings, 3 m/s | 125 m | 467 m | 43.5 % | 10 % | 1.01 [0.90-1.15] |

**2 km spacing (169 towers, 10.5 % undetected)**

| estimator assumes | median error | 90th pct | within 100 m | worse than correct | rate ratio |
|---|---|---|---|---|---|
| D, 3 m/s (correct) | 719 m | 1936 m | 7.5 % | - | 1.01 [0.71-1.27] |
| C, 3 m/s (one class less stable) | 1073 m | 2214 m | 2 % | 66.5 % | 2.90 [2.03-3.97] |
| E, 3 m/s (one class more stable) | 1147 m | 2253 m | 2 % | 64.5 % | 0.49 [0.31-0.64] |
| B, 3 m/s (two classes less stable) | 1501 m | 2499 m | 0 % | 78.5 % | 6.22 [4.20-9.56] |
| F, 3 m/s (two classes more stable) | 1503 m | 2785 m | 3 % | 71.5 % | 0.23 [0.12-0.33] |
| D, 2.4 m/s (wind 20 % too slow) | 719 m | 1936 m | 7.5 % | 0 % | 0.81 [0.56-1.01] |
| D, 3.6 m/s (wind 20 % too fast) | 719 m | 1936 m | 7.5 % | 0 % | 1.21 [0.85-1.52] |
| class fitted from the readings, 3 m/s | 955 m | 2234 m | 4 % | 42 % | 1.09 [0.40-3.90] |

**A wind-speed error is a rate error and nothing else.** With the speed 20 % wrong the position estimates are identical to the metre in all 400 trials and the rate is exactly 0.80× or 1.20× the correctly specified estimate - not approximately, exactly (largest deviation 0 in the result file). The plume depends on Q/u only, so concentrations alone cannot separate the two: the position fit never sees u, and the rate absorbs the whole error. It also means the class fit in the last row cannot be extended to the wind speed; that number has to come from a measurement.

**A class error is a position error and a factor in the rate.** One class off at 1 km spacing, the median position error goes from 113 m to 451 m (D read as C) or 289 m (D read as E) - four times and two and a half times - the share of sources placed within 100 m from 45 % to 11 % / 21.5 %, within 500 m from 97 % to 57 % / 70.5 %, and the estimate is worse than the correctly specified one on 86.5 % / 81.5 % of the same trials. One class is worth about as much as the 5° wind-direction error above (387 m). Two classes off (B or F) the median is 759 / 687 m and 3 % / 8.5 % of the sources are within 100 m. The rate is off by a factor: ×2.70 [2.38-3.13] for C, ×0.52 [0.44-0.59] for E, ×4.91 and ×0.23 two classes out - the ±10 % of the headline cell is the known-class figure. The factor is the ratio of the two classes' centreline concentrations at the tower spacing (D over C at 1 km 2.65, D over E 0.46; the medians agree with these to within 15 % at both spacings), so it can be read off the forward table: the estimator is told the plume spreads more (or less) than it does and makes up the difference in rate. At 2 km the same errors cost less in position - 719 → 1073 / 1147 m, worse on 66.5 % / 64.5 % of trials - because the position was already uncertain by a third of a cell, and more in rate (×2.90, ×0.49; two classes out ×6.22 / ×0.23).

**Fitting the class works at 1 km and not at 2 km.** With 625 towers the readings pick the true class in 177 of 200 trials (C in 15, E in 8); the cost of fitting it is 113 → 125 m median, 385 → 467 m at the 90th percentile, the rate interval widening from [0.90-1.09] to [0.90-1.15], and a worse position on 10 % of the trials. With 169 towers the class is right in 80 of 200 trials - 48 trials pick C and 46 pick E, 26 go two or three classes out - and the rate interval becomes [0.40-3.90], a factor of ten wide where the known class gave [0.71-1.27]; the position is worse than with the known class on 42 % of the trials and better on 18 %. At the spacing of a dense telecom grid the towers do not tell the class, so they do not tell the rate within a factor of three either way; that number has to come from a met measurement or from the satellite column, which sees the total the towers cannot - the first concrete case on this page for the combination the project proposes.

**What this does not show.** One true class (D) and one wind speed; the estimator's error is systematic (the same wrong class in every trial), not random; the σ coefficients within a class are taken as exact, so a dispersion scheme other than Briggs rural was not tried; class and wind-speed errors were not combined; the 5 km and 10 km cells were not re-run (with 69-92.5 % of sources undetected there is little left for a wrong class to spoil); and the class fit compares residuals only - no prior on the class and no met data, which a real estimator would have.

## Status

This is a project page for work at concept stage, and the solver source is not public. There is no measurement campaign yet. What the repository does hold is the two Gaussian-plume reference calculations above (`analysis/`), their result files (`results/`) and the tests that pin this page to them (`tests/`); see `CHANGELOG.md`.

## Licence

Documentation, figures, result files and the reference calculation in `analysis/` in this repository: CC BY 4.0. Source code is held in a private repository, all rights reserved, and is available under NDA.
