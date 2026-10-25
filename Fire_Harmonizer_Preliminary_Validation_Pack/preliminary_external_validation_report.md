# Fire Harmonizer — Preliminary External Consistency Check

**Scope:** Uploaded `harmonized_science_all_years.csv`; annual comparison against published MapBiomas annual burned-area totals for Brazil.

## Executive result

The dataset passes key internal integrity checks, and its annual C-active rate shows a **moderate positive association** with the MapBiomas Brazil-wide annual burned-area series over the 13 full overlapping project years 2013–2025:

- Pearson correlation: **r = 0.753**.
- Spearman rank correlation: **ρ = 0.687**.
- Excluding 2024, Pearson **r = 0.534** and Spearman **ρ = 0.601** across 12 years.

This is promising as an **exploratory external consistency check**, not proof of operational accuracy or full independent validation.

## Internal integrity checks on the uploaded M7 file

- Rows: **131,900**.
- Dates: **5,276**, from **2012-01-20** through **2026-06-30**.
- Sectors: **25**.
- Expected date-sector combinations: 5,276 × 25 = **131,900**; matches the actual **131,900** rows.
- Duplicate date-sector rows: **0**.
- MODIS detection count summed from the file: **6,333,169**.
- VIIRS detection count summed from the file: **25,266,028**.
- Total: **31,599,197**, matching the documented total.

These are internal data-integrity checks, not independent scientific validation.

## External reference

The external comparison uses MapBiomas annual burned-area totals for Brazil published in its RAF 2025 results table. MapBiomas describes its annual burned-area maps as classifications based on annual Landsat surface-reflectance mosaics at 30 m resolution, using a deep-learning method. This is a different measurement approach from Fire Harmonizer's MODIS/VIIRS active-hotspot input.

Sources:
- MapBiomas annual burned-area method and data coverage: https://brasil.mapbiomas.org/iniciativas-e-produtos/fogo/mapeamento-anual/anual/
- MapBiomas RAF 2025 annual burned-area table (Brazil totals and biome rows): https://raf.ia.mapbiomas.org/raf2025/publication-web-resources/html/publication-23.html

## How the comparison was calculated

For every year, C-active rate = count of sector-day rows with **m7_C ≥ 2** divided by all sector-day rows for that year. It is expressed as a percentage to account for partial/full record length. The external variable is the published total area burned across all of Brazil in that calendar year, in million hectares.

The correlation calculation uses **2013–2025** only:
- 2012 is excluded from the correlations because the project record begins on 20 January 2012.
- 2026 is excluded because the project input ends on 30 June 2026 and there is no comparable full-year MapBiomas total.
- Annual values are observations, not independent random samples; the correlation coefficients are descriptive and should not be overinterpreted.

## What matches

- **2024 is the peak in both series.** Fire Harmonizer's C-active rate was **9.56%**, its highest full-year value. MapBiomas reports **30.22 million ha** burned nationally in Brazil in 2024. Relative to 2023, the C-active rate increased **+36.3%** and the MapBiomas total increased **+88.5%**.
- **2025 is lower in both series.** The C-active rate changed by **-83.4%** from 2024 to 2025, while the published MapBiomas total changed by **-58.0%**.
- Broadly, years with higher C-active rates tend to align with years with greater reported national burned area; this is reflected by the positive correlation coefficients.

## What does not match perfectly

- **2022 → 2023 diverges:** C-active rate changed by **+34.7%**, while MapBiomas Brazil-wide burned area changed by **-6.0%**.
- **2019 → 2020 differs in magnitude:** C-active rate rose by about 105.5%, while MapBiomas Brazil-wide burned area rose by about 6.0%.
- C-active rates summarize unusual sector-days across the entire Fire Harmonizer bounding box; MapBiomas values are country-wide totals for Brazil. The project window includes parts of countries outside Brazil and excludes some areas of Brazil. Therefore the spatial domains are not the same, and annual national totals cannot validate specific sectors or events.

## Annual comparison table

| Year | Project coverage | C-active sector-days | C-active rate | MapBiomas Brazil burned area (million ha) |
|---:|---|---:|---:|---:|
| 2012 | Partial: begins 20 Jan | 288 | 3.32% | 18.22 |
| 2013 | Full calendar year | 181 | 1.98% | 10.82 |
| 2014 | Full calendar year | 214 | 2.35% | 17.36 |
| 2015 | Full calendar year | 417 | 4.57% | 22.14 |
| 2016 | Full calendar year | 516 | 5.64% | 20.27 |
| 2017 | Full calendar year | 272 | 2.98% | 21.42 |
| 2018 | Full calendar year | 175 | 1.92% | 11.23 |
| 2019 | Full calendar year | 265 | 2.90% | 19.33 |
| 2020 | Full calendar year | 546 | 5.97% | 20.50 |
| 2021 | Full calendar year | 242 | 2.65% | 14.21 |
| 2022 | Full calendar year | 475 | 5.21% | 17.06 |
| 2023 | Full calendar year | 640 | 7.01% | 16.03 |
| 2024 | Full calendar year | 875 | 9.56% | 30.22 |
| 2025 | Full calendar year | 145 | 1.59% | 12.70 |
| 2026 | Partial: ends 30 Jun | 71 | 1.57% | N/A |

## Conclusion and appropriate claim

**Conclusion:** The uploaded M7 dataset is structurally consistent with its documented 25-sector daily grid and exhibits a positive, moderate-to-strong annual association with an independently produced Landsat-based burned-area series. There are also visible year-to-year disagreements, and the geographical footprints do not match exactly.

Use wording such as:

> “We performed a preliminary external consistency check by comparing annual Fire Harmonizer C-active sector-day rates with MapBiomas Landsat-based burned-area totals for Brazil. The annual series showed a positive correlation over 2013–2025, while some years diverged. Because MapBiomas's totals cover all of Brazil rather than the exact Fire Harmonizer study window, this is preliminary external consistency evidence, not full spatial validation.”

Do **not** claim that this validates the whole 14.1-million-km² box, each M8 event, or hotspot-level accuracy. The next stronger test would clip the MapBiomas burned-area raster to the exact study bounding box (or individual sector polygons) and compare at spatially and temporally matched scale, then inspect selected cases against imagery without viewing their C scores first.
