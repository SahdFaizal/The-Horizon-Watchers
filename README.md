# The Horizon Watchers

## Fire Harmonizer

**Fire Harmonizer — The Horizon Watchers** is a scientific data-processing and visualization system built around NASA FIRMS active-fire observations from **MODIS** and **VIIRS**.


Instead of treating a satellite fire detection as a complete description of a fire, Fire Harmonizer asks a harder question:

> **How can long-running MODIS and VIIRS active-fire observations be placed into one transparent analytical system without hiding sensor disagreement, spatial context, or observation limitations?**

The repository contains the scientific pipeline, derived datasets, event reconstruction engine, offline visualization generator, and the final Streamlit observatory.

---

## What Fire Harmonizer does

At its core, the system transforms millions of satellite-derived observations into a structured **date × sector** scientific record.

```text
NASA FIRMS MODIS + VIIRS observations
                │
                ▼
      M1 — Acquisition / preprocessing
                │
                ▼
      M2 — Validation / diagnostics
                │
                ▼
      M3 — Historical science foundation
                │
                ▼
    M4 — Critical-period detection
                │
                ▼
      M5 — Scientific visual analytics
                │
                ▼
    M7 — Multi-evidence science core
                │
                ▼
    M8 — Reconstructed Fire Events
                │
                ▼
    M9 — Progressive Disclosure app
                │
                ├───────────────┐
                ▼               ▼
        Streamlit observatory   M10 offline plates
```

The architecture is deliberately **data-first**. The web application is a presentation layer over precomputed scientific outputs rather than a hidden research model that recomputes the analysis during every interaction.

---

## Project scale

| Quantity | Current project figure |
|---|---:|
| Raw MODIS + VIIRS detections | **31,599,197** |
| MODIS detections | 6,333,169 |
| VIIRS detections | 25,266,028 |
| Dates | **5,276** |
| Spatial sectors | **25 (5×5)** |
| Core M7 sector-day rows | **131,900** |
| Temporal coverage | **20 Jan 2012 – 30 Jun 2026** |
| Study extent | **~14.1 million km²** |
| M7 evidence dimensions | **C, F, G, S, U** |
| Reconstructed Fire Events | **1,767** |

The study area is represented as a project-defined Amazon Basin **bounding-box framework**, not as an exact hydrographic polygon.

---

# Scientific architecture

## The common spatial framework

All analysis uses the same fixed 5×5 sector geometry:

```text
C01  C02  C03  C04  C05
C06  C07  C08  C09  C10
C11  C12  C13  C14  C15
C16  C17  C18  C19  C20
C21  C22  C23  C24  C25
```

This common geometry allows the two satellite streams to be compared using the same spatial units across the full historical period.

The central analytical object is therefore not an individual satellite point, but a **sector-day**.

---

## M1 — Acquisition and preprocessing

M1 establishes the common data foundation.

The preprocessing stage:

- reads or acquires NASA FIRMS active-fire observations;
- keeps MODIS and VIIRS as separate sensor streams;
- cleans and standardizes fields;
- filters observations to the project study extent;
- assigns each observation to one of the 25 project sectors;
- aggregates observations to daily sector-level values;
- preserves relevant observation attributes for later diagnostics;
- creates the common date × sector structure.

The historical acquisition workflow can use chunked FIRMS retrieval. API credentials should be supplied through an environment variable rather than committed to the repository.

Example:

```bash
# Windows PowerShell
$env:FIRMS_MAP_KEY="YOUR_KEY"

# Linux / macOS
export FIRMS_MAP_KEY="YOUR_KEY"
```

Never commit a FIRMS API key to Git.

---

## M2 — Validation and diagnostics

M2 is the quality-control layer.

It checks the structural and statistical integrity of the processed data, including:

- date coverage;
- expected sector coverage;
- expected date × sector matrix size;
- missing-value behavior;
- zero/detection behavior;
- detection and FRP summaries;
- MODIS vs VIIRS raw-count relationships;
- standardized-value relationships;
- sensor overlap and discrepancy.

A key principle is that **sensor absence and sensor disagreement are not silently converted into false certainty**.

---

## M3 — Historical science foundation

M3 converts descriptive sensor counts into historically interpretable quantities.

The science layer uses sensor-specific historical reference distributions rather than assuming that the raw MODIS and VIIRS streams are directly interchangeable.

The historical framework includes:

- sensor-specific standardization;
- sector and month context;
- leave-one-year-out reference construction;
- harmonized activity;
- sensor discrepancy;
- anomaly detection;
- explicit severity interpretation.

The original M3 fields are retained as part of the later M7 lineage rather than discarded.

---

# M7 — Multi-Evidence Science Core

M7 is the main scientific expansion of the project.

Rather than collapsing every observation into a single “confidence score,” Fire Harmonizer keeps multiple evidence dimensions visible:

```text
E = (C, F, G, S, U)
```

| Dimension | Meaning |
|---|---|
| **C** | Activity anomaly evidence |
| **F** | FRP anomaly evidence |
| **G** | Cross-sensor concordance |
| **S** | Spatial coherence |
| **U** | Observation / sampling context |

### C — Activity anomaly

Measures how unusual sector-day activity is relative to its historical reference.

### F — FRP anomaly

Measures how unusual Fire Radiative Power is relative to historical context.

**FRP remains a physical power quantity; it is not total fire energy.**

### G — Cross-sensor concordance

Quantifies agreement between standardized MODIS and VIIRS evidence.

One implementation is:

```text
G = 1 - |Z_MODIS - Z_VIIRS|
        -------------------------
        |Z_MODIS| + |Z_VIIRS| + ε
```

G is an **observational concordance measure**. It is not ground truth and is not a probability that a fire is real.

### S — Spatial coherence

Measures the proportion of available neighboring sectors whose count-anomaly evidence also crosses the project anomaly threshold.

This distinguishes isolated anomalies from anomalies that develop broader neighboring-sector support.

### U — Observation context

U is deliberately kept separate from the numerical evidence dimensions.

It can retain context such as:

- sensor state;
- observation support;
- sampling availability;
- retained observational metadata;
- other fields needed to understand whether evidence is well supported.

U is **not collapsed into an arbitrary confidence percentage**.

---

# M8 — Reconstructed Fire Events

M8 changes the unit of interpretation from isolated anomalous sector-days to **spatiotemporal Fire Events**.

A seed node is:

```text
(date, sector) where C >= 2
```

Two anomaly nodes can be connected when they satisfy:

```text
next calendar day
        +
same sector OR one of the 8 neighboring sectors
```

The connected components of this graph become reconstructed Fire Events.

This allows the system to describe:

- emergence;
- decay;
- duration;
- spatial footprint;
- peak activity;
- peak FRP anomaly;
- mean cross-sensor concordance;
- spatial coherence;
- maximum daily spatial footprint;
- sectors involved;
- expansion behavior.

These are **analytical/statistical event constructs**, not guaranteed ground-truth fire perimeters and not emergency declarations.

---

## Historical analogues

M8 also computes deterministic historical analogues for reconstructed events.

The event comparison uses normalized:

- duration;
- peak C;
- peak F;
- spatial footprint.

A Euclidean distance is calculated in this event-feature space, with the target event excluded from its own nearest-neighbour list.

The Streamlit application exposes the nearest historical analogues as an additional interpretive layer.

---

# M9 — Progressive Disclosure Observatory

The final application is built in **Streamlit** using a Progressive Disclosure design.

The interface is intentionally layered:

```text
1. Macro understanding
        ↓
2. Burning Activity Calendar
        ↓
3. Long-term context
        ↓
4. Advanced scientific filters
        ↓
5. Guided event investigation
        ↓
6. Event overview + spatial footprint
        ↓
7. Lifecycle progression
        ↓
8. Deep science context + analogues
        ↓
9. Offline scientific plates
```

## Burning Activity Calendar

The calendar is a first-class part of the application.

A user can select a year and a scientific metric, then inspect the temporal structure of burning activity without having to understand the entire underlying model first.

The application also presents context for early-warning-oriented monitoring while clearly stating the boundary:

> Fire Harmonizer is a historical scientific decision-support system, not a certified live emergency-alert service.

## Guided event investigation

Rather than presenting an unexplained event selector, the application first surfaces the largest reconstructed events for the selected year.

Each selected event can then be explored through:

### Event Overview

- event KPIs;
- emergence and decay;
- duration;
- spatial footprint;
- 5×5 event footprint.

### Lifecycle Progression

Daily evolution of:

- C;
- F;
- G;
- active sectors.

### Deep Science Context

- evidence interpretation;
- observation context;
- deterministic historical analogues;
- analogue distances;
- event-level scientific descriptors.

---

# M10 — Offline Scientific Plates

M10 is deliberately kept outside the Streamlit interaction loop.

`generate_charts.py` reads precomputed science/event outputs and generates reusable high-resolution PNG plates with Plotly + Kaleido.

Current plates:

```text
output/charts/
├── 01_longitudinal_evidence.png
├── 02_m4_vs_m8_paradigm_shift.png
├── 03_concordance_breakdown.png
└── 04_mega_event_lifecycle.png
```

### Plate 01 — Longitudinal Evidence

Shows the long-term C/F/G evidence structure across the full record.

### Plate 02 — M4 → M8 Paradigm Shift

Contrasts the legacy single-sector critical-period representation with the newer spatiotemporal event representation.

### Plate 03 — Concordance Breakdown

Shows the relationship between activity and FRP evidence while preserving G concordance as a distinct dimension.

### Plate 04 — Featured Event Lifecycle

Shows the daily evolution of a featured reconstructed event through C/F/G and active-sector structure.

Keeping M10 offline prevents expensive static figure generation from running again whenever a visitor changes an application filter.

---

# Repository structure

The repository is organized around the scientific pipeline and final observatory. Legacy milestone files may remain alongside the current core scripts.

```text
The-Horizon-Watchers/
│
├── README.md
├── requirements.txt
│
├── build_science.py
│       M7 multi-evidence science build
│
├── critical_periods.py
│       M8 reconstructed Fire Events
│       + deterministic analogue search
│
├── generate_charts.py
│       M10 offline scientific plates
│
├── appv2.py
│       Final Streamlit observatory / M9
│
├── output/
│   ├── harmonized_science_all_years.csv
│   ├── critical_periods.csv
│   ├── m7_science_summary.txt
│   ├── ...
│   └── charts/
│       ├── 01_longitudinal_evidence.png
│       ├── 02_m4_vs_m8_paradigm_shift.png
│       ├── 03_concordance_breakdown.png
│       └── 04_mega_event_lifecycle.png
│
└── data/                    # optional local research inputs
    ├── raw/
    └── processed/
```

The large raw/processed observation archive is not required by the deployed Streamlit front end once the derived outputs have already been generated.

---

# Core files

| File | Role |
|---|---|
| `build_science.py` | Builds the M7 multi-evidence scientific record |
| `critical_periods.py` | Reconstructs M8 Fire Events and computes event analogues |
| `generate_charts.py` | Generates the four offline M10 scientific plates |
| `appv2.py` | Runs the final Streamlit observatory |
| `output/harmonized_science_all_years.csv` | Main precomputed science dataset consumed by the application |
| `output/critical_periods.csv` | Reconstructed Fire Event dataset consumed by the application |
| `output/charts/` | Pre-rendered M10 scientific figures |

---

# Installation

## 1. Clone the repository

```bash
git clone https://github.com/<YOUR-USERNAME>/The-Horizon-Watchers.git
cd The-Horizon-Watchers
```

## 2. Install dependencies

The current Streamlit application uses:

```text
streamlit==1.50.0
pandas>=2.0,<3
numpy>=1.26,<3
plotly>=5.20,<7
```

For offline M10 image generation, install the Kaleido dependency used by Plotly.

For acquisition/preprocessing workflows, `requests` may also be required.

A complete local research environment may therefore contain:

```bash
pip install -r requirements.txt
pip install kaleido requests
```

---

# Running the application

The final application is a precomputed-data Streamlit front end.

It expects the required derived files under `output/`.

Run:

```bash
streamlit run appv2.py
```

The main application reads the scientific outputs rather than downloading NASA data during normal visitor interaction.

At minimum, the deployment should include the required `output/` files used by the application, especially:

```text
output/harmonized_science_all_years.csv
output/critical_periods.csv
output/charts/*.png
```

---

# Rebuilding the scientific pipeline

A full research rebuild follows the dependency chain:

```text
Raw / processed FIRMS data
        ↓
M1 / M2 foundation
        ↓
build_science.py
        ↓
M7 all-years science
        ↓
critical_periods.py
        ↓
M8 Fire Events
        ↓
generate_charts.py
        ↓
M10 PNG plates
        ↓
streamlit run appv2.py
```

The preferred workflow is:

```bash
python build_science.py
python critical_periods.py
python generate_charts.py
streamlit run appv2.py
```

For the complete M1–M6 historical foundation, use the corresponding preprocessing, validation, harmonization, critical-period, and visualization scripts retained in the repository/version history.

---

# Design principles

## 1. Harmonize before interpreting

MODIS and VIIRS are not simply concatenated into a single raw-count series.

They are treated as separate observation systems until sensor-specific historical standardization is performed.

## 2. Do not hide disagreement

Sensor disagreement is a scientific signal.

G exists specifically so agreement and disagreement remain visible.

## 3. Do not invent confidence

The project does not reduce C/F/G/S/U into one unexplained confidence percentage.

Different evidence dimensions answer different questions.

## 4. Preserve observation context

A lack of detections is not automatically proof that no fire existed.

Observation opportunity and sensor availability matter.

## 5. Reconstruct events in space and time

Persistent activity is more informative when interpreted as a connected spatiotemporal structure rather than only as isolated spikes.

## 6. Separate science from presentation

The Streamlit application presents precomputed scientific outputs.

The scientific model is not hidden inside the user interface.

## 7. Keep heavy static generation offline

M10 figures are generated once and displayed as artifacts rather than regenerated during every user interaction.

---

# Responsible interpretation

Fire Harmonizer is an analytical system based on remotely sensed active-fire observations.

It does **not** claim to:

- detect every fire;
- produce exact ground-truth fire perimeters;
- measure burned area directly;
- measure wildfire severity directly;
- calculate total fire energy from FRP alone;
- convert sensor agreement into ground truth;
- provide certified emergency alerts.

Important distinctions:

**Detection ≠ unique ground fire**

A satellite detection is an observation of thermal activity, not necessarily one unique physical fire.

**Missing detection ≠ no burning**

The observing system has finite sampling and may miss activity.

**FRP ≠ energy**

FRP is a power quantity. Energy requires integration through time.

**G ≠ probability**

G describes concordance between sensor-derived evidence.

**S ≠ fire severity**

S describes neighboring-sector anomaly coherence.

**M8 event ≠ official fire perimeter**

M8 events are statistical constructs produced by the project’s spatial-temporal connectivity rules.

Extremely large standardized values may also occur when historical baselines have sparse or near-zero variance. These should be interpreted as methodological signals requiring caution rather than literal physical multiples of fire intensity.

---

# AI use

AI was used as a **development and documentation aid**, not as the scientific inference mechanism of Fire Harmonizer.

AI-assisted work may include:

- code drafting and refactoring;
- debugging;
- documentation organization;
- interface wording;
- implementation review;
- analytical discussion.

The core scientific outputs are generated by deterministic Python/Pandas/NumPy processing, statistical transformations, explicit rules, and precomputed datasets.

The repository should therefore be judged from the code and data products rather than from an implied machine-learning model.

---

# Reproducibility

A central design goal of The Horizon Watchers is that a reviewer should be able to trace the project from observation to interpretation:

```text
NASA FIRMS observation
        ↓
cleaned sensor record
        ↓
common sector-day
        ↓
historical reference
        ↓
sensor-specific evidence
        ↓
C / F / G / S / U
        ↓
spatiotemporal event
        ↓
interactive interpretation
```

The project is therefore structured as an **evidence pipeline**, not simply as a dashboard.

---

# Challenge alignment

Fire Harmonizer was designed for the NASA Space Apps Challenge 2026 problem:

**Harmonization of MODIS and VIIRS Hot Spots**

The project responds by providing:

- a common historical representation of MODIS and VIIRS active-fire activity;
- a structured burning-activity calendar;
- historically contextualized anomaly evidence;
- explicit cross-sensor agreement/disagreement;
- spatial coherence;
- observation context;
- reconstructed spatiotemporal Fire Events;
- event lifecycle analysis;
- historical analogues;
- a public-friendly scientific interface with deeper technical layers.

The emphasis is on **interpretability, traceability, and responsible use of Earth-observation evidence**.

---

# Team

## The Horizon Watchers

**Project:** Fire Harmonizer  
**Challenge:** NASA Space Apps Challenge 2026 — Harmonization of MODIS and VIIRS Hot Spots

---

# Data sources

## NASA FIRMS

Fire Information for Resource Management System (FIRMS):

https://firms.modaps.eosdis.nasa.gov/

FIRMS provides the active-fire observation records used as the primary data foundation of this project.

## NASA Space Apps Challenge

https://www.spaceappschallenge.org/

Project challenge context:

https://www.spaceappschallenge.org/2026/challenges/harmonization-of-modis-and-viirs-hot-spots/

---

# License and data note

Add the final repository license here before publishing.

Large raw observation archives and private credentials should not be committed to the public repository.

Recommended public-release pattern:

```text
CODE
+
DERIVED / REQUIRED APP OUTPUTS
+
DOCUMENTATION
+
VISUALS
```

rather than committing every large local acquisition file.

---

# Project status

The scientific and application architecture currently includes:

- M1–M6 historical foundation;
- M7 multi-evidence science;
- M8 reconstructed Fire Events;
- M9 Progressive Disclosure observatory;
- M10 offline scientific plates.

The public repository should be frozen around the same code/output version used for the final application, screenshots, demonstration video, and NASA submission materials.

---

## Fire Harmonizer

**Observe the signal. Preserve the disagreement. Reconstruct the event.**

**The Horizon Watchers**
