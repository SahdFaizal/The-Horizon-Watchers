# ============================================================
# FIRE HARMONIZER — ANNUAL COMPARISON ANALYSIS
# NASA SPACE APPS CHALLENGE 2026
#
# INPUT:
#   output/harmonized_science_all_years.csv
#   output/critical_periods.csv
#
# OUTPUT:
#   output/annual_comparison_2012_2026.csv
#   output/annual_comparison_table.txt
#   output/charts/annual_comparison_2012_2026.png
#
# SCIENCE FRAMEWORK:
#
# C = Activity anomaly
# F = FRP anomaly evidence
# G = Cross-sensor concordance
# S = Spatial coherence
# U = Observation context
#
# IMPORTANT:
#   G = observational concordance, NOT ground truth.
#   F = standardized FRP anomaly evidence, NOT physical energy.
#   S = spatial coherence from neighboring anomalous sectors.
#   U = observation context, NOT a confidence score.
# ============================================================

from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

SCIENCE_FILE = Path(
    "output/harmonized_science_all_years.csv"
)

EVENT_FILE = Path(
    "output/critical_periods.csv"
)

OUTPUT_DIR = Path("output")
CHART_DIR = OUTPUT_DIR / "charts"

ANNUAL_CSV = (
    OUTPUT_DIR /
    "annual_comparison_2012_2026.csv"
)

ANNUAL_TXT = (
    OUTPUT_DIR /
    "annual_comparison_table.txt"
)

ANNUAL_PNG = (
    CHART_DIR /
    "annual_comparison_2012_2026.png"
)

START_YEAR = 2012
END_YEAR = 2026


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def require_column(df, column):
    """
    Require an exact column name.
    """

    if column not in df.columns:
        raise ValueError(
            f"\nRequired column '{column}' was not found.\n\n"
            f"Available columns:\n{list(df.columns)}"
        )

    return column


def numeric(df, column):
    """
    Safely convert a column to numeric.
    """

    return pd.to_numeric(
        df[column],
        errors="coerce"
    )


def safe_mean(series):
    values = pd.to_numeric(
        series,
        errors="coerce"
    ).dropna()

    if len(values) == 0:
        return np.nan

    return values.mean()


def safe_median(series):
    values = pd.to_numeric(
        series,
        errors="coerce"
    ).dropna()

    if len(values) == 0:
        return np.nan

    return values.median()


def safe_max(series):
    values = pd.to_numeric(
        series,
        errors="coerce"
    ).dropna()

    if len(values) == 0:
        return np.nan

    return values.max()


def percentage(part, total):
    if total == 0:
        return np.nan

    return 100.0 * part / total


def fmt(value, decimals=2):

    if pd.isna(value):
        return "N/A"

    return f"{value:.{decimals}f}"


def relative_interpretation(
    value,
    baseline,
    positive,
    negative
):
    """
    Compare annual mean against the long-term median.
    """

    if pd.isna(value) or pd.isna(baseline):
        return "Insufficient data"

    if baseline == 0:
        return "Near-zero baseline"

    ratio = value / baseline

    if ratio >= 1.50:
        return f"Strongly elevated {positive}"

    if ratio >= 1.15:
        return f"Elevated {positive}"

    if ratio <= 0.67:
        return f"Strongly reduced {negative}"

    if ratio <= 0.85:
        return f"Reduced {negative}"

    return "Near long-term median"


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

CHART_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CHECK INPUT FILES
# ============================================================

if not SCIENCE_FILE.exists():

    raise FileNotFoundError(
        f"\nScience file not found:\n"
        f"{SCIENCE_FILE}\n\n"
        f"Expected:\n"
        f"output/harmonized_science_all_years.csv"
    )


if not EVENT_FILE.exists():

    raise FileNotFoundError(
        f"\nEvent file not found:\n"
        f"{EVENT_FILE}\n\n"
        f"Expected:\n"
        f"output/critical_periods.csv"
    )


# ============================================================
# LOAD M7 SCIENCE DATA
# ============================================================

print("=" * 75)
print("FIRE HARMONIZER — ANNUAL COMPARISON")
print("=" * 75)

print("\nLoading M7 science dataset...")

science = pd.read_csv(
    SCIENCE_FILE
)

print(
    f"Rows loaded: {len(science):,}"
)

print(
    f"Columns loaded: {len(science.columns):,}"
)


# ============================================================
# VERIFY EXACT COLUMNS FROM YOUR DATASET
# ============================================================

require_column(
    science,
    "acq_date"
)

require_column(
    science,
    "sector"
)

require_column(
    science,
    "m7_C"
)

require_column(
    science,
    "m7_F"
)

require_column(
    science,
    "m7_G"
)

require_column(
    science,
    "m7_S"
)

require_column(
    science,
    "u_sensor_state"
)


# ============================================================
# PARSE DATE
# ============================================================

print("\nParsing acquisition dates...")

science["_date"] = pd.to_datetime(
    science["acq_date"],
    errors="coerce"
)

bad_dates = science["_date"].isna().sum()

if bad_dates > 0:

    print(
        f"WARNING: {bad_dates:,} rows have invalid dates."
    )

science["_year"] = (
    science["_date"]
    .dt.year
)


# ============================================================
# CONVERT M7 EVIDENCE VARIABLES
# ============================================================

science["_C"] = numeric(
    science,
    "m7_C"
)

science["_F"] = numeric(
    science,
    "m7_F"
)

science["_G"] = numeric(
    science,
    "m7_G"
)

science["_S"] = numeric(
    science,
    "m7_S"
)


# ============================================================
# RESTRICT TO 2012–2026
# ============================================================

science = science[
    science["_year"].between(
        START_YEAR,
        END_YEAR
    )
].copy()


print(
    f"Rows within {START_YEAR}–{END_YEAR}: "
    f"{len(science):,}"
)


# ============================================================
# C-ACTIVE SECTOR-DAYS
# ============================================================

# M7's anomaly evidence threshold.
#
# C >= 2 means the sector-day has meaningful
# activity-anomaly evidence.

science["_C_active"] = (
    science["_C"] >= 2.0
)


active = science[
    science["_C_active"]
].copy()


print(
    f"C-active sector-days: "
    f"{len(active):,}"
)


# ============================================================
# U — SENSOR STATE
# ============================================================

science["_U"] = (
    science["u_sensor_state"]
    .astype(str)
    .str.strip()
    .str.lower()
)


print(
    "\nObserved U sensor states:"
)

print(
    science["_U"]
    .value_counts(dropna=False)
)


# ============================================================
# LOAD M8 RECONSTRUCTED FIRE EVENTS
# ============================================================

print(
    "\nLoading M8 reconstructed fire events..."
)

events = pd.read_csv(
    EVENT_FILE
)

print(
    f"M8 rows: {len(events):,}"
)


# ============================================================
# DISPLAY EVENT COLUMNS
# ============================================================

print(
    "\nM8 columns detected:"
)

print(
    list(events.columns)
)


# ============================================================
# FIND EVENT DATE COLUMNS
# ============================================================

possible_start_columns = [
    "emergence_date",
    "start_date",
    "event_start",
    "start",
    "emergence"
]

possible_end_columns = [
    "decay_date",
    "end_date",
    "event_end",
    "end",
    "decay"
]

possible_id_columns = [
    "event_id",
    "Event_ID",
    "id",
    "ID"
]


event_start_col = None

for col in possible_start_columns:

    if col in events.columns:
        event_start_col = col
        break


event_end_col = None

for col in possible_end_columns:

    if col in events.columns:
        event_end_col = col
        break


event_id_col = None

for col in possible_id_columns:

    if col in events.columns:
        event_id_col = col
        break


print(
    f"\nEvent start column: "
    f"{event_start_col}"
)

print(
    f"Event end column: "
    f"{event_end_col}"
)

print(
    f"Event ID column: "
    f"{event_id_col}"
)


# ============================================================
# PREPARE EVENT DATES
# ============================================================

if event_start_col is not None:

    events["_event_start"] = pd.to_datetime(
        events[event_start_col],
        errors="coerce"
    )

else:

    events["_event_start"] = pd.NaT


if event_end_col is not None:

    events["_event_end"] = pd.to_datetime(
        events[event_end_col],
        errors="coerce"
    )

else:

    events["_event_end"] = events["_event_start"]


# ============================================================
# EVENT IDs
# ============================================================

if event_id_col is not None:

    events["_event_id"] = (
        events[event_id_col]
        .astype(str)
    )

else:

    events["_event_id"] = [
        f"EVT-{i + 1:04d}"
        for i in range(len(events))
    ]


# ============================================================
# COUNT M8 EVENTS TOUCHING EACH YEAR
# ============================================================

event_year_records = []


for _, row in events.iterrows():

    start = row["_event_start"]
    end = row["_event_end"]

    if pd.isna(start):
        continue

    if pd.isna(end):
        end = start

    if end < start:
        end = start

    first_year = max(
        START_YEAR,
        start.year
    )

    last_year = min(
        END_YEAR,
        end.year
    )

    for year in range(
        first_year,
        last_year + 1
    ):

        event_year_records.append(
            {
                "year": year,
                "event_id": row["_event_id"]
            }
        )


event_year_df = pd.DataFrame(
    event_year_records
)


if len(event_year_df) > 0:

    events_per_year = (
        event_year_df
        .groupby("year")["event_id"]
        .nunique()
        .to_dict()
    )

else:

    events_per_year = {}


# ============================================================
# LONG-TERM BASELINES
# ============================================================

baseline_C = safe_median(
    active["_C"]
)

baseline_F = safe_median(
    active["_F"]
)

baseline_G = safe_median(
    active["_G"]
)

baseline_S = safe_median(
    active["_S"]
)


print("\nLong-term active-sector medians:")

print(
    f"  C = {fmt(baseline_C)}"
)

print(
    f"  F = {fmt(baseline_F)}"
)

print(
    f"  G = {fmt(baseline_G)}"
)

print(
    f"  S = {fmt(baseline_S)}"
)


# ============================================================
# ANNUAL ANALYSIS
# ============================================================

print(
    "\nCalculating annual summaries..."
)

annual_records = []


for year in range(
    START_YEAR,
    END_YEAR + 1
):

    year_all = science[
        science["_year"] == year
    ].copy()


    year_active = year_all[
        year_all["_C_active"]
    ].copy()


    total_rows = len(
        year_all
    )

    active_rows = len(
        year_active
    )


    # ========================================================
    # TEMPORAL COVERAGE
    # ========================================================

    if total_rows > 0:

        start_date = (
            year_all["_date"].min()
        )

        end_date = (
            year_all["_date"].max()
        )

        observed_days = (
            year_all["_date"]
            .dt.normalize()
            .nunique()
        )

    else:

        start_date = pd.NaT
        end_date = pd.NaT
        observed_days = 0


    # ========================================================
    # C
    # ========================================================

    mean_C = safe_mean(
        year_active["_C"]
    )

    median_C = safe_median(
        year_active["_C"]
    )

    max_C = safe_max(
        year_active["_C"]
    )


    # ========================================================
    # F
    # ========================================================

    mean_F = safe_mean(
        year_active["_F"]
    )

    median_F = safe_median(
        year_active["_F"]
    )

    max_F = safe_max(
        year_active["_F"]
    )


    # ========================================================
    # G
    # ========================================================

    mean_G = safe_mean(
        year_active["_G"]
    )

    median_G = safe_median(
        year_active["_G"]
    )

    max_G = safe_max(
        year_active["_G"]
    )


    # ========================================================
    # S
    # ========================================================

    mean_S = safe_mean(
        year_active["_S"]
    )

    median_S = safe_median(
        year_active["_S"]
    )

    max_S = safe_max(
        year_active["_S"]
    )


    # ========================================================
    # U
    # ========================================================

    if total_rows > 0:

        u_counts = (
            year_all["_U"]
            .value_counts()
        )

        both_pct = percentage(
            u_counts.get(
                "both_sensors",
                0
            ),
            total_rows
        )

        viirs_pct = percentage(
            u_counts.get(
                "viirs_only",
                0
            ),
            total_rows
        )

        modis_pct = percentage(
            u_counts.get(
                "modis_only",
                0
            ),
            total_rows
        )

        neither_pct = percentage(
            u_counts.get(
                "neither",
                0
            ),
            total_rows
        )

    else:

        both_pct = np.nan
        viirs_pct = np.nan
        modis_pct = np.nan
        neither_pct = np.nan


    # ========================================================
    # M8 EVENTS
    # ========================================================

    event_count = events_per_year.get(
        year,
        0
    )


    # ========================================================
    # C ACTIVE RATE
    # ========================================================

    active_rate = percentage(
        active_rows,
        total_rows
    )


    # ========================================================
    # INTERPRETATIONS
    # ========================================================

    c_interpretation = relative_interpretation(
        mean_C,
        baseline_C,
        "activity anomaly",
        "activity anomaly"
    )


    f_interpretation = relative_interpretation(
        mean_F,
        baseline_F,
        "FRP anomaly",
        "FRP anomaly"
    )


    g_interpretation = relative_interpretation(
        mean_G,
        baseline_G,
        "sensor concordance",
        "sensor concordance"
    )


    s_interpretation = relative_interpretation(
        mean_S,
        baseline_S,
        "spatial coherence",
        "spatial coherence"
    )


    # ========================================================
    # ANNUAL INTERPRETATION
    # ========================================================

    interpretation_parts = []


    if event_count > 0:

        interpretation_parts.append(
            f"{event_count} reconstructed "
            f"fire event"
            + (
                "s"
                if event_count != 1
                else ""
            )
        )

    else:

        interpretation_parts.append(
            "No reconstructed fire events"
        )


    if active_rows > 0:

        interpretation_parts.append(
            c_interpretation.lower()
        )

        interpretation_parts.append(
            f_interpretation.lower()
        )

        interpretation_parts.append(
            g_interpretation.lower()
        )

        interpretation_parts.append(
            s_interpretation.lower()
        )


    final_interpretation = (
        "; ".join(
            interpretation_parts
        )
    )


    # ========================================================
    # STORE RECORD
    # ========================================================

    annual_records.append(
        {
            "Year": year,

            # Coverage
            "Start date":
                (
                    start_date.strftime(
                        "%Y-%m-%d"
                    )
                    if not pd.isna(start_date)
                    else "N/A"
                ),

            "End date":
                (
                    end_date.strftime(
                        "%Y-%m-%d"
                    )
                    if not pd.isna(end_date)
                    else "N/A"
                ),

            "Observed days":
                observed_days,

            "Sector-days":
                total_rows,

            # C
            "C-active sector-days":
                active_rows,

            "C-active rate (%)":
                active_rate,

            "C mean":
                mean_C,

            "C median":
                median_C,

            "C maximum":
                max_C,

            "C interpretation":
                c_interpretation,

            # F
            "F mean":
                mean_F,

            "F median":
                median_F,

            "F maximum":
                max_F,

            "F interpretation":
                f_interpretation,

            # G
            "G mean":
                mean_G,

            "G median":
                median_G,

            "G maximum":
                max_G,

            "G interpretation":
                g_interpretation,

            # S
            "S mean":
                mean_S,

            "S median":
                median_S,

            "S maximum":
                max_S,

            "S interpretation":
                s_interpretation,

            # U
            "U both sensors (%)":
                both_pct,

            "U VIIRS only (%)":
                viirs_pct,

            "U MODIS only (%)":
                modis_pct,

            "U neither (%)":
                neither_pct,

            # M8
            "M8 reconstructed events":
                event_count,

            # Interpretation
            "Final annual interpretation":
                final_interpretation
        }
    )


# ============================================================
# CREATE DATAFRAME
# ============================================================

annual = pd.DataFrame(
    annual_records
)


# ============================================================
# SAVE FULL NUMERICAL CSV
# ============================================================

annual.to_csv(
    ANNUAL_CSV,
    index=False
)

print(
    "\nSaved:"
)

print(
    f"  {ANNUAL_CSV}"
)


# ============================================================
# NASA-READY TEXT STORIES
# ============================================================

def make_C_story(row):

    if row["C-active sector-days"] == 0:

        return (
            "No C-active sector-days."
        )

    return (
        f"{int(row['C-active sector-days']):,} "
        f"C-active sector-days; "
        f"mean C = {fmt(row['C mean'])}; "
        f"maximum C = {fmt(row['C maximum'])}; "
        f"{row['C interpretation']}."
    )


def make_F_story(row):

    if pd.isna(row["F mean"]):

        return "N/A."

    return (
        f"mean F = {fmt(row['F mean'])}; "
        f"median = {fmt(row['F median'])}; "
        f"maximum = {fmt(row['F maximum'])}; "
        f"{row['F interpretation']}."
    )


def make_G_story(row):

    if pd.isna(row["G mean"]):

        return "N/A."

    return (
        f"mean G = {fmt(row['G mean'])}; "
        f"median = {fmt(row['G median'])}; "
        f"{row['G interpretation']}."
    )


def make_S_story(row):

    if pd.isna(row["S mean"]):

        return "N/A."

    return (
        f"mean S = {fmt(row['S mean'])}; "
        f"median = {fmt(row['S median'])}; "
        f"{row['S interpretation']}."
    )


def make_U_story(row):

    if pd.isna(
        row["U both sensors (%)"]
    ):

        return "Unavailable."

    return (
        f"Both sensors = "
        f"{fmt(row['U both sensors (%)'])}%; "
        f"VIIRS-only = "
        f"{fmt(row['U VIIRS only (%)'])}%; "
        f"MODIS-only = "
        f"{fmt(row['U MODIS only (%)'])}%; "
        f"neither = "
        f"{fmt(row['U neither (%)'])}%."
    )


# ============================================================
# BUILD DOCUMENTATION TABLE
# ============================================================

documentation_table = pd.DataFrame(
    {
        "Year":
            annual["Year"],

        "C — anomaly story":
            annual.apply(
                make_C_story,
                axis=1
            ),

        "F — FRP":
            annual.apply(
                make_F_story,
                axis=1
            ),

        "G — concordance":
            annual.apply(
                make_G_story,
                axis=1
            ),

        "S — spatial coherence":
            annual.apply(
                make_S_story,
                axis=1
            ),

        "U — context":
            annual.apply(
                make_U_story,
                axis=1
            ),

        "Final annual interpretation":
            annual[
                "Final annual interpretation"
            ]
    }
)


# ============================================================
# SAVE NASA-READY TEXT FILE
# ============================================================

with open(
    ANNUAL_TXT,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "FIRE HARMONIZER — ANNUAL COMPARISON\n"
    )

    f.write(
        "NASA SPACE APPS CHALLENGE 2026\n"
    )

    f.write(
        "=" * 120 + "\n\n"
    )

    f.write(
        "SCIENCE DEFINITIONS\n"
    )

    f.write(
        "C = activity anomaly evidence.\n"
    )

    f.write(
        "F = standardized FRP anomaly evidence, "
        "not physical energy.\n"
    )

    f.write(
        "G = observational cross-sensor concordance, "
        "not ground truth.\n"
    )

    f.write(
        "S = spatial coherence from neighboring "
        "anomalous sectors.\n"
    )

    f.write(
        "U = observation context, not a confidence score.\n"
    )

    f.write(
        "M8 event counts represent reconstructed events "
        "touching each calendar year.\n"
    )

    f.write(
        "2012 and 2026 are partial years because the "
        "study period begins on 2012-01-20 and ends "
        "on 2026-06-30.\n\n"
    )

    f.write(
        documentation_table.to_string(
            index=False,
            max_colwidth=60
        )
    )

    f.write("\n")


print(
    f"  {ANNUAL_TXT}"
)


# ============================================================
# GRAPH DATA
# ============================================================

years = annual["Year"].to_numpy()

event_values = (
    annual[
        "M8 reconstructed events"
    ].to_numpy()
)

C_values = (
    annual[
        "C mean"
    ].to_numpy()
)

F_values = (
    annual[
        "F mean"
    ].to_numpy()
)

G_values = (
    annual[
        "G mean"
    ].to_numpy()
)

S_values = (
    annual[
        "S mean"
    ].to_numpy()
)


# ============================================================
# CREATE FIGURE
# ============================================================

print(
    "\nGenerating annual comparison graph..."
)

fig = plt.figure(
    figsize=(15, 12)
)

grid = fig.add_gridspec(
    3,
    1,
    hspace=0.35
)


# ============================================================
# PANEL 1 — M8 EVENT COUNT
# ============================================================

ax1 = fig.add_subplot(
    grid[0]
)

ax1.plot(
    years,
    event_values,
    marker="o",
    linewidth=2
)

ax1.set_title(
    "Reconstructed Fire Events by Year",
    fontsize=15,
    fontweight="bold"
)

ax1.set_xlabel(
    "Year"
)

ax1.set_ylabel(
    "M8 reconstructed events"
)

ax1.set_xticks(
    years
)

ax1.grid(
    True,
    alpha=0.25
)


# ============================================================
# PANEL 2 — C AND F
# ============================================================

ax2 = fig.add_subplot(
    grid[1]
)

ax2.plot(
    years,
    C_values,
    marker="o",
    linewidth=2,
    label="Mean C — activity anomaly"
)

ax2.plot(
    years,
    F_values,
    marker="s",
    linewidth=2,
    label="Mean F — FRP anomaly"
)

ax2.axhline(
    baseline_C,
    linestyle="--",
    linewidth=1.5,
    label="C long-term median"
)

ax2.axhline(
    baseline_F,
    linestyle=":",
    linewidth=1.5,
    label="F long-term median"
)

ax2.set_title(
    "Annual Activity and FRP Anomaly Evidence",
    fontsize=15,
    fontweight="bold"
)

ax2.set_xlabel(
    "Year"
)

ax2.set_ylabel(
    "Evidence value"
)

ax2.set_xticks(
    years
)

ax2.grid(
    True,
    alpha=0.25
)

ax2.legend(
    loc="best"
)


# ============================================================
# PANEL 3 — G AND S
# ============================================================

ax3 = fig.add_subplot(
    grid[2]
)

ax3.plot(
    years,
    G_values,
    marker="o",
    linewidth=2,
    label="Mean G — sensor concordance"
)

ax3.plot(
    years,
    S_values,
    marker="s",
    linewidth=2,
    label="Mean S — spatial coherence"
)

ax3.axhline(
    baseline_G,
    linestyle="--",
    linewidth=1.5,
    label="G long-term median"
)

ax3.axhline(
    baseline_S,
    linestyle=":",
    linewidth=1.5,
    label="S long-term median"
)

ax3.set_title(
    "Annual Cross-Sensor and Spatial Evidence",
    fontsize=15,
    fontweight="bold"
)

ax3.set_xlabel(
    "Year"
)

ax3.set_ylabel(
    "Evidence value"
)

ax3.set_xticks(
    years
)

ax3.grid(
    True,
    alpha=0.25
)

ax3.legend(
    loc="best"
)


# ============================================================
# OVERALL TITLE
# ============================================================

fig.suptitle(
    "Fire Harmonizer — Annual Evidence Comparison, 2012–2026",
    fontsize=19,
    fontweight="bold",
    y=0.995
)


# ============================================================
# FOOTNOTE
# ============================================================

fig.text(
    0.5,
    0.005,
    (
        "2012 and 2026 are partial years. "
        "C/F/G/S are summarized over C-active "
        "sector-days (C ≥ 2). "
        "G represents observational concordance, "
        "not ground truth."
    ),
    ha="center",
    fontsize=9
)


# ============================================================
# SAVE PNG
# ============================================================

plt.savefig(
    ANNUAL_PNG,
    dpi=300,
    bbox_inches="tight"
)

plt.close(
    fig
)

print(
    f"  {ANNUAL_PNG}"
)


# ============================================================
# PRINT ANNUAL SUMMARY
# ============================================================

print("\n")
print("=" * 120)
print("NASA-READY ANNUAL SUMMARY")
print("=" * 120)
print()


for _, row in documentation_table.iterrows():

    print(
        f"{int(row['Year'])}: "
        f"{row['Final annual interpretation']}"
    )


# ============================================================
# FINAL STATUS
# ============================================================

print("\n")
print("=" * 75)
print("ANNUAL ANALYSIS COMPLETE")
print("=" * 75)

print(
    """
Generated:

1. output/annual_comparison_2012_2026.csv
   Complete numerical annual dataset.

2. output/annual_comparison_table.txt
   NASA documentation-ready annual comparison.

3. output/charts/annual_comparison_2012_2026.png
   Annual comparison graph.

Science interpretation:

C = activity anomaly
F = FRP anomaly evidence
G = cross-sensor concordance
S = spatial coherence
U = observation context

C/F/G/S are summarized on C-active sector-days
where C >= 2.

U is retained as observation context rather than
being converted into an arbitrary confidence score.
"""
)