"""
FIRE HARMONIZER - MILESTONE 8
================================

SPATIOTEMPORAL EVENT RECONSTRUCTION

Purpose
-------
Upgrade the original M4 critical-period detector from an isolated,
sector-by-sector temporal sequence into a spatiotemporal event graph.

The M7 science file provides the multi-evidence fields:

    m7_C : activity anomaly
    m7_F : FRP anomaly
    m7_G : cross-sensor concordance
    m7_S : spatial coherence

M8 then:

    1. Selects every DATE × SECTOR node with C >= 2.
    2. Connects nodes across consecutive calendar days when they occupy
       the same sector or an adjacent sector in the 5×5 grid.
    3. Extracts connected components as reconstructed Fire Events.
    4. Calculates an empirical event lifecycle and evidence summary.
    5. Searches the full event archive for deterministic historical analogues.

No machine learning is used.
No API calls are made.
No Streamlit/UI code is required.

Input
-----
output/harmonized_science_all_years.csv

Output
------
data/output/critical_periods.csv

The output path can be overridden with --output.

Scientific interpretation
--------------------------
A reconstructed event is a connected component of statistically anomalous
sector-days under the explicit C >= 2 threshold and continuity rule. It is
therefore an observational/statistical event construct, not a guaranteed
ground-truth fire perimeter or emergency declaration.
"""

from __future__ import annotations

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import networkx as nx


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = (
    BASE_DIR
    / "output"
    / "harmonized_science_all_years.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "output"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "critical_periods.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

GRID_SIZE = 5

# M4 seed threshold carried into M8 exactly as specified.
ANOMALY_THRESHOLD = 2.0

# Event analogue features that are normalized to 0–1.
ANALOGUE_FEATURES = [
    "duration_days",
    "peak_C",
    "peak_F",
    "spatial_footprint",
]

# Small numerical tolerance for deterministic comparisons.
EPSILON = 1e-12

# Fixed sector vocabulary used throughout M1–M8.
EXPECTED_SECTORS = [
    f"C{i:02d}"
    for i in range(1, GRID_SIZE * GRID_SIZE + 1)
]


# ============================================================
# REQUIRED INPUT COLUMNS
# ============================================================

# M7 uses explicit m7_ prefixes so the new science layer does not
# silently overwrite the original M3 metrics.
#
# To make M8 robust to a simplified exported file, aliases C/F/G/S
# are also accepted and normalized internally.
COLUMN_ALIASES = {
    "C": ["m7_C", "C"],
    "F": ["m7_F", "F"],
    "G": ["m7_G", "G"],
    "S": ["m7_S", "S"],
}


# ============================================================
# INPUT / VALIDATION HELPERS
# ============================================================


def resolve_column(
    df: pd.DataFrame,
    aliases: list[str],
    canonical_name: str,
) -> str:
    """
    Return the first available input column matching the supplied aliases.

    M7's authoritative names are m7_C/m7_F/m7_G/m7_S, but accepting
    the unprefixed C/F/G/S names makes the M8 script portable across
    equivalent exports without changing the science logic.
    """
    for column in aliases:
        if column in df.columns:
            return column

    raise ValueError(
        f"Could not find the required M7 field '{canonical_name}'. "
        f"Expected one of: {aliases}"
    )


def require_input_columns(df: pd.DataFrame) -> dict[str, str]:
    """
    Validate and return the actual source column names used for C/F/G/S.
    """
    required_base = ["acq_date", "sector"]

    missing_base = [
        column
        for column in required_base
        if column not in df.columns
    ]

    if missing_base:
        raise ValueError(
            "M7 science file is missing required columns:\n  "
            + "\n  ".join(missing_base)
        )

    resolved = {
        canonical: resolve_column(
            df,
            aliases,
            canonical,
        )
        for canonical, aliases in COLUMN_ALIASES.items()
    }

    return resolved


def load_science(input_file: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    """
    Load the complete M7 all-years science matrix and normalize its
    date/sector/evidence fields for M8.
    """
    if not input_file.exists():
        raise FileNotFoundError(
            f"Missing M7 input file:\n{input_file}"
        )

    df = pd.read_csv(input_file)

    if df.empty:
        raise ValueError(
            f"M7 input file is empty:\n{input_file}"
        )

    columns = require_input_columns(df)

    # --------------------------------------------------------
    # Normalize dates.
    # --------------------------------------------------------
    df["acq_date"] = pd.to_datetime(
        df["acq_date"],
        errors="coerce",
    )

    if df["acq_date"].isna().any():
        bad_count = int(df["acq_date"].isna().sum())
        raise ValueError(
            f"M7 input contains {bad_count:,} invalid acquisition dates."
        )

    # --------------------------------------------------------
    # Normalize sector labels.
    # --------------------------------------------------------
    df["sector"] = (
        df["sector"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    invalid_sectors = sorted(
        set(df["sector"].dropna()) - set(EXPECTED_SECTORS)
    )

    if invalid_sectors:
        raise ValueError(
            "M8 found unexpected sector codes:\n  "
            + "\n  ".join(invalid_sectors)
        )

    # --------------------------------------------------------
    # Copy evidence fields into canonical M8 names.
    # --------------------------------------------------------
    for canonical_name, source_name in columns.items():
        df[canonical_name] = pd.to_numeric(
            df[source_name],
            errors="coerce",
        )

    # C is mandatory for node construction.
    if df["C"].isna().any():
        bad_count = int(df["C"].isna().sum())
        raise ValueError(
            f"M7 activity anomaly C contains {bad_count:,} missing values."
        )

    # F/G/S are allowed to have missing values because an event can still
    # be reconstructed from C; their event-level summaries use finite
    # values only. This avoids silently replacing unavailable evidence
    # with zeros.
    if (df["G"].dropna() < 0).any() or (df["G"].dropna() > 1).any():
        raise ValueError(
            "M7 G values must remain within the documented 0–1 range."
        )

    if (df["S"].dropna() < 0).any() or (df["S"].dropna() > 1).any():
        raise ValueError(
            "M7 S values must remain within the documented 0–1 range."
        )

    # --------------------------------------------------------
    # One node must correspond to one DATE × SECTOR position.
    # Duplicate positions would create ambiguous graph nodes and
    # therefore are rejected rather than silently aggregated.
    # --------------------------------------------------------
    duplicate_mask = df.duplicated(
        subset=["acq_date", "sector"],
        keep=False,
    )

    if duplicate_mask.any():
        duplicate_rows = df.loc[
            duplicate_mask,
            ["acq_date", "sector"],
        ].drop_duplicates()

        preview = duplicate_rows.head(10).to_string(index=False)

        raise ValueError(
            "Duplicate DATE × SECTOR positions found in the M7 science "
            "matrix. M8 requires one row per node.\n"
            f"Examples:\n{preview}"
        )

    # --------------------------------------------------------
    # Deterministic ordering.
    # --------------------------------------------------------
    df = (
        df.sort_values(
            ["acq_date", "sector"],
        )
        .reset_index(drop=True)
    )

    return df, columns


# ============================================================
# 5×5 GRID GEOMETRY
# ============================================================


def sector_to_position(sector: str) -> tuple[int, int]:
    """
    Convert C01–C25 to zero-based row/column coordinates.

    Existing project numbering:

        C01 C02 C03 C04 C05
        C06 C07 C08 C09 C10
        C11 C12 C13 C14 C15
        C16 C17 C18 C19 C20
        C21 C22 C23 C24 C25

    Adjacency is 8-neighbor adjacency, because "adjacent sector"
    includes horizontal, vertical, and diagonal neighbors.
    """
    number = int(str(sector).replace("C", ""))
    zero_based = number - 1

    return (
        zero_based // GRID_SIZE,
        zero_based % GRID_SIZE,
    )


SECTOR_POSITIONS = {
    sector: sector_to_position(sector)
    for sector in EXPECTED_SECTORS
}


def are_spatially_contiguous(
    sector_a: str,
    sector_b: str,
) -> bool:
    """
    Return True when two sectors are the same or adjacent in the 5×5 grid.

    The Euclidean/row-column rule is:

        max(|dr|, |dc|) <= 1

    which gives the same sector plus its eight immediate neighbors.
    """
    row_a, col_a = SECTOR_POSITIONS[sector_a]
    row_b, col_b = SECTOR_POSITIONS[sector_b]

    return (
        max(
            abs(row_a - row_b),
            abs(col_a - col_b),
        )
        <= 1
    )


# ============================================================
# GRAPH CONSTRUCTION
# ============================================================


def build_event_graph(
    df: pd.DataFrame,
) -> tuple[nx.Graph, pd.DataFrame]:
    """
    Construct the M8 spatiotemporal event graph.

    Nodes
    -----
    Every DATE × SECTOR row with C >= 2.

    Edges
    ------
    Two qualifying nodes are connected only when:

        1. Date difference is exactly one calendar day.
        2. Sectors are the same or spatially adjacent.

    Since the edge rule always points between consecutive dates,
    connected components become reconstructed events.
    """
    graph = nx.Graph()

    qualifying = (
        df.loc[
            df["C"] >= ANOMALY_THRESHOLD,
            [
                "acq_date",
                "sector",
                "C",
                "F",
                "G",
                "S",
            ],
        ]
        .copy()
        .sort_values(
            ["acq_date", "sector"],
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Add every qualifying DATE × SECTOR row as a graph node.
    #
    # Node id is the tuple (Timestamp, sector), which is:
    #   - unique by validation,
    #   - deterministic,
    #   - naturally sortable,
    #   - directly traceable to the science matrix.
    # --------------------------------------------------------
    for row in qualifying.itertuples(index=False):
        node_id = (
            pd.Timestamp(row.acq_date),
            str(row.sector),
        )

        graph.add_node(
            node_id,
            date=pd.Timestamp(row.acq_date),
            sector=str(row.sector),
            C=float(row.C) if np.isfinite(row.C) else np.nan,
            F=float(row.F) if np.isfinite(row.F) else np.nan,
            G=float(row.G) if np.isfinite(row.G) else np.nan,
            S=float(row.S) if np.isfinite(row.S) else np.nan,
        )

    # --------------------------------------------------------
    # Efficient edge construction.
    #
    # We only need to inspect adjacent calendar dates. For every
    # date T, compare its qualifying sectors against date T+1.
    # No all-pairs graph search is necessary.
    # --------------------------------------------------------
    date_sector_lookup: dict[pd.Timestamp, list[str]] = {}

    for row in qualifying[
        ["acq_date", "sector"]
    ].itertuples(index=False, name=None):
        date_value, sector = row
        date_sector_lookup.setdefault(
            pd.Timestamp(date_value),
            [],
        ).append(str(sector))

    sorted_dates = sorted(date_sector_lookup)

    for current_date in sorted_dates:
        next_date = current_date + pd.Timedelta(days=1)

        current_sectors = date_sector_lookup.get(
            current_date,
            [],
        )
        next_sectors = date_sector_lookup.get(
            next_date,
            [],
        )

        if not next_sectors:
            continue

        # For each qualifying sector at T, connect to all qualifying
        # same/adjacent sectors at T+1.
        for sector_a in current_sectors:
            for sector_b in next_sectors:
                if are_spatially_contiguous(
                    sector_a,
                    sector_b,
                ):
                    graph.add_edge(
                        (current_date, sector_a),
                        (next_date, sector_b),
                    )

    return graph, qualifying


# ============================================================
# EVENT UTILITY FUNCTIONS
# ============================================================


def normalize_series_0_1(
    values: pd.Series,
) -> pd.Series:
    """
    Min-max normalize a series to [0, 1].

    When every value is identical, there is no measurable variation
    in that feature. The deterministic convention here is 0.0 for
    every member so that the feature contributes zero distance.
    """
    numeric = pd.to_numeric(
        values,
        errors="coerce",
    )

    finite = numeric[np.isfinite(numeric)]

    if finite.empty:
        return pd.Series(
            np.nan,
            index=values.index,
            dtype=float,
        )

    minimum = float(finite.min())
    maximum = float(finite.max())

    if abs(maximum - minimum) <= EPSILON:
        return pd.Series(
            0.0,
            index=values.index,
            dtype=float,
        )

    return (
        (numeric - minimum)
        / (maximum - minimum)
    )


def peak_date_by_normalized_c_and_f(
    nodes_df: pd.DataFrame,
) -> pd.Timestamp:
    """
    Select the event peak date using the sum of normalized C and F.

    C and F are normalized independently within the event's nodes.
    This ensures the larger numeric scale of one evidence channel does
    not dominate the other.

    Missing F is not fabricated. Its contribution is treated as zero
    after normalization while C remains fully represented.

    Ties are resolved deterministically by choosing the earliest date,
    then the lexicographically smallest sector.
    """
    work = nodes_df.copy()

    work["C_norm"] = normalize_series_0_1(
        work["C"]
    )

    work["F_norm"] = normalize_series_0_1(
        work["F"]
    )

    work["C_norm"] = work["C_norm"].fillna(0.0)
    work["F_norm"] = work["F_norm"].fillna(0.0)

    work["combined_peak_score"] = (
        work["C_norm"]
        + work["F_norm"]
    )

    work = work.sort_values(
        [
            "combined_peak_score",
            "date",
            "sector",
        ],
        ascending=[
            False,
            True,
            True,
        ],
    )

    return pd.Timestamp(
        work.iloc[0]["date"]
    )


def daily_footprint(
    nodes_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return unique active sectors per event date.

    The output contains:
        date
        daily_spatial_footprint
        cumulative_spatial_footprint
    """
    daily = (
        nodes_df.groupby(
            "date",
            as_index=False,
        )["sector"]
        .nunique()
        .rename(
            columns={
                "sector": "daily_spatial_footprint"
            }
        )
        .sort_values("date")
        .reset_index(drop=True)
    )

    # Cumulative union of unique sectors encountered by that date.
    seen: set[str] = set()
    cumulative_values: list[int] = []

    for date_value in daily["date"]:
        sectors_today = set(
            nodes_df.loc[
                nodes_df["date"] == date_value,
                "sector",
            ]
        )

        seen.update(sectors_today)
        cumulative_values.append(
            len(seen)
        )

    daily["cumulative_spatial_footprint"] = cumulative_values

    return daily


def calculate_expansion_flag(
    nodes_df: pd.DataFrame,
) -> bool:
    """
    Return True when the event's spatial footprint increased after
    its emergence date.

    Operational interpretation:
        compare the number of unique sectors on emergence_date
        with the footprint observed on later event dates.

    This follows the requested "footprint increased after emergence"
    definition without inventing a continuous perimeter.
    """
    footprint = daily_footprint(nodes_df)

    if len(footprint) <= 1:
        return False

    emergence_date = footprint.iloc[0]["date"]

    emergence_footprint = int(
        footprint.iloc[0]["daily_spatial_footprint"]
    )

    later = footprint.loc[
        footprint["date"] > emergence_date
    ]

    if later.empty:
        return False

    return bool(
        later["daily_spatial_footprint"]
        .max()
        > emergence_footprint
    )


def event_nodes_dataframe(
    graph: nx.Graph,
    component: set,
) -> pd.DataFrame:
    """
    Convert one connected component into a deterministic node table.
    """
    records = []

    for node_id in sorted(
        component,
        key=lambda value: (
            value[0],
            value[1],
        ),
    ):
        attributes = graph.nodes[node_id]

        records.append(
            {
                "date": pd.Timestamp(
                    attributes["date"]
                ),
                "sector": attributes["sector"],
                "C": attributes["C"],
                "F": attributes["F"],
                "G": attributes["G"],
                "S": attributes["S"],
            }
        )

    return (
        pd.DataFrame(records)
        .sort_values(
            ["date", "sector"]
        )
        .reset_index(drop=True)
    )


# ============================================================
# EVENT EXTRACTION
# ============================================================


def reconstruct_events(
    graph: nx.Graph,
) -> pd.DataFrame:
    """
    Convert connected components into reconstructed Fire Events.

    Event IDs are assigned after sorting all components by:

        emergence date
        decay date
        first sector

    This makes event identifiers fully deterministic.
    """
    components = list(
        nx.connected_components(graph)
    )

    event_payloads: list[dict] = []

    for component in components:
        nodes_df = event_nodes_dataframe(
            graph,
            component,
        )

        if nodes_df.empty:
            continue

        emergence_date = pd.Timestamp(
            nodes_df["date"].min()
        )

        decay_date = pd.Timestamp(
            nodes_df["date"].max()
        )

        duration_days = (
            decay_date
            - emergence_date
        ).days + 1

        spatial_footprint = int(
            nodes_df["sector"].nunique()
        )

        peak_date = peak_date_by_normalized_c_and_f(
            nodes_df
        )

        peak_C = float(
            pd.to_numeric(
                nodes_df["C"],
                errors="coerce",
            ).max()
        )

        peak_F_values = pd.to_numeric(
            nodes_df["F"],
            errors="coerce",
        )

        peak_F = (
            float(peak_F_values.max())
            if peak_F_values.notna().any()
            else np.nan
        )

        mean_G_values = pd.to_numeric(
            nodes_df["G"],
            errors="coerce",
        )

        mean_G = (
            float(mean_G_values.mean())
            if mean_G_values.notna().any()
            else np.nan
        )

        expansion_flag = calculate_expansion_flag(
            nodes_df
        )

        # Additional deterministic event descriptors help downstream
        # M9 without replacing any required field.
        mean_S_values = pd.to_numeric(
            nodes_df["S"],
            errors="coerce",
        )

        mean_S = (
            float(mean_S_values.mean())
            if mean_S_values.notna().any()
            else np.nan
        )

        max_daily_footprint = int(
            daily_footprint(nodes_df)[
                "daily_spatial_footprint"
            ].max()
        )

        sectors_involved = ";".join(
            sorted(
                nodes_df["sector"].unique()
            )
        )

        event_payloads.append(
            {
                "_nodes_df": nodes_df,
                "_sort_key": (
                    emergence_date,
                    decay_date,
                    sectors_involved,
                ),
                "emergence_date": emergence_date,
                "decay_date": decay_date,
                "duration_days": int(duration_days),
                "spatial_footprint": spatial_footprint,
                "peak_date": peak_date,
                "peak_C": peak_C,
                "peak_F": peak_F,
                "mean_G": mean_G,
                "expansion_flag": bool(expansion_flag),
                "mean_S": mean_S,
                "max_daily_spatial_footprint": max_daily_footprint,
                "node_count": int(len(nodes_df)),
                "sectors_involved": sectors_involved,
            }
        )

    # --------------------------------------------------------
    # Deterministic component ordering and event IDs.
    # --------------------------------------------------------
    event_payloads.sort(
        key=lambda payload: payload["_sort_key"]
    )

    rows: list[dict] = []

    for index, payload in enumerate(
        event_payloads,
        start=1,
    ):
        emergence_date = payload["emergence_date"]

        event_id = (
            f"EVT-{emergence_date.year}-"
            f"{index:03d}"
        )

        rows.append(
            {
                "event_id": event_id,
                "emergence_date": payload["emergence_date"],
                "decay_date": payload["decay_date"],
                "duration_days": payload["duration_days"],
                "spatial_footprint": payload["spatial_footprint"],
                "peak_date": payload["peak_date"],
                "peak_C": payload["peak_C"],
                "peak_F": payload["peak_F"],
                "mean_G": payload["mean_G"],
                "expansion_flag": payload["expansion_flag"],
                "mean_S": payload["mean_S"],
                "max_daily_spatial_footprint": payload[
                    "max_daily_spatial_footprint"
                ],
                "node_count": payload["node_count"],
                "sectors_involved": payload["sectors_involved"],
                "_nodes_df": payload["_nodes_df"],
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# HISTORICAL ANALOGUE SEARCH
# ============================================================


def add_analogue_search(
    events: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add normalized event-feature values and top-three historical analogues.

    Event-level normalization:
        duration_days
        peak_C
        peak_F
        spatial_footprint

    Each feature is independently min-max normalized to [0, 1].
    Pairwise distance is Euclidean in this four-dimensional space.

    The event itself is excluded from its own nearest-neighbour list.

    Top-three analogue IDs are stored in a semicolon-delimited string.
    """
    result = events.copy()

    if result.empty:
        result["top_3_analog_event_ids"] = []
        return result

    # --------------------------------------------------------
    # Normalize each requested event-level feature across the full
    # reconstructed-event archive.
    # --------------------------------------------------------
    normalized_columns = []

    for feature in ANALOGUE_FEATURES:
        normalized_name = f"{feature}_normalized"

        result[normalized_name] = normalize_series_0_1(
            result[feature]
        )

        # Keep missing peak_F explicitly missing rather than inventing
        # a value. For distance calculations, unavailable dimensions
        # contribute zero distance only after pairwise masking.
        normalized_columns.append(
            normalized_name
        )

    # --------------------------------------------------------
    # Pairwise deterministic Euclidean distances.
    #
    # If one event is missing a feature, that dimension is omitted from
    # the pairwise distance and the distance is renormalized by the
    # number of available dimensions. This prevents one missing FRP
    # value from automatically making an event either identical or
    # infinitely distant.
    # --------------------------------------------------------
    feature_matrix = result[
        normalized_columns
    ].to_numpy(
        dtype=float
    )

    event_ids = result[
        "event_id"
    ].tolist()

    analogue_ids: list[str] = []
    analogue_distances: list[str] = []

    for i, source_id in enumerate(event_ids):
        candidate_rows: list[tuple[float, str]] = []

        for j, target_id in enumerate(event_ids):
            if i == j:
                continue

            left = feature_matrix[i]
            right = feature_matrix[j]

            valid = (
                np.isfinite(left)
                & np.isfinite(right)
            )

            valid_count = int(
                valid.sum()
            )

            if valid_count == 0:
                distance = np.nan
            else:
                squared = (
                    left[valid]
                    - right[valid]
                ) ** 2

                # Root-mean-square distance keeps the scale comparable
                # even if one pair has a missing feature.
                distance = float(
                    np.sqrt(
                        squared.mean()
                    )
                )

            if np.isfinite(distance):
                candidate_rows.append(
                    (
                        distance,
                        target_id,
                    )
                )

        candidate_rows.sort(
            key=lambda item: (
                item[0],
                item[1],
            )
        )

        top_candidates = candidate_rows[:3]

        analogue_ids.append(
            ";".join(
                target_id
                for _, target_id in top_candidates
            )
        )

        analogue_distances.append(
            ";".join(
                f"{distance:.6f}"
                for distance, _ in top_candidates
            )
        )

    result[
        "top_3_analog_event_ids"
    ] = analogue_ids

    result[
        "top_3_analog_distances"
    ] = analogue_distances

    return result


# ============================================================
# SERIALIZATION
# ============================================================


def prepare_output(
    events: pd.DataFrame,
) -> pd.DataFrame:
    """
    Remove internal Python-only node tables and order output columns.
    """
    result = events.copy()

    if "_nodes_df" in result.columns:
        result = result.drop(
            columns=["_nodes_df"]
        )

    ordered_columns = [
        "event_id",
        "emergence_date",
        "decay_date",
        "duration_days",
        "spatial_footprint",
        "peak_date",
        "peak_C",
        "peak_F",
        "mean_G",
        "expansion_flag",
        "mean_S",
        "max_daily_spatial_footprint",
        "node_count",
        "sectors_involved",
        "duration_days_normalized",
        "peak_C_normalized",
        "peak_F_normalized",
        "spatial_footprint_normalized",
        "top_3_analog_event_ids",
        "top_3_analog_distances",
    ]

    existing = [
        column
        for column in ordered_columns
        if column in result.columns
    ]

    result = result[
        existing
    ].copy()

    # --------------------------------------------------------
    # Dates are written in ISO YYYY-MM-DD form for stable CSV output.
    # --------------------------------------------------------
    for column in [
        "emergence_date",
        "decay_date",
        "peak_date",
    ]:
        if column in result.columns:
            result[column] = pd.to_datetime(
                result[column],
                errors="coerce",
            ).dt.strftime(
                "%Y-%m-%d"
            )

    # --------------------------------------------------------
    # Stable ordering by event ID.
    # --------------------------------------------------------
    result = (
        result.sort_values(
            ["event_id"]
        )
        .reset_index(drop=True)
    )

    return result


# ============================================================
# CONSOLE SUMMARY
# ============================================================


def print_summary(
    events: pd.DataFrame,
    graph: nx.Graph,
    qualifying: pd.DataFrame,
    output_file: Path,
) -> None:
    """
    Print the exact summary requested by the M8 specification.
    """
    print("\n" + "=" * 70)
    print("MILESTONE 8 COMPLETE")
    print("SPATIOTEMPORAL EVENT RECONSTRUCTION")
    print("=" * 70)

    total_events = int(
        len(events)
    )

    print(
        f"\nQualifying anomaly nodes (C >= {ANOMALY_THRESHOLD:.1f}): "
        f"{len(qualifying):,}"
    )

    print(
        f"Graph nodes: {graph.number_of_nodes():,}"
    )

    print(
        f"Graph edges: {graph.number_of_edges():,}"
    )

    print(
        f"Total events reconstructed: "
        f"{total_events:,}"
    )

    if total_events:
        average_duration = float(
            pd.to_numeric(
                events["duration_days"],
                errors="coerce",
            ).mean()
        )

        print(
            f"Average event duration: "
            f"{average_duration:.2f} days"
        )

        largest_event = (
            events.sort_values(
                [
                    "spatial_footprint",
                    "duration_days",
                    "event_id",
                ],
                ascending=[
                    False,
                    False,
                    True,
                ],
            )
            .iloc[0]
        )

        print(
            "Largest event by spatial footprint: "
            f"{largest_event['event_id']} "
            f"({int(largest_event['spatial_footprint'])} sectors, "
            f"{int(largest_event['duration_days'])} days)"
        )

        print(
            f"  Emergence: {largest_event['emergence_date']}"
        )
        print(
            f"  Decay:     {largest_event['decay_date']}"
        )
        print(
            f"  Peak:      {largest_event['peak_date']}"
        )

    else:
        print(
            "Average event duration: 0.00 days"
        )

        print(
            "Largest event by spatial footprint: "
            "None — no qualifying connected components."
        )

    print(
        f"\nSaved upgraded event table:\n  {output_file}"
    )


# ============================================================
# MAIN M8 PIPELINE
# ============================================================


def run_m8(
    input_file: Path = INPUT_FILE,
    output_file: Path = OUTPUT_FILE,
) -> pd.DataFrame:
    """
    Execute the complete M8 headless event-reconstruction pipeline.
    """
    print("=" * 70)
    print("FIRE HARMONIZER - MILESTONE 8")
    print("SPATIOTEMPORAL EVENT RECONSTRUCTION")
    print("=" * 70)

    print(
        f"\nInput:\n  {input_file}"
    )

    print(
        f"Output:\n  {output_file}"
    )

    # --------------------------------------------------------
    # STEP 1 — Load M7 science.
    # --------------------------------------------------------
    print(
        "\n[1/4] Loading M7 all-years science..."
    )

    df, resolved_columns = load_science(
        input_file
    )

    print(
        f"Rows loaded: {len(df):,}"
    )

    print(
        "M7 evidence columns:"
    )

    for canonical_name, source_name in resolved_columns.items():
        print(
            f"  {canonical_name} <- {source_name}"
        )

    # --------------------------------------------------------
    # STEP 2 — Build graph.
    # --------------------------------------------------------
    print(
        "\n[2/4] Building spatiotemporal event graph..."
    )

    graph, qualifying = build_event_graph(
        df
    )

    print(
        f"Qualifying nodes: {graph.number_of_nodes():,}"
    )

    print(
        f"Continuity edges: {graph.number_of_edges():,}"
    )

    # --------------------------------------------------------
    # STEP 3 — Connected components → Fire Events.
    # --------------------------------------------------------
    print(
        "\n[3/4] Reconstructing Fire Events..."
    )

    events = reconstruct_events(
        graph
    )

    print(
        f"Events reconstructed: {len(events):,}"
    )

    # --------------------------------------------------------
    # STEP 4 — Historical analogue search.
    # --------------------------------------------------------
    print(
        "\n[4/4] Computing deterministic historical analogues..."
    )

    events = add_analogue_search(
        events
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    final_output = prepare_output(
        events
    )

    final_output.to_csv(
        output_file,
        index=False,
    )

    print_summary(
        events=events,
        graph=graph,
        qualifying=qualifying,
        output_file=output_file,
    )

    return final_output


# ============================================================
# CLI
# ============================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fire Harmonizer M8 — "
            "Spatiotemporal Event Reconstruction"
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=INPUT_FILE,
        help=(
            "M7 all-years science CSV. "
            "Default: output/harmonized_science_all_years.csv"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_FILE,
        help=(
            "M8 reconstructed event CSV. "
            "Default: data/output/critical_periods.csv"
        ),
    )

    args = parser.parse_args()

    run_m8(
        input_file=args.input,
        output_file=args.output,
    )


if __name__ == "__main__":
    main()
