"""
Computes area stats across levels and inventories,
analyze errors when translating levels between ADD, RGI and IRR
"""
import sys
import argparse
from itertools import combinations
from pathlib import Path
import ast

import geopandas as gpd
import pandas as pd
import numpy as np
import shapely

AREA_CRS = "EPSG:6932"
LEVELS = [0, 1, 2, 3]
MIN_OVERLAP_KM2 = 0.01

FILES = {
    "ADD": "attachment_level_classification_v1.gpkg",
    "RGIv7": "RGI2000-v7.0-G-19_subantarctic_antarctic_islands_with_levels.gpkg",
    "IRRv1": "IRRv1_with_levels.gpkg",
}

ID_FIELDS = {
    "ADD": "analysis_id",
    "RGIv7": "rgi_id",
    "IRRv1": "id_icerise",
}

OVERLAP_CLASSES = [
    "RGI only",
    "IRR only",
    "RGI + IRR",
    "Neither",
]


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Calculate attachment-level areas for ADD, RGIv7 and IRRv1, "
            "identify cross-level propagation conflicts, remove affected IDs, "
            "and calculate the cleaned spatial union."
        )
    )

    parser.add_argument(
        "--data_path",
        type=Path,
        required=True,
        help="Directory containing the input GeoPackages.",
    )

    return parser.parse_args()


def validate_data_path(data_path):
    """Check that the data directory and required files exist."""

    if not data_path.exists():
        raise FileNotFoundError(
            f"Data directory does not exist: {data_path}"
        )

    for filename in FILES.values():
        path = data_path / filename

        if not path.exists():
            raise FileNotFoundError(
                f"Required input file not found: {path}"
            )


def load_datasets(data_path):
    """Read the three inventory GeoPackages."""

    return {
        name: gpd.read_file(data_path / filename)
        for name, filename in FILES.items()
    }


def inspect_dataset(name, gdf):
    """Print basic information about one inventory."""

    print(f"\n{name}")
    print("-" * len(name))
    print("CRS:", gdf.crs)
    print("Features:", len(gdf))
    print("Levels:", sorted(gdf["level"].dropna().unique()))
    print("Missing levels:", gdf["level"].isna().sum())


def validate_dataset(name, gdf, id_field):
    """Validate fields and attachment levels."""

    required = {"level", "geometry", id_field}
    missing = required.difference(gdf.columns)

    if missing:
        raise ValueError(
            f"{name} is missing required fields: {sorted(missing)}"
        )

    if gdf.crs is None:
        raise ValueError(f"{name} has no CRS.")

    if gdf["level"].isna().any():
        raise ValueError(
            f"{name} contains missing attachment levels."
        )

    levels = set(
        pd.to_numeric(gdf["level"], errors="raise")
        .astype(int)
        .unique()
    )

    unexpected = levels.difference(LEVELS)

    if unexpected:
        raise ValueError(
            f"{name} contains unexpected levels: {sorted(unexpected)}"
        )


def prepare_equal_area(gdf, extra_columns=None):
    """
    Reproject to EPSG:6932 and calculate polygon area.
    The original GeoDataFrame is not modified.
    """

    extra_columns = extra_columns or []

    columns = list(
        dict.fromkeys(
            [*extra_columns, "level", "geometry"]
        )
    )

    out = (
        gdf[columns]
        .to_crs(AREA_CRS)
        .copy()
    )

    out["level"] = (
        pd.to_numeric(out["level"], errors="raise")
        .astype(int)
    )

    out["area_km2"] = (
        out.geometry.area / 1e6
    )

    return out


def area_stats_by_level(gdf):
    """Sum polygon area independently for Levels 0–3."""

    return (
        gdf
        .groupby("level")["area_km2"]
        .sum()
        .reindex(LEVELS, fill_value=0.0)
    )


def build_area_table(stats, union_stats=None):
    """
    Construct the Table A1 structure.

    stats:
        {
            "ADD": Series,
            "RGIv7": Series,
            "IRRv1": Series
        }
    """

    columns = {}

    for name, values in stats.items():

        columns[f"{name} area (km2)"] = values

        columns[f"{name} (%)"] = (
            100 * values / values.sum()
        )

    if union_stats is not None:

        columns["Union area (km2)"] = union_stats

        columns["Union (%)"] = (
            100 * union_stats / union_stats.sum()
        )

    table = pd.DataFrame(columns)

    table.index = [
        f"Level {level}"
        for level in table.index
    ]

    # Explicit totals rather than summing percentages
    totals = []

    for values in stats.values():
        totals.extend([
            values.sum(),
            100.0,
        ])

    if union_stats is not None:
        totals.extend([
            union_stats.sum(),
            100.0,
        ])

    table.loc["Total"] = totals

    return table


def cross_level_overlap_with_ids(
    gdf_a,
    gdf_b,
    id_a,
    id_b,
    name_a,
    name_b,
    min_area_km2=MIN_OVERLAP_KM2,
):
    """
    Find spatial intersections where two inventories carry
    different propagated attachment levels.
    """

    a = gdf_a[
        [id_a, "level", "geometry"]
    ].copy()

    b = gdf_b[
        [id_b, "level", "geometry"]
    ].copy()

    a = a.rename(
        columns={
            "level": f"{name_a}_level"
        }
    )

    b = b.rename(
        columns={
            "level": f"{name_b}_level"
        }
    )

    overlaps = gpd.overlay(
        a,
        b,
        how="intersection",
        keep_geom_type=False,
    )

    overlaps = overlaps[
        overlaps[f"{name_a}_level"]
        != overlaps[f"{name_b}_level"]
    ].copy()

    overlaps["overlap_km2"] = (
        overlaps.geometry.area / 1e6
    )

    overlaps = overlaps[
        overlaps["overlap_km2"]
        >= min_area_km2
    ].copy()

    return overlaps.sort_values(
        "overlap_km2",
        ascending=False,
    )


def find_all_conflicts(area_datasets):
    """Run the three pairwise cross-level comparisons."""

    conflicts = {}

    conflicts["ADD_RGI"] = cross_level_overlap_with_ids(
        area_datasets["ADD"],
        area_datasets["RGIv7"],
        id_a="analysis_id",
        id_b="rgi_id",
        name_a="ADD",
        name_b="RGI",
    )

    conflicts["ADD_IRR"] = cross_level_overlap_with_ids(
        area_datasets["ADD"],
        area_datasets["IRRv1"],
        id_a="analysis_id",
        id_b="id_icerise",
        name_a="ADD",
        name_b="IRR",
    )

    conflicts["RGI_IRR"] = cross_level_overlap_with_ids(
        area_datasets["RGIv7"],
        area_datasets["IRRv1"],
        id_a="rgi_id",
        id_b="id_icerise",
        name_a="RGI",
        name_b="IRR",
    )

    return conflicts


def get_problem_ids(conflicts):
    """
    Return unique IDs involved in any cross-level conflict.
    """

    return {
        "ADD": (
            set(conflicts["ADD_RGI"]["analysis_id"])
            | set(conflicts["ADD_IRR"]["analysis_id"])
        ),

        "RGIv7": (
            set(conflicts["ADD_RGI"]["rgi_id"])
            | set(conflicts["RGI_IRR"]["rgi_id"])
        ),

        "IRRv1": (
            set(conflicts["ADD_IRR"]["id_icerise"])
            | set(conflicts["RGI_IRR"]["id_icerise"])
        ),
    }


def remove_problem_ids(datasets, problem_ids):
    """Remove complete source features involved in conflicts."""

    cleaned = {}

    for name, gdf in datasets.items():

        id_field = ID_FIELDS[name]

        cleaned[name] = gdf[
            ~gdf[id_field].isin(problem_ids[name])
        ].copy()

    return cleaned


def union_geometry_by_level(gdf):
    """Dissolve all polygons belonging to each attachment level."""

    geoms = shapely.force_2d(
        shapely.make_valid(
            gdf.geometry.values
        )
    )

    levels = gdf["level"].to_numpy()

    return {
        level: shapely.union_all(
            geoms[levels == level]
        )
        for level in LEVELS
    }


def combine_inventory_levels(area_datasets):
    """
    Union ADD + RGIv7 + IRRv1 independently for each level.
    """

    inventory_level_geoms = {
        name: union_geometry_by_level(gdf)
        for name, gdf in area_datasets.items()
    }

    combined = {}

    for level in LEVELS:

        combined[level] = shapely.union_all([
            inventory_level_geoms[name][level]
            for name in inventory_level_geoms
        ])

    return combined


def check_cross_level_overlap(combined_levels):
    """Check residual overlap between final level geometries."""

    records = []

    for level_a, level_b in combinations(
        LEVELS, 2
    ):

        overlap = (
            combined_levels[level_a]
            .intersection(
                combined_levels[level_b]
            )
        )

        records.append({
            "level_a": level_a,
            "level_b": level_b,
            "overlap_km2": (
                overlap.area / 1e6
            ),
        })

    return pd.DataFrame(records)


def calculate_union_stats(combined_levels):
    """
    Calculate the final spatial union.

    Higher attachment levels retain priority as a safety rule
    if any residual cross-level overlap remains.
    """

    union_area = {}
    higher = shapely.GeometryCollection()

    for level in sorted(
        LEVELS,
        reverse=True,
    ):

        level_geom = combined_levels[level]

        unique_geom = (
            level_geom
            .difference(higher)
        )

        union_area[level] = (
            unique_geom.area / 1e6
        )

        higher = shapely.union_all([
            higher,
            level_geom,
        ])

    return (
        pd.Series(union_area)
        .sort_index()
        .reindex(LEVELS)
    )


def parse_id_list(x):
    """Convert ID fields into a consistent Python list."""

    if x is None:
        return []

    if isinstance(x, list):
        return x

    if isinstance(x, str):
        x = x.strip()

        if x in ("", "None", "nan", "[]"):
            return []

        try:
            parsed = ast.literal_eval(x)
        except (ValueError, SyntaxError):
            return [x]

        return parsed if isinstance(parsed, list) else [parsed]

    if pd.isna(x):
        return []

    return [x]


def prepare_add_overlap(add_gdf):

    add_overlap = (
        add_gdf[
            [
                "analysis_id",
                "level",
                "rgi_ids",
                "id_icerise",
                "ice_type",
                "geometry",
            ]
        ]
        .to_crs(AREA_CRS)
        .copy()
    )

    add_overlap["level"] = (
        pd.to_numeric(
            add_overlap["level"],
            errors="raise"
        ).astype(int)
    )

    add_overlap["rgi_ids"] = (
        add_overlap["rgi_ids"]
        .apply(parse_id_list)
    )

    add_overlap["id_icerise"] = (
        add_overlap["id_icerise"]
        .apply(parse_id_list)
    )

    add_overlap["has_rgi"] = (
        add_overlap["rgi_ids"]
        .map(bool)
    )

    add_overlap["has_irr"] = (
        add_overlap["id_icerise"]
        .map(bool)
    )

    conditions = [
        add_overlap["has_rgi"] & add_overlap["has_irr"],
        add_overlap["has_rgi"] & ~add_overlap["has_irr"],
        ~add_overlap["has_rgi"] & add_overlap["has_irr"],
    ]

    choices = [
        "RGI + IRR",
        "RGI only",
        "IRR only",
    ]

    add_overlap["inventory_overlap"] = np.select(
        conditions,
        choices,
        default="Neither",
    )

    add_overlap["inventory_overlap"] = pd.Categorical(
        add_overlap["inventory_overlap"],
        categories=OVERLAP_CLASSES,
        ordered=True,
    )

    # Equal-area calculation
    add_overlap["area_km2"] = (
        add_overlap.geometry.area / 1e6
    )

    return add_overlap


def main():

    args = parse_arguments()
    data_path = args.data_path

    validate_data_path(data_path)

    print(f"Data path: {data_path}")

    # --------------------------------------------------------
    # 1. Load and validate inputs
    # --------------------------------------------------------

    datasets = load_datasets(data_path)

    for name, gdf in datasets.items():
        validate_dataset(
            name,
            gdf,
            ID_FIELDS[name],
        )

        inspect_dataset(
            name,
            gdf,
        )

    # --------------------------------------------------------
    # 2. Original area statistics
    # --------------------------------------------------------

    original_area = {
        name: prepare_equal_area(gdf)
        for name, gdf in datasets.items()
    }

    original_stats = {
        name: area_stats_by_level(gdf)
        for name, gdf in original_area.items()
    }

    original_table = build_area_table(
        original_stats
    )

    print("\nOriginal area statistics")
    print(original_table.round(2))

    # --------------------------------------------------------
    # 3. Prepare datasets with IDs for conflict detection
    # --------------------------------------------------------

    area_with_ids = {
        name: prepare_equal_area(
            gdf,
            extra_columns=[ID_FIELDS[name]],
        )
        for name, gdf in datasets.items()
    }

    # --------------------------------------------------------
    # 4. Detect cross-level conflicts
    # --------------------------------------------------------

    conflicts = find_all_conflicts(
        area_with_ids
    )

    print("\nCross-level conflict records")

    for name, conflict_df in conflicts.items():
        print(
            f"{name}: "
            f"{len(conflict_df)} intersections"
        )

    # --------------------------------------------------------
    # 5. Identify problematic source IDs
    # --------------------------------------------------------

    problem_ids = get_problem_ids(
        conflicts
    )

    print("\nProblematic source IDs")

    for name, ids in problem_ids.items():
        print(
            f"{name}: "
            f"{len(ids)} IDs"
        )

        print(sorted(ids))

    # --------------------------------------------------------
    # 6. Remove problematic IDs
    # --------------------------------------------------------

    cleaned_datasets = remove_problem_ids(
        datasets,
        problem_ids,
    )

    print("\nFeatures removed")

    for name in datasets:
        removed = (
                len(datasets[name])
                - len(cleaned_datasets[name])
        )

        print(
            f"{name}: {removed}"
        )

    # --------------------------------------------------------
    # 7. Reproject cleaned datasets and calculate areas
    # --------------------------------------------------------

    cleaned_area = {
        name: prepare_equal_area(gdf)
        for name, gdf in cleaned_datasets.items()
    }

    cleaned_stats = {
        name: area_stats_by_level(gdf)
        for name, gdf in cleaned_area.items()
    }

    # --------------------------------------------------------
    # 8. Show area removed by exclusions
    # --------------------------------------------------------

    removed_area = pd.DataFrame({
        f"{name} removed (km2)":
            original_stats[name]
            - cleaned_stats[name]

        for name in datasets
    })

    print("\nArea removed by level")
    print(removed_area.round(2))

    # --------------------------------------------------------
    # 9. Construct cleaned union
    # --------------------------------------------------------

    combined_levels = combine_inventory_levels(
        cleaned_area
    )

    residual_overlap = check_cross_level_overlap(
        combined_levels
    )

    print("\nResidual cross-level overlap")
    print(residual_overlap.round(6))

    union_stats = calculate_union_stats(
        combined_levels
    )

    # --------------------------------------------------------
    # 10. Final Table A1
    # --------------------------------------------------------

    table_A1 = build_area_table(
        cleaned_stats,
        union_stats=union_stats,
    )

    print("\nFinal Table A1")
    print(table_A1.round(2))

    # --------------------------------------------------------
    # 11. Save outputs
    # --------------------------------------------------------

    original_table.round(2).to_csv(
        data_path / "table_A1_original.csv"
    )

    table_A1.round(2).to_csv(
        data_path / "table_A1_cleaned.csv"
    )

    removed_area.round(2).to_csv(
        data_path / "table_A1_removed_area.csv"
    )

    residual_overlap.round(6).to_csv(
        data_path / "table_A1_residual_cross_level_overlap.csv",
        index=False,
    )

    print(
        "\nSaved:"
        "\n  table_A1_original.csv"
        "\n  table_A1_cleaned.csv"
        "\n  table_A1_removed_area.csv"
        "\n  table_A1_residual_cross_level_overlap.csv"
    )


    ## TABLE A2
    # We keep all features here as the question is different:
    # For each ADD attachment level, how much ADD area is associated with an RGI ID, an IRR ID, both, or neither?
    datasets["ADD"]
    cleaned_datasets["ADD"]

    add_overlap = prepare_add_overlap(
        datasets["ADD"]
    )

    comparison_area = (
        add_overlap
        .groupby(
            ["level", "inventory_overlap"],
            observed=False,
        )["area_km2"]
        .sum()
        .unstack(fill_value=0)
        .reindex(
            index=LEVELS,
            columns=OVERLAP_CLASSES,
            fill_value=0,
        )
    )

    comparison_pct = (
            comparison_area
            .div(
                comparison_area.sum(axis=1),
                axis=0,
            )
            * 100
    )

    inventory_overlap_table = pd.DataFrame(
        index=LEVELS
    )

    for category in OVERLAP_CLASSES:
        inventory_overlap_table[
            f"{category} area (km2)"
        ] = comparison_area[category]

        inventory_overlap_table[
            f"{category} (%)"
        ] = comparison_pct[category]

    inventory_overlap_table.index = [
        f"Level {level}"
        for level in inventory_overlap_table.index
    ]

    print(
        inventory_overlap_table.round(2)
    )

    inventory_overlap_table.round(2).to_csv(
        data_path / "table_A2.csv"
    )


if __name__ == "__main__":
    main()

