"""
This script preforms a spatial linkage assessment between
three masks:

1) Primary Ice-Type Mask (interaction_mask.gpkg)
This mask represents contiguous ice that may interact with or influence glacier complexes on
Antarctic islands, ice rises and rumples.
It includes:
- Ice sheet extent (from ADD coastline)
- Mainland ice-shelf areas (from ADD coastline)
- Ice tongue areas (from ADD coastline)
We only include ice-shelf areas that originate from the mainland ice sheet in this primary mask.

2) Secondary ice-shelves mask (remaining_shelves_mask.gpkg)
Mask of ice shelves that originate from ice on Antarctic islands
(i.e., shelves not connected to the mainland ice sheet, this is use for post-processing).

3) Assessment mask (ADD_polys_with_RGI-GCv7_IRRv1_combined.gpkg)
Polygons to be assessed (glacier complexes, ice rises, etc.).
This assessment computes the percentage of each **polygon perimeter from the assessment mask**
that is shared with the primary interaction mask.
- A detachment scoring is assigned from that perimeter-overlap percentage.
- An additional buttressing context is derived from the secondary (island-origin) shelf mask.

| Perimeter Overlap      | Score | Linkage Type            | Buttress Code | Buttress Source ID                        |
|------------------------|-------|--------------------------|----------------|-----------------------------------------|
| 90–100% (or >100%)     | 1.1   | Strong linkage           | None           | Buttressed by ice sheet/shelves (mainland) |
| 80–90%                 | 1.2   |                          | None           | Buttressed by ice sheet/shelves (mainland) |
| ...                    | ...   | ...                      | None           | Buttressed by ice sheet/shelves (mainland) |
| 10–20%                 | 1.9   | Weak linkage             | None           | Buttressed by ice sheet/shelves (mainland) |
| 0%                     | 2.0   | Completely detached       | 1.0           | Buttressed by non-mainland ice shelves  |
| 0% + ≤10% shelf overlap| 2.0   | Completely detached       | 0.0           | Unclassified (not buttressed)           |

Notes / conventions
- Perimeter Overlap is computed as percentage of the assessment polygon’s perimeter
that touches the primary interaction mask.
- Score uses the decimal detachment scheme: 1.1 … 1.9 (progressively weaker linkage), 2.0 = fully detached.
- Buttress Code: 1.0 = buttressed by a non-mainland (island-origin) shelf; 0.0 = not buttressed / unclassified.

Script done by B. Recinos (NERC IRF, U. Edinburgh)
Portions of this code were generated or optimized using
Microsoft Copilot
"""
import sys
import argparse
from pathlib import Path
import geopandas as gpd
import pandas as pd
import numpy as np

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Parse geopackage and data paths for geospatial processing."
    )
    parser.add_argument(
        "--data_path",
        type=Path,
        required=True,
        help="Path to the output or data directory"
    )
    return parser.parse_args()


def validate_paths(args):
    if not args.data_path.exists():
        print(f"Data path does not exist. Creating: {args.data_path}")
        args.data_path.mkdir(parents=True, exist_ok=True)


def bucket_overlap(ratio=float):
    """
    Converts perimeter overlap ratios into detachment scores,
    where lower overlap means higher detachment (higher score).
    :param ratio, np.float()
    :return: detachment score as np.float()
    """
    # Treat <=0 as fully detached
    if ratio is None or ratio <= 0:
        return 2.0

    # keeps a maximum ratio of 100.0% overlap to
    # avoid cases with > 100 % due to buffer inaccuracies
    capped = min(ratio, 100.0)

    # Calculate to which bucket ratio the % belongs to
    # // 10 means integer division into bins of 10 (e.g., 85 → bin 8)
    # - 1e-9 prevents rounding errors like 10.0 being mistakenly placed in the next bin
    # + 1 shifts the range from 0-based to 1-based
    bucket = int((capped - 1e-9) // 10) + 1 # e.g., 0–10% → bucket 1
    bucket = min(bucket, 10) #Handles ratio=100
    bucket = max(bucket, 2) #Handles ratio in (0, 10]:

    reversed_bucket = 11 - bucket  # 10->1 (1.1), 2->9 (1.9)
    return round(1 + reversed_bucket / 10, 2)


def classify_polygon(row):
    """
    This function takes each row from a geopandas.Dataframe
    (combined_final) and extracts the ratio value in order to
    classify it.

    :param row
    :return: tuple with score, label, None, None
    The last two None's are placeholders for buttress properties
    to be added on the second classification.
    """
    ratio = row["ratio"]

    # Check if the ratio is missing, that means that
    # this is already classified as completely detached
    # (no intersection → NaN ratio)
    if pd.isna(ratio):
        return (
            2.0,
            "Completely detached (0% shared perimeter)",
            None,
            row.get("source_id", None),
        )

    # ratio == 0 -> fully detached (per your table)
    if ratio <= 0:
        return (
            2.0,
            "Completely detached (0% shared perimeter)",
            None,
            row.get("source_id", None),
        )


    # Calls the helper function to compute the detachment score
    # based on perimeter overlap
    score = bucket_overlap(ratio)

    # Determines to which 10% bucket the scores belongs to
    # // 10 means integer division into bins of 10 (e.g., 85 → bin 8)
    # - 1e-9 prevents rounding errors like 10.0 being mistakenly placed in the next bin
    # + 1 shifts the range from 0-based to 1-based
    bucket = int((min(ratio, 100.0) - 1e-9) // 10) + 1
    bucket = min(bucket, 10)

    # We probably need some message to translate the code above
    label = f"Perimeter overlap {int((bucket-1)*10)}–{int(bucket*10)}%"

    return score, label, None, None


def assign_buttress_code(row, ratio_map):
    """
    Assigns a buttress_code to polygons with detachment_score == 2.0
    based on overlap ratio with a secondary ice shelf mask.

    :param row: A row from the GeoDataFrame (combined_final)
    :param ratio_map: Dict mapping analysis_id to shared perimeter ratio
    :return: 1.0 if overlap > 10%, else 0.0; None if not detached
    """
    if row["detachment_score"] != 2.0:
        return None
    ratio = ratio_map.get(row["analysis_id"], 0.0)
    return 1.0 if ratio > 10 else 0.0


def main():
    args = parse_arguments()
    validate_paths(args)

    print("Paths parsed successfully:")
    print(f"Data path: {args.data_path}")

    # Proceed with reading files
    interaction_mask_fp = Path(args.data_path) / "interaction_mask.gpkg"
    assert interaction_mask_fp.exists(), "You need to run first the preprocessing script"
    interaction_mask = gpd.read_file(interaction_mask_fp)

    combined_fp = Path(args.data_path) / "ADD_polys_with_RGI-GCv7_IRRv1_combined.gpkg"
    assert combined_fp.exists(), "You need to run first the preprocessing script"
    combined = gpd.read_file(combined_fp)

    # Assert all projections match
    assert interaction_mask.crs == combined.crs, \
        "CRS mismatch after reprojecting to EPSG:3031"

    print("Shapefiles loaded and all reprojected to EPSG:3031 successfully.")

    # We make a big buffer in the interaction mask
    interaction_mask['geometry'] = interaction_mask.geometry.buffer(50)

    combined['total_area'] = combined.geometry.area
    combined['perimeter'] = combined.geometry.length

    # Spatial intersection with inter_mask
    intersection_result = gpd.overlay(combined, interaction_mask, how='intersection')

    # union of the interaction mask geometry
    mask_union = interaction_mask.union_all()
    # find which combined geometries intersect with the mask
    intersects_mask = combined.geometry.intersects(mask_union)
    # rows that did NOT intersect need to be dropped in intersection_result
    no_intersection_combined = combined[~intersects_mask]
    print('Already these many polygons have no interaction with the main Ice sheet')
    print(no_intersection_combined.head())
    print(no_intersection_combined.shape)

    # We save the shared perimeter for each geometry
    intersection_result['perimeter_shared'] = intersection_result.geometry.length

    # And the ratio
    intersection_result['ratio'] = (intersection_result['perimeter_shared'] / intersection_result['perimeter']) * 100

    # Aggregate intersection results from the overlap per polygon
    intersection_summary = (
        intersection_result
        .groupby("analysis_id", as_index=False)
        .agg({
            "perimeter_shared": "sum",
            "ratio": "max"
        })
    )

    # Build the combined final geopandas dataframe
    combined_final = combined.merge(
        intersection_summary,
        on="analysis_id",
        how="left"
    )

    # We add classification fields that we will fill later
    combined_final["detachment_score"] = np.nan
    combined_final["detachment_desc"] = "unclassified"
    combined_final["buttress_code"] = None
    combined_final["buttress_source_id"] = None

    # Assign 2.0 to polygons with no interaction
    combined_final.loc[
        combined_final["analysis_id"].isin(no_intersection_combined["analysis_id"]),
        ["detachment_score", "detachment_desc"]
        ] = [2.0, "Completely detached (no shared perimeter)"]

    combined_final[[
        "detachment_score",
        "detachment_desc",
        "buttress_code",
        "buttress_source_id"
    ]] = combined_final.apply(classify_polygon, axis=1, result_type="expand")

    # No polygon should lose geometry
    assert combined_final.geometry.notna().all()

    # Detached polygons must have NaN ratio
    detached_ratio = combined_final.loc[combined_final.detachment_score == 2.0, "ratio"]
    assert (detached_ratio.isna() | (detached_ratio <= 0)).all()

    # Overlapping polygons must be 1.x
    assert combined_final.loc[combined_final.ratio.notna(),
                             "detachment_score"].between(1.1, 1.9).all()

    ##----------------- Next classification ------------------------------
    secondary_shelf_fp = Path(args.data_path) / "remaining_shelves_mask.gpkg"
    assert secondary_shelf_fp.exists(), "Secondary ice shelf mask not found"
    secondary_shelf_mask = gpd.read_file(secondary_shelf_fp)

    # Optional: buffer it
    secondary_shelf_mask['geometry'] = secondary_shelf_mask.geometry.buffer(50)

    # Ensure CRS match
    assert secondary_shelf_mask.crs == combined.crs

    # Filter only fully detached polygons
    fully_detached = combined_final[combined_final["detachment_score"] == 2.0].copy()

    # Now let's do again another subclassification if they are buttressed
    # by a different ice shelf that does not come from the main land
    # Spatial intersection
    intersection_with_shelves = gpd.overlay(
        fully_detached,
        secondary_shelf_mask,
        how="intersection"
    )

    intersection_with_shelves["perimeter_shared"] = intersection_with_shelves.geometry.length
    intersection_with_shelves["ratio"] = (intersection_with_shelves["perimeter_shared"] /
                                          intersection_with_shelves["perimeter"]
                                          ) * 100

    # Keep max ratio per analysis_id
    ratio_map = intersection_with_shelves.groupby("analysis_id")["ratio"].max().to_dict()

    combined_final["buttress_code"] = combined_final.apply(
        assign_buttress_code,
        axis=1,
        args=(ratio_map,)
    )

    combined_final.loc[
        (combined_final["detachment_score"] == 2.0) & (combined_final["buttress_code"] == 1.0),
        "buttress_source_id"
    ] = "buttressed by non-mainland ice shelves"

    combined_final["buttress_source_id"] = combined_final["buttress_source_id"].fillna("unclassified")

    print(combined_final["detachment_score"].value_counts(dropna=False).sort_index())

    filename = Path(args.data_path) / f"final_classification_buckets.gpkg"
    combined_final.to_file(filename, driver="GPKG")

if __name__ == "__main__":
    main()