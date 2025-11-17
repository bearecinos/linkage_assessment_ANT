"""
This script preforms a spatial linkage assessment between
two masks:

1) A primary Ice-Type Mask (Interaction Mask)
This mask represents the contiguous ice areas that may interact
with or influence smaller glacial and ice caps systems (or ice rises).
It includes:
- Ice sheet extent (from ADD coastline)
- Ice shelf areas (from ADD coastline)
- Ice tongue areas (from ADD coastline)
- Ice rumple areas (also mapped by ADD coastline)

2) Assessment mask: ADD_polys_with_RGI-GCv7_IRRv1_combined.gpkg

This assessment is done based on how much a polygon in the
 Assessment mask shares its perimeter with the primary mask.

Classification can be customised

Script done by B. Recinos (NERC IRF U. Edinburgh)

"""
import sys
import argparse
from pathlib import Path
import geopandas as gpd
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
    parser.add_argument(
        "--weak_max",
        type=float,
        default=10.0,
        help="Upper bound for weak linkage (e.g. 10.0)"
    )
    parser.add_argument(
        "--medium_max",
        type=float,
        default=50.0,
        help="Upper bound for medium linkage (e.g. 50.0)"
    )
    return parser.parse_args()


def validate_paths(args):
    if not args.data_path.exists():
        print(f"Data path does not exist. Creating: {args.data_path}")
        args.data_path.mkdir(parents=True, exist_ok=True)


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
    interaction_mask['geometry'] = interaction_mask.geometry.buffer(100)

    combined['total_area'] = combined.geometry.area
    combined['perimeter'] = combined.geometry.length

    # Spatial intersection with inter_mask
    intersection_result = gpd.overlay(combined, interaction_mask, how='intersection')

    # union of the interaction mask geometry
    mask_union = interaction_mask.unary_union
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

    # We separate the dataset according to the ratio above
    # Weak: 0 < ratio <= 10
    weak_linkage = intersection_result[
        (intersection_result["ratio"] > 0) & (intersection_result["ratio"] <= args.weak_max)
        ]

    # Medium: 10 < ratio <= 50
    medium_linkage = intersection_result[
        (intersection_result["ratio"] > args.weak_max) & (intersection_result["ratio"] <= args.medium_max)
        ]

    # Strong: ratio > 50
    strong_linkage = intersection_result[
        intersection_result["ratio"] > args.medium_max
        ]

    # Just in case lets make a copy or the combined "original" analysis mask
    combined_final = combined.copy()

    # Initialize classification column with default
    combined_final["detachment_score"] = np.nan
    combined_final["detachment_desc"] = "unclassified"

    # Update classification using analysis_id from each subset
    combined_final.loc[
        combined_final["analysis_id"].isin(no_intersection_combined["analysis_id"]),
        ["detachment_score", "detachment_desc"]] = [2.0, "Completely detached (no shared perimeter)"]

    combined_final.loc[combined_final["analysis_id"].isin(weak_linkage["analysis_id"]),
        ["detachment_score", "detachment_desc"]] = [1.7, "Weak linkage (low shared perimeter ratio)"]

    combined_final.loc[combined_final["analysis_id"].isin(medium_linkage["analysis_id"]),
        ["detachment_score", "detachment_desc"]] = [1.5, "Partial linkage (moderate shared perimeter ratio)"]

    combined_final.loc[combined_final["analysis_id"].isin(strong_linkage["analysis_id"]),
        ["detachment_score", "detachment_desc"]] = [1.0, "Strong linkage (high shared perimeter ratio)"]

    file_suffix = f"weak{int(args.weak_max)}_medium{int(args.medium_max)}"
    filename = Path(args.data_path) / f"final_classification_{file_suffix}.gpkg"
    combined_final.to_file(filename, driver="GPKG")

if __name__ == "__main__":
    main()