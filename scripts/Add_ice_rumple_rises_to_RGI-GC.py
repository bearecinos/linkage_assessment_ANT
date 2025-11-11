"""
Spatial Linkage Assessment of Glacier Complexes and Major Ice Types

This script performs a spatial linkage assessment using three geospatial datasets
to evaluate the degree of connection between glacier complexes and major ice types
(ice sheet, ice shelves, ice tongues, ice rises, and ice rumples).

The script generates two key spatial masks:

1) Primary Ice-Type Mask (Interaction Mask)
This mask represents the contiguous ice areas that may interact
with or influence smaller glacial systems.
It includes:
- Ice sheet extent (from a high-resolution polygon dataset by BAS)
- Ice shelf areas (from BAS)
- Ice tongue areas (from BAS)

This composite mask defines the spatial footprint of large ice masses
potentially connected to peripheral or isolated glacier systems.

2) Glacier Complex and Ice Rise/Rumple Mask (Assessment Layer)
This layer includes:
- Glacier complexes from the Randolph Glacier Inventory v7 (RGIv7)
- Ice rises and ice rumples from a dedicated V1 dataset

Since some ice rises and rumples are already included in RGIv7
and others are only in the dedicated dataset,
the script also resolves duplication and double counting to produce a unified,
enhanced assessment layer.
"""

import sys
import os
import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path
import argparse



def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Parse shapefile and data paths for geospatial processing."
    )
    parser.add_argument(
        "--coast-file",
        type=Path,
        required=True,
        help="Path to the coastline shapefile (.shp)"
    )
    parser.add_argument(
        "--glacier-complex",
        type=Path,
        required=True,
        help="Path to the glacier complex shapefile (.shp)"
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        required=True,
        help="Path to the output or data directory"
    )
    parser.add_argument(
        "--ice-rumples",
        type=Path,
        required=True,
        help="Path to the ice rumples shapefile (.shp)"
    )
    return parser.parse_args()


def validate_paths(args):
    if not args.coast_file.exists():
        sys.exit(f"Coast file not found: {args.coast_file}")
    if not args.glacier_complex.exists():
        sys.exit(f"Glacier complex file not found: {args.glacier_complex}")
    if not args.ice_rumples.exists():
        sys.exit(f"Ice rumples file not found: {args.ice_rumples}")
    if not args.data_path.exists():
        print(f"Data path does not exist. Creating: {args.data_path}")
        args.data_path.mkdir(parents=True, exist_ok=True)


def main():
    args = parse_arguments()
    validate_paths(args)

    print("Paths parsed successfully:")
    print(f"Coast file: {args.coast_file}")
    print(f"Glacier complex: {args.glacier_complex}")
    print(f"Data path: {args.data_path}")
    print(f"Ice rumples: {args.ice_rumples}")

    # Proceed with reading files
    coast = gpd.read_file(args.coast_file).to_crs("EPSG:3031")
    glacier_complex = gpd.read_file(args.glacier_complex).to_crs("EPSG:3031")
    rise_rumple = gpd.read_file(args.ice_rumples).to_crs("EPSG:3031")

    # Assert all projections match
    assert coast.crs == glacier_complex.crs == rise_rumple.crs, \
        "CRS mismatch after reprojecting to EPSG:3031"

    print("Shapefiles loaded and all reprojected to EPSG:3031 successfully.")

    # We only select what is land
    land_gdf = coast[coast['surface'] == 'land']
    assert np.all(land_gdf.surface.values == "land")

    # Let's select the largest land feature: Antarctic ice sheet
    areas = land_gdf.geometry.area
    # find index of largest
    largest_idx = areas.idxmax()

    print(largest_idx)

    # we remove it from the coastal polygons dataset
    no_icesheet = land_gdf.drop(index=largest_idx).copy()

    # TODO: this polygon could be replace with a Grounding line dataset
    # Now select only the ice sheet
    only_icesheet = land_gdf.loc[[largest_idx]].copy()

    # Define ice types of interest
    ice_types = ['ice shelf', 'ice tongue', 'rumple']

    # Create separate GeoDataFrames for each ice type
    ice_type_gdfs = {
        ice_type: coast[coast['surface'].str.lower() == ice_type].copy()
        for ice_type in ice_types
    }

    # Example: access each separately
    # TODO: ice shelf database could be replace with ice shelf extend / area
    ice_shelf_gdf = ice_type_gdfs['ice shelf']
    ice_tongue_gdf = ice_type_gdfs['ice tongue']
    # We leave ice rumples from the coastal polygon dataset out
    # of the analysis because we have the other database
    ice_rumple_gdf = ice_type_gdfs['rumple']


    # 1. Concatenate all GeoDataFrames
    combined_gdf = gpd.GeoDataFrame(
        pd.concat([only_icesheet, ice_shelf_gdf, ice_tongue_gdf], ignore_index=True),
        crs=only_icesheet.crs
    )

    # 2. Dissolve all into one single polygon (mask)
    combined_gdf["mask"] = 1  # Dummy column for dissolve
    interaction_mask = combined_gdf.dissolve(by="mask")

    inter_mask_fp = Path(args.data_path) / "inter_mask.gpkg"
    interaction_mask.to_file(inter_mask_fp, driver="GPKG")

    # 2. Tag origin
    glacier_complex["source"] = "RGIv7_C"
    rise_rumple["source"] = "icerises_inventory_v1"

    # Rename rise_rumple's area column to avoid conflict
    rise_rumple = rise_rumple.rename(columns={"Area_km2": "area_km2_icerises_inventory_v1"})
    glacier_complex = glacier_complex.rename(columns={'area_km2': "area_km2_RGIv7_C"})

    # Perform spatial join to find RGI-Glacier-complex polygons that intersect rise_rumple
    intersecting_rgi = gpd.sjoin(glacier_complex,
                                 rise_rumple[["geometry"]],
                                 how="inner", predicate="intersects")

    # Drop duplicates (if any) and get RGI rows to exclude
    intersecting_rgi_ids = intersecting_rgi.index.unique()

    # Drop intersecting from original RGI
    rgi_only = glacier_complex.drop(index=intersecting_rgi_ids)

    # Combine all columns across all three datasets
    all_cols = set(rgi_only.columns) | set(glacier_complex.columns) | set(rise_rumple.columns)

    # Ensure all dataframes have all columns
    for gdf in [rgi_only, rise_rumple]:
        for col in all_cols:
            if col not in gdf.columns:
                gdf[col] = None

    # Reorder columns consistently
    rgi_only = rgi_only[list(all_cols)]
    rise_rumple = rise_rumple[list(all_cols)]

    # Final spatially-merged GDF
    merged = gpd.GeoDataFrame(
        pd.concat([rgi_only, rise_rumple], ignore_index=True),
        crs=glacier_complex.crs
    )

    linkage_assessment_fp = Path(args.data_path) / "linkage.gpkg"
    merged.to_file(linkage_assessment_fp, driver="GPKG")


if __name__ == "__main__":
    main()