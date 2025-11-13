import sys
import os
import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path
from shapely.geometry import Polygon, MultiPolygon
from shapely import union_all
import argparse

def remove_holes(geom):
    if geom.type == 'Polygon':
        return Polygon(geom.exterior)  # keep outer ring only
    elif geom.type == 'MultiPolygon':
        return MultiPolygon([Polygon(p.exterior) for p in geom.geoms])
    return geom  # skip others


def is_real_glacier_overlap(island_geom, union_polygon, min_ratio=0.05):
    """Identify if the glacier overlays an island by over a minimum
    ratio e.g. 0.05 -> the glacier outline covers the island by at least 5%
    :param island_geom: geopandas.GeoDataFrame of island geometries
    :param union_polygon: polygon for all RGIv7
    :param min_ratio: minimum ratio between island and RGI
    :return: bool: True if if the glacier covers the island by more than min_ratio,
    else False.
    """
    overlap_area = union_polygon.intersection(island_geom).area
    ratio = overlap_area / island_geom.area
    return ratio > min_ratio


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
        pd.concat([only_icesheet, ice_shelf_gdf, ice_tongue_gdf, ice_rumple_gdf], ignore_index=True),
        crs=only_icesheet.crs
    )

    # 2. Dissolve all into one single polygon (mask)
    combined_gdf["mask"] = 1  # Dummy column for dissolve
    interaction_mask = combined_gdf.dissolve(by="mask")

    interaction_mask['geometry'] = interaction_mask['geometry'].apply(remove_holes)

    inter_mask_fp = Path(args.data_path) / "inter_mask.gpkg"
    interaction_mask.to_file(inter_mask_fp, driver="GPKG")

    glacier_complex['geometry'] = glacier_complex['geometry'].apply(remove_holes)

    # Fix geometries
    no_icesheet['geometry'] = no_icesheet.geometry.buffer(0)
    glacier_complex['geometry'] = glacier_complex.geometry.buffer(0)

    # Remove empty geometries
    no_icesheet = no_icesheet[~no_icesheet.geometry.is_empty]
    glacier_complex = glacier_complex[~glacier_complex.geometry.is_empty]

    glaciers_union = glacier_complex.geometry.copy().union_all()

    islands_with_glacier_complex = no_icesheet[
        no_icesheet.geometry.apply(lambda g: is_real_glacier_overlap(g, glaciers_union, min_ratio=0.05))
    ].copy()

    island_glacier_join = gpd.sjoin(
        islands_with_glacier_complex[['geometry']],  # left: islands
        glacier_complex[['rgi_id', 'geometry']],  # right: glacier complexes
        how='left',
        predicate='intersects'
    )

    rgi_list_per_island = (
        island_glacier_join.groupby(island_glacier_join.index)['rgi_id']
        .apply(lambda s: list(s.dropna().unique()))
        .rename("rgi_ids")
    )

    islands_with_glacier_complex = islands_with_glacier_complex.join(rgi_list_per_island)

    rgi_with_add = Path(args.data_path) / "islands_with_glacier_complex.gpkg"
    islands_with_glacier_complex.to_file(rgi_with_add, driver="GPKG")

    ## Now we do the same with ice rumples
    # Fix geometries
    rise_rumple['geometry'] = rise_rumple.geometry.buffer(0)

    # Remove empty geometries
    rise_rumple = rise_rumple[~rise_rumple.geometry.is_empty]

    rise_rumple_union = rise_rumple.geometry.copy().union_all()

    islands_with_rumples_rises = no_icesheet[
        no_icesheet.geometry.apply(lambda g: is_real_glacier_overlap(g, rise_rumple_union, min_ratio=0.05))
    ].copy()

    island_rumple_join = gpd.sjoin(
        islands_with_rumples_rises[['geometry']],  # left: islands
        rise_rumple[['id_icerise', 'geometry']],  # right: glacier complexes
        how='left',
        predicate='intersects'
    )

    # ---------------------------------------------
    # 6. Aggregate glacier IDs per island into a list
    # ---------------------------------------------
    rumple_list_per_island = (
        island_rumple_join.groupby(island_rumple_join.index)['id_icerise']
        .apply(lambda s: list(s.dropna().unique()))
        .rename("id_icerise")
    )

    # ---------------------------------------------
    # 7. Attach the glacier ID list back to the islands GeoDataFrame
    # ---------------------------------------------
    islands_with_rumples_rises = islands_with_rumples_rises.join(rumple_list_per_island)

    rises_rumples_with_add = Path(args.data_path) / "islands_with_ice_rumples_and_rises.gpkg"
    islands_with_rumples_rises.to_file(rises_rumples_with_add, driver="GPKG")

    # Adding unmatch geometries
    matched_glacier_idxs = island_glacier_join['index_right'].dropna().unique()
    unmatched_glaciers = glacier_complex.drop(index=matched_glacier_idxs).copy()
    unmatched_glaciers['rgi_ids'] = unmatched_glaciers.apply(lambda row: [row['rgi_id']], axis=1)
    unmatched_glaciers = unmatched_glaciers[['geometry', 'rgi_ids']]

    glaciers_final = pd.concat([islands_with_glacier_complex[['geometry', 'rgi_ids']], unmatched_glaciers],
                               ignore_index=True)

    rgi_with_add = Path(args.data_path) / "islands_with_glacier_complex_AND_unmatched.gpkg"
    gpd.GeoDataFrame(glaciers_final, geometry='geometry', crs="EPSG:3031").to_file(rgi_with_add, driver="GPKG")


    # Adding unmatch geometries
    matched_rumple_idxs = island_rumple_join['index_right'].dropna().unique()
    unmatched_rumples = rise_rumple.drop(index=matched_rumple_idxs).copy()
    unmatched_rumples['id_icerise'] = unmatched_rumples.apply(lambda row: [row['id_icerise']], axis=1)
    unmatched_rumples = unmatched_rumples[['geometry', 'id_icerise']]

    rumples_final = pd.concat([islands_with_rumples_rises[['geometry', 'id_icerise']], unmatched_rumples],
                              ignore_index=True)

    rises_rumples_with_add = Path(args.data_path) / "islands_with_ice_rumples_AND_unmatched.gpkg"
    gpd.GeoDataFrame(rumples_final, geometry='geometry',
                     crs="EPSG:3031").to_file(rises_rumples_with_add, driver="GPKG")

    # 1. Ensure both have the same columns
    # If one doesn't have the other's ID column, fill with None
    if 'id_icerise' not in glaciers_final.columns:
        glaciers_final['id_icerise'] = None
    if 'rgi_ids' not in rumples_final.columns:
        rumples_final['rgi_ids'] = None

    # 2. Combine both into one GeoDataFrame
    combined = pd.concat([glaciers_final, rumples_final], ignore_index=True)

    # 3. Drop duplicate geometries
    # We drop geometries that are exactly equal (bitwise comparison)
    combined = combined[~combined.geometry.duplicated(keep='first')].copy()

    # 4. Save final combined layer
    final_output_fp = Path(args.data_path) / "islands_with_glacier_and_rumples_combined.gpkg"
    gpd.GeoDataFrame(combined, geometry='geometry', crs="EPSG:3031").to_file(final_output_fp, driver="GPKG")

if __name__ == "__main__":
    main()



