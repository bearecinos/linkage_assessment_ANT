"""
This script splits the database of High resolution vector polygons of
the Antarctic coastline (from SCAR - ADD) into different masks
to conduct a linkage assessments.

1) Primary Ice-Type Mask (Interaction Mask)
This mask represents the contiguous ice areas that may interact
with or influence smaller glacial and ice caps systems (or ice rises).
It includes:
- Ice sheet extent (from ADD coastline)
- Ice shelf areas (from ADD coastline)
- Ice tongue areas (from ADD coastline)
- Ice rumple areas (also mapped by ADD coastline)

2) Assessment mask
The code sub-selects coastline polygons and assigns to each polygon
covering a glacier complex or ice rise with an ID. This ID is based on
the Randolph Glacier Inventory (v7) glacier complexes `rgi_id` column
or the Ice rises and rumples database v1.0 (`id_icerise`).
If the data set only appears in RGI, the polygon has only `rgi_id`
and the `id_icerise` is set to None (same the other way around).

Script done by B. Recinos (NERC IRF U. Edinburgh)

"""
import sys
import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path
from shapely.geometry import Polygon, MultiPolygon
from shapely import union_all
import argparse

def remove_holes(geom):
    """
    Rock outcrops are removed from the RGIv7 glacier complex data
    to allow a fast and accurate spatial joint with the ADD polygons.
    This function removes all holes inside a MultiPolygon or Polygon.

    :param geom:
    :return: geom with no holes.
    """
    if geom.geom_type == 'Polygon':
        return Polygon(geom.exterior)  # keep outer ring only
    elif geom.geom_type == 'MultiPolygon':
        return MultiPolygon([Polygon(p.exterior) for p in geom.geoms])
    return geom  # skip others


def is_real_glacier_overlap(island_geom, union_polygon, min_ratio=0.05):
    """
    Identifies if the glacier outline or glacier complex outline overlays an island
    or coastal polygon from the ADD database, by over a minimum ratio
    e.g. 0.05 -> The glacier outline covers the island by at least 5%

    :param island_geom: geopandas.GeoDataFrame of island geometries (from ADD)
    :param union_polygon: polygon for all RGIv7 (unified via union_all)
    :param min_ratio: minimum ratio between island and RGI
    :return: bool: True if the glacier covers the island by more than min_ratio,
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
        "--coast_file",
        type=Path,
        required=True,
        help="Path to the coastline shapefile (.shp)"
    )
    parser.add_argument(
        "--glacier_complex",
        type=Path,
        required=True,
        help="Path to the glacier complex shapefile (.shp)"
    )
    parser.add_argument(
        "--data_path",
        type=Path,
        required=True,
        help="Path to the output or data directory"
    )
    parser.add_argument(
        "--ice_rumples",
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

    # From the ADD coastline product, we only select what is land
    land_gdf = coast[coast['surface'] == 'land']
    assert np.all(land_gdf.surface.values == "land")

    # Let's select the largest land feature: the Antarctic Ice Sheet
    areas = land_gdf.geometry.area
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
    # We include ice rumples from the ADD coastal polygon dataset
    # because otherwise we endup with white spaces between
    # the Ice rises and rumples dataset and the primary mask
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

    inter_mask_fp = Path(args.data_path) / "interaction_mask.gpkg"
    interaction_mask.to_file(inter_mask_fp, driver="GPKG")

    print('-------------------------')
    print('Finished with first mask')
    print('-------------------------')

    # Remove holes inside glacier complexes
    glacier_complex['geometry'] = glacier_complex['geometry'].apply(remove_holes)

    # Fix geometries
    no_icesheet['geometry'] = no_icesheet.geometry.buffer(0)
    glacier_complex['geometry'] = glacier_complex.geometry.buffer(0)

    # Remove empty geometries
    no_icesheet = no_icesheet[~no_icesheet.geometry.is_empty]
    glacier_complex = glacier_complex[~glacier_complex.geometry.is_empty]

    glaciers_union = glacier_complex.geometry.copy().union_all()

    # Find or crop polygons from the ADD coastline data set that overlaps
    # a glacier complex polygon from RGIv7 (we keep their RGI-ids)
    islands_with_glacier_complex = no_icesheet[
        no_icesheet.geometry.apply(lambda g: is_real_glacier_overlap(g, glaciers_union, min_ratio=0.05))
    ].copy()

    # Make a spatial joint on those with the glacier_complex
    island_glacier_join = gpd.sjoin(
        islands_with_glacier_complex[['geometry']],  # left: islands
        glacier_complex[['rgi_id', 'geometry']],  # right: glacier complexes
        how='left',
        predicate='intersects'
    )

    # Assigned the corresponding rgi_id of the glacier complex
    rgi_list_per_island = (
        island_glacier_join.groupby(island_glacier_join.index)['rgi_id']
        .apply(lambda s: list(s.dropna().unique()))
        .rename("rgi_ids")
    )

    islands_with_glacier_complex = islands_with_glacier_complex.join(rgi_list_per_island)
    print('Island with glacier complexes')
    print(islands_with_glacier_complex.head())

    # Adding unmatch geometries for a complete data set
    matched_glacier_idxs = island_glacier_join['index_right'].dropna().unique()
    unmatched_glaciers = glacier_complex.drop(index=matched_glacier_idxs).copy()
    unmatched_glaciers['rgi_ids'] = unmatched_glaciers.apply(lambda row: [row['rgi_id']],
                                                             axis=1)
    unmatched_glaciers = unmatched_glaciers[['geometry', 'rgi_ids']]

    glaciers_final = pd.concat([islands_with_glacier_complex[['geometry', 'rgi_ids']],
                                unmatched_glaciers],
                               ignore_index=True)

    glacier_final = gpd.GeoDataFrame(glaciers_final,
                                     geometry='geometry',
                                     crs="EPSG:3031")

    print('Final ADD coastal polygons with glacier complexes + '
          'the unmatched come from RGIv7')
    print(glacier_final.head())


    ## Now we do the same to the ice rises and rumples database
    ## Fix geometries
    rise_rumple['geometry'] = rise_rumple.geometry.buffer(0)
    # Remove empty geometries
    rise_rumple = rise_rumple[~rise_rumple.geometry.is_empty]
    rise_rumple_union = rise_rumple.geometry.copy().union_all()

    # Select ADD coastal polygons which overlap with polygons
    # in the ice rises and rumples database keep `id_icerise`
    islands_with_rumples_rises = no_icesheet[
        no_icesheet.geometry.apply(lambda g: is_real_glacier_overlap(g,
                                                                     rise_rumple_union,
                                                                     min_ratio=0.05))
    ].copy()

    # Do a spatial join on those with the ice rumple and rises database
    island_rumple_join = gpd.sjoin(
        islands_with_rumples_rises[['geometry']],  # left: islands
        rise_rumple[['id_icerise', 'geometry']],  # right: glacier complexes
        how='left',
        predicate='intersects'
    )

    ## Add to each polygon from ADD coastline the corresponding
    # ice rise id
    rumple_list_per_island = (
        island_rumple_join.groupby(island_rumple_join.index)['id_icerise']
        .apply(lambda s: list(s.dropna().unique()))
        .rename("id_icerise")
    )

    # Attach the glacier ID list back to the islands GeoDataFrame
    islands_with_rumples_rises = islands_with_rumples_rises.join(rumple_list_per_island)

    print('Coastline polygons (island) that are ice rises and rumples')
    print(islands_with_rumples_rises.head())

    # The same happens below for ice rises and rumples
    # Adding unmatch geometries
    matched_rumple_idxs = island_rumple_join['index_right'].dropna().unique()
    unmatched_rumples = rise_rumple.drop(index=matched_rumple_idxs).copy()
    unmatched_rumples['id_icerise'] = unmatched_rumples.apply(lambda row: [row['id_icerise']],
                                                              axis=1)
    unmatched_rumples = unmatched_rumples[['geometry', 'id_icerise']]

    rumples_final = pd.concat([islands_with_rumples_rises[['geometry', 'id_icerise']],
                               unmatched_rumples],
                              ignore_index=True)

    rumples_final = gpd.GeoDataFrame(rumples_final, geometry='geometry',
                                     crs="EPSG:3031")

    print('Final ADD coastal polygons which are ice rises according to'
          'ice rises and rumples database v1.0 + '
          'the unmatched come from ice rises and rumples database v1.0')
    print(rumples_final.head())

    # Ensure that both of the geopandas.Dataframes generated above have the same columns
    # If one doesn't have the other's ID column, fill with None
    # This is essential to track the linkage assessment done with the ADD
    # coastline back to the RGIv7 and the ice rise and rumples v1.0
    if 'id_icerise' not in glaciers_final.columns:
        glaciers_final['id_icerise'] = None
    if 'rgi_ids' not in rumples_final.columns:
        rumples_final['rgi_ids'] = None

    # Combine both into one GeoDataFrame
    combined = pd.concat([glaciers_final, rumples_final], ignore_index=True)

    # Drop duplicate geometries
    # We drop geometries that are exactly equal (bitwise comparison)
    combined = combined[~combined.geometry.duplicated(keep='first')].copy()

    combined['id_icerise'] = combined['id_icerise'].apply(
        lambda lst: [int(x) for x in lst if x is not None] if isinstance(lst, list) else None
    )

    combined['analysis_id'] = combined.index

    # Save final combined layer
    final_output_fp = Path(args.data_path) / "ADD_polys_with_RGI-GCv7_IRRv1_combined.gpkg"
    gpd.GeoDataFrame(combined, geometry='geometry',
                     crs="EPSG:3031").to_file(final_output_fp,
                                              driver="GPKG")

if __name__ == "__main__":
    main()