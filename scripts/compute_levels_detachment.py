"""
This script assigns detachment / attachment levels (0–3) to all Antarctic coastal
and glacier polygons. The workflow integrates ADD coastline data, RGI glacier
complex dataset, the ice rises and rumples dataset and additional ice-shelf masks that
help with the final scoring.

Outputs:
1. final_classification_buckets_w_levels.gpkg
2. RGI-GCv7_with_levels.gpkg
3. IRRv1_with_levels.gpkg
4. Stats CSVs: level_area_stats.csv, level_area_metrics.csv

Reversed level convention:
- Level 0: Fully detached (0% perimeter overlap), detachment_score == 2.0
- Level 1: Weak attachment, detachment_score in [1.7, 1.9]
- Level 2: Strong attachment, detachment_score in [1.1, 1.6]
- Level 3: Attached / part of ice sheet, detachment_score in [1.0, 1.0]

Notes:
- The indirect-connection logic (buffered shelves) is preserved: some polygons with
  detachment_score==2 and buttress_code==1 may be reclassified from Level 0 to Level 1.
- Per-RGI-ID and per-ice-rise aggregation keeps the MOST ATTACHED state, which is MAX(level)
  under this reversed convention.
"""
import sys
import os
import argparse
from pathlib import Path
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Polygon, MultiPolygon
import ast

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Assign reversed attachment/detachment levels (0–3) to Antarctic polygons."
    )
    parser.add_argument(
        "--data_path",
        type=Path,
        required=True,
        help="Path to the output or data directory"
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
        "--ice_rumples",
        type=Path,
        required=True,
        help="Path to the ice rumples shapefile (.shp)"
    )
    parser.add_argument("--level3_min_score", type=float, default=1.0, help="Level 3 (attached) min detachment_score")
    parser.add_argument("--level3_max_score", type=float, default=1.0, help="Level 3 (attached) max detachment_score")

    parser.add_argument("--level2_min_score", type=float, default=1.1, help="Level 2 (strong attachment) min detachment_score")
    parser.add_argument("--level2_max_score", type=float, default=1.6, help="Level 2 (strong attachment) max detachment_score")

    parser.add_argument("--level1_min_score", type=float, default=1.7, help="Level 1 (weak attachment) min detachment_score")
    parser.add_argument("--level1_max_score", type=float, default=1.9, help="Level 1 (weak attachment) max detachment_score")

    parser.add_argument("--level0_score", type=float, default=2.0, help="Level 0 (fully detached) exact detachment_score")

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


def norm_icerise(x):
    """
    Ice rises and rumples ids are sometimes
    store as a list or an interger value
    :param x: item from the df_final dataset
    :return: if list is empty None, if list has
    a value returns int or the int.
    """
    # empty list returns None
    if isinstance(x, list):
        return x if len(x) > 0 else None
    # single integer returns keep
    if isinstance(x, int):
        return x
    return None


def apply_level(out_df, level_df, key="analysis_id"):
    """
    Assign the final level of attachment
    - **Level 0** - Fully detached island 0% perimeter overlap score -> 2.0
    - **Level 1** - Weak attachment: 39-1% perimeter overlap scores -> 1.9 - 1.7 if chosen default
    - **Level 2** - Strong attachment: 40-99% perimeter overlap scores -> 1.6 - 1.1 if chosen default
    - **Level 3** - Attached: 100% perimeter overlap scores -> 1.0
    :param out_df: same file as final_classification_buckets.gpkg but with levels of detachment
    :param level_df: the final grouping for each level
    :param key: column name
    :return: out_df with the right detachment level assigned
    """
    m = level_df[[key, "level", "level_text"]].drop_duplicates(subset=[key])
    tmp = out_df.merge(m, on=key, how="left", suffixes=("", "__new"), validate="m:1")
    tmp["level"] = tmp["level"].combine_first(tmp["level__new"])
    tmp["level_text"] = tmp["level_text"].combine_first(tmp["level_text__new"])
    return tmp.drop(columns=["level__new", "level_text__new"])


def _att_range(dmin: float, dmax: float):
    """
    Define attachment range scores given detachment range scores
    :param dmin:
    :param dmax:
    :return:
    """
    amin = round(2.0 - dmax, 2)
    amax = round(2.0 - dmin, 2)
    return f"attachment_score in [{amin}, {amax}]"


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

    # Now select only the ice sheet
    only_icesheet = land_gdf.loc[[largest_idx]].copy()

    # Define ice types of interest
    ice_types = ['ice shelf', 'ice tongue', 'rumple']

    # Create separate GeoDataFrames for each ice type
    ice_type_gdfs = {
        ice_type: coast[coast['surface'].str.lower() == ice_type].copy()
        for ice_type in ice_types
    }

    # Separate ADD coastline polygons into specific ice-related surface types:
    # - ice shelf
    # - ice tongue
    # - rumple
    ice_shelf_gdf = ice_type_gdfs['ice shelf']
    ice_tongue_gdf = ice_type_gdfs['ice tongue']
    ice_rumple_gdf = ice_type_gdfs['rumple']

    # These are needed to understand what is buttressed by main land ice
    # or other
    remaining_shelves = os.path.join(args.data_path, 'remaining_shelves_mask.gpkg')
    other_shelves = gpd.read_file(remaining_shelves)

    final_classification = os.path.join(args.data_path, 'final_classification_buckets.gpkg')
    final_df = gpd.read_file(final_classification)

    only_icesheet['surface'] = 'Ice sheet'

    # Combine in a mask the first version of Level 3
    combined_gdf = gpd.GeoDataFrame(
        pd.concat([only_icesheet, ice_shelf_gdf, ice_tongue_gdf, ice_rumple_gdf],
                  ignore_index=True),
                  crs=only_icesheet.crs
    )

    # We need to remove the other_shelves that are not connected via
    # anything to the mainland ice
    mask = other_shelves.dissolve()  # single (multi)polygon row

    # Use the 'remaining_shelves_mask' dataset to erase shelves that are not
    # physically connected to the main land. This produces a cleaned mask
    # for the initial Level 3 dataset.
    combined_erased = gpd.overlay(combined_gdf,
                                  mask,
                                  how="difference",
                                  keep_geom_type=True)

    # Add extra columns to combine the geopandas.Dataframes easier
    # later
    combined_erased = combined_erased.assign(
        rgi_ids=[["RGI2000-v7.0-C-20-00000"]] * len(combined_erased),  # list per row
        id_icerise=None,
        ice_type=None,
        ice_type_text=None,
        analysis_id=None,
        perimeter_shared=None,
        ratio=100,
        detachment_score=1.0,
        attachment_score=1.0,
        detachment_desc="Perimeter overlap 100%",
        buttress_code=None,
        buttress_source_id="Buttressed by mainland ice",
    )

    combined_erased["total_area"] = combined_erased.geometry.area
    combined_erased["perimeter"] = combined_erased.geometry.length

    # Make sure final_df also has a surface column so we can combined
    # ice bodies classified as Level 3 with the main Level 3 mask
    final_df['surface'] = None

    # Partition polygons from final_df into Level 3 and 0
    # using detachment_score and buttress_code.
    # These groups come straight from perimeter overlap metrics prior to
    # from running this code. We start with the easy group Level 3 and 0.
    Level_3 = final_df.loc[final_df["detachment_score"].between(args.level3_min_score,
                                                                args.level3_max_score,
                                                                inclusive="both")].copy()
    Level_3_from_final_df = pd.concat([Level_3, combined_erased], ignore_index=True)

    # Now lets pick those completly detached and buttressed by ice from the mainland: detachment_score == 2.0
    Level_0_from_final_df = final_df.loc[final_df["detachment_score"].eq(args.level0_score)
                                         & final_df["buttress_code"].eq(0)].copy()

    # Now everything with a perimeter overlap between level2_min_score and level2_max_score
    Level_2_from_final_df = final_df.loc[final_df["detachment_score"].between(args.level2_min_score,
                                                                              args.level2_max_score,
                                                                              inclusive="both")].copy()

    # Now everything with a perimeter overlap between level1_min_score and level1_max_score (weak attachment)
    # We will still need to add anything in Level 0 (fully detached candidates)
    # that has an indirect connection via intersection with the buffered mask built from Levels 1–3.
    # After doing another interaction mask analysis, those Level 0 polygons become Level 1.
    Level_1_from_final_df = final_df.loc[final_df["detachment_score"].between(args.level1_min_score,
                                                                              args.level1_max_score,
                                                                              inclusive="both")].copy()

    # Interaction mask uses "attached-ish" classes (Levels 1–3) to reclassify buffered detached polygons.
    inter_mask_one = gpd.GeoDataFrame(
        pd.concat([
            Level_3_from_final_df,
            Level_2_from_final_df,
            Level_1_from_final_df
        ], ignore_index=True), crs=only_icesheet.crs)

    inter_mask_one["mask"] = 1 # Dummy column for dissolve
    int_mask_fa = inter_mask_one.dissolve(by="mask")
    int_mask_fa["geometry"] = int_mask_fa["geometry"].apply(remove_holes)

    # We make a big buffer in the interaction mask in order to separate
    # the other ice shelves areas
    int_mask_fa["geometry"] = int_mask_fa.geometry.buffer(50)

    # Union mask into one geometry
    mask_union = int_mask_fa.geometry.union_all()

    # Split "other_shelves" by intersection with the (buffered) mask
    other_shelves_in = other_shelves.loc[other_shelves.intersects(mask_union)].copy()
    other_shelves_out = other_shelves.loc[~other_shelves.intersects(mask_union)].copy()

    # Repeat mask construction but this time including shelves that intersected the first mask.
    # This two-step approach ensures that indirect attachment levels passing through
    # multiple shelves are propagated to those ADD polygons
    # Any Level 0 polygon intersecting this "new mask" becomes Level 1 ("indirectly attached").
    inter_mask_two = gpd.GeoDataFrame(
        pd.concat([
            Level_3_from_final_df,
            Level_2_from_final_df,
            Level_1_from_final_df,
            other_shelves_in
        ], ignore_index=True), crs=only_icesheet.crs)

    inter_mask_two["mask"] = 1 # Dummy column for dissolve
    int_mask_fa = inter_mask_two.dissolve(by="mask")
    int_mask_fa["geometry"] = int_mask_fa["geometry"].apply(remove_holes)

    # We make a big buffer in the new interaction mask in order to separate
    # Level 0 polygons buffered by ice shelves which are truly detached
    int_mask_fa["geometry"] = int_mask_fa.geometry.buffer(50)

    # Union mask into one geometry
    mask_union = int_mask_fa.geometry.union_all()

    Level_0_buffered = final_df.loc[final_df["detachment_score"].eq(args.level0_score)
                                    & final_df["buttress_code"].eq(1)].copy()

    # Identify Level 0 polygons buffered by the interaction mask.
    # These polygons are indirectly connected through shelves → reclassified as Level 1.
    Level_0_to_1 = Level_0_buffered.loc[Level_0_buffered.intersects(mask_union)].copy()

    other_ice_bodies_out = Level_0_buffered.loc[~Level_0_buffered.intersects(mask_union)].copy()

    # Finalize Level 1: direct weak attachment + indirect (from detached buffered)
    Level_1_final = gpd.GeoDataFrame(
        pd.concat([Level_1_from_final_df, Level_0_to_1], ignore_index=True),
        crs=final_df.crs,
    )

    # Finalize Level 0: fully detached (buttress_code 0) + buffered-but-still-detached
    Level_0_final = gpd.GeoDataFrame(
        pd.concat([Level_0_from_final_df, other_ice_bodies_out], ignore_index=True),
        crs=final_df.crs,
    )

    assert (
        len(Level_3)
        + len(Level_2_from_final_df)
        + len(Level_1_final)
        + len(Level_0_final)
        == len(final_df)
    ), (
        f"Count mismatch: sum(levels)={len(Level_3) + 
                                       len(Level_2_from_final_df) + 
                                       len(Level_1_final) + 
                                       len(Level_0_final)} "
        f"!= len(final_df)={len(final_df)}"
    )

    # Add 'level' and 'level_text' to each group (0,1,2,3).
    # Then merge them back into the original final_df using apply_level(),
    # which respects priority order.
    Level_0_final["level"] = int(0)
    Level_0_final["level_text"] = "Fully detached"

    Level_1_final["level"] = int(1)
    Level_1_final["level_text"] = "Weak attachment"

    Level_2_from_final_df["level"] = int(2)
    Level_2_from_final_df["level_text"] = "Strong attachment"

    Level_3_from_final_df["level"] = int(3)
    Level_3_from_final_df["level_text"] = "Attached (ice sheet)"

    out = final_df.copy()
    out["level"] = None
    out["level_text"] = None

    KEY = "analysis_id"

    # Other ice shelves or ice sheet polygons in the interaction mask
    # will be dropped from the dataset since those where not in the original
    # final_df, the goal here is propagate levels to the original RGI and IRR datasets
    out = apply_level(out, Level_0_final, key=KEY)
    out = apply_level(out, Level_1_final, key=KEY)
    out = apply_level(out, Level_2_from_final_df, key=KEY)
    out = apply_level(out, Level_3_from_final_df, key=KEY)

    if out["level"].isna().any():
        missing = out.loc[out["level"].isna(), KEY].head(20).tolist()
        raise ValueError(f"Unassigned rows after merges. Example {KEY}s: {missing}")

    filename = Path(args.data_path) / "final_classification_buckets_w_levels.gpkg"
    out.to_file(filename, driver="GPKG")

    # Assigning levels to RGI data set
    # Some polygons reference the same RGI glacier ID.
    # We explode rgi_ids into 1 row per glacier and then collapse back to get one
    # level per rgi_id polygon using the condition: → keep the MAX level (most attached)
    rgi_out = out.dropna(subset=["rgi_ids"]).copy()
    rgi_out["rgi_ids"] = rgi_out["rgi_ids"].apply(ast.literal_eval)

    rgi_out_exploded = (
        rgi_out.explode("rgi_ids")
        .rename(columns={"rgi_ids": "rgi_id"})
        .reset_index(drop=True)
    )

    tmp = rgi_out_exploded.dropna(subset=["level"]).copy()
    tmp["level"] = pd.to_numeric(tmp["level"], errors="coerce").astype("Int64")

    # Build a 1-row-per-rgi_id mapping using max level as a condition
    # This is necessary for Alexander island only
    tmp["_has_text"] = tmp["level_text"].notna().astype(int)

    # MAX level = most attached
    rgi_level_map = (
        tmp.sort_values(["rgi_id", "level", "attachment_score", "_has_text"], ascending=[True, False, False, False])
        .drop_duplicates(subset=["rgi_id"], keep="first")[["rgi_id", "level", "level_text", "attachment_score"]]
        .reset_index(drop=True)
    )

    # Ensure glacier_complex has one geometry row per rgi_id
    gc = glacier_complex.copy()
    df_rgi = gc.merge(rgi_level_map, on="rgi_id", how="inner", validate="1:1")


    filename = Path(args.data_path) / "RGI-GCv7_with_levels.gpkg"
    df_rgi.to_file(filename, driver="GPKG")

    # Print some stats
    grouped = (
        df_rgi
        .groupby("level", dropna=False)["area_km2"]
        .agg(total_area_km2="sum")
        .reset_index()
    )

    # add counts and percent of total
    counts = df_rgi.groupby("level", dropna=False).size().reset_index(name="n_rows")
    grouped = grouped.merge(counts, on="level")

    grouped["pct_of_total"] = 100 * grouped["total_area_km2"] / grouped["total_area_km2"].sum()

    # sort by largest area first
    grouped = grouped.sort_values("total_area_km2", ascending=False).reset_index(drop=True)

    level_defs = {
        3: f"Level 3 (attached): {_att_range(args.level3_min_score, args.level3_max_score)}",
        2: f"Level 2 (strong attachment): {_att_range(args.level2_min_score, args.level2_max_score)}",
        1: f"Level 1 (weak attachment): {_att_range(args.level1_min_score, args.level1_max_score)} "
           f"+ indirect connection rule",
        0: f"Level 0 (fully detached): attachment_score=={round(2.0 - args.level0_score, 2)} with buttress_code 0",
    }

    grouped["level_definition"] = grouped["level"].map(level_defs).fillna("")

    print(grouped)

    out_stats_csv = args.data_path / "level_area_stats.csv"
    grouped.to_csv(out_stats_csv, index=False)
    print(f"Saved level stats to: {out_stats_csv}")

    # total RGI area
    total_area = glacier_complex["area_km2"].sum()
    # sum area for levels, we remove (1, 2, 3)
    removed_area = grouped.loc[grouped["level"].isin([1, 2, 3]), "total_area_km2"].sum()

    # percent lost
    if total_area == 0:
        pct_lost = 0.0
    else:
        pct_lost = 100 * removed_area / total_area

    print(f"Area that is potentially "
          f"part of the Ice Sheet = {removed_area:.3f} km²")
    print(f"Total RGI area = {total_area:.3f} km²")
    print(f"Percent area part of Ice Sheet = {pct_lost:.2f}%")

    metrics = pd.DataFrame([{
        "removed_area_km2": removed_area,
        "total_rgi_area_km2": total_area,
        "pct_area_part_of_ice_sheet": pct_lost,
    }])
    out_metrics_csv = args.data_path / "level_area_metrics.csv"
    metrics.to_csv(out_metrics_csv, index=False)
    print(f"Saved metrics to: {out_metrics_csv}")

    # --- Distribute levels across IRR dataset (keep MOST attached => MAX(level)) ---
    out_IRR = out.copy()
    out_IRR["id_icerise"] = out_IRR["id_icerise"].apply(ast.literal_eval)
    out_IRR["id_icerise"] = out_IRR["id_icerise"].apply(norm_icerise)

    # Keep only valid rows
    out_IRR = out_IRR.dropna(subset=["id_icerise"]).copy()

    # Explode list
    rise_out_exploded = (
        out_IRR.explode("id_icerise").reset_index(drop=True)
    )

    # Keep rows with assigned level
    tmp = rise_out_exploded.dropna(subset=["level"]).copy()
    tmp["level"] = pd.to_numeric(tmp["level"], errors="coerce").astype("Int64")
    tmp["_has_text"] = tmp["level_text"].notna().astype(int)

    icerise_level_map = (
        tmp.sort_values(["id_icerise", "level", "attachment_score",  "_has_text"],
                        ascending=[True, False, False, False])
        .drop_duplicates(subset=["id_icerise"], keep="first")
        [["id_icerise", "level", "level_text", "attachment_score"]]
        .reset_index(drop=True)
    )

    df_icerise = rise_rumple.merge(
        icerise_level_map,
        on="id_icerise",
        how="inner",
        validate="1:1"
    )

    # Law Dome forced to ATTACHED (Level 3) under reversed convention
    df_icerise.loc[df_icerise["id_icerise"] == 174, "level"] = 3
    df_icerise.loc[df_icerise["id_icerise"] == 174, "level_text"] = "Attached (ice sheet)"
    df_icerise.loc[df_icerise["id_icerise"] == 174, "attachment_score"] = 1.0

    outfile = Path(args.data_path) / "IRRv1_with_levels.gpkg"
    df_icerise.to_file(outfile, driver="GPKG")


if __name__ == "__main__":
    main()













