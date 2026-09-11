"""
Compute per-polygon coverage of a boolean raster mask for multiple polygon datasets.

This script reads an application mask raster where cells with value 1 indicate
that data exist, then computes for each polygon the fraction of its true vector
area covered by valid raster cells. The calculation is done separately for three
polygon databases (ADD, RGI, and IRR), while preserving all original polygon
attributes and appending new coverage metrics.

For each polygon, the workflow:
1. Reads only the raster window around the polygon bounds.
2. Keeps only raster cells where the mask equals 1.
3. Converts those valid raster cells into grid-cell polygons.
4. Intersects them with the input polygon geometry.
5. Sums the overlap area.
6. Computes the percentage of polygon area covered by valid raster cells.

The script parallelizes the polygon processing with multiprocessing and writes
one output GeoPackage per input polygon dataset.

Script done by B. Recinos (NERC IRF, U. Edinburgh)
Portions of this code were generated or optimized using
Microsoft Copilot
"""

from __future__ import annotations

import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path
import math

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import argparse
from rasterio.windows import Window, from_bounds
from shapely.geometry import box, shape
from shapely.ops import unary_union
from rasterio.features import geometry_mask, shapes
from shapely.prepared import prep

_SRC = None
_BAND = 1


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Compute level coverage per application."
    )
    parser.add_argument(
        "--application_mask",
        type=Path,
        required=True,
        help="Path to the application mask (.tif)"
    )
    parser.add_argument(
        "--data_path",
        type=Path,
        required=True,
        help="Path to the output or data directory"
    )
    return parser.parse_args()


def validate_paths(args):
    """
    Makes sure paths to files exist
    :param args:
    :return: errors if files are not there
    """
    if not args.application_mask.exists():
        sys.exit(f"Mask file not found: {args.application_mask}")
    if not args.data_path.exists():
        print(f"Data path does not exist. Creating: {args.data_path}")
        args.data_path.mkdir(parents=True, exist_ok=True)


def empty_result():
    """
    Builds dictionary for empty results when:
    geometry is missing
    geometry is invalid
    geometry has zero area
    polygon does not overlap the raster or raster window is empty
    :return: Dictionary for empty results
    """
    return {
        "polygon_area_m2": np.nan,
        "covered_area_m2": 0.0,
        "covered_area_km2": 0.0,
        "valid_cell_count": 0,
        "percent_coverage": np.nan,
        "window_height": 0,
        "window_width": 0,
        "window_cells": 0,
        "valid_cells_in_window": 0,
        "runtime_s": 0.0,
        "status": "empty",
    }


def zero_result(polygon_area_m2: float, status: str):
    """
    This returns a default result for polygons that should get 0 coverage
    Typical cases are:
    - polygon is outside the mask footprint
    - mask has no valid cells
    - no overlap / no valid cells in the local raster window
    :param polygon_area_m2:
    :param status:
    :return:
    """
    return {
        "polygon_area_m2": polygon_area_m2,
        "covered_area_m2": 0.0,
        "covered_area_km2": 0.0,
        "valid_cell_count": 0,
        "percent_coverage": 0.0,
        "window_height": 0,
        "window_width": 0,
        "window_cells": 0,
        "valid_cells_in_window": 0,
        "runtime_s": 0.0,
        "status": status,
    }


def cell_polygon_from_rowcol(transform, row: int, col: int):
    """
    This converts one raster cell into a Shapely polygon
    :param transform: this is the raster affine transform to get the coordinates
    from the cell corners
    :param row: row indexes from the raster
    :param col: columns indexes from the raster
    :return: a rectangle polygon box
    """
    x_left, y_top = transform * (col, row)
    x_right, y_bottom = transform * (col + 1, row + 1)
    return box(
        min(x_left, x_right),
        min(y_bottom, y_top),
        max(x_left, x_right),
        max(y_bottom, y_top),
    )


def valid_data_mask(arr):
    """
    Identify raster cells where data exist.
    Assumes a boolean-like mask raster with value 1 where data exist.
    """
    return (~arr.mask) & (arr.data == 1)


def build_mask_footprint(raster_path: str | Path, band: int = 1):
    """
    Convert all valid raster cells into a merged Shapely footprint.

    Cells where valid_data_mask() returns True are vectorized and merged.
    The resulting footprint is used to quickly reject polygons that do not
    intersect any valid raster cell.
    :param raster path
    :param band (default to 1)
    :return: the merged footprint geometry
    """
    with rasterio.open(raster_path) as src:
        arr = src.read(band, masked=True)
        mask_bool = valid_data_mask(arr)

        if not mask_bool.any():
            return None, src.crs

        geoms = [
            shape(geom)
            for geom, value in shapes(
                mask_bool.astype(np.uint8),
                mask=mask_bool,
                transform=src.transform,
            )
            if value == 1
        ]

        if not geoms:
            return None, src.crs

        footprint = unary_union(geoms)
        return footprint, src.crs


def init_worker(raster_path: str, band: int):
    """
    Define global variables for worker deployment
    :param raster_path
    :param band
    """
    global _SRC, _BAND
    _SRC = rasterio.open(raster_path)
    _BAND = band


def compute_single_polygon_valid_coverage(src,
                                          geom,
                                          band: int = 1):
    """
    It computes the fraction of one polygon covered by valid raster cells.
    percent coverage= 100 x (poly area of polygon covered by valid raster cells/ polygon area)
    :param src: Raster source
    :param geom: geometry to compute coverage for
    :param band: raster band
    :return: dictionary with the result coverage for that polygon along with other stats
    of the raster window.
    """
    # Starts timer to check how each polygon takes
    t0 = time.perf_counter()

    if band < 1 or band > src.count:
        raise ValueError(
            f"Invalid band {band}. Raster contains {src.count} band(s)."
        )

    if geom is None or geom.is_empty or not geom.is_valid:
        result = empty_result()
        result["status"] = "invalid_geometry"
        result["runtime_s"] = time.perf_counter() - t0
        return result

    # reject zero-area polygons
    polygon_area_m2 = float(geom.area)
    if polygon_area_m2 <= 0:
        result = empty_result()
        result["polygon_area_m2"] = polygon_area_m2
        result["status"] = "zero_area"
        result["runtime_s"] = time.perf_counter() - t0
        return result

    minx, miny, maxx, maxy = geom.bounds
    raster_bounds = src.bounds

    # Reject polygons outside the raster extent
    if (
        maxx <= raster_bounds.left
        or minx >= raster_bounds.right
        or maxy <= raster_bounds.bottom
        or miny >= raster_bounds.top
    ):
        result = zero_result(
            polygon_area_m2,
            "outside_raster_bounds",
        )
        result["runtime_s"] = time.perf_counter() - t0
        return result

    raw_window = from_bounds(
        left=minx,
        bottom=miny,
        right=maxx,
        top=maxy,
        transform=src.transform,
    )

    # The window must contain the complete polygon bounding box.
    row_start = max(0, math.floor(raw_window.row_off))
    col_start = max(0, math.floor(raw_window.col_off))

    row_stop = min(src.height,
                   math.ceil(raw_window.row_off + raw_window.height)
                   )
    col_stop = min(src.width,
                   math.ceil(raw_window.col_off + raw_window.width)
                   )

    height = row_stop - row_start
    width = col_stop - col_start

    if height <= 0 or width <= 0:
        result = zero_result(
            polygon_area_m2,
            "no_overlap",
        )
        result["runtime_s"] = time.perf_counter() - t0
        return result

    window = Window(
        col_off=col_start,
        row_off=row_start,
        width=width,
        height=height,
    )

    window_cells = int(height * width)

    arr = src.read(
        band,
        window=window,
        masked=True,
    )

    if arr.size == 0:
        result = empty_result()
        result.update(
            {
                "polygon_area_m2": polygon_area_m2,
                "window_height": int(height),
                "window_width": int(width),
                "window_cells": window_cells,
                "status": "empty_read",
                "runtime_s": time.perf_counter() - t0,
            }
        )
        return result

    window_transform = src.window_transform(window)

    valid_mask = valid_data_mask(arr)

    # all_touched=True only selects candidate cells.
    # Exact covered area is still calculated below with Shapely.
    polygon_mask = geometry_mask(
        [geom],
        transform=window_transform,
        invert=True,
        out_shape=arr.shape,
        all_touched=True,
    )

    candidate_mask = valid_mask & polygon_mask

    valid_cells_in_window = int(candidate_mask.sum())

    if valid_cells_in_window == 0:
        result = zero_result(
            polygon_area_m2,
            "no_valid_cells",
        )
        result.update(
            {
                "window_height": int(height),
                "window_width": int(width),
                "window_cells": window_cells,
                "valid_cells_in_window": 0,
                "runtime_s": time.perf_counter() - t0,
            }
        )
        return result

    prepared_geom = prep(geom)

    covered_area_m2 = 0.0
    valid_region_count = 0

    # Instead of looping over every raster cell one by one,
    # it uses shapes(...) to convert connected or non-connected
    # valid raster regions into vector polygons.
    # So if many neighboring valid cells touch each other,
    # they become one larger region polygon.
    # That is much faster than intersecting cell-by-cell
    for geom_dict, value in shapes(candidate_mask.astype(np.uint8),
                                   mask=candidate_mask,
                                   transform=window_transform):

        # ignore anything that is not a valid region
        if value != 1:
            continue

        # convert the raster region into a Shapely geometry
        valid_region = shape(geom_dict)

        # quickly skip non-intersecting regions
        if not prepared_geom.intersects(valid_region):
            continue

        # Calculate the exact overlap
        intersection = valid_region.intersection(geom)
        if intersection.is_empty:
            continue

        intersection_area = float(intersection.area)
        # Boundary-only contact has no area and is not coverage.
        if intersection_area <= 0:
            continue

        covered_area_m2 += intersection_area
        valid_region_count += 1

    percent_coverage = (100.0 * covered_area_m2 / polygon_area_m2)

    # Protect against tiny floating-point errors such as
    # 100.00000000001.
    if math.isclose(percent_coverage,
                    100.0,
                    rel_tol=0.0,
                    abs_tol=1e-9):
        percent_coverage = 100.0

    return {
        "polygon_area_m2": polygon_area_m2,
        "covered_area_m2": float(covered_area_m2),
        "covered_area_km2": float(covered_area_m2 / 1e6),
        "valid_cell_count": valid_cells_in_window,
        "valid_region_count": int(valid_region_count),
        "percent_coverage": float(percent_coverage),
        "window_height": int(height),
        "window_width": int(width),
        "window_cells": window_cells,
        "valid_cells_in_window": valid_cells_in_window,
        "runtime_s": time.perf_counter() - t0,
        "status": "ok",
    }


def process_one_polygon(item):
    """
    Function to deploy each polygon per processor
    :param item:
    :return: status of the run
    """
    idx, geom, poly_label, polygon_area_m2 = item
    result = compute_single_polygon_valid_coverage(_SRC, geom, band=_BAND)
    return idx, poly_label, result


def attach_valid_coverage_to_polygons_parallel_pool(raster_path,
                                                    polygons_path,
                                                    band = 1,
                                                    n_processes=16,
                                                    chunksize= 1,
                                                    id_column: str | None = None,
                                                    diagnostics_csv: str | Path | None = None):
    """
    Pararell function that deploys all the code per worker
    :param raster_path:
    :param polygons_path:
    :param band:
    :param n_processes:
    :param chunksize:
    :param id_column:
    :param diagnostics_csv:
    :return:
    """
    gdf = gpd.read_file(polygons_path)

    original_count = len(gdf)
    print("\n=== FEATURE COUNT CHECK ===")
    print(f"Input file: {polygons_path}")
    print(f"Original feature count: {original_count}")
    print(f"Original non-null geometries: {gdf.geometry.notna().sum()}")
    print(f"Original empty geometries: {gdf.geometry.is_empty.sum()}")
    print(f"Original index unique: {gdf.index.is_unique}")

    mask_footprint, raster_crs = build_mask_footprint(raster_path, band=band)

    if gdf.crs is None:
        raise ValueError("Input polygons have no CRS.")

    if gdf.crs != raster_crs:
        gdf = gdf.to_crs(raster_crs)

    gdf = gdf.copy()
    gdf["_polygon_area_m2_sort"] = gdf.geometry.area

    if id_column is None or id_column not in gdf.columns:
        gdf["_poly_label"] = gdf.index.astype(str)
    else:
        gdf["_poly_label"] = gdf[id_column].astype(str)

    if mask_footprint is None:
        result_df = pd.DataFrame(
            {
                idx: zero_result(float(area), "mask_has_no_valid_cells")
                for idx, area in zip(gdf.index, gdf["_polygon_area_m2_sort"])
            }
        ).T
        out = gdf.join(result_df, how="left")

        if len(out) != original_count:
            raise RuntimeError(
                "Polygon count changed in mask_has_no_valid_cells branch: "
                f"{original_count} -> {len(out)}"
            )

        out = out.drop(columns=["_polygon_area_m2_sort", "_poly_label"], errors="ignore")
        return out

    prepared_mask_footprint = prep(mask_footprint)
    intersects_mask = gdf.geometry.apply(
        lambda geom: geom is not None and not geom.is_empty and prepared_mask_footprint.intersects(geom)
    )

    gdf_outside = gdf.loc[~intersects_mask].copy()
    gdf_inside = gdf.loc[intersects_mask].copy()

    split_count = len(gdf_inside) + len(gdf_outside)

    print("\n--- MASK FOOTPRINT SPLIT ---")
    print(f"Original polygons: {original_count}")
    print(f"Inside footprint: {len(gdf_inside)}")
    print(f"Outside footprint: {len(gdf_outside)}")
    print(f"Inside + outside: {split_count}")

    if split_count != original_count:
        raise RuntimeError(
            "Polygons were lost while splitting into "
            "gdf_inside and gdf_outside: "
            f"{original_count} -> {split_count}"
        )

    outside_results = {
        idx: zero_result(float(geom.area) if geom is not None and not geom.is_empty else np.nan, "outside_mask_footprint")
        for idx, geom in zip(gdf_outside.index, gdf_outside.geometry)
    }

    gdf_inside = gdf_inside.sort_values("_polygon_area_m2_sort", ascending=False).copy()

    print("Largest intersecting polygons first:")
    print(
        gdf_inside[["_poly_label", "_polygon_area_m2_sort"]]
        .head(20)
        .assign(area_km2=lambda x: x["_polygon_area_m2_sort"] / 1e6)
        [["_poly_label", "area_km2"]]
        .to_string(index=True)
    )

    items = [
        (idx, geom, poly_label, polygon_area_m2)
        for idx, geom, poly_label, polygon_area_m2 in zip(
            gdf_inside.index,
            gdf_inside.geometry,
            gdf_inside["_poly_label"],
            gdf_inside["_polygon_area_m2_sort"],
        )
    ]

    print("\n--- MULTIPROCESSING INPUT ---")
    print(f"Inside polygons: {len(gdf_inside)}")
    print(f"Items created: {len(items)}")

    if len(items) != len(gdf_inside):
        raise RuntimeError(
            "Not every inside polygon was converted into a worker item: "
            f"{len(gdf_inside)} inside polygons -> {len(items)} items"
        )

    if n_processes is None:
        n_processes = max(1, (os.cpu_count() or 1) - 1)

    inside_results: dict[int, dict] = {}
    if items:
        with Pool(
            processes=n_processes,
            initializer=init_worker,
            initargs=(str(raster_path), band),
        ) as pool:
            results = pool.map(process_one_polygon, items, chunksize=chunksize)

        inside_results = {
            idx: {"polygon_label": poly_label, **metrics}
            for idx, poly_label, metrics in results
        }

    combined_results = {**outside_results, **inside_results}

    print("\n--- COMBINED RESULTS ---")
    print(f"Outside results: {len(outside_results)}")
    print(f"Inside results: {len(inside_results)}")
    print(f"Combined results: {len(combined_results)}")
    print(f"Expected results: {original_count}")

    result_df = pd.DataFrame.from_dict(combined_results, orient="index").sort_index()

    out = gdf.join(result_df.drop(columns=["polygon_label"], errors="ignore"), how="left")
    out = out.drop(columns=["_polygon_area_m2_sort", "_poly_label"], errors="ignore")

    if diagnostics_csv is not None:
        diagnostics_path = Path(diagnostics_csv)
        diagnostics_path.parent.mkdir(parents=True, exist_ok=True)

        diagnostics_df = (
            out.drop(columns="geometry", errors="ignore")
            .sort_values(
                ["runtime_s", "valid_cells_in_window", "window_cells", "polygon_area_m2"],
                ascending=False,
            )
            .copy()
        )

        diagnostics_df.to_csv(diagnostics_path, index=True)
        print(f"Saved diagnostics: {diagnostics_path}")

    return out

def main():
    args = parse_arguments()
    validate_paths(args)

    print("Paths parsed successfully:")
    print(f"Application mask file: {args.application_mask}")
    print(f"Data path: {args.data_path}")

    # Polygons paths
    data_dir = Path(args.data_path)
    application_mask = Path(args.application_mask)
    add_polygons = data_dir / "final_classification_buckets_w_levels.gpkg"
    rgi_polygons = data_dir / "RGI2000-v7.0-G-19_subantarctic_antarctic_islands_with_levels.gpkg"
    irr_polygons = data_dir / "IRRv1_with_levels.gpkg"

    add_polygons_with_coverage = attach_valid_coverage_to_polygons_parallel_pool(
        raster_path=application_mask,
        polygons_path=add_polygons,
        band=1,
        n_processes=8,
        chunksize=1
    )

    RGI_polygons_with_coverage = attach_valid_coverage_to_polygons_parallel_pool(
        raster_path=application_mask,
        polygons_path=rgi_polygons,
        band=1,
        n_processes=8,
        chunksize=1
    )

    IRR_polygons_with_coverage = attach_valid_coverage_to_polygons_parallel_pool(
        raster_path=application_mask,
        polygons_path=irr_polygons,
        band=1,
        n_processes=8,
        chunksize=1
    )

    output_dir = application_mask.parent

    fname_ADD = application_mask.stem + "_ADD_coverage.gpkg"
    fpath_ADD = output_dir / fname_ADD

    fname_RGI = application_mask.stem + "_RGI_coverage.gpkg"
    fpath_RGI = output_dir / fname_RGI

    fname_IRR = application_mask.stem + "_IRR_coverage.gpkg"
    fpath_IRR = output_dir / fname_IRR

    add_polygons_with_coverage.to_file(fpath_ADD, driver="GPKG")
    RGI_polygons_with_coverage.to_file(fpath_RGI, driver="GPKG")
    IRR_polygons_with_coverage.to_file(fpath_IRR, driver="GPKG")

    print(f"All saved")
    print("Done!")

if __name__ == "__main__":
    main()
