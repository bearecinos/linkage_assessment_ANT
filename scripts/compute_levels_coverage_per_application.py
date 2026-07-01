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
from multiprocessing import Pool
from pathlib import Path
import argparse

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds
from shapely.geometry import box

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
    }


def cell_polygon_from_rowcol(transform, row: int, col: int):
    """
    This converts one raster cell into a Shapely polygon
    :param transform:
    :param row:
    :param col:
    :return:
    """
    x_left, y_top = transform * (col, row)  # Gets the upper left corner of the cell
    x_right, y_bottom = transform * (col + 1, row + 1)  # gets the lower right corner
    # Now we build the rectangular box of that pixel
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


def init_worker(raster_path: str, band: int):
    global _SRC, _BAND
    _SRC = rasterio.open(raster_path)
    _BAND = band


def compute_single_polygon_valid_coverage(src,
                                          geom,
                                          band: int = 1):
    """
    Measures how much of one vector polygon is covered by raster cells where the mask equals 1

    """
    if geom is None or geom.is_empty or not geom.is_valid:
        return empty_result()

    polygon_area_m2 = float(geom.area)
    if polygon_area_m2 <= 0:
        return empty_result()

    # finds the raster subset for this polygon to speed up computation
    minx, miny, maxx, maxy = geom.bounds

    window = from_bounds(
        left=minx,
        bottom=miny,
        right=maxx,
        top=maxy,
        transform=src.transform,
    ).round_offsets().round_lengths()

    # Adjust the raster window so it stays fully inside the raster extent before reading any data.

    row_off = max(0, int(window.row_off))
    col_off = max(0, int(window.col_off))
    height = min(src.height - row_off, int(window.height))
    width = min(src.width - col_off, int(window.width))

    if height <= 0 or width <= 0:
        return empty_result()

    # builds the final pixel window
    # reads only that part of the raster as a masked array
    window = rasterio.windows.Window(
        col_off=col_off,
        row_off=row_off,
        width=width,
        height=height,
    )

    arr = src.read(band, window=window, masked=True)
    if arr.size == 0:
        return empty_result()

    valid_mask = valid_data_mask(arr)
    valid_rows, valid_cols = np.where(valid_mask)

    if len(valid_rows) == 0:
        return {
            "polygon_area_m2": polygon_area_m2,
            "covered_area_m2": 0.0,
            "covered_area_km2": 0.0,
            "valid_cell_count": 0,
            "percent_coverage": 0.0,
        }

    window_transform = src.window_transform(window)

    covered_area_m2 = 0.0
    intersecting_valid_cell_count = 0

    # convert valid raster cells to polygons and intersect
    for row, col in zip(valid_rows, valid_cols):
        cell = cell_polygon_from_rowcol(window_transform, int(row), int(col))

        if not cell.intersects(geom):
            continue

        inter = cell.intersection(geom)
        if not inter.is_empty:
            covered_area_m2 += inter.area
            intersecting_valid_cell_count += 1

    # compute percent coverage
    percent_coverage = 100.0 * covered_area_m2 / polygon_area_m2

    return {
        "polygon_area_m2": polygon_area_m2,
        "covered_area_m2": float(covered_area_m2),
        "covered_area_km2": float(covered_area_m2 / 1e6),
        "valid_cell_count": int(intersecting_valid_cell_count),
        "percent_coverage": float(percent_coverage),
    }
    # polygon_area_m2: full vector area of the polygon
    # covered_area_m2: exact area covered by valid raster cells
    # covered_area_km2: same in km²
    # valid_cell_count: number of valid raster cells that intersect the polygon
    # percent_coverage: percent of polygon covered


def process_one_polygon(item):
    idx, geom = item
    result = compute_single_polygon_valid_coverage(_SRC, geom, band=_BAND)
    return idx, result


def attach_valid_coverage_to_polygons_parallel_pool(raster_path,
                                                    polygons_path,
                                                    band: int = 1,
                                                    n_processes: int | None = None,
                                                    chunksize: int = 50):
    gdf = gpd.read_file(polygons_path)

    with rasterio.open(raster_path) as src:
        if gdf.crs is None:
            raise ValueError("Input polygons have no CRS.")
        if gdf.crs != src.crs:
            gdf = gdf.to_crs(src.crs)

    gdf = gdf.copy()
    items = list(gdf.geometry.items())

    if n_processes is None:
        n_processes = max(1, (os.cpu_count() or 1) - 1)

    with Pool(
            processes=n_processes,
            initializer=init_worker,
            initargs=(str(raster_path), band),
    ) as pool:
        results = pool.map(process_one_polygon, items, chunksize=chunksize)

    result_df = pd.DataFrame(
        {idx: metrics for idx, metrics in results}
    ).T.sort_index()

    result_df.index = gdf.index
    return gdf.join(result_df)

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
    rgi_polygons = data_dir / "RGI-GCv7_with_levels.gpkg"
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
