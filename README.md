## Spatial Linkage Assessment for Glacier Complexes (Antarctic Region, RGIv7)

This project performs a spatial analysis to determine the degree of interaction or "detachment" between glacier complexes and surrounding Antarctic ice types—namely the ice sheet, ice shelves and ice tongues. The linkage classification is also done on Ice rises and rumples.

The workflow consists of two scripts:

- `preprocess_ice_types_masks.py`: prepares the interaction and assessment masks.
- `compute_spatial_linkage.py`: performs the linkage classification based on polygons with shared perimeter ratios.

### Dependencies

```bash
pip install geopandas pandas numpy shapely pyproj
```

or check the `environment.yml`

### Input Files

- `--coast-file`: High-resolution Antarctic coast and surface type shapefile, [click here for download](https://data.bas.ac.uk/full-record.php?id=GB/NERC/BAS/PDC/01391).

- --glacier-complex: Randolph Glacier Inventory (v7) region 19 (Antarctic & Subantarctic Islands Glacier complexes product), [click here for download](https://nsidc.org/data/nsidc-0770/versions/7). More on RGI info [here](https://www.glims.org/rgi_user_guide/welcome.html).

- --data-path: Output directory for generated GPKG files.

- --ice-rumples: Ice rises and rumples database v1.0 (Moholdt, G., & Matsuoka, K. 2015), [click here for download](https://data.npolar.no/dataset/9174e644-3540-44e8-b00b-c629acbf1339).

### Workflow overview

1. Preprocessing step – `preprocess_ice_types_masks.py`

This script:
- Extracts and merges the primary ice types (ice sheet, shelf, tongue) into a unified interaction mask.
- Matches coastline polygons with either glacier complexes or ice rises/rumples.
- Produces a single combined assessment layer (`ADD_polys_with_RGI-GCv7_IRRv1_combined.gpkg`) used for linkage computation, each polygon can be track back to the RGI-ids and IRR-ids.

Outputs:
- `interaction_mask.gpkg`: a unified polygon for the main ice sheet, ice shelves and ice tongues.
- `ADD_polys_with_RGI-GCv7_IRRv1_combined.gpkg`: Combined coastline sections which are polygons representing glacier complexes, ice rises and rumples for linkage analysis.

#### Usage
```bash
python preprocess_ice_types_masks.py \
  --coast-file path/to/add_coastline.shp \
  --glacier-complex path/to/rgi_glacier_complexes.shp \
  --data-path path/to/output_directory \
  --ice-rumples path/to/ice_rumples.shp
```

2. Linkage classification – `compute_spatial_linkage.py`

This script:
- Computes how much of each polygon in the assessment mask overlaps via its perimeter with the interaction mask.
- Assigns each polygon a detachment score and description based on shared perimeter percentage and user defined thresholds.

- **Classification scheme (to be discussed)**

| Detachment score | Description                                               |
|------------------|-----------------------------------------------------------|
| **1.0**          | Strong linkage (high shared perimeter ratio)              |
| **1.5**          | Partial linkage (moderate shared perimeter ratio)         |
| **1.7**          | Weak linkage (low shared perimeter ratio)                 |
| **2.0**          | Completely detached (no shared perimeter with the interaction mask) |

Thresholds for “weak” and “medium” linkage are configurable at runtime via `--weak_max` and `--medium_max` params. 

Outputs:
- `final_classification_weak{X}_medium{Y}.gpkg`: GeoPackage with one row per polygon, including its detachment_score and detachment_description and original RGI-ids or IRR-ids.

#### Usage
```bash
python compute_spatial_linkage.py \
  --data_path path/to/output_directory \
  --weak_max 10 \
  --medium_max 50
```

Contributors of data, guidance or code
------------
Celia Baumhoer, Beatriz Recinos Rivas, Bertie Miles, Fabien Maussion, Ken Mankoff and Regine Hock
