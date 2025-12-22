## Spatial linkage assessment for Glacier Complexes (Antarctic Region, RGIv7)

This project performs a spatial analysis to determine the degree of spatial connectivity or "detachment" between glacier complexes and surrounding Antarctic ice types—namely the ice sheet, ice shelves and ice tongues. The linkage classification is also done on Ice rises and rumples.

The workflow consists of two scripts:

- `preprocess_ice_types_masks.py`: generates core geospatial masks for analysis.
- `compute_spatial_linkage.py`: computes connectivity ("detachment") scores between each polygon and surrounding ice using shared perimeter logic.

### Dependencies

```bash
pip install geopandas pandas numpy shapely pyproj
```

or check the `environment.yml`

### Input Files

- `--coast_file`: High-resolution Antarctic coast and surface type shapefile, [click here for download](https://data.bas.ac.uk/full-record.php?id=GB/NERC/BAS/PDC/01391).

- `--glacier_complex`: Randolph Glacier Inventory (v7) region 19 (Antarctic & Subantarctic Islands Glacier complexes product), [click here for download](https://nsidc.org/data/nsidc-0770/versions/7). More on RGI info [here](https://www.glims.org/rgi_user_guide/welcome.html).

- `--data_path`: Output directory for generated GPKG files.

- `--ice_rumples`: Ice rises and rumples database v1.0 (Moholdt, G., & Matsuoka, K. 2015), [click here for download](https://data.npolar.no/dataset/9174e644-3540-44e8-b00b-c629acbf1339).

### Workflow overview

1. Preprocessing step – `preprocess_ice_types_masks.py`
This script builds the following:
    - Interaction mask, a merged mask including:
      - The Antarctic Ice Sheet
      - Mainland-connected ice shelves
      - Ice tongues
      - Ice rumples (from ADD)
      This represents the continuous, connected floating or grounded ice originating from the main ice sheet.
    - Secondary ice shelf mask:
      - We save those ice shelves not coming from the mainland in a different mask for post-processing and to assing a buttressing score to those classified as detached but within an ice shelf. 
    - Assessment mask:
      - Matches coastline polygons with either glacier complexes or ice rises/rumples, saves `rgi_id` or `id_icerise`.
      - Each polygon is assigned an `analysis_id` for traceability.

Outputs:
- `interaction_mask.gpkg`: a unified polygon for the main ice sheet, ice shelves and ice tongues.
- `remaining_shelves_mask.gpkg`: island shelves
- `ADD_polys_with_RGI-GCv7_IRRv1_combined.gpkg`: Combined coastline sections which are polygons representing glacier complexes, ice rises and rumples for linkage analysis.

#### Usage
```bash
python preprocess_ice_types_masks.py \
  --coast_file path/to/add_coastline.shp \
  --glacier_complex path/to/rgi_glacier_complexes.shp \
  --ice_rumples path/to/ice_rumples.shp \
  --data_path path/to/output_directory
```

2. Linkage classification – `compute_spatial_linkage.py`
This script performs two main operations:

🌟 Primary detachment scoring:
- Computes how much of each polygon in the assessment mask overlaps via its perimeter with the interaction mask.
- Assigns a detachment score ranging from 1.1 (strong linkage) to 2.0 (completely detached) based on the percentage of shared perimeter.

🧊 Buttress classification:
- For polygons scored as 2.0 (fully detached), a secondary analysis is performed:
  - Checks whether they are buttressed by island-sourced ice shelves (`remaining_shelves_mask.gpkg`).
  - Assigns a buttress code to indicate this context.

- **Classification scheme (to be discussed)**

**Please provide community feedback on the [Github Issues](https://github.com/bearecinos/linkage_assessment_ANT/issues)**

| Perimeter Overlap (%)   | Detachment Score | Description    | Buttress Code | Buttress Source ID               |
| ----------------------- | ---------------- | -------------- | ------------- | -------------------------------- |
| 90–100%                 | 1.1              | Strong linkage | None          | Buttressed by mainland ice       |
| 80–90%                  | 1.2              |                | None          | Buttressed by mainland ice       |
| ...                     | ...              | ...            | None          | ...                              |
| 10–20%                  | 1.9              | Weak linkage   | None          | Buttressed by mainland ice       |
| 0% (no overlap)         | 2.0              | Fully detached | 1.0           | Buttressed by non-mainland shelf |
| 0% + ≤10% shelf overlap | 2.0              | Fully detached | 0.0           | Unclassified (no buttress)       |


Outputs:
- `final_classification_buckets.gpkg`: GeoPackage with one row per polygon, including its `detachment_score` and `detachment_description` and original RGI-ids or IRR-ids.

#### Usage
```bash
python compute_spatial_linkage.py \
  --data_path path/to/output_directory
```

Contributors of data, guidance or code
------------
Celia Baumhoer, Beatriz Recinos Rivas, Bertie Miles, Fabien Maussion, Ken Mankoff and Regine Hock
