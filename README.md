## Attachment levels of Antarctic and Sub-Antarctic ice bodies from the Ice Sheet: a perimeter-overlap classification

This project performs a spatial analysis to determine the degree of spatial connectivity or "attachment" between glacier complexes and surrounding Antarctic ice types—namely the ice sheet, ice shelves and ice tongues. The linkage classification is also done on Ice rises and rumples.

The workflow consists of two scripts:

- `preprocess_ice_types_masks.py`: generates core geospatial masks for analysis.
- `compute_spatial_linkage.py`: computes connectivity (attachment and detachment scores) between each polygon and surrounding ice (ice sheet and ice shelves) using shared perimeter logic.
- `compute_levels_detachment.py`: quantifies how attached peripheral Antarctic coastal polygons are to the ice-sheet and ice-shelf interaction mask by grouping high-resolution attachment scores into attachment levels 0–3 using community-derived thresholds; the script also supports sensitivity studies to test how different threshold groupings affect polygon classification.

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

## 1. Preprocessing step – `preprocess_ice_types_masks.py`

This script builds the following:

- Interaction mask, a merged mask including:
  - The Antarctic Ice Sheet
  - Mainland-connected ice shelves
  - Ice tongues

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

## 2. Linkage classification – `compute_spatial_linkage.py`

This script performs the **core spatial linkage assessment** by quantifying how much of each polygon’s perimeter is shared with the main Antarctic ice system.

### Key components

The analysis combines three masks:

- #### Interaction mask (`interaction_mask.gpkg`)
   Used as the reference for spatial linkage  

- #### Secondary shelf mask (`remaining_shelves_mask.gpkg`)
   Ice shelves originating from islands (not connected to the mainland). Used to evaluate buttressing of detached features  

#### Assessment mask (`ADD_polys_with_RGI-GCv7_IRRv1_combined.gpkg`)
Polygons to evaluate:
- Glacier complexes (RGIv7)  
- Ice rises and rumples (IRRv1)  

---

### Method

#### Perimeter-overlap linkage

For each polygon, the script computes:
- Total perimeter  
- Shared perimeter with the interaction mask (buffered)  

The percentage overlap is converted into **attachment scores**, while an equivalent **detachment scale** is also computed to provide an alternative framing of the same relationship.:

- **Classification scheme (to be discussed)**

**Please provide community feedback on the [Github Issues](https://github.com/bearecinos/linkage_assessment_ANT/issues)**

| Perimeter Overlap (%)   | Detachment Score | Attachment Score | Description (attachment-oriented) | Buttress Code | Buttress Source ID               |
| ----------------------- | ---------------: | ---------------: | --------------------------------- | ------------- | -------------------------------- |
| 100%                    |              1.0 |              1.0 | Fully attached (ice sheet)        | None          | Buttressed by mainland ice       |
| 90–99%                  |              1.1 |              0.9 | Strong attachment                 | None          | Buttressed by mainland ice       |
| 80–90%                  |              1.2 |              0.8 | Strong attachment                 | None          | Buttressed by mainland ice       |
| 70–80%                  |              1.3 |              0.7 | Moderate attachment               | None          | Buttressed by mainland ice       |
| 60–70%                  |              1.4 |              0.6 | Moderate attachment               | None          | Buttressed by mainland ice       |
| 50–60%                  |              1.5 |              0.5 | Moderate attachment               | None          | Buttressed by mainland ice       |
| 40–50%                  |              1.6 |              0.4 | Moderate/weak attachment          | None          | Buttressed by mainland ice       |
| 30–40%                  |              1.7 |              0.3 | Weak attachment                   | None          | Buttressed by mainland ice       |
| 20–30%                  |              1.8 |              0.2 | Weak attachment                   | None          | Buttressed by mainland ice       |
| 1–20%                   |              1.9 |              0.1 | Weak attachment                   | None          | Buttressed by mainland ice       |
| 0% (no overlap)         |              2.0 |              0.0 | Fully detached                    | 1.0           | Buttressed by non-mainland shelf |
| 0% + ≤10% shelf overlap |              2.0 |              0.0 | Fully detached                    | 0.0           | Unclassified (no buttress)       |

Notes / conventions
- Perimeter Overlap is computed as percentage of the assessment polygon’s perimeter that touches the primary interaction mask.
- Score uses the decimal detachment scheme: 1.0 … 1.9 (progressively weaker linkage), 2.0 = fully detached.
- Buttress Code: 1.0 = buttressed by a non-mainland (island-origin) shelf; 0.0 = not buttressed / unclassified.
- An attachment score is introduced to convert categories into connectivity levels comparable to Greenland. **Computed to provide an alternative framing of the same relationship.**

Outputs:
- `final_classification_buckets.gpkg`: GeoPackage with one row per polygon, including its `attachment_score`, a `descriptor` and original RGI-ids or IRR-ids.

#### Usage
```bash
python compute_spatial_linkage.py --data_path .../linkage_assessment_ANT/data_files
```

## 3. Attachment levels – `compute_level_detachment.py`

This script converts detachment scores into **discrete attachment levels (0–3)** and propagates them to glacier complexes and ice rises.

---

### Level definition (reversed convention)

| Level | Meaning               | Detachment Score | Attachment Score |
|------|----------------------|------------------|------------------|
| 3    | Attached (ice sheet) | 1.0              | 1.0              |
| 2    | Strong attachment    | 1.1 – 1.6        | 0.9 – 0.4        |
| 1    | Weak attachment      | 1.7 – 1.9        | 0.3 – 0.1        |
| 0    | Fully detached       | 2.0              | 0.0              |

This follows a **“most attached = highest level”** logic in line with Greenland levels.

---

### Method

#### 1. Initial classification
- Uses `final_classification_buckets.gpkg`  
- Assigns polygons into Levels 0–3 based on score thresholds  

#### 2. Indirect connectivity (key feature ⭐)
- Some **Level 0 (detached)** polygons may still be indirectly connected via ice shelves  
- A **two-step buffered interaction mask** is constructed:
  - Includes Levels 1–3  
  - Then expands to include intersecting shelves  

**Result:**  
- Detached polygons intersecting this mask → reclassified to Level 1  

This preserves **indirect ice-flow pathways via shelves.**

---

#### 3. Final level assignment

Each polygon receives:
- `level` (0–3)  
- `level_text`  

---

#### 4. Aggregation to datasets

Levels are propagated to:

**RGI glacier complexes**
- Exploded by `rgi_id`

**Ice rises & rumples (IRRv1)**
- Same logic applied per `id_icerise`  

---

### Outputs

- `final_classification_buckets_w_levels.gpkg`  
  → Polygon-level results with levels  

- `RGI-GCv7_with_levels.gpkg`  
  → Glacier complexes with assigned level  

- `IRRv1_with_levels.gpkg`  
  → Ice rises and rumples with assigned level  

**Summary statistics:**
- `level_area_stats.csv`  
- `level_area_metrics.csv`  

---


### Usage

```bash
python compute_level_detachment.py \
  --data_path path/to/output_directory \
  --coast_file path/to/add_coastline.shp \
  --glacier_complex path/to/rgi_glacier_complexes.shp \
  --ice_rumples path/to/ice_rumples.shp


Contributors of data, guidance or code
------------
Celia Baumhoer, Beatriz Recinos Rivas, Bertie Miles, Mathieu Morlighem, Fabien Maussion, Ken Mankoff and Regine Hock

Citation
--------
Recinos, B., & Baumhoer, C. (2026). bearecinos/linkage_assessment_ANT: Spatial linkage assessment for Glacier Complexes and Ice rises and rumples (Antarctic Region, RGIv7) — Pre-release (v1.0.0-beta.1). Zenodo.
[![DOI](https://zenodo.org/badge/1091078829.svg)](https://doi.org/10.5281/zenodo.18175061)
