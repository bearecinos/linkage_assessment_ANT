## Spatial Linkage Assessment for Glacier Complexes (RGIv7 - Antarctic Region)

This script assesses spatial linkages between glacier complexes (from RGIv7) and major Antarctic ice types including ice sheets, shelves, tongues, rises, and rumples. It produces:

- `inter_mask.gpkg`: A primary interaction mask combining ice sheet, ice shelf, and ice tongue extents.
- `linkage.gpkg`: A merged assessment layer combining glacier complexes and ice rises/rumples (deduplicated).

### 📁 Input Files

- --coast-file: High-resolution Antarctic coast and surface type shapefile (must include surface column).

- --glacier-complex: RGIv7 shapefile for region C-19 (Antarctic & Subantarctic Islands).

- --data-path: Output directory for generated GPKG files.

- --ice-rumples: Ice rises and rumples shapefile (must include `Area_km2` and geometry).

### 📦 Dependencies

```bash
pip install geopandas pandas numpy shapely pyproj
```

### 🔧 Usage for `Add_ice_rumple_rises_to_RGI-GC.py`

```bash
python Add_ice_rumple_rises_to_RGI-GC.py \
  --coast-file ../add_coastline_high_res_polygon_v7_10.shp/add_coastline_high_res_polygon_v7_10.shp \
  --glacier-complex ../RGI2000-v7.0-C-19_subantarctic_antarctic_islands/RGI2000-v7.0-C-19_subantarctic_antarctic_islands.shp \
  --data-path ../linkage_assessment_ANT/data_files \
  --ice-rumples ../Ice_rise_rumples/icerises_inventory_v1.shp
```

### 🔧 Usage for `linkage_assesment.py`
