# Codebase Analysis: Mangrove-to-Aquaculture Transition Detection System

## Project Overview

A **physics-guided, multi-sensor, object-based** pipeline built on **Google Earth Engine (GEE)** that detects and tracks the conversion of mangrove forests into aquaculture ponds across a **5-stage lifecycle** (S1→S5). The study area is the **Coringa Wildlife Sanctuary / Godavari Delta**, Andhra Pradesh, India — a well-documented mangrove→shrimp-pond conversion hotspot.

**Time span:** 1988–2025 (historical mode), or near-real-time (operational mode)

---

## Architecture (15 Modules)

| Module | File | Role |
|--------|------|------|
| M0  | [m00_aoi.py](file:///m:/DIP/DIP/modules/m00_aoi.py) | AOI initialization, elevation mask, GeoJSON export |
| M1  | [m01_acquisition.py](file:///m:/DIP/DIP/modules/m01_acquisition.py) | Multi-sensor data loading (Landsat 5/7, HLS L30/S30, Sentinel-1, JRC, GLO30, GMW, SoilGrids) |
| M2  | [m02_optical_preprocess.py](file:///m:/DIP/DIP/modules/m02_optical_preprocess.py) | Cloud masking, band renaming, radiometric scaling, SLC gap fill, Roy 2016 harmonization |
| M3  | [m03_sar_preprocess.py](file:///m:/DIP/DIP/modules/m03_sar_preprocess.py) | Border noise masking, dB→linear, Sigma0→Gamma0, Enhanced Lee Sigma speckle filter, tidal normalization |
| M4  | [m04_normalization.py](file:///m:/DIP/DIP/modules/m04_normalization.py) | Safe spatial normalization (no forced reproject), composite creation (median/p25/mean/mosaic) |
| M5  | [m05_features.py](file:///m:/DIP/DIP/modules/m05_features.py) | 13+ spectral indices, SAR features, edge density, temporal features, elevation, JRC water history |
| M6  | [m06_stage_engine.py](file:///m:/DIP/DIP/modules/m06_stage_engine.py) | Rule-based 5-stage pixel classifier with adaptive thresholds, CCDC integration, transition constraints |
| M7  | [m07_polygons.py](file:///m:/DIP/DIP/modules/m07_polygons.py) | Water detection → connected components → polygon extraction with shape filters |
| M8  | [m08_validator.py](file:///m:/DIP/DIP/modules/m08_validator.py) | 5-component weighted validation (mangrove, water, elevation, SAR, multi-evidence) |
| M9  | [m09_alerts.py](file:///m:/DIP/DIP/modules/m09_alerts.py) | Persistence engine + alert generation for confirmed stage transitions |
| M10 | [m10_web_export.py](file:///m:/DIP/DIP/modules/m10_web_export.py) | Thumbnail download (12 layer types), JSON export, DIP image enhancement |
| M11 | [m11_accuracy.py](file:///m:/DIP/DIP/modules/m11_accuracy.py) | Accuracy assessment vs GMW (mangrove) and JRC (water), confusion matrix |
| M12 | [m12_pond_registry.py](file:///m:/DIP/DIP/modules/m12_pond_registry.py) | Per-pond persistent tracking, lifecycle stage history, transition constraints |
| M13 | [m13_object_matcher.py](file:///m:/DIP/DIP/modules/m13_object_matcher.py) | Cross-epoch centroid-based (Haversine) spatial matching of pond polygons |
| M14 | [m14_per_pond_classifier.py](file:///m:/DIP/DIP/modules/m14_per_pond_classifier.py) | Per-object multi-evidence scoring classifier (S1–S5) + confidence scorer |
| DIP | [dip_pipeline.py](file:///m:/DIP/DIP/dip_pipeline.py) | Digital Image Processing: Bilateral filter → CLAHE → Unsharp mask |

---

## Complete Preprocessing Inventory

### 1. Optical Preprocessing (m02)

| Step | Technique | Details |
|------|-----------|---------|
| **Cloud masking** | QA_PIXEL bit-masking | Bits 1 (dilated cloud), 3 (cloud), 4 (cloud shadow) for Landsat; Bits 1 (cloud), 3 (shadow) for HLS Fmask |
| **Band renaming** | Sensor-specific mapping | L5: SR_B1→blue…SR_B7→swir2; L7: same; HLS L30: B2→blue…B7→swir2; HLS S30: B2→blue…B8A→nir |
| **Radiometric scaling** | DN → surface reflectance | Landsat: `DN × 0.0000275 − 0.2`, clamp [0, 1]; HLS: `DN / 10000`, clamp [0, 1] |
| **SLC-off gap filling** | Two-pass focal median | L7 post-2003: Pass 1 radius=10px, Pass 2 radius=20px (circle kernel); preserves class boundaries |
| **Cross-sensor harmonization** | Roy et al. 2016 coefficients | Linear transform (slope + intercept) per band to align L5/L7 with HLS/OLI baseline |
| **Cloud gap filling** | Focal median interpolation | Radius=2px, fills small residual cloud gaps |
| **Export fill** | Type-aware NoData replacement | Optical index→0.0, RGB→0.02, SAR→−25dB, DEM→0.0 (visualization only) |

### 2. SAR Preprocessing (m03)

| Step | Technique | Details |
|------|-----------|---------|
| **Border noise masking** | Incidence angle + intensity filter | Mask angle < 29° or > 46°; mask VV < −40 dB or VH < −40 dB |
| **dB → linear conversion** | Power transformation | `linear = 10^(dB/10)` |
| **Sigma0 → Gamma0** | Incidence angle correction | `Gamma0 = Sigma0 / cos(θ)` — removes flat-terrain incidence bias |
| **Speckle filtering** | Enhanced Lee Sigma filter | Operates in linear domain; ENL=4.4; adaptive weight = `1 − (V_speckle / V_local)`; preserves edges |
| **linear → dB conversion** | Logarithmic | `dB = 10 × log10(linear)` |
| **Tidal normalization** | Water-fraction-based filtering | Computes per-image water fraction (VV < −14 dB); retains only images ≤ median water fraction (low tide) |

### 3. Spatial Normalization (m04)

| Step | Technique | Details |
|------|-----------|---------|
| **AOI clipping** | Geometry clip | No forced `.reproject()` (avoids bounding-box artifacts) |
| **Temporal compositing** | Multiple methods | `median` (default), `p25` (tidal suppression), `mean`, `mosaic` |
| **Empty collection guard** | Server-side safe | Injects dummy image if collection is empty |
| **Sensor-specific resampling** | Type-aware | Bilinear for optical/SAR/DEM; nearest for categorical (JRC monthly, GMW) |

### 4. Digital Image Processing — DIP Pipeline (dip_pipeline.py)

| Step | Technique | Details |
|------|-----------|---------|
| **Noise reduction** | Bilateral filter | `d=9, sigmaColor=75, sigmaSpace=75` — edge-preserving |
| **Contrast enhancement** | CLAHE | `clipLimit=2.0, tileGridSize=(8,8)` — applied on L-channel of LAB color space |
| **Edge sharpening** | Unsharp mask | `1.5×original − 0.5×Gaussian(9×9, σ=10)` |

### 5. Dry-Season Temporal Filtering (m01)

| Step | Technique | Details |
|------|-----------|---------|
| **Seasonal constraint** | Calendar month filter | Nov–Mar (months 11, 12, 1, 2, 3) for Godavari Delta dry season |
| **Multi-temporal gap filling** | ±1 year window | Each epoch uses a 3-year window to maximize cloud-free pixel availability |

---

## Complete Feature Engineering Inventory (m05)

### Vegetation Indices (8)
| Index | Formula | Purpose |
|-------|---------|---------|
| **NDVI** | [(NIR−Red)/(NIR+Red)](file:///m:/DIP/DIP/modules/m12_pond_registry.py#30-34) | Canopy density, primary mangrove indicator |
| **EVI** | `2.5×(NIR−Red)/(NIR+6×Red−7.5×Blue+1)` | Atmospheric-corrected vegetation |
| **NDMI** | [(NIR−SWIR2)/(NIR+SWIR2)](file:///m:/DIP/DIP/modules/m12_pond_registry.py#30-34) | Canopy moisture content |
| **SAVI** | [(NIR−Red)×1.5/(NIR+Red+0.5)](file:///m:/DIP/DIP/modules/m12_pond_registry.py#30-34) | Soil-adjusted vegetation (sparse canopy) |
| **GNDVI** | [(NIR−Green)/(NIR+Green)](file:///m:/DIP/DIP/modules/m12_pond_registry.py#30-34) | Early chlorophyll stress |
| **CMRI** | `NDVI − NDWI` | Canopy Mangrove Recognition Index |
| **MMRI** | `|MNDWI|/(|MNDWI|+|NDVI|)` | Modular Mangrove Recognition Index |
| **NDBI** | [(SWIR1−NIR)/(SWIR1+NIR)](file:///m:/DIP/DIP/modules/m12_pond_registry.py#30-34) | Built-up / bare soil / embankments |

### Water Indices (4)
| Index | Formula | Purpose |
|-------|---------|---------|
| **NDWI** | [(Green−NIR)/(Green+NIR)](file:///m:/DIP/DIP/modules/m12_pond_registry.py#30-34) | Water body detection |
| **MNDWI** | [(Green−SWIR1)/(Green+SWIR1)](file:///m:/DIP/DIP/modules/m12_pond_registry.py#30-34) | Modified water (suppresses built-up) |
| **AWEI** | `4×(Green−SWIR1)−0.25×NIR−2.75×SWIR2` | Shadow-resistant water extraction |
| **Veg-Water Diff** | `NDVI − MNDWI` | Vegetation vs water separator |

### SAR Features (6)
| Feature | Method | Purpose |
|---------|--------|---------|
| **VV mean** | dB backscatter | Water/vegetation discrimination |
| **VH mean** | dB backscatter | Cross-pol canopy volume |
| **VV/VH ratio** | dB subtraction | Cross-pol ratio |
| **VV texture** | Local stdDev (3×3) | Surface roughness |
| **RVI** | `4×VH_lin/(VV_lin+VH_lin)` | Radar Vegetation Index |
| **VV homogeneity** | GLCM IDM (3×3) | Smooth water surfaces |

### Temporal Features (6)
| Feature | Method | Purpose |
|---------|--------|---------|
| **NDVI trend slope** | Linear regression over time | Canopy degradation signal |
| **NDVI CV** | Coefficient of variation | Canopy instability/disturbance |
| **VV temporal mean** | Time-series mean | Baseline backscatter |
| **VV temporal std** | Time-series stdDev | Seasonal flooding changes |
| **VH temporal mean** | Time-series mean | Cross-pol baseline |
| **VH/VV ratio mean** | Time-series mean | Stability indicator |

### Contextual/Ancillary Features (6)
| Feature | Source | Purpose |
|---------|--------|---------|
| **JRC occurrence** | JRC Global Surface Water | Long-term water persistence (%) |
| **JRC seasonality** | JRC GSW | Months/year with water |
| **JRC change** | JRC GSW | Absolute water change |
| **Low elevation mask** | GLO30 DEM | Elevation < 10m constraint |
| **Slope** | GLO30 DEM terrain | Flat terrain = pond-feasible |
| **TWI proxy** | [(10−elevation)/(slope+0.01)](file:///m:/DIP/DIP/modules/m12_pond_registry.py#30-34) | Topographic Wetness Index proxy |

### Shape/Edge Features (1+3)
| Feature | Method | Purpose |
|---------|--------|---------|
| **Edge density** | Canny edge on NDVI + focal mean | Rectilinear pond boundaries |
| **Rectangularity** | Area / BBox area | Geometric regularity |
| **Compactness** | 4π×Area / Perimeter² | Shape circularity |
| **Elongation** | Perimeter / √Area | River vs pond discrimination |

> **Total: 34+ features** extracted per pixel/polygon

---

## Comparison with Similar Works

| Aspect | Existing Literature | **This Project (DIP)** |
|--------|-------------------|----------------------|
| **Classification** | Typically ML-based (Random Forest, SVM, deep learning U-Net) producing binary or multi-class land cover maps | **Physics-guided rule-based** 5-stage lifecycle classifier with literature-calibrated thresholds — no training data needed |
| **Temporal scope** | Typically postclassification change detection between 2–3 dates | **Multi-epoch historical stream** (1988–2022) + **operational near-real-time** mode |
| **Object tracking** | Pixel-based or static polygon comparison | **Persistent object-based tracking** with cross-epoch centroid matching (PondRegistry) + lifecycle history per pond |
| **Stage model** | Binary (mangrove/non-mangrove) or simple 3-class | **5-stage biophysical model**: S1 (intact mangrove) → S2 (degradation) → S3 (clearing) → S4 (water filling) → S5 (operational pond) |
| **Sensor fusion** | Optical only OR SAR only OR simple composite | **Optical + SAR + DEM + JRC + GMW + SoilGrids** fused at feature level with 34+ features |
| **SAR preprocessing** | Basic speckle filter, often in dB domain | **Enhanced Lee Sigma filter in linear domain** → Gamma0 conversion → **tidal normalization** via water-fraction filtering |
| **Change detection** | Map comparison or simple differencing | **CCDC temporal segmentation** + epoch-to-epoch NDVI/MNDWI trend + persistence engine with biologically valid transition constraints |
| **Validation** | Confusion matrix from field points | **5-component weighted automated validation** (mangrove, water, elevation, SAR, multi-evidence) + GMW/JRC cross-comparison |
| **Alert system** | None | **Real-time alert engine** with per-object persistence + biologically valid transition gates |
| **Image enhancement** | None | **DIP pipeline** (bilateral filter → CLAHE → unsharp mask) for output visualization |
| **Cross-sensor harmony** | Raw comparison across sensors | **Roy et al. 2016 harmonization** (L5/L7 → OLI); **NASA HLS v2.0** (pre-harmonized L8/S2) |

---

## Key Novelty Aspects

### 1. 🔬 Physics-Guided 5-Stage Lifecycle Model
Unlike ML-based approaches that require labeled training data and produce static classifications, this system models the **biophysical conversion process** as a 5-stage state machine:
- **S1 → S2 → S3 → S4 → S5** with enforced transition constraints
- Each stage has **multi-evidence thresholds** calibrated from published literature (Li et al., Xu 2006, Feyisa 2014, Sun et al. 2020)
- No training data required — fully governed by physical spectral/SAR indicators

### 2. 🏗️ Object-Based Persistent Pond Tracking
No other known system maintains a **per-pond registry** across decades:
- Each pond has a unique ID, stage history, area history, confidence history, and alert history
- Cross-epoch **Haversine centroid matching** re-identifies ponds across time gaps of 5–10 years
- Per-object persistence engine ensures transitions are **biologically valid** before confirmation

### 3. 📡 7-Sensor Fusion (34+ Features)
Most studies use 1–2 sensors. This system fuses:
- **4 optical:** Landsat 5, Landsat 7, HLS L30 (Landsat), HLS S30 (Sentinel-2)
- **1 SAR:** Sentinel-1 GRD
- **3 ancillary:** JRC Global Surface Water, Copernicus GLO30 DEM, Global Mangrove Watch
- Plus optional SoilGrids

### 4. 🌊 Tidal-Aware SAR Preprocessing
Unique tidal normalization: computes per-image water fraction over the AOI, then retains only low-tide images (≤ median water fraction). Combined with p25 compositing, this suppresses transient tidal inundation — a critical issue in the Godavari Delta that other studies address poorly.

### 5. 🔄 CCDC Temporal Segmentation Integration
Uses Google Earth Engine's native **CCDC algorithm** to detect structural breaks in NDVI/MNDWI time series. CCDC breaks are used to **reinforce stage transitions**: where a structural break coincides with water signatures, pixels are promoted from S1/S2 → S4 (or S3). This is a unique integration of CCDC into a stage-based classification framework.

### 6. 🖼️ DIP Enhancement Pipeline
A traditional Digital Image Processing pipeline (bilateral filter → CLAHE → unsharp mask) is applied to exported satellite thumbnails, improving visual sharpness and contrast for the web dashboard — not seen in typical GEE-based remote sensing systems.

### 7. 📊 Dual-Mode Architecture (Historical + Operational)
Historical mode uses decade-spaced epochs (1988→2022) with relaxed jump-transitions. Operational mode processes recent HLS imagery with strict sequential transitions and multi-pass persistence — enabling the same system for both retrospective analysis and near-real-time monitoring.

### 8. ⚡ Fully Server-Side GEE Computation
The entire pixel-level classification, feature extraction, and polygon extraction runs **server-side** on Google Earth Engine — no local GPU/ML infrastructure required. Only polygon-level classification and tracking run client-side in Python.

---

## Data Sources Summary

| Dataset | GEE ID | Resolution | Use |
|---------|--------|-----------|-----|
| Landsat 5 SR | `LANDSAT/LT05/C02/T1_L2` | 30m | 1984–2012 optical |
| Landsat 7 SR | `LANDSAT/LE07/C02/T1_L2` | 30m | 1999–present optical |
| HLS L30 | `NASA/HLS/HLSL30/v002` | 30m | 2013+ harmonized Landsat |
| HLS S30 | `NASA/HLS/HLSS30/v002` | 30m | 2015+ harmonized Sentinel-2 |
| Sentinel-1 GRD | `COPERNICUS/S1_GRD` | 10m | 2014+ C-band SAR |
| JRC GSW | `JRC/GSW1_4/GlobalSurfaceWater` | 30m | Static water occurrence |
| JRC Monthly | `JRC/GSW1_4/MonthlyHistory` | 30m | Water persistence time series |
| GLO30 DEM | `COPERNICUS/DEM/GLO30` | 30m | Elevation/slope constraint |
| GMW v3 | `projects/.../GMW_V3` | 25m | Mangrove baseline validation |
| SoilGrids | `projects/soilgrids-isric/...` | 250m | Optional soil data |
