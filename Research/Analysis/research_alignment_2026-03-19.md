# Research Alignment - 2026-03-19

## Why pond extraction matters

Aquaculture detection is not only a pixel-threshold problem. The recent literature consistently shows that pond extraction improves when we combine:

- water-sensitive spectral or SAR evidence
- object geometry such as compactness, rectangularity, elongation, and area
- temporal context so short-lived tidal water is not confused with managed ponds

This matches the project direction in `modules/m07_polygons.py`, `modules/m13_object_matcher.py`, and `modules/m14_per_pond_classifier.py`.

## Key takeaways from local research files

### `Research/stage_detection_audit_2026-03-19.md`

- The codebase had drift between pixel and object classifiers.
- Historical mangrove context was not being used strongly enough.
- The pipeline was over-detecting "water in the AOI" instead of "mangrove-to-aquaculture conversion".
- Object metrics and mixed-pixel handling were explicitly identified as needed fixes.

### `Research/Research Paper/Research.docx`

- The project paper argues for a 5-stage lifecycle:
  - S1 intact mangrove
  - S2 degrading mangrove
  - S3 cleared land
  - S4 pond formation / water filling
  - S5 active aquaculture
- It also emphasizes:
  - composite water evidence (`CWI = MNDWI + NDWI - NDVI`)
  - NDBI gating to reduce soil-water confusion
  - object geometry to separate artificial ponds from natural water

### `Research/Base papers/Feature Engineering for Mangrove-Aquaculture Monitoring.pdf`

- Recommends combining NDVI, MNDWI, AWEI, CMRI, MMRI, SAVI, NDBI, SAR texture/homogeneity, JRC water history, and geometry.
- Reinforces that edge density and rectangularity are useful for artificial pond discrimination.

### `Research/Base papers/Fishpond-change-detection-based-on-short-term-time-series-of-RADARSAT-images-and-object-oriented-method.pdf`

- Supports object-oriented fishpond change detection.
- Reinforces the value of texture, shape, and contextual information over pixel-only change detection.

## External literature used for alignment

### Object-based pond extraction

- Hu et al. 2024, Remote Sensing:
  https://www.mdpi.com/2072-4292/16/7/1217
- Liu et al. 2024, Remote Sensing of Environment (OptiSAR-POM):
  https://doi.org/10.1016/j.rse.2024.114484

Practical implication:

- keep object extraction and classification geometry-aware
- avoid one-to-many object matching across epochs
- expose polygon outputs cleanly to the dashboard

### Optical plus SAR fusion

- Sun et al. 2020, Remote Sensing:
  https://www.mdpi.com/2072-4292/12/18/3086
- Joshi et al. 2025, Results in Earth Sciences:
  https://doi.org/10.1016/j.rines.2025.100114

Practical implication:

- keep Sentinel-1 plus optical fusion for pond discrimination
- retain SAR-derived homogeneity/texture and temporal stability for S4/S5 support

### Mixed pixels and mangrove context

- The local project paper and audit both emphasize mixed-pixel water/vegetation confusion.
- Huang et al. 2025, GIScience and Remote Sensing:
  https://doi.org/10.1080/15481603.2025.2480422
- Global Mangrove Watch baseline paper:
  https://www.mdpi.com/2072-4292/10/10/1669

Practical implication:

- historical mangrove priors should be epoch-aware, not fixed to the first available GMW layer
- 30 m historical scenes need context-aware summaries instead of pretending they are 10 m geometry maps

## Code changes driven by this review

- Switched GMW injection to year-specific epochs in `main.py`.
- Tightened object matching to one-to-one nearest assignments in `modules/m13_object_matcher.py`.
- Reduced export-side graph load by removing redundant server-side polygon styling when local annotation is already present.
- Synced stage metadata, image assets, and AOI metadata into `stats.json` and `timeline.json`.
- Added `polygons.geojson` export so the web map consumes real geospatial features.
- Updated the web app to read the actual exported data structure, use dynamic year sliders, and load `rgb_*.png` correctly.
- Added neutral-score handling in the validator for empty stage domains and empty JRC yearly slices.

## Remaining recommended direction

- Keep historical stage interpretation primarily pixel-based and context-driven.
- Keep pond delineation strongest from 2002 onward and especially from Sentinel-2/Sentinel-1 years.
- If future work is allowed, the next research-grade upgrade should be adaptive object seeding from SAR/optical water cores rather than direct vectorization of a single combined mask.
