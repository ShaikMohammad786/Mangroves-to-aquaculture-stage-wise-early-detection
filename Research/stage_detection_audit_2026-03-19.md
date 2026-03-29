# Stage Detection Audit - 2026-03-19

## Aim

Build a defensible stage-wise mangrove-to-aquaculture detection workflow that:

- detects early conversion risk, not just final ponds
- uses historical mangrove context to suppress false positives
- handles 30 m mixed pixels without pretending Landsat is 10 m
- keeps module logic, config, and outputs synchronized

## Key findings from the code audit

1. The historical stream was too sparse.
   Only 2018, 2022, and 2024 were active, so the pipeline could not show a real S1 -> S2 -> S3 -> S4 -> S5 progression.

2. Optical compositing was inconsistent with stage logic.
   The code still allowed percentile compositing across all optical bands when tidal normalization was enabled. That can distort reflectance ratios and directly damage NDVI/MNDWI-based stage boundaries.

3. Pixel and object classifiers had drifted apart.
   The object classifier could still force ambiguous water-like shapes into stages even when the pixel classifier had been tightened.

4. Validation and GMW alignment were not fully synchronized.
   Validation still used partially hardcoded thresholds, and GMW matching often selected the first image in a time window rather than the closest epoch.

5. The pipeline was detecting "water in the AOI" more than "mangrove-to-aquaculture conversion".
   Historical mangrove context was not being used strongly enough outside the S1 anchor.

## Research-backed design direction

### 1. Object features matter for aquaculture ponds

Recent object-based Sentinel-2 work shows aquaculture ponds are better separated from rivers and dikes using object metrics such as area, compactness, aspect ratio, slope, and shape index, not spectral features alone.

Source:
- Hu et al. 2024, Remote Sensing
  https://www.mdpi.com/2072-4292/16/7/1217

### 2. Mixed pixels are a real conversion-mapping problem

Mixed-pixel analysis has been used explicitly in mangrove-to-aquaculture conversion studies because moderate-resolution imagery blends vegetation, soil, and water during transition.

Source:
- Rahman et al. 2013, Remote Sensing of Environment
  https://doi.org/10.1016/j.rse.2012.11.014

### 3. Sentinel-1 + Sentinel-2 fusion improves pond discrimination

Recent studies continue to show that optical + SAR improves aquaculture classification because water state, texture, and management signals are complementary.

Sources:
- Joshi et al. 2025, Results in Earth Sciences
  https://doi.org/10.1016/j.rines.2025.100114
- Hu et al. 2025, Expert Systems with Applications
  https://doi.org/10.1016/j.eswa.2025.128740

### 4. GMW is a strong historical mangrove anchor

Global Mangrove Watch reports strong baseline accuracy and is suitable as a prior or validation layer for historical mangrove extent.

Source:
- Bunting et al. 2018, Remote Sensing
  https://www.mdpi.com/2072-4292/10/10/1669

### 5. Godavari has documented mangrove-to-aquaculture conversion

Independent Godavari estuary work reports large aquaculture expansion and direct conversion of mangrove vegetation to aquaculture, supporting the project rationale and hotspot choice.

Source:
- Rao et al. 2023, Journal of the Indian Society of Remote Sensing
  https://link.springer.com/article/10.1007/s12524-023-01698-w

## Implemented corrections

1. Historical epochs were expanded to restore actual progression analysis.
2. Optical composites were forced back to median instead of p25 for full-band stage classification.
3. NDVI coefficient-of-variation was stabilized to avoid invalid behavior around low or negative means.
4. Pixel classifier now uses a historical mangrove context prior for S2-S5.
5. Object classifier now penalizes candidates outside mangrove context.
6. GMW epoch selection now uses closest available epoch rather than first-in-window.
7. Validator thresholds were aligned to the shared stage specification.

## Remaining strategic recommendation

For high-confidence proof panels, use:

- Landsat only for historical context and broad progression
- Sentinel-2 as the main modern delineation source
- Sentinel-1 as the water/texture complement
- GMW conversion polygons plus verified site panels for defensible evidence

Historical sub-10 m imagery is not available as a free, continuous long-term source in the same way Landsat is, so the correct strategy is not to "pretend" 30 m is high resolution, but to:

- use spectral unmixing and temporal context on Landsat/HLS
- switch to Sentinel-2/Sentinel-1 when available
- keep proof outputs explicit about sensor resolution
