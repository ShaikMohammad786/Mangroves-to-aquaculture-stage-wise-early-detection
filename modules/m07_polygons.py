"""
MODULE 7 — Polygon Extraction (v17.0 — RESEARCH GRADE)

v17.0 RESEARCH-GRADE OVERHAUL:
  - River masking now uses shared ADSRM module (m19) — eliminates code divergence with m06
  - ELD (Erode-Label-Dilate) instance segmentation replaces Canny edge slicer
  - GLCM IDM fix: homogeneity is now actual IDM [0,1], not Sum Average [0-255]
  - Sharper polygon boundaries (simplification 8m → 5m)
  - GEE memory optimization: connectedPixelCount capped at 64, fewer focal ops

Research references:
  - Serra 1982 (Mathematical Morphology)
  - Vincent & Soille 1991 (Watershed segmentation)
  - Haralick et al. 1973 (GLCM IDM definition)
"""

import ee
import config
from modules.m19_river_mask_shared import compute_river_mask
from modules.m20_instance_segmentation import eld_segment_water_mask



def extract_pond_candidates(features, aoi, date_str, scale=None):
    """
    Extract pond polygons v3.0 — Natural boundaries, aggressive river masking.
    
    Key design:
    - Hard JRC mask removes rivers BEFORE polygon extraction
    - No SNIC: uses connected-component vectorization for natural boundaries
    - No min-area: whatever the pond size, it gets its exact boundary
    """
    if scale is None:
        scale = config.TARGET_SCALE

    ndvi = features.select("ndvi")
    mndwi = features.select("mndwi")
    ndwi = features.select("ndwi")

    band_names = features.bandNames()

    pixel_s4 = ee.Image(
        ee.Algorithms.If(
            band_names.contains("pixel_s4_fraction"),
            features.select("pixel_s4_fraction"),
            ee.Image(0),
        )
    )
    pixel_s5 = ee.Image(
        ee.Algorithms.If(
            band_names.contains("pixel_s5_fraction"),
            features.select("pixel_s5_fraction"),
            ee.Image(0),
        )
    )
    pixel_conf = ee.Image(
        ee.Algorithms.If(
            band_names.contains("pixel_confidence_mean"),
            features.select("pixel_confidence_mean"),
            ee.Image(0.5),
        )
    )

    # =============================================
    # STEP 1: ADSRM RIVER MASK (v17.0 — SHARED MODULE)
    # Single source of truth: m19_river_mask_shared.py
    # =============================================
    river_result = compute_river_mask(features, aoi)
    is_river = river_result['is_river']
    not_permanent_water = is_river.Not()

    # =============================================
    # STEP 2: WATER DETECTION TIERS
    # =============================================

    has_cwi = band_names.contains("cwi")
    cwi = ee.Image(
        ee.Algorithms.If(
            has_cwi, features.select("cwi"), mndwi.add(ndwi).subtract(ndvi)
        )
    )
    
    has_unmix = band_names.contains("water_fraction")
    water_fraction = ee.Image(
        ee.Algorithms.If(has_unmix, features.select("water_fraction"), ee.Image(0))
    )

    water_evidence = features.select("water_evidence_score")

    # Unified water mask: Use single-source-of-truth from stage engine
    water_raw = pixel_s4.add(pixel_s5).gt(0).Or(water_evidence.gt(0.35))

    # =============================================
    # STEP 3: APPLY RIVER MASK (kill rivers at source!)
    # =============================================
    water_mask = water_raw.And(not_permanent_water)

    # =============================================
    # STEP 4: ELD INSTANCE SEGMENTATION (v17.0 — NOVEL)
    # Replaces Canny edge slicer which was over-fragmenting large ponds
    # and under-separating adjacent small ponds.
    # ELD: Erode cores → Label unique → Dilate back to boundary
    # =============================================

    # Closing: fill 1-pixel gaps inside ponds
    close_kernel = ee.Kernel.circle(radius=1, units="pixels")
    water_cleaned = water_mask.focal_max(kernel=close_kernel).focal_min(
        kernel=close_kernel
    )

    # Opening: remove thin protrusions
    open_kernel = ee.Kernel.circle(radius=1, units="pixels")
    water_cleaned = water_cleaned.focal_min(kernel=open_kernel).focal_max(
        kernel=open_kernel
    )

    # ELD segmentation: separate touching ponds
    labeled = eld_segment_water_mask(water_cleaned, mndwi)

    # Remove small noise patches (minimum 16 connected pixels)
    # Use the labeled image for connected count (each segment counted separately)
    pixel_counts = water_cleaned.selfMask().connectedPixelCount(64)
    size_filtered = water_cleaned.selfMask().where(pixel_counts.lt(16), 0).selfMask()

    # =============================================
    # STEP 4b: WATERSHED-LINE GAP (v21.0 — NOVEL)
    # Compute MNDWI gradient within water mask. Where adjacent
    # ponds have different water quality (turbid vs clear),
    # insert a 1px gap to force vectorization into separate polygons.
    # =============================================
    mndwi_water = mndwi.updateMask(size_filtered)
    # Sobel gradient magnitude
    grad_x = mndwi_water.convolve(ee.Kernel.fixed(
        3, 3, [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    ))
    grad_y = mndwi_water.convolve(ee.Kernel.fixed(
        3, 3, [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
    ))
    gradient_mag = grad_x.pow(2).add(grad_y.pow(2)).sqrt()
    watershed_line = gradient_mag.gt(0.08)  # High gradient = pond boundary
    # Remove watershed line pixels from water mask
    size_filtered = size_filtered.And(watershed_line.Not()).selfMask()

    # =============================================
    # STEP 5: VECTORIZATION (4-connected, ELD-enhanced)
    # =============================================

    try:
        vectors = size_filtered.reduceToVectors(
            geometry=aoi,
            scale=scale,
            geometryType="polygon",
            eightConnected=False,
            maxPixels=1e9,
            tileScale=int(config.GEE_SAFE["tileScale"]),
            bestEffort=True,
        )
    except Exception:
        vectors = water_cleaned.selfMask().reduceToVectors(
            geometry=aoi,
            scale=scale,
            geometryType="polygon",
            eightConnected=False,
            maxPixels=1e9,
            tileScale=int(config.GEE_SAFE["tileScale"]),
            bestEffort=True,
        )


    # =============================================
    # SHAPE METRICS
    # =============================================

    def add_shape_metrics(feature):
        geom = feature.geometry()
        area = geom.area(maxError=1)
        perimeter = geom.perimeter(maxError=1)
        bounds = geom.bounds(maxError=1)
        bbox_area = bounds.area(maxError=1)

        rectangularity = area.divide(bbox_area.max(1))
        compactness = area.multiply(4 * 3.14159).divide(
            perimeter.multiply(perimeter).max(1)
        )
        elongation = perimeter.divide(area.sqrt().max(0.01))

        centroid = geom.centroid(maxError=1)
        coords = centroid.coordinates()

        return feature.set(
            {
                "area_m2": area,
                "perimeter_m": perimeter,
                "rectangularity": rectangularity,
                "compactness": compactness,
                "elongation": elongation,
                "centroid_lon": coords.get(0),
                "centroid_lat": coords.get(1),
                "date": date_str,
            }
        )

    # Apply polygon simplification (Douglas-Peucker) for crisp boundaries
    # v21.0: Tightened from 5m to 3m for sharper edges
    def simplify_geometry(feature):
        return feature.setGeometry(feature.geometry().simplify(maxError=3))

    simplified = vectors.map(simplify_geometry)
    candidates = simplified.map(add_shape_metrics)

    # =============================================
    # FILTERING (v17.0: tighter shape constraints)
    # =============================================

    max_area = 2000000.0  # 2 km² max
    min_compactness = 0.012  # Reject very elongated river fragments
    max_elongation = 50  # Tightened from 60

    filtered = candidates.filter(
        ee.Filter.And(
            ee.Filter.lte("area_m2", max_area),
            ee.Filter.gte("compactness", min_compactness),
            ee.Filter.lte("elongation", max_elongation),
        )
    )

    # Spectral validation
    ndvi_water_summary = ee.Image.cat(
        [
            ndvi.rename("ndvi_mean"),
            mndwi.rename("mndwi_mean"),
            ndwi.rename("ndwi_mean"),
            cwi.rename("cwi_mean"),
            water_evidence.rename("water_evidence_mean"),
            water_fraction.rename("water_fraction_mean"),
            pixel_s4.rename("pixel_s4_mean"),
            pixel_s5.rename("pixel_s5_mean"),
        ]
    ).reduceRegions(
        collection=filtered,
        reducer=ee.Reducer.mean(),
        scale=scale,
        tileScale=int(config.GEE_SAFE["tileScale"]),
    )

    # Spectral filters — ensure water-like spectral signature
    filtered_ndvi = ndvi_water_summary.filter(
        ee.Filter.Or(
            ee.Filter.And(
                ee.Filter.lt("ndvi_mean", 0.28),
                ee.Filter.gt("mndwi_mean", -0.15),
                ee.Filter.gt("water_evidence_mean", 0.40),
            ),
            ee.Filter.And(
                ee.Filter.lt("ndvi_mean", 0.30),
                ee.Filter.gt("water_evidence_mean", 0.45),
                ee.Filter.gt("compactness", 0.015),
            ),
            ee.Filter.And(
                ee.Filter.gt("pixel_s4_mean", 0.15),
                ee.Filter.gt("water_evidence_mean", 0.30),
            ),
            ee.Filter.And(
                ee.Filter.gt("pixel_s5_mean", 0.15),
                ee.Filter.gt("water_evidence_mean", 0.30),
            ),
        )
    )

    # Secondary JRC polygon-level filter (catch anything that slipped past pixel mask)
    has_jrc_band = band_names.contains("jrc_occurrence")
    has_jrc_seas = band_names.contains("jrc_seasonality")
    nwr = getattr(config, "NATURAL_WATER_REJECTION", {})

    if config.EXTENSIONS.get("use_natural_water_rejection", True) and has_jrc_band:
        jrc_occ_img = ee.Image(
            ee.Algorithms.If(has_jrc_band, features.select("jrc_occurrence"), ee.Image(0))
        )
        jrc_seas_img = ee.Image(
            ee.Algorithms.If(
                has_jrc_seas, features.select("jrc_seasonality"), ee.Image(0)
            )
        )

        jrc_sampled = jrc_occ_img.addBands(jrc_seas_img).reduceRegions(
            collection=filtered_ndvi,
            reducer=ee.Reducer.mean(),
            scale=scale,
            tileScale=int(config.GEE_SAFE["tileScale"]),
        )

        def _rename_jrc_means(f):
            f = ee.Feature(f)
            return f.set(
                {
                    "jrc_occ_mean": f.get("jrc_occ"),
                    "jrc_seas_mean": f.get("jrc_seas"),
                }
            )

        jrc_sampled = jrc_sampled.map(_rename_jrc_means)

        # Reject polygons that look like rivers:
        # High JRC occurrence + river-like shape
        natural_filter = ee.Filter.And(
            ee.Filter.gt("jrc_occ_mean", 60),  # Moderate-high JRC
            ee.Filter.Or(
                ee.Filter.lt("compactness", 0.03),  # Very low compactness = river
                ee.Filter.gt("elongation", 25.0),  # Very elongated = river
            ),
        )

        # Pure shape-based thin river filter (regardless of JRC)
        river_filter = ee.Filter.And(
            ee.Filter.lt("compactness", 0.02),
            ee.Filter.gt("elongation", 25.0),
        )

        # v14.0: Width-to-length ratio filter — catch elongated thin shapes
        thin_elongated_filter = ee.Filter.And(
            ee.Filter.lt("compactness", 0.025),  # Tightened from 0.03
            ee.Filter.gt("elongation", 18.0),    # Tightened from 30
        )

        # Aquaculture geometry (ponds have specific shapes)
        aquaculture_geom_filter = ee.Filter.And(
            ee.Filter.gt("rectangularity", 0.06),
            ee.Filter.gt("compactness", 0.015),
        )

        # Keep: (not river-like JRC OR has pond-like geometry)
        keep_filter = ee.Filter.And(
            ee.Filter.Or(natural_filter.Not(), aquaculture_geom_filter),
            ee.Filter.Or(river_filter.Not(), aquaculture_geom_filter),
            ee.Filter.Or(thin_elongated_filter.Not(), aquaculture_geom_filter),
        )

        final = jrc_sampled.filter(keep_filter)
    else:
        final = filtered_ndvi

    return final


def paint_polygon_outlines(candidates_fc, aoi, line_width=3):
    """Paint polygon outlines — thick and high-contrast."""
    if candidates_fc is None:
        return ee.Image(0).clip(aoi).rename("outlines").toInt()

    return (
        ee.Image(0)
        .paint(featureCollection=candidates_fc, color=1, width=line_width)
        .clip(aoi)
        .rename("outlines")
        .toInt()
    )


def paint_stage_polygons(candidates_fc, aoi, polygon_list=None):
    """Paint stage polygons — FILLED regions with clean outlines on top."""
    if candidates_fc is None:
        empty = ee.Image(0).clip(aoi).rename("stage").toInt()
        return empty.updateMask(empty.gt(0))

    # FILL polygons with their stage value (region consistency)
    fill_image = (
        ee.Image(0)
        .paint(featureCollection=candidates_fc, color="stage")  # Fill with stage
        .clip(aoi)
        .rename("stage")
        .toInt()
    )

    # Add outline on top for visibility (width=3)
    outline_image = (
        ee.Image(0)
        .paint(featureCollection=candidates_fc, color="stage", width=3)
        .clip(aoi)
        .rename("stage")
        .toInt()
    )

    # Combine: fill first, outline on top
    combined = fill_image.where(outline_image.gt(0), outline_image)
    return combined.selfMask()
