"""
MODULE 20 — Erode-Label-Dilate (ELD) Instance Segmentation v18.0

NOVEL CONTRIBUTION: Server-side pond instance segmentation for GEE.

v18.0 CRITICAL FIXES:
  - SIZE-ADAPTIVE EROSION: radius=2 for large ponds, radius=1 for small ponds
    prevents small ponds (<5×5 px) from vanishing during erosion while still
    separating adjacent large ponds that share 1-2px bund boundaries.
  - GRADIENT-BASED SEPARATION: uses MNDWI gradient to detect spectral
    transitions between adjacent ponds (turbid vs clear water) and suppresses
    those boundary pixels before erosion.
  - INCREASED DILATION (4 passes) to match the larger erosion reach.
  - MINIMUM CORE FILTER: labeled cores < 4px are noise; suppressed before dilation.

Problem: Adjacent aquaculture ponds share bund (dyke) boundaries that are
1-2 pixels wide. Standard connected-component vectorization merges them
into a single polygon, losing individual pond identity.

ELD Algorithm:
  1. GRADIENT SUPPRESS: Detect spectral boundaries between adjacent ponds.
  2. ERODE: Size-adaptive morphological erosion removes shared bund pixels.
     Only "core" interior pixels survive.
  3. LABEL: Connected component labeling on eroded cores assigns unique IDs.
  4. FILTER: Remove noise cores (< 4 pixels).
  5. DILATE: Each labeled region grows back to the original water mask
     boundary. Adjacent ponds with separate cores get separate labels.

GEE OPTIMIZATION:
  - Uses focal_min/focal_max (O(1) per pixel) instead of Canny (O(n²))
  - connectedComponents maxSize capped at 256 labels
  - connectedPixelCount for adaptive erosion capped at 128
  - Total computation: 3-4 focal ops + 1 connectedComponents

Research references:
  Vincent & Soille 1991 (Watershed segmentation)
  Beucher & Lantuéjoul 1979 (Marker-controlled watershed)
  Serra 1982 (Mathematical Morphology)
"""

import ee
import config


def eld_segment_water_mask(water_mask, mndwi=None):
    """
    Apply Erode-Label-Dilate segmentation to separate touching ponds.

    Args:
        water_mask: ee.Image (binary, 1=water)
        mndwi: ee.Image (optional, used for gradient-based separation)

    Returns:
        ee.Image with unique integer labels per pond segment
    """
    # =============================================
    # STEP 0: GRADIENT-BASED BOUNDARY SUPPRESSION
    # Detect spectral transitions between adjacent ponds.
    # Where MNDWI changes sharply within the water mask, there is
    # likely a bund or boundary between two ponds with different
    # water properties (turbid vs clear).
    # =============================================
    if mndwi is not None:
        # Compute MNDWI gradient magnitude within water mask
        mndwi_masked = mndwi.updateMask(water_mask)
        # Sobel gradient magnitude
        grad_x = mndwi_masked.convolve(ee.Kernel.fixed(
            3, 3, [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        ))
        grad_y = mndwi_masked.convolve(ee.Kernel.fixed(
            3, 3, [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
        ))
        gradient_mag = grad_x.pow(2).add(grad_y.pow(2)).sqrt()
        # High gradient = boundary between ponds → suppress these pixels
        # Threshold: gradient > 0.08 (empirically calibrated for MNDWI range ~0.5)
        boundary_pixels = gradient_mag.gt(0.08).And(water_mask)
        # Remove boundary pixels from water mask before erosion
        water_for_erosion = water_mask.And(boundary_pixels.Not())
    else:
        water_for_erosion = water_mask

    # =============================================
    # STEP 1: SIZE-ADAPTIVE EROSION
    # Large ponds (>= 25 connected pixels): radius=2 erosion
    #   → separates ponds sharing 1-3px wide bunds
    # Small ponds (< 25 connected pixels): radius=1 erosion
    #   → preserves small ponds that would vanish with radius=2
    # =============================================
    # Count connected water pixels for size classification
    water_connected = water_for_erosion.selfMask().connectedPixelCount(128, True).unmask(0)
    is_large_water = water_connected.gte(25)
    is_small_water = water_connected.gt(0).And(water_connected.lt(25))

    # Erode large water bodies with radius=2
    eroded_large = water_for_erosion.focal_min(
        radius=2, kernelType='circle', units='pixels'
    ).And(is_large_water)

    # Erode small water bodies with radius=1 (preserve them)
    eroded_small = water_for_erosion.focal_min(
        radius=1, kernelType='circle', units='pixels'
    ).And(is_small_water)

    # Combine: either large-eroded or small-eroded cores
    eroded = eroded_large.Or(eroded_small)

    # =============================================
    # STEP 2: LABEL — connected components on eroded cores
    # Each isolated core gets a unique ID
    # maxSize=256 limits memory; typical AOI has <200 ponds
    # =============================================
    cores_int = eroded.selfMask().toInt()
    labeled_cores = cores_int.connectedComponents(
        connectedness=ee.Kernel.plus(1),
        maxSize=256,
    )
    labels = labeled_cores.select("labels")

    # =============================================
    # STEP 3: FILTER — remove noise cores (< 4 pixels)
    # Very small cores are noise from erosion artifacts,
    # not real pond interiors.
    # =============================================
    core_sizes = eroded.selfMask().connectedPixelCount(128, True).unmask(0)
    valid_cores = core_sizes.gte(4)
    labels = labels.updateMask(valid_cores)

    # =============================================
    # STEP 4: DILATE — grow labels back to original boundaries
    # Each pixel in the original water mask inherits the label
    # of its nearest core via focal_max on the label image.
    # 4 passes of radius=1 = effective reach of 4 pixels
    # (increased from 3 to match larger erosion radius)
    # =============================================
    # Mask labels to only exist within the original water mask
    labels_in_water = labels.updateMask(water_mask)

    # Iterative dilation: grow labels outward within the water mask
    grown = labels_in_water
    for _ in range(4):
        expanded = grown.focal_max(radius=1, units='pixels')
        # Only grow into water mask pixels that don't have a label yet
        grown = grown.unmask(0).where(
            grown.unmask(0).eq(0).And(water_mask).And(expanded.gt(0)),
            expanded
        )

    # Final: mask to only water pixels with valid labels
    final_labels = grown.updateMask(water_mask.And(grown.gt(0)))

    return final_labels.rename("pond_label").toInt()


def eld_to_vectors(labeled_image, aoi, scale=None):
    """
    Vectorize ELD-labeled image into polygon features.

    Each unique label becomes a separate polygon, ensuring
    previously-merged adjacent ponds are now individual features.

    Args:
        labeled_image: ee.Image from eld_segment_water_mask()
        aoi: ee.Geometry
        scale: int (meters)

    Returns:
        ee.FeatureCollection of pond polygons
    """
    if scale is None:
        scale = config.TARGET_SCALE

    vectors = labeled_image.reduceToVectors(
        geometry=aoi,
        scale=scale,
        geometryType='polygon',
        eightConnected=False,
        maxPixels=1e9,
        tileScale=int(config.GEE_SAFE["tileScale"]),
        bestEffort=True,
        labelProperty='pond_label',
    )

    return vectors


def apply_eld_segmentation(water_mask, features, aoi, scale=None):
    """
    Full ELD pipeline: segment → vectorize → add shape metrics.

    This is the main entry point replacing the old Canny-based slicer.

    Args:
        water_mask: ee.Image (binary water detection mask)
        features: ee.Image (feature stack with mndwi band)
        aoi: ee.Geometry
        scale: int (meters)

    Returns:
        ee.FeatureCollection with shape metrics
    """
    if scale is None:
        scale = config.TARGET_SCALE

    mndwi = features.select("mndwi")

    # Apply ELD segmentation
    labeled = eld_segment_water_mask(water_mask, mndwi)

    # Vectorize
    vectors = eld_to_vectors(labeled, aoi, scale)

    # Add shape metrics (identical to m07)
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

        return feature.set({
            'area_m2': area,
            'perimeter_m': perimeter,
            'rectangularity': rectangularity,
            'compactness': compactness,
            'elongation': elongation,
            'centroid_lon': coords.get(0),
            'centroid_lat': coords.get(1),
        })

    # Simplify boundaries (3m tolerance for crisp edges — tightened from 5m)
    def simplify_geometry(feature):
        return feature.setGeometry(feature.geometry().simplify(maxError=3))

    simplified = vectors.map(simplify_geometry)
    with_metrics = simplified.map(add_shape_metrics)

    # Filter: remove noise and river fragments
    max_area = 2000000.0  # 2 km²
    min_compactness = 0.012
    max_elongation = 50

    filtered = with_metrics.filter(
        ee.Filter.And(
            ee.Filter.lte("area_m2", max_area),
            ee.Filter.gte("compactness", min_compactness),
            ee.Filter.lte("elongation", max_elongation),
        )
    )

    return filtered
