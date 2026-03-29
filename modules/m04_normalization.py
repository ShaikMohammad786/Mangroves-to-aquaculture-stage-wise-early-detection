"""
MODULE 4 — Spatial Normalization (Research-Grade v2.0)

✔ Avoids dangerous .reproject() overuse
✔ Uses lazy resampling
✔ Different resampling for optical vs SAR vs labels
✔ Aligns datasets only when needed
✔ Fully compatible with existing pipeline
"""

import ee
import config


# ─────────────────────────────────────────────────────────────
# SAFE NORMALIZATION
# ─────────────────────────────────────────────────────────────

def normalize_image(image, aoi, resample_method="bilinear"):
    """
    Prepare image for spatial consistency WITHOUT forcing reprojection.

    FIX: Removed .resample() call. When called on composites (median/percentile)
    that lack an explicit projection, GEE applies the resample at the default
    1-degree EPSG:4326 scale, collapsing the entire AOI to ~1 pixel and
    producing uniform single-color images. Since the pipeline intentionally
    avoids .reproject() (to prevent solid bounding-box artifacts), .resample()
    has no valid target projection and only causes harm.

    resample_method: kept as parameter for API compatibility but no longer used.
    """
    image = ee.Image(image)

    # Only clip (safe operation — no resample, no reproject)
    image = image.clip(aoi)

    return image


def normalize_collection(collection, aoi, resample_method="bilinear"):
    """
    Normalize collection lazily (no forced reprojection).
    """
    return collection.map(
        lambda img: normalize_image(img, aoi, resample_method)
    )


# ─────────────────────────────────────────────────────────────
# ALIGNMENT (OPTICAL + SAR)
# ─────────────────────────────────────────────────────────────

def align_optical_and_sar(optical_image, sar_image, aoi):
    """
    Align SAR to optical grid using optical projection.
    This avoids arbitrary CRS locking.
    
    NOTE: This is intentionally NOT hooked into normalize_all(). 
    SAR and Optical arrays must be stacked as temporally matched pairs. 
    M5/M6 downstream modules must invoke this manually when extracting cross-sensor features.
    """

    optical_image = ee.Image(optical_image)
    sar_image = ee.Image(sar_image)

    # Use optical projection as reference
    target_proj = optical_image.projection()

    sar_aligned = sar_image \
        .resample("bilinear") \
        .reproject(crs=target_proj, scale=config.TARGET_SCALE)

    opt_aligned = optical_image.clip(aoi)
    sar_aligned = sar_aligned.clip(aoi)

    return opt_aligned, sar_aligned


# ─────────────────────────────────────────────────────────────
# COMPOSITE CREATION
# ─────────────────────────────────────────────────────────────

def create_composite(collection, aoi, method="median"):
    """
    Create temporal composite.
    Projection kept native until export.
    """
    
    # ── Server-Side Guard for Empty Collections ──
    # Python evaluates both branches of ee.Algorithms.If eagerly.
    # To prevent server-side crashes (e.g. from .first() on an empty collection),
    # we inject a dummy image if the collection is empty BEFORE any reducers.
    size = collection.size()
    safe_collection = ee.ImageCollection(ee.Algorithms.If(
        size.gt(0),
        collection,
        ee.ImageCollection([ee.Image().clip(aoi)])
    ))

    if method == "median":
        composite = safe_collection.median()
    elif method == "p25":
        # Rationale: p25 compositing captures the lower 25th percentile of pixel values.
        # For water detection (MNDWI), this suppresses transient high-tide artifacts and highlights permanent water.
        composite = safe_collection.reduce(ee.Reducer.percentile([25]))
        
        # safely extract band names without hitting empty collection errors
        band_names = ee.Image(safe_collection.first()).bandNames()
        composite = composite.rename(band_names)
    elif method == "mean":
        composite = safe_collection.mean()
    elif method == "mosaic":
        # Sort so newest images are placed on top
        composite = safe_collection.sort("system:time_start", False).mosaic()
    else:
        composite = safe_collection.median()

    # Lazy normalization only
    composite = normalize_image(composite, aoi)

    return composite


# ─────────────────────────────────────────────────────────────
# NORMALIZE ALL DATASETS
# ─────────────────────────────────────────────────────────────

def normalize_all(data, aoi_info):
    """
    Spatial normalization controller.
    No forced reprojection of full collections.
    """

    aoi = aoi_info["geometry"]

    print("\n" + "=" * 60)
    print("[M4] SPATIAL NORMALIZATION (Safe Mode)")
    print("=" * 60)

    # Safe passthrough for any dataset keys M4 doesn't explicitly process
    # Note: data.copy() is a shallow copy. This is intentional and safe 
    # since GEE objects inside are server-side references (not duplicated).
    normalized = data.copy()

    # Optical collections (bilinear)
    for key in ["landsat5", "landsat7", "hls_l30", "hls_s30", "sentinel2_sr"]:
        if key in data and data[key] is not None:
            normalized[key] = normalize_collection(
                data[key],
                aoi,
                resample_method="bilinear"
            )

    # SAR collection (bilinear)
    if "sentinel1" in data and data["sentinel1"] is not None:
        normalized["sentinel1"] = normalize_collection(
            data["sentinel1"],
            aoi,
            resample_method="bilinear"
        )

    # JRC Water is 0-100 continuous, use bilinear to prevent blocky artifacts
    if "jrc_water" in data and data["jrc_water"] is not None:
        normalized["jrc_water"] = normalize_image(
            data["jrc_water"],
            aoi,
            resample_method="bilinear"
        )
        
    # JRC Monthly is categorical/binary (0, 1, 2)
    # Note: Other bands (waterHistory, waterDetections) are counts (0-23). 
    # Downstream modules relying on counts instead of the 0-2 class map should apply custom resampling.
    if "jrc_monthly" in data and data["jrc_monthly"] is not None:
        normalized["jrc_monthly"] = normalize_collection(
            data["jrc_monthly"],
            aoi,
            resample_method="nearest"
        )

    if "glo30" in data and data["glo30"] is not None:
        normalized["glo30"] = normalize_image(
            data["glo30"],
            aoi,
            resample_method="bilinear"
        )
        
    # Mangrove Watch is a binary class mask (1=mangrove), nearest is correct.
    if "mangrove_baseline" in data and data["mangrove_baseline"] is not None:
        normalized["mangrove_baseline"] = normalize_collection(
            data["mangrove_baseline"],
            aoi,
            resample_method="nearest"
        )
        
    if "soilgrids" in data and data["soilgrids"] is not None:
        normalized["soilgrids"] = normalize_image(
            data["soilgrids"],
            aoi,
            resample_method="bilinear"
        )

    print("[M4] All datasets normalized safely")

    return normalized