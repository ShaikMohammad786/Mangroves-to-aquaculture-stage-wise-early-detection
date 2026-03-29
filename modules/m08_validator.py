"""
MODULE 8 — Multi-Evidence Validator (Fixed v3.2)

FIXES:
  ALL reduceRegion calls now include tileScale=4 and maxPixels=1e8.
  Without tileScale, complex computation graphs (8+ band feature images)
  exceed GEE's 80 MiB per-tile memory limit and crash with:
  "Output of image computation is too large".

  All scores gracefully handle missing data (no GMW/JRC -> neutral 0.5).
  Final: validation_score = weighted sum (0-1). Fully server-side.
"""

import ee
import config
from modules.stage_spec import DEFAULT_STAGE_SPEC

# Shared reduceRegion kwargs
_RR = dict(scale=config.TARGET_SCALE, **config.GEE_SAFE)


def _fraction_or_neutral(agree_val, total_obj, neutral=0.5):
    total_val = ee.Number(ee.Algorithms.If(total_obj, total_obj, 0)).max(0)
    return ee.Number(ee.Algorithms.If(
        total_val.gt(0),
        ee.Number(agree_val).max(0).divide(total_val),
        neutral
    ))


# ---------------------------------------------
# A) MANGROVE VALIDATION SCORE
# ---------------------------------------------

def compute_mangrove_score(features, stage_image, aoi):
    """S1 pixels should have high NDVI + low MNDWI. Self-consistent check."""
    spec = DEFAULT_STAGE_SPEC
    ndvi   = features.select("ndvi")
    mndwi  = features.select("mndwi")
    stage  = stage_image.select("stage")
    s1_mask = stage.eq(1)

    good = ndvi.gt(spec.s1_ndvi_min).And(mndwi.lt(spec.s1_water_index_max)).And(s1_mask)

    agree_sum = good.reduceRegion(
        reducer=ee.Reducer.sum(), geometry=aoi, **_RR)
    total_sum = s1_mask.reduceRegion(
        reducer=ee.Reducer.sum(), geometry=aoi, **_RR)

    a_val = agree_sum.values().get(0)
    t_val = total_sum.values().get(0)
    agree_val = ee.Number(ee.Algorithms.If(a_val, a_val, 0)).max(0)
    return _fraction_or_neutral(agree_val, t_val)


# ---------------------------------------------
# B) WATER VALIDATION SCORE
# ---------------------------------------------

def compute_water_score(features, stage_image, aoi, jrc_monthly=None, image_year=None):
    """
    S4/S5 pixels should have high MNDWI + NDWI, low NDVI.
    Optionally validates long-term temporal persistence using JRC Monthly History
    (filters tidal transients vs permanent water/aquaculture).
    """
    spec = DEFAULT_STAGE_SPEC
    mndwi = features.select("mndwi")
    ndwi  = features.select("ndwi")
    ndvi  = features.select("ndvi")
    stage = stage_image.select("stage")
    water_mask = stage.gte(4)

    good = (
        mndwi.gt(spec.s4_mndwi_min)
        .And(ndwi.gt(spec.s4_ndwi_min))
        .And(ndvi.lt(spec.s4_ndvi_max))
        .And(water_mask)
    )

    agree_sum = good.reduceRegion(
        reducer=ee.Reducer.sum(), geometry=aoi, **_RR)
    total_sum = water_mask.reduceRegion(
        reducer=ee.Reducer.sum(), geometry=aoi, **_RR)

    a_val = agree_sum.values().get(0)
    t_val = total_sum.values().get(0)
    agree_val = ee.Number(ee.Algorithms.If(a_val, a_val, 0)).max(0)
    base_score = _fraction_or_neutral(agree_val, t_val)

    # Apply JRC persistence
    if config.EXTENSIONS.get("use_tidal_normalization") and jrc_monthly is not None and image_year is not None:
        jrc_year = ee.ImageCollection(jrc_monthly).filterDate(f"{image_year}-01-01", f"{image_year}-12-31")
        jrc_size = jrc_year.size()

        months_water = jrc_year.map(lambda img: img.eq(2)).sum()
        months_valid = jrc_year.map(lambda img: img.gt(0)).sum()

        water_ratio = months_water.divide(months_valid.max(1))

        avg_water_ratio = water_ratio.updateMask(water_mask).reduceRegion(
            reducer=ee.Reducer.mean(), geometry=aoi, **_RR
        )

        scene_persistence = ee.Number(ee.Algorithms.If(
            jrc_size.gt(0),
            ee.Algorithms.If(avg_water_ratio.values().get(0), avg_water_ratio.values().get(0), 0.5),
            0.5
        ))

        min_water = config.FALSE_POSITIVE.get("jrc_persistence", {}).get("min_water_months_ratio", 0.6)
        max_tidal = config.FALSE_POSITIVE.get("jrc_persistence", {}).get("max_tidal_months_ratio", 0.3)

        jrc_multiplier = ee.Number(ee.Algorithms.If(
            scene_persistence.lt(max_tidal),
            0.5,
            ee.Algorithms.If(
                scene_persistence.gt(min_water),
                1.2,
                1.0
            )
        ))

        final_score = base_score.multiply(jrc_multiplier).clamp(0, 1)
        return final_score

    return base_score


# ---------------------------------------------
# C) ELEVATION PLAUSIBILITY SCORE
# ---------------------------------------------

def compute_elevation_score(stage_image, glo30, aoi):
    """S4/S5 at elevation > 10m = violation. Score = 1 - violation_ratio."""
    if glo30 is None:
        return ee.Number(0.5)

    water_stages = stage_image.select("stage").gte(4)
    elevation    = glo30.select("DEM")
    high_elev    = elevation.gt(config.AOI["max_elevation_m"])
    violations   = water_stages.And(high_elev)
    land_mask    = elevation.gt(0)
    valid_domain = water_stages.And(land_mask)

    v_count = violations.reduceRegion(
        reducer=ee.Reducer.sum(), geometry=aoi, **_RR)
    t_count = valid_domain.reduceRegion(
        reducer=ee.Reducer.sum(), geometry=aoi, **_RR)

    v_val_obj = v_count.values().get(0)
    t_val_obj = t_count.values().get(0)
    v_val = ee.Number(ee.Algorithms.If(v_val_obj, v_val_obj, 0)).max(0)
    t_val = ee.Number(ee.Algorithms.If(t_val_obj, t_val_obj, 0)).max(0)
    return ee.Number(ee.Algorithms.If(
        t_val.gt(0),
        ee.Number(1).subtract(v_val.divide(t_val)).max(0),
        0.5
    ))


# ---------------------------------------------
# D) SAR CONSISTENCY SCORE
# ---------------------------------------------

def compute_sar_score(features, stage_image, aoi):
    """
    S1 -> expect high VH (> -15 dB).
    S4/S5 -> expect low VV (< -18 dB).
    No SAR (pre-2015) -> neutral 0.5.
    """
    band_names = features.bandNames()
    has_sar = band_names.contains("vv_mean")

    # Safe band access - falls back to default when SAR bands missing
    vv = ee.Image(ee.Algorithms.If(
        band_names.contains("vv_mean"),
        features.select("vv_mean"),
        ee.Image(-15)
    )).rename("vv_s")
    vh = ee.Image(ee.Algorithms.If(
        band_names.contains("vh_mean"),
        features.select("vh_mean"),
        ee.Image(-15)
    )).rename("vh_s")

    stage      = stage_image.select("stage")
    s1_mask    = stage.eq(1)
    water_mask = stage.gte(4)

    s1_good = vh.gt(-15).And(s1_mask)
    w_good  = vv.lt(-18).And(water_mask)

    def _ratio(numerator_img, mask):
        a_obj = numerator_img.reduceRegion(
            reducer=ee.Reducer.sum(), geometry=aoi, **_RR
        ).values().get(0)
        t_obj = mask.reduceRegion(
            reducer=ee.Reducer.sum(), geometry=aoi, **_RR
        ).values().get(0)

        a = ee.Number(ee.Algorithms.If(a_obj, a_obj, 0)).max(0)
        return a, t_obj

    s1_a, s1_t = _ratio(s1_good, s1_mask)
    w_a,  w_t  = _ratio(w_good,  water_mask)
    total_support = ee.Number(ee.Algorithms.If(s1_t, s1_t, 0)).add(ee.Number(ee.Algorithms.If(w_t, w_t, 0)))
    sar_result = ee.Number(ee.Algorithms.If(
        total_support.gt(0),
        s1_a.add(w_a).divide(total_support),
        0.5
    ))

    return ee.Number(ee.Algorithms.If(has_sar, sar_result, 0.5))


# ---------------------------------------------
# E) COMBINED VALIDATION SCORE
# ---------------------------------------------

def compute_validation_score(
        stage_image, confidence_image, features,
        mangrove_baseline, jrc_water, glo30, aoi,
        jrc_monthly=None, image_year=None):
    """
    Final per-image validation score.
    5 independent sub-scores -> weighted sum -> single number in [0, 1].
    """
    w = config.VALIDATION["weights"]

    mangrove_s = compute_mangrove_score(features, stage_image, aoi)
    water_s    = compute_water_score(features, stage_image, aoi, jrc_monthly, image_year)
    elevation_s = compute_elevation_score(stage_image, glo30, aoi)
    sar_s      = compute_sar_score(features, stage_image, aoi)

    # Multi-evidence = mean classifier confidence over AOI
    conf_mean = confidence_image.reduceRegion(
        reducer=ee.Reducer.mean(), geometry=aoi, **_RR)
    multi_conf = ee.Number(conf_mean.values().get(0))
    multi_conf = ee.Number(ee.Algorithms.If(multi_conf, multi_conf, 0.5))

    validation_score = (
        mangrove_s.multiply(w["mangrove"])
        .add(water_s.multiply(w["water"]))
        .add(elevation_s.multiply(w["elevation"]))
        .add(sar_s.multiply(w["sar"]))
        .add(multi_conf.multiply(w["multi_evidence"]))
    )

    return {
        "validation_score":          validation_score,
        "mangrove_score":            mangrove_s,
        "water_score":               water_s,
        "elevation_score":           elevation_s,
        "sar_score":                 sar_s,
        "multi_evidence_confidence": multi_conf,
    }
