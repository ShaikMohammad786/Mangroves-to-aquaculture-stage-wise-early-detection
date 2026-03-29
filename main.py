"""
MAIN PIPELINE - Fixed v3.2
"""

import ee
import json
import argparse
import os
import glob
import time as _time
from datetime import datetime

os.makedirs(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs"), exist_ok=True
)

ee.Initialize(project="claude-491607")
print("[INIT] Google Earth Engine initialized")

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "outputs", "pipeline.log"
            ),
            mode="w",
        ),
    ],
)
pipe_log = logging.getLogger("PIPELINE")

import config
from modules.m00_aoi import initialize_aoi, export_aoi_geojson
from modules.m01_acquisition import acquire_all
from modules.m02_optical_preprocess import preprocess_optical_collection
from modules.m03_sar_preprocess import preprocess_s1_collection
from modules.m05_features import extract_all_features
from modules.m06_stage_engine import (
    classify_image,
    precompute_sar_thresholds,
    precompute_optical_thresholds,
    apply_ccdc,
    merge_ccdc_features,
)
from modules.m04_normalization import create_composite
from modules.m07_polygons import (
    extract_pond_candidates,
    paint_polygon_outlines,
    paint_stage_polygons,
)
from modules.m08_validator import compute_validation_score
from modules.m09_alerts import (
    AlertStore,
    PersistenceEngine,
    create_alert,
    record_image_event,
)
from modules.m12_pond_registry import PondRegistry
from modules.m13_object_matcher import match_polygons_to_registry, haversine
from modules.m14_per_pond_classifier import (
    classify_pond_features,
    compute_pond_confidence,
    classify_pond_observation,
)
from modules.m15_ground_truth import (
    load_converted_ponds,
    get_gmw_epoch_extent,
    get_historical_mangrove_anchor,
)
from modules.m16_feature_audit import export_feature_audit
from modules.m11_accuracy import (
    compare_detected_ponds_to_converted_gt,
    compare_with_gmw,
)
from modules.m10_web_export import (
    export_stage_thumbnail,
    export_rgb_thumbnail,
    export_detection_overlay_thumbnail,
    export_ground_truth_overlay_thumbnail,
    export_ndvi_thumbnail,
    export_savi_thumbnail,
    export_ndwi_thumbnail,
    export_awei_thumbnail,
    export_mndwi_thumbnail,
    export_ndbi_thumbnail,
    export_sar_thumbnail,
    export_rvi_thumbnail,
    export_jrc_thumbnail,
    export_glo30_thumbnail,
    export_all_web_data,
    reset_static_export_flags,
    STAGE_LABELS,
    STAGE_COLORS_HEX,
    stage_label,
)


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(
        description="Mangrove->Aquaculture Transition Detection"
    )
    parser.add_argument(
        "--mode", choices=["operational", "historical"], default="historical"
    )
    parser.add_argument(
        "--years",
        type=str,
        default=None,
        help="Comma-separated historical years to process, e.g. 1990,2010,2024",
    )
    return parser.parse_args()


def _parse_year_list(years_arg):
    if not years_arg:
        return None
    years = []
    for token in str(years_arg).split(","):
        token = token.strip()
        if not token:
            continue
        try:
            year = int(token)
        except ValueError as exc:
            raise ValueError(f"Invalid year '{token}' in --years") from exc
        years.append(year)
    if not years:
        return None
    return sorted(dict.fromkeys(years))


# ─────────────────────────────────────────────
# Polygon Tracking (Object-Based)
# ─────────────────────────────────────────────


def upload_painted_ponds(pond_dicts, aoi):
    """
    Creates an ee.Image by taking the centroid points of ponds and buffering them
    to visualize the pond stages on the thumbnail export.
    """
    if not pond_dicts:
        empty = ee.Image(0).clip(aoi).rename("stage").toInt()
        return empty.updateMask(empty.gt(0))

    features = []
    for p in pond_dicts:
        geom_dict = p.get("geometry")
        lon = p.get("lon")
        lat = p.get("lat")
        stage = p.get("confirmed_stage", 0)

        if geom_dict and stage > 0:
            geom = ee.Geometry(geom_dict)
            features.append(ee.Feature(geom, {"stage": stage}))
        elif lon and lat and stage > 0:
            geom = ee.Geometry.Point([lon, lat]).buffer(30)
            features.append(ee.Feature(geom, {"stage": stage}))

    if not features:
        empty = ee.Image(0).clip(aoi).rename("stage").toInt()
        return empty.updateMask(empty.gt(0))

    fc = ee.FeatureCollection(features)
    painted = ee.Image(0).paint(fc, "stage").clip(aoi).rename("stage").toInt()
    return painted.updateMask(painted.gt(0))


def _build_object_feature_image(features, include_sar=False, include_previous=False):
    keep_bands = [
        "ndvi",
        "ndwi",
        "mndwi",
        "evi",
        "savi",
        "ndbi",
        "awei",
        "lswi",
        "cmri",
        "mmri",
        "gndvi",
        "veg_water_diff",
        "cwi",
        "ndti",
        "ndci",
        "turbidity_proxy",
        "water_fraction",
        "vegetation_fraction",
        "slope",
        "low_elevation_mask",
        "ndvi_cv",
        "ndvi_trend_slope",
        "ndvi_amplitude",
        "ndvi_iqr",
        "mndwi_amplitude",
        "mndwi_iqr",
        "water_persistence",
        "seasonal_water_entropy",
        "water_transition_frequency",
        "hydro_connectivity",
        "tidal_exposure_proxy",
        "jrc_occurrence",
        "jrc_seasonality",
        "jrc_transition",  # v20.0: JRC transition for river discrimination
        "gmw_mangrove",
        "gmw_historical_mangrove",
        "ccdc_break",
        "ccdc_change_prob",
        "ccdc_ndvi_break_magnitude",
        "ccdc_mndwi_break_magnitude",
        "ccdc_pre_ndvi_slope",
        "ccdc_post_ndvi_slope",
        "ccdc_pre_mndwi_slope",
        "ccdc_post_mndwi_slope",
        "ccdc_recovery_direction",
        "ccdc_recent_break",
        "ccdc_break_recency_years",
        "pixel_stage_mean",
        "pixel_confidence_mean",
        "water_evidence_score",
        "veg_evidence_score",
        "bare_soil_score",
        "is_river",
        "river_probability",
        "is_near_river",
        "pixel_s1_fraction",
        "pixel_s2_fraction",
        "pixel_s3_fraction",
        "pixel_s4_fraction",
        "pixel_s5_fraction",
        "pixel_water_fraction",
        # v12.0: Previously disconnected features now connected
        "smri",
        "mavi",
        "soil_fraction",
        "nir_texture_contrast",
        "nir_texture_correlation",
        "nir_texture_entropy",
        "nir_texture_variance",
        "nir_texture_homogeneity",
        "nir_texture_asm",  # v17.0: ASM uniformity feature
    ]
    if include_sar:
        keep_bands.extend(
            [
                "vv_mean",
                "vv_texture",
                "vv_homogeneity",
                "vv_temporal_std",
                "vv_temporal_iqr",
                "sar_water_persistence",
                "sar_water_entropy",
                "sar_seasonality",
                "rvi",
                # v12.0: Previously disconnected SAR features
                "sdwi",
                "sar_water_likelihood",
            ]
        )
    if include_previous:
        keep_bands.extend(["previous_ndvi", "previous_mndwi"])
    band_names = features.bandNames()

    def safe_select(band_name, default_val=0):
        return ee.Image(
            ee.Algorithms.If(
                band_names.contains(band_name),
                features.select(band_name),
                ee.Image(default_val).rename(band_name),
            )
        ).rename(band_name)

    return ee.Image.cat([safe_select(band) for band in keep_bands]).toFloat()


def _build_stage_support_image(stage_conf):
    stage_image = ee.Image(stage_conf).select("stage").toInt()
    confidence = (
        ee.Image(stage_conf)
        .select("confidence")
        .rename("pixel_confidence_mean")
        .toFloat()
    )
    bands = [
        stage_image.rename("pixel_stage_mean").toFloat(),
        confidence,
        stage_image.gte(4).rename("pixel_water_fraction").toFloat(),
    ]
    for stage_id in range(1, 6):
        bands.append(
            stage_image.eq(stage_id).rename(f"pixel_s{stage_id}_fraction").toFloat()
        )
    for ext_band in ["water_evidence_score", "veg_evidence_score", "bare_soil_score"]:
        bands.append(ee.Image(stage_conf).select(ext_band).toFloat())
    return ee.Image.cat(bands)


def _add_indices_for_ccdc(img):
    from modules.m05_features import extract_optical_features

    feats = extract_optical_features(img)
    return ee.Image(img).addBands(feats.select(["ndvi", "mndwi"]))


def _compute_ccdc_breaks(optical_collection, aoi):
    if not config.EXTENSIONS.get("use_ccdc"):
        return None
    return apply_ccdc(optical_collection.map(_add_indices_for_ccdc), aoi)


def _is_transient_gee_error(exc):
    text = str(exc).lower()
    retry_tokens = [
        "503",
        "429",
        "quota",
        "deadline exceeded",
        "timed out",
        "temporarily unavailable",
        "internal error",
        "connection reset",
        "connection aborted",
        "service unavailable",
    ]
    return any(token in text for token in retry_tokens)


def _get_info_with_retry(ee_obj, label, retries=4, base_delay=1.5):
    last_error = None
    for attempt in range(retries):
        try:
            return ee_obj.getInfo()
        except Exception as exc:
            last_error = exc
            if attempt >= retries - 1 or not _is_transient_gee_error(exc):
                raise
            wait_s = base_delay * (2**attempt)
            print(f"  [RETRY] {label}: transient GEE error, retrying in {wait_s:.1f}s")
            _time.sleep(wait_s)
    raise last_error


def _compute_pixel_stage_summary(stage_image, aoi, scale):
    try:
        hist = _get_info_with_retry(
            stage_image.reduceRegion(
                reducer=ee.Reducer.frequencyHistogram(),
                geometry=aoi,
                scale=scale,
                maxPixels=1e9,
                bestEffort=True,
                tileScale=int(config.GEE_SAFE["tileScale"]),
            ),
            "pixel stage summary",
        )
        raw = (hist or {}).get("stage", {}) or {}
        counts = {}
        for k, v in raw.items():
            try:
                stage_id = int(float(k))
                count = int(v)
            except Exception:
                continue
            if 1 <= stage_id <= 5 and count > 0:
                counts[stage_id] = count
        if not counts:
            return 0, {}, 0
        total = sum(counts.values())
        dist = {s: round(c / total * 100, 1) for s, c in counts.items()}
        dominant = max(counts, key=counts.get)
        return dominant, dist, total
    except Exception as e:
        pipe_log.debug(f"Pixel stage summary failed: {e}")
        return 0, {}, 0


def _compute_object_stage_summary(polygons):
    if not polygons:
        return 0, {}, 0

    weighted = {}
    for polygon in polygons:
        try:
            stage_id = int(
                polygon.get("confirmed_stage") or polygon.get("raw_stage") or 0
            )
        except Exception:
            stage_id = 0
        if stage_id < 1 or stage_id > 5:
            continue
        area_m2 = float(polygon.get("area_m2", 0) or 0.0)
        confidence = float(polygon.get("confidence", 0) or 0.0)
        weight = max(0.25, confidence) * max(1.0, min(area_m2, 50000.0) / 5000.0)
        weighted[stage_id] = weighted.get(stage_id, 0.0) + weight

    if not weighted:
        return 0, {}, 0

    total = sum(weighted.values())
    dominant = max(weighted, key=weighted.get)
    distribution = {
        stage_id: round(weight / total * 100.0, 1)
        for stage_id, weight in weighted.items()
    }
    return dominant, distribution, len(polygons)


def _geometry_centroid_lonlat(geometry):
    if not isinstance(geometry, dict):
        return None, None

    coords = geometry.get("coordinates") or []
    gtype = geometry.get("type")
    xs = []
    ys = []

    if gtype == "Point" and len(coords) >= 2:
        return coords[0], coords[1]

    def _collect(points):
        for point in points:
            if (
                isinstance(point, (list, tuple))
                and len(point) >= 2
                and isinstance(point[0], (int, float))
                and isinstance(point[1], (int, float))
            ):
                xs.append(float(point[0]))
                ys.append(float(point[1]))

    if gtype == "Polygon":
        for ring in coords:
            _collect(ring)
    elif gtype == "MultiPolygon":
        for polygon in coords:
            for ring in polygon:
                _collect(ring)

    if not xs or not ys:
        return None, None
    return sum(xs) / len(xs), sum(ys) / len(ys)


def _feature_collection_to_local_geojson(fc, output_path, label, max_features=600):
    if fc is None:
        return []
    try:
        info = (
            _get_info_with_retry(
                ee.FeatureCollection(fc).limit(max_features),
                label,
                retries=4,
                base_delay=2.0,
            )
            or {}
        )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2)
        return info.get("features", []) or []
    except Exception as exc:
        pipe_log.warning(f"{label} export skipped: {exc}")
        return []


def _compute_scene_feature_summary(features, aoi, scale):
    summary_bands = ["ndvi", "mndwi", "cwi", "water_fraction"]
    try:
        available = set(features.bandNames().getInfo())
        chosen = [band for band in summary_bands if band in available]
        if not chosen:
            return {}
        stats = (
            _get_info_with_retry(
                features.select(chosen).reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=aoi,
                    scale=scale,
                    maxPixels=1e9,
                    bestEffort=True,
                    tileScale=int(config.GEE_SAFE["tileScale"]),
                ),
                "scene feature summary",
            )
            or {}
        )
        result = {}
        for band in chosen:
            value = stats.get(band)
            result[f"{band}_mean"] = float(value) if value is not None else None
        return result
    except Exception as e:
        pipe_log.debug(f"Scene feature summary failed: {e}")
        return {}


def _compute_mean_image_value(image, band_name, aoi, scale):
    try:
        stats = (
            _get_info_with_retry(
                ee.Image(image)
                .select(band_name)
                .reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=aoi,
                    scale=scale,
                    maxPixels=1e9,
                    bestEffort=True,
                    tileScale=int(config.GEE_SAFE["tileScale"]),
                ),
                f"{band_name} mean",
            )
            or {}
        )
        value = stats.get(band_name)
        return float(value) if value is not None else None
    except Exception as e:
        pipe_log.debug(f"Mean image value failed for {band_name}: {e}")
        return None


def _build_aoi_tiles(scale_m=None):
    tile_cfg = getattr(config, "TILE_PROCESSING", {})
    if not tile_cfg.get("enabled", False) or not config.EXTENSIONS.get(
        "use_tile_processing", False
    ):
        return [
            ee.Geometry.Rectangle(
                [
                    config.AOI["lon_min"],
                    config.AOI["lat_min"],
                    config.AOI["lon_max"],
                    config.AOI["lat_max"],
                ]
            )
        ]

    lon_min = float(config.AOI["lon_min"])
    lon_max = float(config.AOI["lon_max"])
    lat_min = float(config.AOI["lat_min"])
    lat_max = float(config.AOI["lat_max"])
    coarse_mode = scale_m is not None and float(scale_m) >= 30
    max_tile_width = max(
        0.005,
        float(
            tile_cfg.get(
                "coarse_max_tile_width_deg", tile_cfg.get("max_tile_width_deg", 0.03)
            )
        )
        if coarse_mode
        else float(tile_cfg.get("max_tile_width_deg", 0.03)),
    )
    max_tile_height = max(
        0.005,
        float(
            tile_cfg.get(
                "coarse_max_tile_height_deg", tile_cfg.get("max_tile_height_deg", 0.03)
            )
        )
        if coarse_mode
        else float(tile_cfg.get("max_tile_height_deg", 0.03)),
    )
    overlap = max(
        0.0,
        float(
            tile_cfg.get(
                "coarse_tile_overlap_deg", tile_cfg.get("tile_overlap_deg", 0.0)
            )
        )
        if coarse_mode
        else float(tile_cfg.get("tile_overlap_deg", 0.0)),
    )

    tiles = []
    lat_cursor = lat_min
    while lat_cursor < lat_max:
        next_lat = min(lat_max, lat_cursor + max_tile_height)
        lon_cursor = lon_min
        while lon_cursor < lon_max:
            next_lon = min(lon_max, lon_cursor + max_tile_width)
            tiles.append(
                ee.Geometry.Rectangle(
                    [
                        max(lon_min, lon_cursor - overlap),
                        max(lat_min, lat_cursor - overlap),
                        min(lon_max, next_lon + overlap),
                        min(lat_max, next_lat + overlap),
                    ]
                )
            )
            lon_cursor = next_lon
        lat_cursor = next_lat

    max_tiles = (
        int(tile_cfg.get("coarse_max_tiles", tile_cfg.get("max_tiles", 25)))
        if coarse_mode
        else int(tile_cfg.get("max_tiles", 25))
    )
    return (
        tiles[:max_tiles]
        if tiles
        else [ee.Geometry.Rectangle([lon_min, lat_min, lon_max, lat_max])]
    )


def _dedupe_candidate_dicts(candidates, distance_threshold_m=20.0):
    deduped = []
    for cand in candidates:
        lon = cand.get("centroid_lon")
        lat = cand.get("centroid_lat")
        if lon is None or lat is None:
            deduped.append(cand)
            continue

        replaced = False
        for idx, existing in enumerate(deduped):
            ex_lon = existing.get("centroid_lon")
            ex_lat = existing.get("centroid_lat")
            if ex_lon is None or ex_lat is None:
                continue
            dist = haversine(lon, lat, ex_lon, ex_lat)
            if dist <= distance_threshold_m:
                existing_area = float(existing.get("area_m2", 0) or 0)
                current_area = float(cand.get("area_m2", 0) or 0)
                if current_area > existing_area:
                    deduped[idx] = cand
                replaced = True
                break

        if not replaced:
            deduped.append(cand)
    return deduped


def _candidate_priority(candidate):
    area = float(candidate.get("area_m2", 0) or 0.0)
    rectangularity = float(candidate.get("rectangularity", 0) or 0.0)
    compactness = float(candidate.get("compactness", 0) or 0.0)
    water_fraction = float(candidate.get("water_fraction", 0) or 0.0)
    mndwi = float(candidate.get("mndwi", candidate.get("mndwi_mean", 0)) or 0.0)
    ndwi = float(candidate.get("ndwi", candidate.get("ndwi_mean", 0)) or 0.0)
    cwi = float(candidate.get("cwi", candidate.get("cwi_mean", 0)) or 0.0)
    pixel_s4 = float(candidate.get("pixel_s4_fraction", 0) or 0.0)
    pixel_s5 = float(candidate.get("pixel_s5_fraction", 0) or 0.0)
    pixel_conf = float(candidate.get("pixel_confidence_mean", 0) or 0.0)

    water_score = (
        max(0.0, mndwi)
        + max(0.0, ndwi)
        + max(0.0, cwi)
        + max(0.0, water_fraction)
        + max(0.0, pixel_s4 + pixel_s5) * max(0.25, pixel_conf)
    )
    shape_score = max(0.0, rectangularity) * 1.2 + max(0.0, compactness) * 0.6
    area_score = min(max(area, 0.0), 40000.0) / 40000.0
    return water_score * 2.0 + shape_score + area_score


def _stage_distribution_probabilities(stage_dist, confidence=0.5):
    probs = {}
    total = 0.0
    for stage_id in range(1, 6):
        value = (
            float(stage_dist.get(stage_id, stage_dist.get(str(stage_id), 0.0)) or 0.0)
            / 100.0
        )
        probs[stage_id] = max(0.0, value)
        total += probs[stage_id]

    if total <= 0:
        dominant = 0
        conf = max(0.05, min(0.95, float(confidence or 0.5)))
        spread = (1.0 - conf) / 5.0
        return {str(stage_id): round(spread, 4) for stage_id in range(1, 6)}

    normalized = {stage_id: probs[stage_id] / total for stage_id in probs}
    return {str(stage_id): round(prob, 4) for stage_id, prob in normalized.items()}


# (Removing old compute_stage_summary and extract_polygon_info)


# ─────────────────────────────────────────────
# Historical Image Stream (FIX BUG-03, BUG-13, BUG-14)
# ─────────────────────────────────────────────


def _default_historical_years():
    return [2024]


def _historical_sensor_priority(year):
    if year <= 1999:
        return ["landsat5"]
    if year <= 2012:
        return ["landsat5", "landsat7"]
    if year <= 2016:
        return ["hls_l30", "hls_s30", "landsat7"]
    return ["sentinel2_sr", "hls_s30", "hls_l30"]


def _build_historical_stream(data, aoi, years=None):
    """
    Build chronological stream of best-available cloud-free composites.

    FIX BUG-03: Landsat 7 post-2003 (SLC-off) removed from defaults.
    FIX BUG-13: Single median reducer across all optical bands.
                Previous split (p25 water / p50 veg) broke spectral ratios.
    FIX BUG-14: Dry-season filter now Nov–Feb per config.DRY_SEASON_MONTHS.

    Epoch design rationale:
      1988 L5  - pre-aquaculture baseline (Coringa WLS had intact mangroves)
      1993 L5  - early aquaculture expansion period
      2000 L7  - pre-SLC failure, good quality
      2003 L7  - last good L7 year (SLC failed May 31 2003)
      2013 L8  - post-cloud-free Landsat 8 era
      2017 L8  - continued expansion
      2022 S2  - modern era high-res reference
    """
    # (year, sensor_priority_list)
    # Landsat 8 launched April 11 2013 - its FIRST dry-season data is
    # Nov 2013 – Feb 2014. Use epoch year=2014 so the Nov-Feb window
    # captures those images. Epoch year=2013 would open Nov 2012 – Mar 2013,
    # a period with zero L8 data (launches are too late in 2013).
    # Minimal but meaningful chronology for stage progression.
    # Note: for 2017+ we will later prefer 10m Sentinel-2 SR where available,
    # but HLS S30 provides a consistent baseline today.
    selected_years = years or _default_historical_years()
    epochs = [(year, _historical_sensor_priority(year)) for year in selected_years]

    stream = []
    all_bands = ["blue", "green", "red", "nir", "swir1", "swir2"]

    for year, sensors in epochs:
        # Multi-Temporal Gap Filling (±1 year window)
        # Captures moving clouds across a 3-year span and filters naturally via median.
        start = f"{year - 1}-01-01"
        end = f"{year + 1}-12-31"
        found = False

        for sensor in sensors:
            if sensor not in data or data[sensor] is None:
                continue
            try:
                # 1. Expand to 3-year window
                col_3yr = data[sensor].filterDate(start, end)

                # 2. Filter strictly by dry season (single source of truth)
                from modules.m01_acquisition import get_dry_season_filter

                col = col_3yr.filter(get_dry_season_filter())

                proc_temporal = preprocess_optical_collection(col_3yr, sensor, aoi)
                proc = preprocess_optical_collection(col, sensor, aoi)
                size = proc.size().getInfo()

                if size == 0:
                    continue

                # Cross-sensor harmonization for L5/L7 to HLS baseline
                if sensor in ["landsat5", "landsat7"] and config.EXTENSIONS.get(
                    "use_harmonization", True
                ):
                    from modules.m02_optical_preprocess import harmonize_to_hls

                    proc = harmonize_to_hls(proc, sensor, aoi)
                    proc_temporal = harmonize_to_hls(proc_temporal, sensor, aoi)

                ccdc_breaks = _compute_ccdc_breaks(proc_temporal, aoi)

                # Optical Temporal Features (extracted from full window before compositing)
                from modules.m05_features import compute_optical_temporal_feature_cube

                temporal_source = (
                    proc_temporal
                    if config.TEMPORAL_FEATURES.get(
                        "use_full_year_temporal_context", True
                    )
                    else proc
                )
                opt_temporal = compute_optical_temporal_feature_cube(temporal_source)

                # Use a physically consistent optical composite.
                # Percentile compositing across all bands distorts spectral ratios and
                # degrades stage logic, especially NDVI/MNDWI boundaries.
                image = create_composite(
                    proc.select(all_bands), aoi, method="median"
                ).rename(all_bands)

                # Check valid pixel fraction (from the composite mask)
                # to decide if we should drop this epoch due to too many clouds
                try:
                    mask_mean = (
                        image.select("blue")
                        .mask()
                        .reduceRegion(
                            reducer=ee.Reducer.mean(),
                            geometry=aoi,
                            scale=120,  # coarse scale to save computation
                            maxPixels=1e9,
                            bestEffort=True,
                            tileScale=8,
                        )
                        .getInfo()
                        .get("blue")
                    )

                    if mask_mean is not None and mask_mean < 0.50:
                        print(
                            f"  [{year}] {sensor}: Dropped composite due to >50% gaps (only {mask_mean * 100:.1f}% valid pixels)"
                        )
                        continue
                except Exception as e:
                    print(
                        f"  [{year}] {sensor}: Error checking cloud cover, skipping. {e}"
                    )
                    continue

                label_map = {
                    "landsat5": "Landsat 5",
                    "landsat7": "Landsat 7",
                    "hls_l30": "HLS L30",
                    "hls_s30": "HLS S30",
                    "sentinel2_sr": "Sentinel-2 SR",
                }

                stream.append(
                    {
                        "image": image,
                        "opt_temporal": opt_temporal,
                        "date": f"{year}-01-15",
                        "sensor": label_map.get(sensor, sensor),
                        "sensor_key": sensor,
                        "year": year,
                        "image_count": size,
                        "ccdc_breaks": ccdc_breaks,
                    }
                )
                print(
                    f"  [{year}] {label_map.get(sensor, sensor)}: "
                    f"{size} images -> stream (composite ready)"
                )
                found = True
                break

            except Exception as e:
                print(f"  [{year}] {sensor} error: {e}")
                continue

        if not found:
            print(f"  [{year}] No data - skipped")

    print(f"[STREAM] {len(stream)} images in chronological stream")
    return stream


def _build_operational_stream(data, aoi):
    from modules.m01_acquisition import get_hls_operational
    from modules.m05_features import compute_optical_temporal_feature_cube

    s2 = get_hls_operational(aoi)
    s2_proc = preprocess_optical_collection(s2, "hls_s30", aoi)
    size = s2_proc.size().getInfo()
    if size == 0:
        s2 = get_hls_operational(aoi, days_back=90)
        s2_proc = preprocess_optical_collection(s2, "hls_s30", aoi)

    all_bands = ["blue", "green", "red", "nir", "swir1", "swir2"]

    comp_method = (
        "p25" if config.EXTENSIONS.get("use_tidal_normalization") else "median"
    )
    latest = create_composite(
        s2_proc.select(all_bands), aoi, method=comp_method
    ).rename(all_bands)
    opt_temporal = compute_optical_temporal_feature_cube(s2_proc)
    ccdc_breaks = None
    if config.EXTENSIONS.get("use_ccdc"):
        ccdc_days_back = max(
            90,
            int(
                getattr(config, "TEMPORAL_FEATURES", {}).get(
                    "operational_ccdc_days_back", 365
                )
            ),
        )
        try:
            ccdc_source = preprocess_optical_collection(
                get_hls_operational(aoi, days_back=ccdc_days_back),
                "hls_s30",
                aoi,
            )
            if ccdc_source.size().getInfo() >= 6:
                ccdc_breaks = _compute_ccdc_breaks(ccdc_source, aoi)
            else:
                pipe_log.info(
                    "Operational CCDC skipped: fewer than 6 observations in lookback window"
                )
        except Exception as ccdc_err:
            pipe_log.warning(f"Operational CCDC unavailable: {ccdc_err}")

    return [
        {
            "image": latest,
            "opt_temporal": opt_temporal,
            "date": datetime.utcnow().strftime("%Y-%m-%d"),
            "sensor": "Sentinel-2",
            "sensor_key": "sentinel2",
            "year": datetime.utcnow().year,
            "image_count": 1,
            "ccdc_breaks": ccdc_breaks,
        }
    ]


def build_image_stream(data, aoi, mode="historical", years=None):
    print("\n[STREAM] Building chronological image stream...")
    if mode == "operational":
        return _build_operational_stream(data, aoi)
    return _build_historical_stream(data, aoi, years=years)


# ─────────────────────────────────────────────
# Closest SAR
# ─────────────────────────────────────────────


def find_closest_sar(data, date_str, aoi):
    if "sentinel1" not in data or data["sentinel1"] is None:
        return None, None
    year = date_str[:4]
    s1_temporal = data["sentinel1"].filterDate(f"{year}-01-01", f"{year}-12-31")
    s1 = data["sentinel1"].filterDate(f"{year}-11-01", f"{int(year) + 1}-03-31")
    size = s1.size().getInfo()
    if size == 0:
        return None, None
    s1_proc = preprocess_s1_collection(s1, aoi)
    s1_temporal_proc = preprocess_s1_collection(
        s1_temporal if s1_temporal.size().getInfo() > 0 else s1, aoi
    )

    from modules.m05_features import compute_sar_temporal_features

    sar_temporal = compute_sar_temporal_features(s1_temporal_proc)

    return s1_proc.median(), sar_temporal


# ─────────────────────────────────────────────
# STREAM PROCESSOR
# ─────────────────────────────────────────────


def process_stream(stream, data, aoi_info, mode="historical"):
    """
    Object-Based Stream Processor (v3.0)
    Extracts geometric properties, matches to PondRegistry, and assigns stages per-object.
    All major steps logged to outputs/pipeline.log via pipe_log.
    """
    pipe_log.info(f"Starting stream processing: mode={mode}, {len(stream)} epoch(s)")
    stream_start = _time.time()
    aoi = aoi_info["geometry"]
    reset_run_state = mode == "historical"
    registry = PondRegistry(mode=mode, reset=reset_run_state)
    alerts = AlertStore(reset=reset_run_state)
    scene_persistence = PersistenceEngine(
        required_count=config.PERSISTENCE_REQUIRED_COUNT.get(mode, 1),
        mode=mode,
    )
    all_results = []
    all_polygons = []

    previous_ndvi = None
    previous_mndwi = None
    previous_stage_image = None
    gmw_epoch_cache = {}
    historical_gmw_anchor = None
    gt_fc = None
    gt_meta = {"available": False}
    gt_overlay_features = []
    gt_polygon_list = []

    if data.get("mangrove_baseline") is not None:
        try:
            historical_gmw_anchor = get_historical_mangrove_anchor(aoi)
        except Exception as anchor_err:
            pipe_log.warning(f"Historical GMW anchor unavailable: {anchor_err}")

    if not getattr(config, "PERFORMANCE", {}).get("skip_ground_truth_eval", True):
        try:
            gt_fc, gt_meta = load_converted_ponds(aoi=aoi)
            gt_overlay_features = _feature_collection_to_local_geojson(
                gt_fc,
                os.path.join(config.WEB_DATA_DIR, "ground_truth.geojson"),
                "ground truth export",
                max_features=500,
            )
            for feature in gt_overlay_features:
                geometry = feature.get("geometry")
                lon, lat = _geometry_centroid_lonlat(geometry)
                gt_polygon_list.append(
                    {
                        "geometry": geometry,
                        "lon": lon,
                        "lat": lat,
                        "label": (feature.get("properties") or {}).get("name") or "GT",
                        "show_label": False,
                    }
                )
        except Exception as gt_load_err:
            pipe_log.warning(f"Ground truth preload skipped: {gt_load_err}")

    print("\n" + "=" * 70)
    print("  STREAM PROCESSING (OBJECT-BASED TRACKING)")
    print("=" * 70)

    for i, item in enumerate(stream):
        image = item["image"]
        opt_temporal = item.get("opt_temporal")
        date_str = item["date"]
        year = item["year"]
        sensor = item["sensor"]
        perf = getattr(config, "PERFORMANCE", {})

        print(f"\n[{i + 1}/{len(stream)}] {date_str} ({sensor})")

        # ── 1. Feature Extraction ──
        if year >= 2015:
            sar_image, sar_temporal = find_closest_sar(data, date_str, aoi)
        else:
            sar_image, sar_temporal = None, None

        srtm = data.get("glo30") or data.get(
            "srtm"
        )  # FIX: acquire_all uses "glo30" key
        soilgrids = data.get("soilgrids")
        jrc_water = data.get("jrc_water")
        features = extract_all_features(
            image,
            sar_image,
            aoi,
            srtm,
            soilgrids,
            jrc_water,
            sar_temporal,
            opt_temporal,
        )

        # ── GMW Baseline Anchor (Phase 2: Classification anchor for S1) ──
        # If Global Mangrove Watch data is available, add as binary band
        # This gives the stage engine prior knowledge about where mangroves existed
        gmw_baseline = data.get("mangrove_baseline")
        if gmw_baseline is not None:
            try:
                if year not in gmw_epoch_cache:
                    gmw_epoch_cache[year] = get_gmw_epoch_extent(aoi, year)
                gmw_binary = ee.Image(gmw_epoch_cache[year]).rename("gmw_mangrove")
                features = features.addBands(gmw_binary, overwrite=True)
                if historical_gmw_anchor is not None:
                    features = features.addBands(
                        ee.Image(historical_gmw_anchor), overwrite=True
                    )
            except Exception as gmw_err:
                pipe_log.warning(
                    f"  [{year}] GMW baseline injection skipped: {gmw_err}"
                )

        # ── Diagnostic Logging ──
        # Avoid heavy client-side calls (.getInfo()) in the main loop, which can
        # trigger GEE timeouts on slower connections. Keep only lightweight logs.
        try:
            pipe_log.info(f"  [{year}] Feature stack prepared")
        except Exception:
            pass

        threshold_scale = max(config.TARGET_SCALE, int(perf.get("threshold_scale", 30)))
        if sar_image is not None:
            features = precompute_sar_thresholds(features, aoi, scale=threshold_scale)
        features = precompute_optical_thresholds(features, aoi, scale=threshold_scale)
        print(f"  Thresholds prepared at {threshold_scale} m")

        if previous_ndvi is not None:
            features = features.addBands(previous_ndvi.rename("previous_ndvi"))
        if previous_mndwi is not None:
            features = features.addBands(previous_mndwi.rename("previous_mndwi"))

        # ── Pixel-wise stage map (S1-S5) ──
        # This is the core "stage-wise mangrove -> aquaculture" signal.
        # Object-based ponds are derived separately and overlaid for inspection.
        ccdc_breaks = item.get("ccdc_breaks")
        features = merge_ccdc_features(features, ccdc_breaks)
        stage_conf = classify_image(
            features,
            aoi,
            previous_stage=previous_stage_image,
            ccdc_breaks=ccdc_breaks,
            mode=mode,
        )

        stage_support = _build_stage_support_image(stage_conf)
        detection_features = features.addBands(stage_support, overwrite=True)
        pixel_stage_image = stage_conf.select("stage")
        print("  Pixel stage graph built")
        pixel_summary_scale = max(
            config.get_scale_for_sensor(item.get("sensor_key", "landsat5")), 20
        )
        pixel_dominant_stage, pixel_stage_dist, pixel_stage_pixels = (
            _compute_pixel_stage_summary(pixel_stage_image, aoi, pixel_summary_scale)
        )
        if pixel_stage_dist:
            print(f"  Pixel summary: dominant S{pixel_dominant_stage} from raster")
        scene_feature_summary = {}
        pixel_confidence_mean = _compute_mean_image_value(
            stage_conf, "confidence", aoi, pixel_summary_scale
        )
        if perf.get("compute_scene_feature_summary", False):
            scene_feature_summary = _compute_scene_feature_summary(
                features, aoi, pixel_summary_scale
            )

        # ── 2. Candidate Polygon Extraction ──
        # Keep pixel classification at native quality, but use a coarser scale
        # for object extraction to avoid hour-long reduceRegions stalls.
        sensor_key = item.get("sensor_key", "landsat5")
        vector_scale = max(
            config.get_scale_for_sensor(sensor_key),
            int(perf.get("min_object_scale", 20)),
        )
        aoi_tiles = _build_aoi_tiles(vector_scale)
        candidates_fc = None
        object_features = None
        max_reduce_candidates = int(perf.get("max_reduce_candidates", 220))
        coarse_reduce_candidates = int(
            perf.get("max_reduce_candidates_coarse", max_reduce_candidates)
        )
        max_reduce_candidates_per_tile = int(
            perf.get("max_reduce_candidates_per_tile", 60)
        )
        max_total_reduce_candidates = int(
            perf.get("max_total_reduce_candidates", max_reduce_candidates)
        )
        if vector_scale >= 30:
            max_reduce_candidates_per_tile = min(
                max_reduce_candidates_per_tile,
                int(
                    perf.get(
                        "max_reduce_candidates_per_tile_coarse",
                        max_reduce_candidates_per_tile,
                    )
                ),
            )
            max_total_reduce_candidates = min(
                max_total_reduce_candidates,
                int(
                    perf.get(
                        "max_total_reduce_candidates_coarse",
                        max_total_reduce_candidates,
                    )
                ),
            )
        reduce_tile_scale = max(
            1, min(16, int(perf.get("reduce_tile_scale", config.GEE_SAFE["tileScale"])))
        )
        # Avoid candidates_fc.size().getInfo() (can exceed GEE client-side limits).
        # We treat any failure during downstream reductions as "0 ponds".
        count = 0

        polygon_list = []
        display_polygons = []
        render_polygons = []
        dominant_stage = 0
        stage_dist = {}
        mean_conf = 0.5
        alerts_triggered_scene = 0
        min_object_conf = max(0.15, config.VALIDATION["confidence_threshold"] * 0.5)
        display_conf_threshold = max(
            0.20, config.VALIDATION["confidence_threshold"] * 0.6
        )
        object_extraction_start_year = int(
            perf.get("object_extraction_start_year", 2002)
        )

        if year < object_extraction_start_year:
            print(f"  Skipping object extraction for {year} (pixel-stage mode)")
        else:
            try:
                object_features = _build_object_feature_image(
                    detection_features,
                    include_sar=sar_image is not None,
                    include_previous=(
                        previous_ndvi is not None and previous_mndwi is not None
                    ),
                )
                effective_max_reduce_candidates = max_reduce_candidates
                if vector_scale >= 30:
                    effective_max_reduce_candidates = min(
                        max_reduce_candidates, coarse_reduce_candidates
                    )
                per_tile_limit = min(
                    max_reduce_candidates_per_tile, effective_max_reduce_candidates
                )
                print(
                    f"  Candidate extraction queued at {vector_scale} m across {len(aoi_tiles)} tile(s)"
                )
                # ── 3. Object Feature Reduction ──
                if len(aoi_tiles) > 1:
                    print(
                        f"  Reducing object features at {vector_scale} m "
                        f"(per-tile limit {per_tile_limit}, total cap {max_total_reduce_candidates})..."
                    )
                else:
                    print(
                        f"  Reducing object features at {vector_scale} m (max {effective_max_reduce_candidates} candidates)..."
                    )
                reduce_start = _time.time()
                pond_dicts = []
                tile_processing_enabled = len(aoi_tiles) > 1
                tile_limit = (
                    per_tile_limit
                    if tile_processing_enabled
                    else effective_max_reduce_candidates
                )
                for tile_idx, tile_geom in enumerate(aoi_tiles, start=1):
                    if tile_processing_enabled:
                        print(
                            f"    [Tile {tile_idx}/{len(aoi_tiles)}] extracting candidates..."
                        )
                    tile_candidates_fc = extract_pond_candidates(
                        detection_features, tile_geom, date_str, scale=vector_scale
                    )
                    reduced = object_features.reduceRegions(
                        collection=tile_candidates_fc.limit(tile_limit),
                        reducer=ee.Reducer.mean(),
                        scale=vector_scale,
                        tileScale=reduce_tile_scale,
                    )

                    reduced_info = _get_info_with_retry(
                        reduced,
                        f"{year} object reduction tile {tile_idx}",
                        retries=4,
                        base_delay=2.0,
                    )
                    features_list = reduced_info.get("features", [])
                    for cand in features_list:
                        props = cand.get("properties", {})
                        props["geometry"] = cand.get("geometry")
                        props["tile_id"] = tile_idx
                        pond_dicts.append(props)

                pond_dicts = _dedupe_candidate_dicts(
                    pond_dicts, distance_threshold_m=max(15.0, vector_scale * 1.5)
                )
                if len(pond_dicts) > max_total_reduce_candidates:
                    pond_dicts = sorted(
                        pond_dicts, key=_candidate_priority, reverse=True
                    )[:max_total_reduce_candidates]

                elapsed_s = _time.time() - reduce_start
                pipe_log.info(
                    f"[{year}] Extracted {len(pond_dicts)} candidate ponds in {elapsed_s:.1f}s"
                )
                print(
                    f"  Extracted {len(pond_dicts)} candidate ponds in {elapsed_s:.1f}s."
                )
                count = len(pond_dicts)

                # ── 4. ID Matching ──
                matched, unmatched = match_polygons_to_registry(
                    pond_dicts, registry.ponds
                )
                all_scene_ponds = matched + unmatched

                # ── 5. Classification & Lifecycle ──
                for pond_data in all_scene_ponds:
                    pid = pond_data.get("pond_id")
                    observation = classify_pond_observation(pond_data)
                    stage = observation["stage"]
                    confidence = observation["confidence"]
                    if stage <= 0 or confidence < min_object_conf:
                        continue

                    pond_data["stage"] = stage
                    pond_data["confidence"] = confidence
                    pond_data["validation_score"] = confidence
                    pond_data["stage_probabilities"] = observation.get(
                        "stage_probabilities", {}
                    )
                    pond_data["stage_scores"] = observation.get("stage_scores", {})
                    pond_data["uncertain"] = observation.get("uncertain", False)
                    pond_data["uncertainty_reason"] = observation.get(
                        "uncertainty_reason", ""
                    )

                    new_pid, confirmed, triggered = registry.register_or_update(
                        pid, date_str, pond_data
                    )
                    pond_data["confirmed_stage"] = confirmed

                    polygon_list.append(
                        {
                            "pond_id": new_pid,
                            "lat": pond_data.get("centroid_lat"),
                            "lon": pond_data.get("centroid_lon"),
                            "area_m2": pond_data.get("area_m2", 0),
                            "rectangularity": pond_data.get("rectangularity", 0),
                            "raw_stage": stage,
                            "confirmed_stage": confirmed,
                            "confidence": confidence,
                            "stage_probabilities": observation.get(
                                "stage_probabilities", {}
                            ),
                            "stage_scores": observation.get("stage_scores", {}),
                            "uncertain": observation.get("uncertain", False),
                            "uncertainty_reason": observation.get(
                                "uncertainty_reason", ""
                            ),
                            "date": date_str,
                            "geometry": pond_data.get("geometry"),
                            "merge_from_ids": pond_data.get("merge_from_ids", []),
                            "split_from_id": pond_data.get("split_from_id"),
                            "cluster_neighbor_count": pond_data.get(
                                "cluster_neighbor_count", 0
                            ),
                            "cluster_recent_water_ratio": pond_data.get(
                                "cluster_recent_water_ratio", 0
                            ),
                            "cluster_recent_conversion_ratio": pond_data.get(
                                "cluster_recent_conversion_ratio", 0
                            ),
                            "previous_confirmed_stage": pond_data.get(
                                "previous_confirmed_stage"
                            ),
                            "match_score": pond_data.get("match_score"),
                        }
                    )

                    if triggered:
                        print(
                            f"  [ALERT] Pond {new_pid} confirmed stage transition to S{confirmed}"
                        )
                        old_stage = (
                            registry.ponds[new_pid]["stage_history"][-2][
                                "confirmed_stage"
                            ]
                            if len(registry.ponds[new_pid]["stage_history"]) > 1
                            else 0
                        )
                        alerts.add_alert(
                            create_alert(
                                old_stage,
                                confirmed,
                                date_str,
                                1.0,
                                confidence,
                                stage_probability=observation.get(
                                    "stage_probabilities", {}
                                ).get(str(confirmed)),
                                uncertain=observation.get("uncertain", False),
                                uncertainty_reason=observation.get(
                                    "uncertainty_reason", ""
                                ),
                            )
                        )
                        alerts_triggered_scene += 1

                render_polygons = [
                    p for p in polygon_list if p.get("confirmed_stage", 0) > 0
                ]
                display_polygons = [
                    p
                    for p in render_polygons
                    if p.get("confidence", 0) >= display_conf_threshold
                ]
                all_polygons.extend(render_polygons)

                if len(render_polygons) > 0:
                    count = len(render_polygons)
                    mean_conf = sum(p["confidence"] for p in render_polygons) / len(
                        render_polygons
                    )
                    dominant_stage, stage_dist, _ = _compute_object_stage_summary(
                        render_polygons
                    )
            except Exception as e:
                print(f"  Error extracting ponds (GEE timeout/limit): {e}")
                print("  Continuing with pixel-stage outputs for this epoch.")

        if (
            not stage_dist
            and perf.get("use_pixel_stage_summary", True)
            and pixel_stage_dist
        ):
            dominant_stage = pixel_dominant_stage
            stage_dist = pixel_stage_dist
        if pixel_confidence_mean is not None and not display_polygons:
            mean_conf = pixel_confidence_mean

        # ── 6. Export thumbnails ──
        try:
            fc_features = []
            for p in display_polygons:
                geom = p.get("geometry")
                if geom:
                    fc_features.append(
                        ee.Feature(geom, {"stage": p.get("confirmed_stage", 0)})
                    )
                elif p.get("lon") and p.get("lat"):
                    pt = ee.Geometry.Point([p.get("lon"), p.get("lat")]).buffer(30)
                    fc_features.append(
                        ee.Feature(pt, {"stage": p.get("confirmed_stage", 0)})
                    )

            polygons_fc = ee.FeatureCollection(fc_features) if fc_features else None
            stage_export_image = ee.Image(pixel_stage_image).select("stage").toInt()
            if polygons_fc is not None:
                try:
                    # v12.0: Use FILLED paint for region consistency
                    polygon_stage_fill = (
                        ee.Image(0)
                        .paint(polygons_fc, "stage")  # Fill with stage value
                        .clip(aoi)
                        .rename("stage")
                        .toInt()
                    )
                    stage_export_image = (
                        stage_export_image.where(
                            polygon_stage_fill.gt(0), polygon_stage_fill
                        )
                        .rename("stage")
                        .toInt()
                    )
                except Exception as stage_overlay_err:
                    pipe_log.warning(
                        f"[{year}] Stage display fusion skipped: {stage_overlay_err}"
                    )

            # Export true pixel-wise stage map (S1-S5).
            # Early epochs can exceed GEE memory when exported at full size; export them coarser.
            if year < 2013:
                hist_stage_size = int(perf.get("historical_stage_thumb_size", 768))
                hist_rgb_size = int(perf.get("historical_rgb_thumb_size", 768))
                hist_feat_size = int(perf.get("historical_feature_thumb_size", 768))
                # v23.0: Use explicit coarse scale for historical exports
                hist_stage_scale = int(perf.get("historical_stage_export_scale", 60))
                export_stage_thumbnail(
                    stage_export_image,
                    aoi,
                    f"{year}.png",
                    polygon_list=display_polygons,
                    thumb_size=hist_stage_size,
                    scale_m=hist_stage_scale,  # v23.0: Force coarse scale
                )
                export_rgb_thumbnail(
                    image,
                    aoi,
                    f"{year}.png",
                    None,
                    None,
                    thumb_size=hist_rgb_size,
                    scale_m=None,
                )
                export_detection_overlay_thumbnail(
                    image,
                    aoi,
                    f"{year}.png",
                    polygon_list=display_polygons,
                    thumb_size=hist_rgb_size,
                    scale_m=None,
                )
                if gt_polygon_list:
                    export_ground_truth_overlay_thumbnail(
                        image,
                        aoi,
                        f"{year}.png",
                        gt_polygon_list=gt_polygon_list,
                        thumb_size=hist_rgb_size,
                        scale_m=None,
                    )
                export_ndvi_thumbnail(
                    features,
                    aoi,
                    f"{year}.png",
                    polygons_fc,
                    render_polygons,
                    thumb_size=hist_feat_size,
                    scale_m=None,
                )
                export_mndwi_thumbnail(
                    features,
                    aoi,
                    f"{year}.png",
                    polygons_fc,
                    render_polygons,
                    thumb_size=hist_feat_size,
                    scale_m=None,
                )
            else:
                modern_stage_scale = max(
                    10, int(perf.get("modern_stage_export_scale", 20))
                )
                modern_feature_scale = max(
                    10, int(perf.get("modern_feature_export_scale", modern_stage_scale))
                )
                export_stage_thumbnail(
                    stage_export_image,
                    aoi,
                    f"{year}.png",
                    polygon_list=display_polygons,
                    thumb_size=None,
                    scale_m=modern_stage_scale,
                )
                export_rgb_thumbnail(image, aoi, f"{year}.png", None, None)
                export_detection_overlay_thumbnail(
                    image, aoi, f"{year}.png", polygon_list=display_polygons
                )
                if gt_polygon_list:
                    export_ground_truth_overlay_thumbnail(
                        image, aoi, f"{year}.png", gt_polygon_list=gt_polygon_list
                    )
                export_ndvi_thumbnail(
                    features,
                    aoi,
                    f"{year}.png",
                    polygons_fc,
                    render_polygons,
                    scale_m=modern_feature_scale,
                )
                export_mndwi_thumbnail(
                    features,
                    aoi,
                    f"{year}.png",
                    polygons_fc,
                    render_polygons,
                    scale_m=modern_feature_scale,
                )

            if not perf.get("export_core_features_only", True):
                extra_feature_scale = None if year < 2013 else modern_feature_scale
                export_ndwi_thumbnail(
                    features,
                    aoi,
                    f"{year}.png",
                    polygons_fc,
                    render_polygons,
                    scale_m=extra_feature_scale,
                )
                export_savi_thumbnail(
                    features,
                    aoi,
                    f"{year}.png",
                    polygons_fc,
                    render_polygons,
                    scale_m=extra_feature_scale,
                )
                export_ndbi_thumbnail(
                    features,
                    aoi,
                    f"{year}.png",
                    polygons_fc,
                    render_polygons,
                    scale_m=extra_feature_scale,
                )
                export_awei_thumbnail(
                    features,
                    aoi,
                    f"{year}.png",
                    polygons_fc,
                    render_polygons,
                    scale_m=extra_feature_scale,
                )

            if sar_image is not None and not perf.get(
                "export_core_features_only", True
            ):
                extra_feature_scale = None if year < 2013 else modern_feature_scale
                export_sar_thumbnail(
                    sar_image,
                    aoi,
                    f"{year}.png",
                    polygons_fc,
                    render_polygons,
                    scale_m=extra_feature_scale,
                )
                export_rvi_thumbnail(
                    features,
                    aoi,
                    f"{year}.png",
                    polygons_fc,
                    render_polygons,
                    scale_m=extra_feature_scale,
                )

            # v3.1: POLYGON-ONLY THUMBNAIL — separate from stage map
            if display_polygons:
                try:
                    # Build ee.FeatureCollection from display polygon dicts
                    poly_fc_features = []
                    for p in display_polygons:
                        geom = p.get("geometry")
                        stage = p.get("confirmed_stage", 0)
                        if geom and stage > 0:
                            poly_fc_features.append(
                                ee.Feature(ee.Geometry(geom), {"stage": stage})
                            )
                        elif p.get("lon") and p.get("lat") and stage > 0:
                            pt = ee.Geometry.Point([p["lon"], p["lat"]]).buffer(30)
                            poly_fc_features.append(ee.Feature(pt, {"stage": stage}))
                    if poly_fc_features:
                        poly_fc = ee.FeatureCollection(poly_fc_features)
                        poly_outline = paint_polygon_outlines(poly_fc, aoi)
                        poly_fname = f"polygons_{year}.png"
                        poly_scale = modern_stage_scale if year >= 2013 else None
                        poly_size = 1280 if year < 2013 else None
                        export_stage_thumbnail(
                            poly_outline.selfMask().rename("stage"),
                            aoi,
                            poly_fname,
                            poly_fc,
                            polygon_list=display_polygons,
                            thumb_size=poly_size,
                            scale_m=poly_scale,
                        )
                        pipe_log.info(
                            f"[{year}] Exported polygon-only thumbnail: {poly_fname} ({len(poly_fc_features)} polygons)"
                        )
                except Exception as poly_err:
                    pipe_log.warning(
                        f"[{year}] Polygon-only thumbnail failed: {poly_err}"
                    )

            export_jrc_thumbnail(data.get("jrc_water"), aoi)
            export_glo30_thumbnail(data.get("glo30") or data.get("srtm"), aoi)
            pipe_log.info(f"[{year}] Thumbnail exports completed")
        except Exception as e:
            pipe_log.error(f"[{year}] Thumbnail error: {e}")
            print(f"  Thumbnail error: {e}")

        # Phase 1: Full validation with Kappa (m17/m11)
        # v22.0: Gated behind config flag — stratified sampling + Kappa is expensive
        # (~15-30s per epoch). Only run for the LAST epoch or when explicitly enabled.
        kappa_value = 0.0
        kappa_result = {'available': False}
        gt_eval = {"available": False, "reason": "skipped"}
        run_per_epoch_kappa = (
            not perf.get("skip_per_epoch_kappa", True)
            or i == len(stream) - 1  # Always run for the last epoch
        )
        if run_per_epoch_kappa:
            try:
                from modules.m11_accuracy import run_validation
                full_metrics = run_validation(pixel_stage_image, data, aoi, gt_fc)
                gt_eval = full_metrics.get('polygon_matching', {"available": False})
                kappa_result = full_metrics.get('kappa_analysis', {'available': False})
                kappa_value = float(kappa_result.get('kappa', 0))
                if kappa_result.get('available', False):
                    print(f"  [KAPPA] v2.0: {kappa_value:.3f} ({kappa_result.get('n_samples', 0)} pts)")
                else:
                    print(f"  [KAPPA] unavailable: {kappa_result.get('reason', 'unknown')}")
            except Exception as val_err:
                print(f"  [VALIDATE] error: {val_err}")
                gt_eval = {"available": False, "reason": str(val_err)}
        else:
            print(f"  [KAPPA] skipped (non-final epoch; set skip_per_epoch_kappa=False to enable)")

        # ── 6c. Per-epoch GMW accuracy comparison ──
        gmw_result = {"gmw_available": False, "reason": "not_run"}
        if not perf.get("skip_gmw_comparison", True):
            try:
                gmw_result = compare_with_gmw(
                    pixel_stage_image,
                    data.get("mangrove_baseline"),
                    aoi,
                    target_year=year,
                )
                if gmw_result.get("gmw_available"):
                    pipe_log.info(
                        f"  [GMW] year={year} accuracy={gmw_result.get('accuracy', 0):.2%} "
                        f"F1={gmw_result.get('f1_score', 0):.2%}"
                    )
            except Exception as gmw_err:
                pipe_log.debug(f"  [GMW] comparison skipped: {gmw_err}")

        # ── 7. Log and track ──
        print(
            f"  Distributed Summary (Among Ponds): Dominant S{dominant_stage} | "
            f"Conf: {mean_conf:.3f} | "
            f"Ponds: {count} | "
            f"Alerts: {alerts_triggered_scene}"
        )

        result = {
            "date": date_str,
            "year": year,
            "sensor": sensor,
            "stage": dominant_stage,
            "stage_name": stage_label(dominant_stage)
            if dominant_stage
            else "Unclassified",
            "confirmed_stage": dominant_stage,
            "confirmed_stage_name": stage_label(dominant_stage)
            if dominant_stage
            else "Unclassified",
            "stage_distribution": stage_dist,
            "stage_probabilities": _stage_distribution_probabilities(
                stage_dist, mean_conf
            ),
            "stage_probability": round(float(mean_conf or 0.5), 3),
            "uncertain": False,
            "uncertainty_reason": "",
            "confidence": mean_conf,
            "alert_triggered": alerts_triggered_scene > 0,
            "polygon_count": count,
            "pixel_stage_distribution": pixel_stage_dist,
            "pixel_stage_pixels": pixel_stage_pixels,
            "pixel_confidence_mean": pixel_confidence_mean,
            "gt_validation": gt_eval,
            "gmw_validation": gmw_result,
            "kappa": kappa_value,
            "kappa_analysis": kappa_result,
        }
        result.update(scene_feature_summary)

        # ── v6.0: Integrate m08_validator for actual validation scoring ──
        # Previously hardcoded to 1.0; now computes real multi-evidence scores.
        if perf.get("skip_expensive_validation", True):
            result["validation_score"] = round(float(mean_conf or 0.5), 3)
            result["mangrove_score"] = 0.5
            result["water_score"] = 0.5
            result["elevation_score"] = 0.5
            result["sar_score"] = 0.5
            print(f"  [VALIDATE] fast mode -> val={result['validation_score']:.3f}")
        else:
            try:
                from modules.m08_validator import compute_validation_score

                confidence_image = stage_conf.select("confidence")
                val_scores = compute_validation_score(
                    pixel_stage_image,
                    confidence_image,
                    features,
                    data.get("mangrove_baseline"),
                    data.get("jrc_water"),
                    data.get("glo30") or data.get("srtm"),
                    aoi,
                    jrc_monthly=data.get("jrc_monthly"),
                    image_year=year,
                )
                # Extract server-side ee.Number values
                val_score_val = val_scores["validation_score"].getInfo()
                mangrove_s = val_scores["mangrove_score"].getInfo()
                water_s = val_scores["water_score"].getInfo()
                elevation_s = val_scores["elevation_score"].getInfo()
                sar_s = val_scores["sar_score"].getInfo()

                result["validation_score"] = round(float(val_score_val or 0.5), 3)
                result["mangrove_score"] = round(float(mangrove_s or 0.5), 3)
                result["water_score"] = round(float(water_s or 0.5), 3)
                result["elevation_score"] = round(float(elevation_s or 0.5), 3)
                result["sar_score"] = round(float(sar_s or 0.5), 3)
                print(
                    f"  [VALIDATE] val={result['validation_score']:.3f} "
                    f"mangrove={result['mangrove_score']:.3f} "
                    f"water={result['water_score']:.3f}"
                )
            except Exception as val_err:
                fallback_val = round(float(mean_conf or 0.5), 3)
                pipe_log.warning(
                    f"  [{year}] Validation scoring error (reverting to confidence={fallback_val:.3f}): {val_err}"
                )
                result["validation_score"] = fallback_val
                result["mangrove_score"] = 0.5
                result["water_score"] = 0.5
                result["elevation_score"] = 0.5
                result["sar_score"] = 0.5

        scene_event = scene_persistence.process(
            date_str,
            dominant_stage,
            result["validation_score"],
            mean_conf,
            stage_distribution=stage_dist,
            stage_probabilities=result.get("stage_probabilities"),
        )
        result["confirmed_stage"] = scene_event["stage"]
        result["confirmed_stage_name"] = stage_label(scene_event["stage"])
        result["stage_probability"] = scene_event.get(
            "stage_probability", result["stage_probability"]
        )
        result["uncertain"] = scene_event.get("uncertain", False)
        result["uncertainty_reason"] = scene_event.get("uncertainty_reason", "")
        result["stage_probabilities"] = scene_event.get(
            "stage_probabilities", result.get("stage_probabilities", {})
        )
        if scene_event.get("alert"):
            alerts.add_alert(
                create_alert(
                    scene_event.get("old_stage"),
                    scene_event["stage"],
                    date_str,
                    result["validation_score"],
                    mean_conf,
                    stage_probability=scene_event.get("stage_probability"),
                    uncertain=scene_event.get("uncertain", False),
                    uncertainty_reason=scene_event.get("uncertainty_reason", ""),
                )
            )
            result["alert_triggered"] = True

        try:
            record_image_event(
                date_str,
                result["confirmed_stage"],
                result["validation_score"],
                mean_conf,
                validation_scores={
                    "mangrove_score": result.get("mangrove_score"),
                    "water_score": result.get("water_score"),
                    "elevation_score": result.get("elevation_score"),
                    "sar_score": result.get("sar_score"),
                },
                polygon_count=count,
                stage_probability=result.get("stage_probability"),
                uncertain=result.get("uncertain", False),
                uncertainty_reason=result.get("uncertainty_reason", ""),
                stage_probabilities=result.get("stage_probabilities", {}),
            )
        except Exception as history_err:
            pipe_log.debug(f"  [{year}] stage history record skipped: {history_err}")

        all_results.append(result)

        previous_ndvi = features.select("ndvi")
        previous_mndwi = features.select("mndwi")
        previous_stage_image = pixel_stage_image

        registry.save()

    return all_results, all_polygons


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────


def main():
    args = parse_args()
    start_time = _time.time()
    selected_years = _parse_year_list(args.years)

    print("=" * 70)
    print("  MANGROVE TO AQUACULTURE TRANSITION DETECTION")
    print(f"  AOI: {config.AOI['name']}")
    print(
        f"  Bounds: {config.AOI['lon_min']}–{config.AOI['lon_max']}°E, "
        f"{config.AOI['lat_min']}–{config.AOI['lat_max']}°N"
    )
    print(f"  Mode: {args.mode.upper()}")
    print("=" * 70)

    # Reset static export flags at pipeline start (FIX BUG-06)
    reset_static_export_flags()

    aoi_info = initialize_aoi()
    export_aoi_geojson(aoi_info)

    data = acquire_all(aoi_info["geometry"])
    if args.mode == "historical" and selected_years:
        print(f"  Selected years: {', '.join(str(year) for year in selected_years)}")
    stream = build_image_stream(
        data, aoi_info["geometry"], args.mode, years=selected_years
    )

    if not stream:
        print("[ERROR] No images in stream.")
        return

    all_results, all_polygons = process_stream(stream, data, aoi_info, mode=args.mode)

    if all_polygons:
        pond_path = os.path.join(config.POLYGON_DIR, "pond_locations.json")
        with open(pond_path, "w") as f:
            json.dump({"count": len(all_polygons), "ponds": all_polygons}, f, indent=2)
        print(f"\n[POLYGONS] {len(all_polygons)} ponds saved")

    print("\n[EXPORT] Generating web data...")
    export_all_web_data(None, all_results)  # Deprecated stage_history
    try:
        export_feature_audit()
    except Exception as audit_err:
        pipe_log.warning(f"Feature audit export skipped: {audit_err}")
    _export_dashboard_data(all_results, all_polygons)
    try:
        from compare_results import create_stage_progression_panel

        create_stage_progression_panel()
    except Exception as compare_err:
        pipe_log.warning(f"Comparison export skipped: {compare_err}")

    elapsed = _time.time() - start_time
    alerts_triggered = sum(1 for r in all_results if r["alert_triggered"])

    print(f"\n{'=' * 70}")
    print(f"  COMPLETE ({elapsed:.0f}s)")
    print(f"  Images processed: {len(all_results)}")
    print(f"  Alerts triggered: {alerts_triggered}")
    print(f"  Ponds detected:   {len(all_polygons)}")
    if all_results:
        print(f"  Date range: {all_results[0]['date']} -> {all_results[-1]['date']}")
        print(f"\n  Stage Timeline:")
        for r in all_results:
            marker = " [STAGE CHANGE]" if r["alert_triggered"] else ""
            print(
                f"    {r['date']} -> S{r['confirmed_stage']} "
                f"({r['sensor']}) "
                f"val={r['validation_score']:.3f}{marker}"
            )
    print("=" * 70)


def _export_dashboard_data(all_results, all_polygons):
    os.makedirs(config.WEB_DATA_DIR, exist_ok=True)

    available_years = sorted(
        {r["year"] for r in all_results if r.get("year") is not None}
    )
    timeline_meta = {
        "stage_names": {str(k): v for k, v in STAGE_LABELS.items()},
        "stage_colors": {str(k): v for k, v in STAGE_COLORS_HEX.items()},
        "available_years": available_years,
        "aoi": {
            "name": config.AOI["name"],
            "bbox": [
                config.AOI["lon_min"],
                config.AOI["lat_min"],
                config.AOI["lon_max"],
                config.AOI["lat_max"],
            ],
            "center_lat": config.WEB.get(
                "map_center_lat", (config.AOI["lat_min"] + config.AOI["lat_max"]) / 2
            ),
            "center_lon": config.WEB.get(
                "map_center_lon", (config.AOI["lon_min"] + config.AOI["lon_max"]) / 2
            ),
            "zoom": config.WEB.get("map_zoom", 13),
        },
    }

    timeline = []
    for r in all_results:
        timeline.append(
            {
                "date": r["date"],
                "year": r["year"],
                "sensor": r["sensor"],
                "stage": r["stage"],
                "confirmed_stage": r["confirmed_stage"],
                "stage_name": r.get("stage_name") or stage_label(r["stage"]),
                "confirmed_stage_name": r.get("confirmed_stage_name")
                or stage_label(r["confirmed_stage"]),
                "confidence": r["confidence"],
                "stage_probability": r.get("stage_probability"),
                "stage_probabilities": r.get("stage_probabilities", {}),
                "uncertain": r.get("uncertain", False),
                "uncertainty_reason": r.get("uncertainty_reason", ""),
                "validation_score": r["validation_score"],
                "mangrove_score": r.get("mangrove_score", 0.5),
                "water_score": r.get("water_score", 0.5),
                "elevation_score": r.get("elevation_score", 0.5),
                "sar_score": r.get("sar_score", 0.5),
                "alert": r["alert_triggered"],
                "polygon_count": r.get("polygon_count", 0),
                "stage_distribution": r.get("stage_distribution", {}),
            }
        )

    with open(os.path.join(config.WEB_DATA_DIR, "timeline.json"), "w") as f:
        json.dump({"timeline": timeline, **timeline_meta}, f, indent=2)
    print("[EXPORT] timeline.json saved")

    with open(os.path.join(config.WEB_DATA_DIR, "polygons.json"), "w") as f:
        json.dump({"ponds": all_polygons}, f, indent=2)
    print(f"[EXPORT] polygons.json saved ({len(all_polygons)} ponds)")

    polygon_features = []
    for p in all_polygons:
        geometry = p.get("geometry")
        if geometry is None and p.get("lon") is not None and p.get("lat") is not None:
            geometry = {
                "type": "Point",
                "coordinates": [p.get("lon"), p.get("lat")],
            }
        if geometry is None:
            continue
        confirmed_stage = p.get("confirmed_stage") or p.get("raw_stage") or 0
        props = {
            "pond_id": p.get("pond_id"),
            "date": p.get("date"),
            "stage": confirmed_stage,
            "stage_code": f"S{confirmed_stage}" if confirmed_stage else "S0",
            "stage_name": stage_label(confirmed_stage)
            if confirmed_stage
            else "Unclassified",
            "confidence": p.get("confidence"),
            "stage_probabilities": p.get("stage_probabilities", {}),
            "stage_scores": p.get("stage_scores", {}),
            "uncertain": p.get("uncertain", False),
            "uncertainty_reason": p.get("uncertainty_reason", ""),
            "area_m2": p.get("area_m2"),
            "rectangularity": p.get("rectangularity"),
            "lat": p.get("lat"),
            "lon": p.get("lon"),
            "cluster_neighbor_count": p.get("cluster_neighbor_count", 0),
            "cluster_recent_water_ratio": p.get("cluster_recent_water_ratio", 0),
            "cluster_recent_conversion_ratio": p.get(
                "cluster_recent_conversion_ratio", 0
            ),
            "previous_confirmed_stage": p.get("previous_confirmed_stage"),
            "match_score": p.get("match_score"),
            "merge_from_ids": p.get("merge_from_ids", []),
            "split_from_id": p.get("split_from_id"),
        }
        polygon_features.append(
            {
                "type": "Feature",
                "geometry": geometry,
                "properties": props,
            }
        )

    with open(os.path.join(config.WEB_DATA_DIR, "polygons.geojson"), "w") as f:
        json.dump(
            {"type": "FeatureCollection", "features": polygon_features}, f, indent=2
        )
    print(f"[EXPORT] polygons.geojson saved ({len(polygon_features)} features)")

    alert_store = AlertStore()
    alerts = alert_store.get_all()
    with open(os.path.join(config.WEB_DATA_DIR, "alerts.json"), "w") as f:
        json.dump({"alerts": alerts}, f, indent=2)
    print(f"[EXPORT] alerts.json saved ({len(alerts)} alerts)")

    if len(all_results) >= 2:
        comparison = {
            "before": {
                "date": all_results[0]["date"],
                "stage": all_results[0]["stage"],
                "stage_name": all_results[0].get("stage_name")
                or stage_label(all_results[0]["stage"]),
                "validation_score": all_results[0]["validation_score"],
            },
            "after": {
                "date": all_results[-1]["date"],
                "stage": all_results[-1]["stage"],
                "stage_name": all_results[-1].get("stage_name")
                or stage_label(all_results[-1]["stage"]),
                "validation_score": all_results[-1]["validation_score"],
            },
        }
        with open(os.path.join(config.WEB_DATA_DIR, "comparison.json"), "w") as f:
            json.dump(comparison, f, indent=2)
        print("[EXPORT] comparison.json saved")

    def _rel_if_exists(path):
        if not path or not os.path.exists(path):
            return None
        return f"../{os.path.relpath(path, config.BASE_DIR).replace('\\', '/')}"

    latest_gmw = next(
        (
            r.get("gmw_validation")
            for r in reversed(all_results)
            if (r.get("gmw_validation") or {}).get("gmw_available")
        ),
        {"gmw_available": False},
    )
    latest_gt = next(
        (
            r.get("gt_validation")
            for r in reversed(all_results)
            if (r.get("gt_validation") or {}).get("available")
        ),
        {"available": False},
    )

    per_year_accuracy = []
    for result in all_results:
        year = result.get("year")
        if year is None:
            continue
        per_year_accuracy.append(
            {
                "date": result.get("date"),
                "year": year,
                "gt_validation": result.get("gt_validation", {"available": False}),
                "gmw_validation": result.get(
                    "gmw_validation", {"gmw_available": False}
                ),
                "assets": {
                    "rgb": _rel_if_exists(
                        os.path.join(config.IMAGE_DIR, f"rgb_{year}.png")
                    ),
                    "detection": _rel_if_exists(
                        os.path.join(config.IMAGE_DIR, f"detection_{year}.png")
                    ),
                    "ground_truth": _rel_if_exists(
                        os.path.join(config.IMAGE_DIR, f"ground_truth_{year}.png")
                    ),
                    "stage": _rel_if_exists(
                        os.path.join(config.IMAGE_DIR, f"stage_{year}.png")
                    ),
                },
            }
        )

    comparison_panels = []
    for panel_path in sorted(
        glob.glob(os.path.join(config.OUTPUT_DIR, "comparisons", "*.png"))
    ):
        comparison_panels.append(
            {
                "label": os.path.splitext(os.path.basename(panel_path))[0].replace(
                    "_", " "
                ),
                "path": _rel_if_exists(panel_path),
            }
        )

    feature_audit = None
    feature_audit_path = os.path.join(config.WEB_DATA_DIR, "feature_audit.json")
    if os.path.exists(feature_audit_path):
        try:
            with open(feature_audit_path, "r", encoding="utf-8") as f:
                feature_audit = json.load(f)
        except Exception:
            feature_audit = None

    accuracy_payload = {
        "gmw": latest_gmw,
        "ground_truth": latest_gt,
        "per_year": per_year_accuracy,
        "comparison_panels": comparison_panels,
        "feature_audit": feature_audit,
        "confusion_matrix": {
            "available": False,
            "note": "Manual reference points not yet supplied",
        },
    }
    with open(
        os.path.join(config.WEB_DATA_DIR, "accuracy.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(accuracy_payload, f, indent=2)
    print("[EXPORT] accuracy.json saved")


if __name__ == "__main__":
    main()
