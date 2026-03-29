"""
MODULE 16 - Feature Audit

Exports a simple pipeline-usage audit so we can see which configured
features are actually connected into the active classification workflow.
"""

from __future__ import annotations

import json
import os

import config


PIXEL_CLASSIFIER_FEATURES = {
    "ndvi", "ndwi", "mndwi", "evi", "awei", "cmri", "mmri", "gndvi", "ndbi",
    "ndmi", "savi", "rvi", "veg_water_diff", "cwi", "ndti", "ndci",
    "turbidity_proxy", "lswi", "water_fraction", "vegetation_fraction",
    "water_persistence", "seasonal_water_entropy", "water_transition_frequency",
    "vv_temporal_mean", "vv_temporal_std", "sar_water_persistence",
    "sar_seasonality", "vv_homogeneity", "slope", "twi_proxy",
    "hydro_connectivity", "tidal_exposure_proxy", "edge_density",
    "jrc_occurrence", "jrc_seasonality", "jrc_change", "low_elevation_mask",
    "ndvi_trend_slope", "ndvi_cv", "ccdc_break", "ccdc_change_prob",
    "ccdc_ndvi_break_magnitude", "ccdc_mndwi_break_magnitude",
    "ccdc_pre_ndvi_slope", "ccdc_post_ndvi_slope", "ccdc_pre_mndwi_slope",
    "ccdc_post_mndwi_slope", "ccdc_recovery_direction", "ccdc_recent_break",
    "ccdc_break_recency_years",
    # v12.0: Previously disconnected features now integrated
    "ndvi_amplitude", "ndvi_iqr", "mndwi_amplitude", "mndwi_iqr",
    "smri", "mavi",
    "nir_texture_contrast", "nir_texture_correlation", "nir_texture_entropy",
    "nir_texture_variance", "nir_texture_homogeneity",
}

OBJECT_REDUCTION_FEATURES = {
    "ndvi", "ndwi", "mndwi", "evi", "savi", "ndbi", "awei", "lswi",
    "cmri", "mmri", "gndvi", "veg_water_diff", "cwi", "ndti", "ndci",
    "turbidity_proxy", "water_fraction", "vegetation_fraction",
    "slope", "low_elevation_mask", "ndvi_cv", "ndvi_trend_slope",
    "ndvi_amplitude", "ndvi_iqr", "mndwi_amplitude", "mndwi_iqr",
    "water_persistence", "seasonal_water_entropy", "water_transition_frequency",
    "hydro_connectivity", "tidal_exposure_proxy", "jrc_occurrence",
    "jrc_seasonality", "ccdc_break", "ccdc_change_prob",
    "ccdc_ndvi_break_magnitude", "ccdc_mndwi_break_magnitude",
    "ccdc_pre_ndvi_slope", "ccdc_post_ndvi_slope", "ccdc_pre_mndwi_slope",
    "ccdc_post_mndwi_slope", "ccdc_recovery_direction", "ccdc_recent_break",
    "ccdc_break_recency_years", "vv_mean", "vv_texture", "vv_homogeneity",
    "vv_temporal_std", "vv_temporal_iqr", "sar_water_persistence",
    "sar_water_entropy", "sar_seasonality", "rvi",
    # v12.0: Previously disconnected features now connected
    "smri", "mavi", "sdwi", "sar_water_likelihood",
    "nir_texture_contrast", "nir_texture_correlation", "nir_texture_entropy",
    "nir_texture_variance", "nir_texture_homogeneity",
}

OBJECT_CLASSIFIER_FEATURES = {
    "ndvi", "ndwi", "mndwi", "evi", "savi", "ndbi", "awei", "lswi", "cmri",
    "mmri", "gndvi", "veg_water_diff", "cwi", "ndti", "ndci",
    "turbidity_proxy", "water_fraction", "vegetation_fraction",
    "water_persistence", "seasonal_water_entropy", "water_transition_frequency",
    "ndvi_trend_slope", "ndvi_cv", "vv_mean", "vv_texture", "vv_homogeneity",
    "vv_temporal_std", "sar_water_persistence", "sar_water_entropy",
    "sar_seasonality", "rvi", "edge_density", "jrc_occurrence",
    "jrc_seasonality", "low_elevation_mask", "hydro_connectivity",
    "tidal_exposure_proxy", "ccdc_break", "ccdc_recent_break",
    "ccdc_break_recency_years", "ccdc_ndvi_break_magnitude",
    "ccdc_mndwi_break_magnitude", "ccdc_recovery_direction", "slope",
    # v12.0: Previously disconnected features now connected
    "ndvi_amplitude", "ndvi_iqr", "mndwi_amplitude", "mndwi_iqr",
    "smri", "mavi", "sdwi", "sar_water_likelihood",
    "nir_texture_contrast", "nir_texture_correlation", "nir_texture_entropy",
    "nir_texture_variance", "nir_texture_homogeneity",
}


def _flatten_feature_config():
    flattened = set()
    for values in getattr(config, "FEATURES", {}).values():
        flattened.update(values)
    return sorted(flattened)


def export_feature_audit():
    all_features = set(_flatten_feature_config())
    active_features = PIXEL_CLASSIFIER_FEATURES | OBJECT_REDUCTION_FEATURES | OBJECT_CLASSIFIER_FEATURES
    payload = {
        "total_features": len(all_features),
        "pixel_classifier_features": sorted(all_features & PIXEL_CLASSIFIER_FEATURES),
        "object_reduction_features": sorted(all_features & OBJECT_REDUCTION_FEATURES),
        "object_classifier_features": sorted(all_features & OBJECT_CLASSIFIER_FEATURES),
        "active_features": sorted(all_features & active_features),
        "unused_features": sorted(all_features - active_features),
    }

    os.makedirs(config.STATS_DIR, exist_ok=True)
    os.makedirs(config.WEB_DATA_DIR, exist_ok=True)
    with open(os.path.join(config.STATS_DIR, "feature_audit.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    with open(os.path.join(config.WEB_DATA_DIR, "feature_audit.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return payload
