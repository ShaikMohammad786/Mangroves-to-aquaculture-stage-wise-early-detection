"""
Configuration for Mangrove-Aquaculture Transition Detection System.
v17.0 RESEARCH-GRADE — ADSRM + ELD + IDM-corrected GLCM

Key changes v17.0:
  - ADSRM dual-scale river masking configuration
  - ELD instance segmentation parameters
  - GLCM IDM recalibration (all texture thresholds updated)
  - GEE memory optimization: disabled SNIC, CCDC for pre-SAR
  - Computation footprint < 80MB per tile
"""

import os

# ─── Target Scale ────────────────────────────────────────────────────
TARGET_SCALE = 10

SENSOR_SCALES = {
    "sentinel2_sr": 10,
    "hls_s30": 30,
    "hls_l30": 30,
    "landsat5": 30,
    "landsat7": 30,
    "landsat8": 30,
}


def get_scale_for_sensor(sensor_key: str) -> int:
    return SENSOR_SCALES.get(sensor_key, TARGET_SCALE)


# ─── GEE Configuration ─────────────────────────────────
GEE_SAFE = {
    "tileScale": 16,
    "maxPixels": 1e9,
    "bestEffort": True,
}

# ─── Paths ───────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
IMAGE_DIR = os.path.join(OUTPUT_DIR, "images")
FEATURE_DIR = os.path.join(OUTPUT_DIR, "features")
POLYGON_DIR = os.path.join(OUTPUT_DIR, "polygons")
ALERT_DIR = os.path.join(OUTPUT_DIR, "alerts")
STATS_DIR = os.path.join(OUTPUT_DIR, "stats")
WEB_DATA_DIR = os.path.join(BASE_DIR, "web", "data")

for d in [
    OUTPUT_DIR,
    IMAGE_DIR,
    FEATURE_DIR,
    POLYGON_DIR,
    ALERT_DIR,
    STATS_DIR,
    WEB_DATA_DIR,
]:
    os.makedirs(d, exist_ok=True)

# ─── AOI ──────────────────────────────────────────────────
AOI = {
    "name": "Coringa Conversion Hotspot",
    "lat_min": 16.60,
    "lat_max": 16.70,
    "lon_min": 82.22,  # Reduced left side
    "lon_max": 82.30,  # Reduced right side
    "buffer_km": 0.5,
    "max_elevation_m": 10,
    "crs": "EPSG:32644",
    "resolution_m": 10,
}

# ─── Date Range ───────────────────────────────────────────────────────
DATE_RANGE = {
    "start": "1980-01-01",
    "end": "2025-12-31",
}

# ─── Dry-Season Filter  ──────────────────────────────────
DRY_SEASON_MONTHS = [1, 2, 3, 11, 12]

# ─── Cloud Thresholds ────────────────────────────────────────────────
CLOUD = {
    "landsat_max_cloud_pct": 40,
    "sentinel2_max_cloud_pct": 35,
    "s2_cloud_prob_max": 40,
}

# ─── Adaptive Thresholds ───────────────────────────────────────────
ADAPTIVE_THRESHOLDS = {
    "use_adaptive": True,
    "ndvi_percentiles": [30, 70],
    "mndwi_percentile": 80,
    "ndvi_p70_clamp": (0.30, 0.55),
    "ndvi_p30_clamp": (0.10, 0.30),
    "mndwi_p80_clamp": (0.00, 0.50),  # Adjusted for turbid water
    "ndvi_p70_fallback": 0.40,
    "ndvi_p30_fallback": 0.20,
    "mndwi_p80_fallback": 0.20,
}

# ─── Water Cluster ─────────────────────────────────────
WATER_CLUSTER = {
    "min_pond_pixels": 8,  # Up from 6
    "max_cluster_pixels": 400,
    "min_pond_area_m2": 300,  # Minimum for shape analysis (3px at 10m)
    "max_component_area_m2": 250000,
    "split_large_component_area_m2": 45000,
}

# ─── Natural Water Rejection (STRENGTHENED v2.0) ───────────────────
NATURAL_WATER_REJECTION = {
    "jrc_occurrence_min": 85,  # Raised back to 85 to prevent masking aquaponds
    "jrc_seasonality_min": 10,  # Raised back to 10
    "compactness_max": 0.025,  # Tightened: less compact = more likely river
    "elongation_min": 15.0,   # Tightened: catch thinner rivers
}

# ─── Stage Transitions ───────────────────────────────────────
ALLOWED_TRANSITIONS = {
    "S1": ["S2"],
    "S2": ["S3"],
    "S3": ["S4"],
    "S4": ["S5"],
    "S5": [],
}

HISTORICAL_ALLOWED_TRANSITIONS = {
    "S1": ["S2", "S3", "S4", "S5"],
    "S2": ["S3", "S4", "S5"],
    "S3": ["S4", "S5"],
    "S4": ["S5"],
    "S5": [],
}

# ─── Persistence ─────────────────────────────────────────
PERSISTENCE_REQUIRED_COUNT = {
    "historical": 1,
    "operational": 3,
}

# ─── Validation ───────────────────────────────────────
VALIDATION = {
    "weights": {
        "mangrove": 0.25,
        "water": 0.25,
        "elevation": 0.15,
        "sar": 0.15,
        "multi_evidence": 0.20,
    },
    "confidence_threshold": 0.25,
    "min_polygon_area_ha": 0.04,
}

# ─── False Positive Rules ────────────────────────────────────────────
FALSE_POSITIVE = {
    "jrc_seasonal_reject_pct": 20,
    "ndvi_recovery_reject_years": 1,
    "max_elevation_m": 10,
    "min_area_ha": 0.04,
    "jrc_persistence": {"min_water_months_ratio": 0.5, "max_tidal_months_ratio": 0.35},
}

# ─── GEE Datasets ─────────────────────────────────────────
GEE_DATASETS = {
    "landsat5_sr": "LANDSAT/LT05/C02/T1_L2",
    "landsat7_sr": "LANDSAT/LE07/C02/T1_L2",
    "hls_l30": "NASA/HLS/HLSL30/v002",
    "hls_s30": "NASA/HLS/HLSS30/v002",
    "sentinel2_sr": "COPERNICUS/S2_SR_HARMONIZED",
    "sentinel1_grd": "COPERNICUS/S1_GRD",
    "jrc_water": "JRC/GSW1_4/GlobalSurfaceWater",
    "jrc_monthly": "JRC/GSW1_4/MonthlyHistory",
    "glo30": "COPERNICUS/DEM/GLO30",
    "mangrove_watch": "projects/earthengine-legacy/assets/projects/sat-io/open-datasets/GMW/extent/GMW_V3",
}

# ─── Band Mappings ───────────────────────────────────────────────────
BAND_MAP = {
    "landsat5": {
        "blue": "SR_B1",
        "green": "SR_B2",
        "red": "SR_B3",
        "nir": "SR_B4",
        "swir1": "SR_B5",
        "swir2": "SR_B7",
        "qa": "QA_PIXEL",
    },
    "landsat7": {
        "blue": "SR_B1",
        "green": "SR_B2",
        "red": "SR_B3",
        "nir": "SR_B4",
        "swir1": "SR_B5",
        "swir2": "SR_B7",
        "qa": "QA_PIXEL",
        "pan": "SR_B8",
    },
    "hls_l30": {
        "blue": "B2",
        "green": "B3",
        "red": "B4",
        "nir": "B5",
        "swir1": "B6",
        "swir2": "B7",
        "qa": "Fmask",
    },
    "hls_s30": {
        "blue": "B2",
        "green": "B3",
        "red": "B4",
        "nir": "B8A",
        "swir1": "B11",
        "swir2": "B12",
        "qa": "Fmask",
    },
    "sentinel2_sr": {
        "blue": "B2",
        "green": "B3",
        "red": "B4",
        "nir": "B8",
        "swir1": "B11",
        "swir2": "B12",
        "qa": "QA60",
        "scl": "SCL",
    },
}

# ─── SAR Config ───────────────────────────────────────────────────────
SAR = {
    "instrument_mode": "IW",
    "polarization": ["VV", "VH"],
    "speckle_filter": "refined_lee",
    "speckle_window": 3,
    "water_threshold_db": -14.0,  # Relaxed for turbid water
}

# ─── Features ────────────────────────────────────────────
FEATURES = {
    "vegetation": ["ndvi", "evi", "ndmi", "gndvi", "savi", "cmri", "mmri", "ndbi"],
    "water": ["ndwi", "mndwi", "awei", "cwi", "ndti", "ndci", "turbidity_proxy"],
    "sar": ["vv_mean", "vh_mean", "vv_vh_ratio", "vv_texture", "rvi", "vv_homogeneity"],
    "shape": ["edge_density"],
    "temporal": [
        "ndvi_trend_slope",
        "ndvi_cv",
        "ndvi_amplitude",
        "ndvi_iqr",
        "mndwi_amplitude",
        "mndwi_iqr",
        "water_persistence",
        "seasonal_water_entropy",
        "water_transition_frequency",
        "vv_temporal_mean",
        "vv_temporal_std",
        "vv_temporal_iqr",
        "vh_temporal_mean",
        "vh_vv_ratio_mean",
        "sar_water_persistence",
        "sar_water_entropy",
        "sar_seasonality",
    ],
    "history": ["jrc_occurrence", "jrc_seasonality", "jrc_change"],
    "terrain": [
        "low_elevation_mask",
        "slope",
        "twi_proxy",
        "hydro_connectivity",
        "tidal_exposure_proxy",
    ],
    "unmixing": [
        "water_fraction",
        "vegetation_fraction",
        "soil_fraction",
        "unmixing_rmse",
    ],
    "ccdc": [
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
    ],
}

# ─── Spectral Unmixing (RELAXED) ─────────────────────────────
SPECTRAL_UNMIXING = {
    "water_endmember": [0.04, 0.06, 0.03, 0.02, 0.01, 0.01],
    "turbid_water_endmember": [0.12, 0.15, 0.10, 0.06, 0.04, 0.03],
    "mangrove_endmember": [0.02, 0.04, 0.03, 0.35, 0.15, 0.06],
    "degraded_mangrove_em": [0.05, 0.08, 0.07, 0.20, 0.12, 0.08],
    "bare_soil_endmember": [0.15, 0.18, 0.22, 0.25, 0.30, 0.28],
    "shade_endmember": [0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
    "water_fraction_threshold": 0.30,  # Relaxed
}

# ─── Extensions ──────────────────────────────────────────
EXTENSIONS = {
    "use_ccdc": False,               # v17.0: DISABLED — GEE memory limit
    "use_hmm": True,
    "use_gnn": False,
    "use_ml_classifier": False,
    "use_tidal_normalization": True,
    "use_modis_fusion": False,
    "use_soilgrids": True,
    "use_dip_enhancement": True,
    "use_temporal_in_pixel_stage": True,
    "use_natural_water_rejection": True,
    "use_tile_processing": False,
    "use_snic_segmentation": False,   # v17.0: DISABLED — GEE memory limit
    "use_temporal_feature_cube": True,
    "use_hydrodynamic_proxies": True,
    "ml_model_path": None,
}

TEMPORAL_FEATURES = {
    "use_full_year_temporal_context": True,
    "quarterly_summary": True,
    "water_threshold_mndwi": -0.15,  # RELAXED for turbid water
    "water_threshold_ndwi": -0.10,  # RELAXED
    "entropy_epsilon": 1e-4,
    "ccdc_recent_break_years": 2.5,
    "operational_ccdc_days_back": 365,
}

MANGROVE_CONTEXT = {
    "historical_anchor_year": 1996,
    "context_buffer_px": 4,
}

HYDROLOGY = {
    "persistent_water_occurrence_min": 80,
    "persistent_water_seasonality_min": 8,
    "tidal_seasonality_min": 5,
    "connectivity_radius_m": 150,
    "tidal_exposure_penalty_threshold": 0.60,  # Relaxed
    "hydro_connectivity_penalty_threshold": 0.50,  # Relaxed
}

TILE_PROCESSING = {
    "enabled": False,
    "max_tile_width_deg": 0.15,
    "max_tile_height_deg": 0.12,
    "coarse_max_tile_width_deg": 0.15,
    "coarse_max_tile_height_deg": 0.12,
    "tile_overlap_deg": 0.002,
    "coarse_tile_overlap_deg": 0.002,
    "max_tiles": 4,
    "coarse_max_tiles": 2,
}

OBJECT_SEGMENTATION = {
    "use_snic": False,              # v17.0: DISABLED — GEE memory limit
    "snic_max_scale_m": 30,
    "snic_seed_spacing_px": 5,      # Coarser (was 3) to reduce computation
    "snic_compactness": 0.10,
    "snic_connectivity": 4,          # 4-connected (was 8) to reduce memory
    "snic_neighborhood_factor": 2,
    "segment_water_threshold": 0.40,
    "segment_min_area_m2": 400,
}

# ─── River Masking v23.0 — POND-PRESERVING (ADSRM) ───────────────
RIVER_MASKING = {
    # ADSRM dual-scale radii
    "coarse_morph_radius_px": 4,       # Coarse pass: catches wide rivers (>8px)
    "fine_morph_radius_px": 2,         # Fine pass: catches thin tributaries (>4px)
    "morphological_opening_radius_px": 3,  # Legacy compat
    # Gap fill
    "gap_fill_max_px": 25,             # Fill stage-0 holes up to this size
    # Expansion control
    # v23.0: DISABLED — expansion was destroying turbid ponds adjacent to rivers
    "conditional_expansion": False,    # v23.0: CRITICAL — disabled to preserve ponds
    "edge_guide_threshold": 0.15,      # Canny threshold for edge-guided smoothing
    # Protection
    "river_adjacent_min_water_evidence": 0.35,  # v23.0: Relaxed from 0.40
}

# ─── Spatial Smoothing v23.0 — Improved for cleaner stage maps ──────────────
SPATIAL_SMOOTHING = {
    "min_cluster_px": 15,              # v23.0: Increased from 10 for less noise
    "super_majority_threshold": 6,
    "use_edge_guided": False,          # DISABLED — GEE memory limit
    "use_snic_vote": False,            # DISABLED — GEE memory limit
    "snic_vote_seed_spacing": 16,      # Coarse spacing if ever re-enabled
    "snic_vote_compactness": 0.05,
}

OBJECT_MATCHING = {
    "distance_threshold_m": 150.0,
    "spatial_bin_m": 200.0,
    "cluster_radius_m": 300.0,
    "max_area_ratio": 8.0,
    "distance_weight": 0.35,
    "area_weight": 0.25,
    "shape_weight": 0.15,
    "bbox_weight": 0.10,
    "polygon_overlap_weight": 0.15,
}

HMM = {
    "enabled": True,
    "scene_enabled": True,
    "pond_enabled": True,
    "duration_min_run": {"1": 1, "2": 1, "3": 1, "4": 1, "5": 1},
    "transition_probs_historical": {
        "1": {"1": 0.62, "2": 0.24, "3": 0.08, "4": 0.04, "5": 0.02},
        "2": {"1": 0.04, "2": 0.48, "3": 0.28, "4": 0.14, "5": 0.06},
        "3": {"1": 0.01, "2": 0.08, "3": 0.42, "4": 0.33, "5": 0.16},
        "4": {"1": 0.00, "2": 0.03, "3": 0.10, "4": 0.49, "5": 0.38},
        "5": {"1": 0.00, "2": 0.00, "3": 0.03, "4": 0.10, "5": 0.87},
    },
    "transition_probs_operational": {
        "1": {"1": 0.82, "2": 0.18},
        "2": {"2": 0.55, "3": 0.35, "1": 0.10},
        "3": {"3": 0.45, "4": 0.45, "2": 0.10},
        "4": {"4": 0.45, "5": 0.45, "3": 0.10},
        "5": {"5": 0.92, "4": 0.08},
    },
}

UNCERTAINTY = {
    "pond_min_probability": 0.50,  # Relaxed
    "scene_min_probability": 0.45,  # Relaxed
    "min_margin": 0.10,
    "low_confidence_threshold": 0.40,
    "scene_low_confidence_threshold": 0.35,
}

# ─── Web Export ───────────────────────────────────────────
WEB = {
    "thumbnail_width": 1536,
    "thumbnail_height": 1536,
    "map_center_lat": 16.65,
    "map_center_lon": 82.26,
    "map_zoom": 13,
}

PERFORMANCE = {
    "threshold_scale": 30,
    "min_object_scale": 20,
    "reduce_tile_scale": 8,
    "max_reduce_candidates": 5000,
    "max_reduce_candidates_per_tile": 2500,
    "max_total_reduce_candidates": 5000,
    "max_reduce_candidates_per_tile_coarse": 80,
    "max_total_reduce_candidates_coarse": 300,
    "max_reduce_candidates_coarse": 300,
    "object_extraction_start_year": 2002,
    "use_pixel_stage_summary": True,
    "compute_scene_feature_summary": False,
    "export_core_features_only": False,
    "skip_expensive_validation": False,
    "skip_ground_truth_eval": False,
    "skip_gmw_comparison": False,
    # v23.0: Reduced thumbnail sizes to stay under GEE 80MB limit
    "historical_stage_thumb_size": 768,  # Was 1280
    "historical_rgb_thumb_size": 768,    # Was 1024
    "historical_feature_thumb_size": 768,  # Was 1024
    "historical_stage_export_scale": 60,  # v23.0: Force coarse scale for large AOIs
    "modern_stage_export_scale": 15,
    "modern_feature_export_scale": 15,
}

EXPORT_RENDER = {
    "apply_dip_to_rgb": False,
    "apply_dip_to_stage": True,
    "apply_dip_to_features": False,
    "max_overlay_labels": 18,
    "label_collision_padding_px": 6,
    "label_min_confidence": 0.50,
    "detection_fill_alpha": 92,
    "ground_truth_fill_alpha": 104,
}
