"""
Single source of truth for stage thresholds and guards.

CRITICAL FIXES v11.0:
  - S4/S5 water thresholds RELAXED for turbid aquaculture water detection
  - MNDWI > -0.15 for S4 (turbid water), MNDWI > 0.0 for S5 (clearer water)
  - Multi-index water detection: MNDWI + CWI + AWEI + water_fraction
  - Aquaculture-optimized thresholds for Godavari delta conditions

This module exists to prevent silent divergence between:
  - pixel-stage classification (modules/m06_stage_engine.py)
  - object/per-pond classification (modules/m14_per_pond_classifier.py)

All numeric thresholds that define S1–S5 should be referenced from here.

Research references:
  - NDVI > 0.33 for healthy mangrove (Li et al., ArcGIS mangrove studies)
  - MNDWI > 0 for CLEAR water (Xu 2006)
  - MNDWI > -0.2 for TURBID water (aquaculture ponds - practical adjustment)
  - NDBI > 0 for built-up/bare soil (Zha et al. 2003)
  - CWI = MNDWI + NDWI - NDVI (2024 research, 94% OA)
  - AWEI: Feyisa et al. 2014 (shadow-resistant water detection)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StageSpec:
    # --- S1 (Intact mangrove) ---
    s1_ndvi_min: float = 0.30
    s1_evi_min: float = 0.16
    s1_water_index_max: float = 0.0  # MNDWI must be negative for vegetation
    s1_cmri_min: float = 0.08
    s1_savi_min: float = 0.22

    # --- S2 (Degrading mangrove) ---
    s2_ndvi_min: float = 0.08
    s2_ndvi_max: float = 0.30
    s2_mndwi_max: float = -0.10  # Relaxed water exclusion
    s2_savi_max: float = 0.35
    s2_gndvi_stress_max: float = 0.40
    s2_ndmi_stress_max: float = 0.38
    s2_ndvi_trend_threshold: float = -0.002
    s2_ndvi_cv_min: float = 0.10

    # --- S3 (Cleared/bare soil/construction) ---
    s3_ndvi_max: float = 0.18
    s3_ndwi_max: float = 0.0
    s3_mndwi_max: float = -0.05  # S3 should not be water
    s3_ndbi_min: float = -0.02  # v13.0: tightened from -0.08 to exclude turbid water

    # =============================================
    # S4 (Pond Formation) - RELAXED for turbid water
    # =============================================
    # CRITICAL: Aquaculture water is often turbid with MNDWI -0.15 to 0.1
    s4_ndwi_min: float = -0.25  # v12.1: Relaxed for turbid water
    s4_mndwi_min: float = -0.30  # v12.1: CRITICAL: Relaxed from -0.15 for very turbid ponds
    s4_ndvi_max: float = 0.30  # v12.1: Allow more mixed pixels
    s4_awei_min: float = -0.45  # v12.1: AWEI for shadow-resistant detection
    s4_mmri_min: float = 0.30  # v12.1: Relaxed MMRI threshold
    s4_veg_water_diff_max: float = 0.35
    s4_cwi_min: float = -0.20  # v12.1: Relaxed CWI
    s4_water_fraction_min: float = 0.20  # v12.1: Lowered for turbid/mixed pixels

    # S4 JRC: New ponds (lower occurrence is OK)
    s4_jrc_occurrence_max: float = 70.0  # Can have some JRC history
    s4_jrc_seasonality_max: float = 10.0  # Can be seasonal

    # =============================================
    # S5 (Established Aquaculture) - Clearer water
    # =============================================
    s5_ndwi_min: float = -0.15  # v12.1: Relaxed for turbid established ponds
    s5_mndwi_min: float = -0.10  # v12.1: Relaxed from 0.0
    s5_ndvi_max: float = 0.25  # v12.1: Relaxed vegetation limit
    s5_awei_min: float = -0.20  # Higher AWEI for clear water
    s5_cwi_min: float = -0.05  # v12.1: Relaxed CWI
    s5_water_fraction_min: float = 0.30  # v12.1: Lowered water fraction

    # S5 JRC: Established water
    s5_jrc_occurrence_min: float = 45.0  # Moderate JRC occurrence
    s5_jrc_seasonality_min: float = 5.0  # Some seasonality OK
    s5_vv_temporal_std_max: float = 2.5  # SAR stability

    # --- Cross-stage guards ---
    ndbi_blocks_water_min: float = 0.12  # Block water if high NDBI
    slope_pond_max_deg: float = 5.0

    # --- Natural Water Body Rejection (v13.0 tightened) ---
    # Tier 1: Definite river — permanent water + high seasonality
    river_tier1_jrc_min: float = 85.0  # target high-confidence rivers to save ponds
    river_tier1_seasonality_min: float = 10.0
    # Tier 2: Probable river — moderate JRC + temporal instability
    river_tier2_jrc_min: float = 70.0
    river_tier2_seasonality_min: float = 8.0
    river_tier2_transition_freq_min: float = 0.25
    # Tier 3: Connectivity-based
    river_tier3_hydro_connectivity_min: float = 0.4
    river_tier3_jrc_min: float = 70.0
    # Aquaculture rescue thresholds
    pond_rescue_ndvi_amplitude_max: float = 0.15
    pond_rescue_water_persistence_min: float = 0.4  # v13.0: 0.5 → 0.4 (rescue more ponds)
    pond_rescue_transition_freq_max: float = 0.20

    # Shape-based river filtering
    natural_water_compactness_max: float = 0.06
    natural_water_elongation_min: float = 12.0
    river_compactness_strict: float = 0.03
    river_elongation_strict: float = 25.0

    # River size filters
    river_area_min_m2: float = 30000.0
    river_perimeter_area_ratio_min: float = 0.15

    # --- S3: Bare soil must be actual bare soil (v13.0) ---
    # s3_ndbi_min raised from -0.08 to -0.02 to exclude turbid water
    # (turbid water has negative NDBI, bare soil has near-zero or positive)

    # --- Temporal feature thresholds (v13.0) ---
    s4_water_persistence_min: float = 0.4  # v13.0: 0.5 → 0.4 (detect more ponds)
    s4_ndvi_amplitude_max: float = 0.15
    s4_ndti_min: float = 0.05
    s5_water_persistence_min: float = 0.5  # v13.0: 0.6 → 0.5
    s5_mndwi_iqr_max: float = 0.15

    # --- SAR heuristics ---
    sar_vv_water_db_max: float = -14.0
    sar_vv_veg_db_min: float = -10.0

    # --- Water confidence ---
    water_evidence_strong_threshold: float = 0.50
    water_evidence_moderate_threshold: float = 0.35

    # =============================================
    # ADSRM parameters (v17.0 — Adaptive Dual-Scale River Masking)
    # =============================================
    river_coarse_morph_radius: int = 4    # Coarse pass: catches wide rivers
    river_fine_morph_radius: int = 2      # Fine pass: catches thin tributaries
    river_soft_threshold_high: float = 0.7  # Above = definite river
    river_soft_threshold_low: float = 0.3   # Below = definite pond
    pond_neighbor_immunity_count: int = 4   # Min water-like neighbors for PNPS

    # =============================================
    # SGTM parameters (v17.0 — Spatial Gradient Transition Model)
    # =============================================
    stage_gradient_max: int = 2        # Max stage jump in 3×3 before denoising
    snic_vote_seed_spacing: int = 16   # Coarse for memory safety

    # =============================================
    # IDM-corrected GLCM thresholds (v17.0)
    # IDM ∈ [0, 1]: 1.0 = perfectly smooth, 0.0 = max texture
    # =============================================
    idm_smooth_threshold: float = 0.25   # IDM above = smooth (pond-like)
    idm_very_smooth_threshold: float = 0.35  # IDM above = very smooth (S5)
    idm_rough_threshold: float = 0.15    # IDM below = rough (river/vegetation)
    texture_variance_smooth_max: float = 400.0  # Variance below = smooth
    texture_variance_rough_min: float = 600.0   # Variance above = rough


# Default instance - import this in all modules
DEFAULT_STAGE_SPEC = StageSpec()

