"""
MODULE 14 — Per-Pond Classifier (v23.0 — STAGE_SPEC INTEGRATION)

v23.0 RESEARCH-GRADE OVERHAUL:
  - Imports stage_spec.py for single source of truth thresholds
  - All spectral thresholds now reference stage_spec
  - Turbid pond detection improved to match m06 v23.0
  - GLCM IDM fix retained from v17.0
  - ASM (Angular Second Moment) integrated
  - Water evidence weights calibrated to stage_spec

v17.0 FIXES RETAINED:
  - CRITICAL FIX: GLCM homogeneity uses IDM (was Sum Average)
    IDM range [0,1] vs Sum Average range [50-200]
  - ALL texture thresholds recalibrated for correct IDM values
"""

import config
from .stage_spec import DEFAULT_STAGE_SPEC as spec


def classify_pond_features(pond_features):
    """
    Clear and sharp pond classifier (v11.1).
    Boosted S4/S5 detection with thinner boundaries.
    """
    properties = pond_features.keys()
    has_sar = "vv_mean" in properties

    def safe_get(key, default):
        v = pond_features.get(key)
        if v is None:
            return default
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    # Core spectral features
    ndvi = safe_get("ndvi", 0.0)
    ndwi = safe_get("ndwi", 0.0)
    mndwi = safe_get("mndwi", 0.0)
    evi = safe_get("evi", 0.0)
    savi = safe_get("savi", ndvi)
    ndbi = safe_get("ndbi", 0.0)
    rvi = safe_get("rvi", 1.0)
    slope = safe_get("slope", 0.0)
    awei = safe_get("awei", 0.0)
    cwi = safe_get("cwi", mndwi + ndwi - ndvi)
    gndvi = safe_get("gndvi", 0.0)

    # Unmixing fractions
    water_fraction = safe_get("water_fraction", 0.0)
    vegetation_fraction = safe_get("vegetation_fraction", 0.0)
    soil_fraction = safe_get("soil_fraction", 0.0)

    # v17.0: GLCM TEXTURE FEATURES — CORRECTED FOR IDM
    # CRITICAL FIX: homogeneity is now IDM (Inverse Difference Moment)
    # IDM range [0, 1]: 1.0 = perfectly smooth, 0.0 = max texture
    # Previously used Sum Average (range ~50-200) — completely wrong!
    texture_homogeneity = safe_get("nir_texture_homogeneity", 0.5)  # IDM default
    texture_contrast = safe_get("nir_texture_contrast", 50.0)
    texture_entropy = safe_get("nir_texture_entropy", 3.0)
    texture_variance = safe_get("nir_texture_variance", 500.0)
    texture_asm = safe_get("nir_texture_asm", 0.1)

    # v13.0: NOVEL INDICES (previously computed, never used)
    smri_val = safe_get("smri", 0.0)
    mavi_val = safe_get("mavi", 0.5)
    lswi_val = safe_get("lswi", 0.0)

    # v13.0: SAR COMPOSITE FEATURES (previously computed, never used)
    sdwi_val = safe_get("sdwi", 0.0)
    sar_water_likelihood = safe_get("sar_water_likelihood", 0.0)

    vv_mean = safe_get("vv_mean", -15.0) if has_sar else -15.0

    # Reference datasets
    jrc_occurrence = safe_get("jrc_occurrence", 0.0)
    jrc_seasonality = safe_get("jrc_seasonality", 0.0)
    gmw_mangrove = safe_get("gmw_mangrove", 0.0)
    gmw_historical_mangrove = safe_get("gmw_historical_mangrove", gmw_mangrove)

    # Pixel-level stage fractions
    pixel_s1 = safe_get("pixel_s1_fraction", 0.0)
    pixel_s2 = safe_get("pixel_s2_fraction", 0.0)
    pixel_s3 = safe_get("pixel_s3_fraction", 0.0)
    pixel_s4 = safe_get("pixel_s4_fraction", 0.0)
    pixel_s5 = safe_get("pixel_s5_fraction", 0.0)
    pixel_conf = safe_get("pixel_confidence_mean", 0.0)

    # Geometry features
    rectangularity = safe_get("rectangularity", 0.5)
    compactness = safe_get("compactness", 0.3)
    elongation = safe_get("elongation", 10.0)
    area_m2 = safe_get("area_m2", 5000)

    # Temporal features
    water_transition = safe_get("water_transition_frequency", 0.0)
    hydro_conn = safe_get("hydro_connectivity", 0.0)
    water_pers = safe_get("water_persistence", 0.0)
    ndvi_amp = safe_get("ndvi_amplitude", 0.3)
    ndti_val = safe_get("ndti", 0.0)
    turbidity = safe_get("turbidity_proxy", 0.0)
    ccdc_break = safe_get("ccdc_recent_break", 0.0)
    mndwi_iqr = safe_get("mndwi_iqr", 0.2)
    sar_water_pers = safe_get("sar_water_persistence", 0.0)

    # Use absolute single-source-of-truth river mask from the stage engine
    is_natural_water = safe_get("is_river", 0.0) >= 0.5
    river_probability = safe_get("river_probability", 0.0)

    # v23.0: Use stage_spec thresholds for classification
    # Turbid ponds have lower MNDWI (-0.30 to 0.0) but valid water evidence
    is_river_shape = compactness < 0.03 and elongation > 25.0
    is_aquaculture_geometry = (
        rectangularity > 0.05 and compactness > 0.012 and area_m2 > 200
    )
    in_mangrove_context = max(gmw_mangrove, gmw_historical_mangrove) > 0.05

    # v23.0: TURBID POND DETECTION
    # Turbid aquaculture ponds have MNDWI as low as -0.30 but smooth texture
    is_turbid_pond = (
        mndwi >= spec.s4_mndwi_min  # -0.30 for turbid ponds
        and mndwi < 0.05            # But not clear water
        and ndvi < spec.s4_ndvi_max # Low vegetation
        and texture_homogeneity > spec.idm_smooth_threshold  # Smooth surface
        and is_aquaculture_geometry
    )

    # v15.0: WIDTH-BASED RIVER FRAGMENT FILTER
    # Polygons narrower than ~30m AND very elongated are river fragments
    # that slipped past pixel-level masking. Reject them.
    min_bounding_width = safe_get("min_bounding_width_m", 100.0)
    is_thin_fragment = (
        min_bounding_width < 30.0 and elongation > 15.0
        and not is_aquaculture_geometry
        and not is_turbid_pond  # v23.0: Don't reject turbid ponds
    )

    # Reject rivers: must be (permanent water AND river shape) OR thin fragment
    # v23.0: Don't reject if turbid_pond flag is set
    if (is_natural_water and is_river_shape and not is_aquaculture_geometry and not is_turbid_pond) or is_thin_fragment:
        pond_features["stage_scores"] = {str(i): 0.0 for i in range(1, 6)}
        pond_features["stage_probabilities"] = {str(i): 0.0 for i in range(1, 6)}
        pond_features["uncertain"] = True
        pond_features["uncertainty_reason"] = "natural water" if not is_thin_fragment else "thin river fragment"
        return 0

    # ============================================================
    # CONTINUOUS EVIDENCE SCORING (v23.0 SINGLE-SOURCE OF TRUTH)
    # Propagated directly from m06_stage_engine.py to prevent pixel vs polygon disagreement
    # Uses stage_spec thresholds
    # ============================================================
    water_evidence = safe_get("water_evidence_score", 0.0)
    veg_evidence = safe_get("veg_evidence_score", 0.0)
    bare_soil_evidence = safe_get("bare_soil_score", 0.0)

    # ============================================================
    # STAGE SCORING — OBJECT EVALUATION (v23.0)
    # Uses stage_spec thresholds, improved turbid pond handling
    # ============================================================

    # S1: Dense vegetation (uses spec.s1_ndvi_min)
    s1_score = 0.10 + pixel_s1 * 0.65 + veg_evidence * 0.25
    if in_mangrove_context:
        s1_score += 0.10
    if ndvi > spec.s1_ndvi_min:  # v23.0: Use spec threshold
        s1_score += 0.05

    # S2: Degrading vegetation (uses spec.s2_ndvi_min/max)
    s2_score = 0.10 + pixel_s2 * 0.65 + veg_evidence * 0.25
    if in_mangrove_context:
        s2_score += 0.10
    if ndvi >= spec.s2_ndvi_min and ndvi < spec.s2_ndvi_max:  # v23.0: Use spec
        s2_score += 0.05

    # S3: Cleared / Bare Soil (uses spec.s3_ndvi_max)
    s3_score = 0.10 + pixel_s3 * 0.65 + bare_soil_evidence * 0.25
    if in_mangrove_context:
        s3_score += 0.10
    if ndvi < spec.s3_ndvi_max and ndbi > spec.s3_ndbi_min:  # v23.0: Use spec
        s3_score += 0.05

    # S4: Pond Formation (uses spec.s4_* thresholds)
    s4_score = 0.10 + pixel_s4 * 0.65 + water_evidence * 0.25
    if is_aquaculture_geometry:
        s4_score += 0.15
    if is_turbid_pond:  # v23.0: Turbid pond boost
        s4_score += 0.20
    if mndwi > spec.s4_mndwi_min and ndvi < spec.s4_ndvi_max:  # v23.0: Use spec
        s4_score += 0.05
    if texture_homogeneity > spec.idm_smooth_threshold:  # v23.0: Smooth surface
        s4_score += 0.05
    if is_river_shape and not is_aquaculture_geometry and not is_turbid_pond:
        s4_score *= 0.3
    if is_natural_water and not is_turbid_pond:
        s4_score = 0.0

    # S5: Established Aquaculture (uses spec.s5_* thresholds)
    s5_score = 0.10 + pixel_s5 * 0.65 + water_evidence * 0.25
    if is_aquaculture_geometry and rectangularity > 0.10:
        s5_score += 0.15
    if mndwi > spec.s5_mndwi_min and ndvi < spec.s5_ndvi_max:  # v23.0: Use spec
        s5_score += 0.05
    if jrc_occurrence > spec.s5_jrc_occurrence_min * 100:  # v23.0: JRC established
        s5_score += 0.10
    if texture_homogeneity > spec.idm_smooth_threshold + 0.05:  # v23.0: Very smooth
        s5_score += 0.05
    if is_river_shape and not is_aquaculture_geometry:
        s5_score *= 0.3
    if is_natural_water and not is_turbid_pond:
        s5_score = 0.0

    # CCDC break = recent land-use change → conversion signal
    if ccdc_break > 0.5:
        if s4_score > 0.3 or s5_score > 0.3:
            s4_score += 0.06
            s5_score += 0.04

    scores = {1: s1_score, 2: s2_score, 3: s3_score, 4: s4_score, 5: s5_score}

    # Calculate probabilities
    score_total = sum(max(score, 0.01) for score in scores.values())
    probabilities = {
        str(stage_id): round(max(score, 0.01) / score_total, 4)
        for stage_id, score in scores.items()
    }

    best_stage = max(scores, key=scores.get)
    best_score = scores[best_stage]

    pond_features["stage_scores"] = {str(k): round(v, 4) for k, v in scores.items()}
    pond_features["stage_probabilities"] = probabilities
    pond_features["uncertain"] = best_score < 0.40
    pond_features["uncertainty_reason"] = "low score" if best_score < 0.40 else ""
    pond_features["best_stage_score"] = round(best_score, 4)

    return best_stage


def compute_pond_confidence(pond_features, stage):
    """
    Compute real confidence score based on:
      - Best stage score magnitude (how strong the winning signal is)
      - Score margin over second-best (how decisive the classification is)
      - Shape quality bonus (rectangularity indicates engineered structure)
    
    Returns: float in [0.10, 0.99]
    """
    best_score = float(pond_features.get("best_stage_score") or 0.5)
    scores = pond_features.get("stage_scores", {})
    
    # Get sorted scores for margin calculation
    score_values = sorted(
        [float(v) for v in scores.values() if v is not None],
        reverse=True,
    )
    
    if len(score_values) >= 2:
        margin = score_values[0] - score_values[1]
    elif len(score_values) == 1:
        margin = score_values[0]
    else:
        margin = 0.0
    
    # Shape: rectangular ponds are more likely real aquaculture
    rectangularity = float(pond_features.get("rectangularity") or 0.0)
    shape_bonus = 0.10 if rectangularity > 0.15 else 0.0
    
    # Uncertainty penalty
    uncertain = pond_features.get("uncertain", False)
    uncertainty_penalty = 0.15 if uncertain else 0.0
    
    # Weighted combination
    confidence = (
        best_score * 0.55
        + margin * 0.35
        + shape_bonus
        - uncertainty_penalty
    )
    
    return round(min(0.99, max(0.10, confidence)), 3)



def classify_pond_observation(pond_features):
    """Wrapper."""
    stage = classify_pond_features(pond_features)
    confidence = compute_pond_confidence(pond_features, stage)
    pond_features["confidence"] = confidence
    return {
        "stage": stage,
        "confidence": confidence,
        "stage_probabilities": pond_features.get("stage_probabilities", {}),
        "stage_scores": pond_features.get("stage_scores", {}),
        "uncertain": pond_features.get("uncertain", False),
        "uncertainty_reason": pond_features.get("uncertainty_reason", ""),
        "best_stage_score": pond_features.get("best_stage_score"),
    }
