"""
MODULE 19 — River Mask v23.0 (RESEARCH-GRADE AQUACULTURE FIX)

CRITICAL FIX v23.0 — POND PRESERVATION PRIORITY:

  v22.0 was OVER-MASKING aquaculture ponds because:
    1. River expansion was too aggressive (2px into ANY water-like pixels)
    2. Turbid ponds have MNDWI -0.30 to 0.0 which overlaps with river edges
    3. Compact water bodies were not adequately protected
  
  v23.0 fixes:
    1. POND-FIRST LOGIC: Any compact water body (< 500 connected pixels) with
       reasonable water signal is ASSUMED to be a pond unless proven otherwise.
    2. STRICT RIVER CRITERIA: Rivers must have JRC transition=1 (permanent water
       since 1984) AND high JRC occurrence (>85%) AND river-like shape.
    3. NO EXPANSION INTO COMPACT WATER: River mask expansion NEVER touches
       connected water bodies < 500 pixels, regardless of spectral signature.
    4. TEXTURE RESCUE STRENGTHENED: Smooth texture (high IDM) = managed pond
       surface, protected from river masking.
    5. MNDWI-BASED RESCUE: Turbid water (MNDWI -0.30 to 0.0) in compact bodies
       is assumed to be aquaculture, not river.

  WHY THIS WORKS: Aquaculture ponds are engineered structures with:
    - Compact shapes (area/perimeter² > 0.012)
    - Defined boundaries (not meandering like rivers)
    - Relatively uniform water (smooth texture)
    - Limited connected extent (rarely > 500 pixels at 10m resolution)

Research references:
  Pekel et al. 2016 (JRC Global Surface Water — Nature 540:418-422)
  Xu 2006 (MNDWI for water detection)
  Xie et al. 2024 (Aquaculture detection with GLCM texture features)
"""

import ee
import config
from modules.stage_spec import DEFAULT_STAGE_SPEC


def compute_river_mask(features, aoi=None):
    """
    Compute river mask using JRC transition as the primary signal.

    Returns dict with keys:
      'is_river': ee.Image (binary 0/1, unmasked for all pixels)
      'river_probability': ee.Image (continuous 0-1)
      'pond_rescue': ee.Image (binary 0/1)
      'texture_is_smooth': ee.Image (binary 0/1)
      'texture_is_rough': ee.Image (binary 0/1)
      'is_near_river': ee.Image (binary 0/1, 3px buffer around river)
    """
    band_names = features.bandNames()

    def safe_select(band_name, default_val):
        return ee.Image(
            ee.Algorithms.If(
                band_names.contains(band_name),
                features.select(band_name),
                ee.Image(default_val),
            )
        ).rename(band_name)

    # Import thresholds from single source of truth
    spec = DEFAULT_STAGE_SPEC

    ndvi = features.select("ndvi")
    mndwi = features.select("mndwi")
    ndwi = features.select("ndwi")

    # JRC layers
    has_jrc = band_names.contains("jrc_occurrence")
    has_jrc_seas = band_names.contains("jrc_seasonality")
    has_jrc_transition = band_names.contains("jrc_transition")
    jrc_occurrence = safe_select("jrc_occurrence", 0)
    jrc_seasonality = safe_select("jrc_seasonality", 0)
    jrc_transition = safe_select("jrc_transition", 0)

    # Temporal layers
    water_persistence = safe_select("water_persistence", 0)

    # Connectivity
    hydro_connectivity = safe_select("hydro_connectivity", 0)

    # GLCM Texture (v23.0: use stage_spec thresholds)
    has_texture = band_names.contains("nir_texture_homogeneity")
    nir_texture_homogeneity = safe_select("nir_texture_homogeneity", 0.5)
    nir_texture_variance = safe_select("nir_texture_variance", 500.0)
    nir_texture_asm = safe_select("nir_texture_asm", 0.1)

    # =============================================
    # TEXTURE DEFINITIONS (v23.0: calibrated for pond detection)
    # IDM (homogeneity) in [0,1]: high = smooth pond surface
    # ASM (uniformity) in [0,1]: high = uniform texture = managed pond
    # =============================================
    texture_is_smooth = ee.Image(
        ee.Algorithms.If(
            has_texture,
            nir_texture_homogeneity.gt(spec.idm_smooth_threshold)
            .Or(nir_texture_variance.lt(spec.texture_variance_smooth_max)),
            ee.Image(0),
        )
    )
    texture_is_very_smooth = ee.Image(
        ee.Algorithms.If(
            has_texture,
            nir_texture_homogeneity.gt(spec.idm_very_smooth_threshold)
            .And(nir_texture_variance.lt(300)),
            ee.Image(0),
        )
    )
    texture_is_rough = ee.Image(
        ee.Algorithms.If(
            has_texture,
            nir_texture_homogeneity.lt(spec.idm_rough_threshold)
            .Or(nir_texture_variance.gt(spec.texture_variance_rough_min)),
            ee.Image(0),
        )
    )

    # =============================================
    # SPECTRAL WATER DETECTION (v23.0: pond-preserving thresholds)
    # Key insight: Turbid aquaculture ponds have MNDWI -0.35 to 0.1
    # Rivers have MNDWI > 0.2 (clearer water)
    # =============================================
    is_spectral_water = mndwi.gt(spec.s4_mndwi_min).And(ndvi.lt(spec.s4_ndvi_max))

    # =============================================
    # CONNECTED WATER BODY ANALYSIS (v23.0: CRITICAL FOR POND RESCUE)
    # Small-to-medium connected water = LIKELY POND
    # Very large connected water = POSSIBLY RIVER
    # =============================================
    water_connected = is_spectral_water.selfMask().connectedPixelCount(1024, True).unmask(0)
    
    # v23.0: Size tiers for protection levels
    # Tier A: Very small (< 100 px) = definite pond, NEVER mask
    is_tiny_water = water_connected.gt(0).And(water_connected.lt(100))
    # Tier B: Small-medium (100-500 px) = likely pond, strong protection  
    is_small_water = water_connected.gte(100).And(water_connected.lt(500))
    # Tier C: Medium (500-1000 px) = possible pond, moderate protection
    is_medium_water = water_connected.gte(500).And(water_connected.lt(1000))
    # Tier D: Large (> 1000 px) = could be river, check other evidence
    is_large_water = water_connected.gte(1000)
    
    # Combined compact water (protect from river masking)
    is_compact_water = is_tiny_water.Or(is_small_water).Or(is_medium_water)

    # =============================================
    # PRIMARY RIVER MASK: STRICT CRITERIA (v23.0)
    #
    # Rivers must satisfy MULTIPLE criteria:
    # - JRC Transition class 1 (permanent water since 1984)
    # - High JRC occurrence (> 85%)
    # - Low vegetation (NDVI < 0.15)
    # - Clear water signal (MNDWI > 0.05)
    # - NOT in compact water body
    # - NOT smooth texture (ponds are smoother than rivers)
    # =============================================

    # Tier A: STRICT PERMANENT WATER (class 1) — definite rivers
    tier_a = ee.Image(
        ee.Algorithms.If(
            has_jrc_transition,
            jrc_transition.eq(1)             # Permanent water since 1984
            .And(jrc_occurrence.gt(spec.river_tier1_jrc_min))  # High occurrence
            .And(ndvi.lt(0.15))              # Very low vegetation
            .And(mndwi.gt(0.05))             # Clear water (NOT turbid pond)
            .And(is_compact_water.Not())     # v23.0: NOT compact = possibly river
            .And(texture_is_smooth.Not()),   # v23.0: Smooth texture = pond
            ee.Image(0),
        )
    )

    # Tier B: SEASONAL WATER (class 4) — tidal channels (STRICT)
    has_hydro = band_names.contains("hydro_connectivity")
    is_hydro_connected = ee.Image(
        ee.Algorithms.If(
            has_hydro,
            hydro_connectivity.gt(spec.river_tier3_hydro_connectivity_min),
            ee.Image(0),
        )
    )

    tier_b = ee.Image(
        ee.Algorithms.If(
            has_jrc_transition,
            jrc_transition.eq(4)                     # Seasonal water in both epochs
            .And(jrc_seasonality.gte(spec.river_tier2_seasonality_min))  # Present most of the year
            .And(jrc_occurrence.gt(spec.river_tier2_jrc_min))  # Moderate-high occurrence
            .And(ndvi.lt(0.12))                      # Very strong water signal
            .And(mndwi.gt(0.10))                     # Clear water
            .And(is_hydro_connected)                 # Connected to river network
            .And(is_compact_water.Not())             # v23.0: NOT compact
            .And(texture_is_very_smooth.Not()),      # v23.0: Very smooth = pond
            ee.Image(0),
        )
    )

    # Tier C: FALLBACK (no transition data available) — very strict
    tier_c = ee.Image(
        ee.Algorithms.If(
            ee.Algorithms.If(has_jrc_transition, ee.Number(0), ee.Number(1)),
            jrc_occurrence.gt(92)           # Very high occurrence
            .And(ndvi.lt(0.10))             # Almost no vegetation
            .And(is_large_water)            # Must be large
            .And(mndwi.gt(0.20))            # Clear water
            .And(is_compact_water.Not()),   # NOT compact
            ee.Image(0),
        )
    )

    # Tier D: Very large + extremely clear water = river
    tier_d = is_large_water.And(
        mndwi.gt(0.40)                      # Very clear water
    ).And(
        ndvi.lt(0.05)                       # No vegetation
    ).And(
        is_hydro_connected
    ).And(
        texture_is_very_smooth.Not()
    )

    # Combine all tiers (v23.0: stricter combination)
    is_river_core = tier_a.Or(tier_b).Or(tier_c).Or(tier_d)

    # =============================================
    # POND RESCUE — AGGRESSIVE PROTECTION (v23.0)
    #
    # v23.0 PHILOSOPHY: Assume water is a POND unless proven to be a RIVER.
    # This reverses the previous logic which assumed rivers by default.
    #
    # Rescue conditions (ANY triggers protection):
    #   1. JRC transition says land became water = aquaculture
    #   2. Compact water body (< 500 connected px) = pond
    #   3. Smooth texture (high IDM) = managed pond surface
    #   4. Turbid water (low MNDWI) in compact body = aquaculture
    #   5. Very small water (< 100 px) = definitely pond
    #   6. High ASM (uniform texture) = engineered structure
    # =============================================

    # PRIMARY RESCUE: JRC transition says this was NEVER permanent water
    jrc_says_aquaculture = ee.Image(
        ee.Algorithms.If(
            has_jrc_transition,
            jrc_transition.eq(2)      # New Permanent (land → water)
            .Or(jrc_transition.eq(5)) # New Seasonal (land → seasonal water)
            .Or(jrc_transition.eq(7)) # Seasonal → Permanent (converted pond)
            .Or(jrc_transition.eq(0)) # Not water in JRC = definitely not river
            .Or(jrc_transition.eq(3)) # Lost Permanent = was water, now land
            .Or(jrc_transition.eq(6)) # Lost Seasonal
            .Or(jrc_transition.eq(9)) # Ephemeral Permanent
            .Or(jrc_transition.eq(10)), # Ephemeral Seasonal
            ee.Image(0),
        )
    )

    # SIZE-BASED RESCUE: All compact water bodies protected
    pond_rescue_size = is_compact_water  # v23.0: ALL compact water = protected

    # TEXTURE RESCUE: Smooth or uniform texture = managed pond
    pond_rescue_texture = ee.Image(
        ee.Algorithms.If(
            has_texture,
            texture_is_smooth.Or(nir_texture_asm.gt(0.15)),  # Smooth OR uniform
            ee.Image(0),
        )
    )

    # TURBID WATER RESCUE: Low MNDWI in compact body = turbid aquaculture pond
    # Rivers have MNDWI > 0.1 (clear water), ponds can be -0.35 to 0.1 (turbid)
    pond_rescue_turbid = is_compact_water.And(
        mndwi.gt(spec.s4_mndwi_min)  # Has some water signal
    ).And(
        mndwi.lt(0.15)               # But not crystal clear (would be river)
    ).And(
        ndvi.lt(spec.s4_ndvi_max)
    )

    # TINY WATER RESCUE: Very small = always pond
    pond_rescue_tiny = is_tiny_water

    # VERY SMOOTH RESCUE: Exceptionally smooth surface = engineered pond
    pond_rescue_very_smooth = texture_is_very_smooth

    # Combined pond rescue — aggressive OR logic
    pond_rescue = (
        jrc_says_aquaculture
        .Or(pond_rescue_size)
        .Or(pond_rescue_texture)
        .Or(pond_rescue_turbid)
        .Or(pond_rescue_tiny)
        .Or(pond_rescue_very_smooth)
    )

    # Apply pond rescue — rescued pixels CANNOT be rivers
    is_river = is_river_core.And(pond_rescue.Not())

    # =============================================
    # v23.0: CONSERVATIVE RIVER EXPANSION
    #
    # Previous versions expanded aggressively into all adjacent water-like pixels.
    # This destroyed aquaculture ponds near rivers.
    #
    # v23.0: Expansion is DISABLED by default. River mask is the CORE only.
    # If enabled, expansion ONLY touches pixels that are:
    #   1. Very large connected water (> 1000 px)
    #   2. Crystal clear (MNDWI > 0.25)
    #   3. No vegetation (NDVI < 0.10)
    #   4. NOT rescued as pond
    #   5. NOT compact water
    # =============================================
    river_cfg = getattr(config, 'RIVER_MASKING', {})
    if river_cfg.get('conditional_expansion', False):  # v23.0: DISABLED by default
        expansion_allowed = (
            mndwi.gt(0.25)               # v23.0: Very clear water ONLY
            .And(ndvi.lt(0.10))          # v23.0: No vegetation
            .And(pond_rescue.Not())      # Must not be a rescued pond
            .And(is_compact_water.Not()) # Must not be compact water
            .And(is_large_water)         # Must be large water body
        )

        # Single 1px expansion with very strict guard
        expansion_1 = is_river.focal_max(radius=1, units='pixels')
        is_river = expansion_1.And(expansion_allowed).Or(is_river)

    # =============================================
    # v21.0: IS_NEAR_RIVER — Disabled
    #
    # The 3px buffer was completely destroying aquaculture
    # classification because ponds are naturally near rivers.
    # The 2px spectrally-guarded expansion is sufficient.
    # =============================================
    is_near_river = ee.Image(0).rename('is_near_river')

    # =============================================
    # RIVER PROBABILITY (continuous 0-1)
    # =============================================
    transition_prob = ee.Image(
        ee.Algorithms.If(
            has_jrc_transition,
            jrc_transition.eq(1).toFloat().multiply(0.7)
            .add(jrc_transition.eq(4).toFloat().multiply(0.4)),
            jrc_occurrence.divide(100).clamp(0, 1).multiply(0.5),
        )
    )
    jrc_weight = ee.Image(
        ee.Algorithms.If(
            has_jrc,
            jrc_occurrence.divide(100).clamp(0, 1),
            ee.Image(0.3),
        )
    )
    river_probability = (
        transition_prob.multiply(0.6)
        .add(jrc_weight.multiply(0.2))
        .add(is_river.toFloat().multiply(0.2))
    ).clamp(0, 1).rename("river_probability")
    river_probability = river_probability.where(pond_rescue, 0)

    # =============================================
    # CRITICAL: unmask ALL outputs to prevent mask propagation
    # =============================================
    return {
        'is_river': is_river.unmask(0).rename('is_river'),
        'river_probability': river_probability.unmask(0),
        'pond_rescue': pond_rescue.unmask(0).rename('pond_rescue'),
        'texture_is_smooth': texture_is_smooth.unmask(0),
        'texture_is_rough': texture_is_rough.unmask(0),
        'is_near_river': is_near_river.unmask(0).rename('is_near_river'),
    }
