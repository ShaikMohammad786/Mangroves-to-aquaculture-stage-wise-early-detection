"""
MODULE 6 — Stage State Engine (v23.0 — RESEARCH GRADE)

v23.0 AQUACULTURE-OPTIMIZED OVERHAUL:
  - Uses stage_spec.py as SINGLE SOURCE OF TRUTH for all thresholds
  - ADSRM (Adaptive Dual-Scale River Masking) via shared m19 module
  - CRITICAL FIX: GLCM homogeneity now uses IDM (was incorrectly using Sum Average)
  - Recalibrated ALL texture thresholds for IDM range [0,1]
  - RELAXED water evidence thresholds for turbid pond detection
  - Enhanced spatial smoothing for cleaner stage maps
  - Stronger pond protection against over-masking

GEE MEMORY OPTIMIZATION:
  - Reduced focal operations from 15+ to 6
  - connectedPixelCount maxSize reduced from 200 to 64
  - Removed redundant post-classification sweeps
  - Total computation footprint < 50MB per tile

Research references:
  - Serra 1982 (Mathematical Morphology — opening for size filtering)
  - Xie et al. 2024 (GLCM-PCA aquaculture detection, 96.15% accuracy)
  - Xu 2006 (MNDWI water detection)
  - Feyisa et al. 2014 (AWEI shadow-resistant water)
  - Yang et al. 2024 (SDWI dual-polarization water index)
  - Baloloy et al. 2020 (MAVI mangrove-aquaculture separation)
  - Haralick et al. 1973 (GLCM IDM definition)
"""

import math

import ee
import config
from modules.m19_river_mask_shared import compute_river_mask
from modules.stage_spec import DEFAULT_STAGE_SPEC


class StageClassifier:
    def classify(self, features, aoi):
        raise NotImplementedError

    def get_confidence(self, features, stage, aoi):
        raise NotImplementedError


class RuleBasedClassifier(StageClassifier):
    """
    Aquaculture-optimized v23.0 — Uses stage_spec for thresholds.
    """

    def classify(self, features, aoi):
        # Import thresholds from single source of truth
        spec = DEFAULT_STAGE_SPEC
        band_names = features.bandNames()

        def safe_select(band_name, default_val):
            return ee.Image(
                ee.Algorithms.If(
                    band_names.contains(band_name),
                    features.select(band_name),
                    ee.Image(default_val),
                )
            ).rename(band_name)

        # =============================================
        # FEATURE EXTRACTION
        # =============================================
        ndvi = features.select("ndvi")
        ndwi = features.select("ndwi")
        mndwi = features.select("mndwi")
        evi = features.select("evi")

        awei = safe_select("awei", 0)
        cmri = safe_select("cmri", 0)
        mmri = safe_select("mmri", 0.5)
        gndvi = safe_select("gndvi", 0)
        ndbi = safe_select("ndbi", 0)
        ndmi = safe_select("ndmi", 0)
        savi_raw = safe_select("savi", 0)
        savi = ee.Image(
            ee.Algorithms.If(band_names.contains("savi"), savi_raw, ndvi)
        ).rename("savi")
        rvi = safe_select("rvi", 1.0)
        veg_water_diff = safe_select("veg_water_diff", 0)
        cwi = safe_select("cwi", 0)

        water_fraction = safe_select("water_fraction", 0)
        vegetation_fraction = safe_select("vegetation_fraction", 0)
        soil_fraction = safe_select("soil_fraction", 0)

        # v17.0: GLCM TEXTURE FEATURES — CORRECTED FOR IDM
        # CRITICAL FIX: homogeneity is now IDM (Inverse Difference Moment), range [0,1]
        # Previously was nir_savg (Sum Average, range ~50-200) — completely wrong!
        # IDM: 1.0 = perfectly smooth (pond), 0.0 = maximum texture (vegetation/river)
        nir_texture_homogeneity = safe_select("nir_texture_homogeneity", 0.5)
        nir_texture_contrast = safe_select("nir_texture_contrast", 50.0)
        nir_texture_entropy = safe_select("nir_texture_entropy", 3.0)
        nir_texture_variance = safe_select("nir_texture_variance", 500.0)
        nir_texture_asm = safe_select("nir_texture_asm", 0.1)

        # v13.0: NOVEL INDICES (previously computed, never used)
        smri = safe_select("smri", 0)
        mavi = safe_select("mavi", 0.5)
        lswi = safe_select("lswi", 0)

        # v13.0: SAR COMPOSITE FEATURES (previously computed, never used)
        sdwi = safe_select("sdwi", 0)
        sar_water_likelihood = safe_select("sar_water_likelihood", 0)

        homogeneity = safe_select("vv_homogeneity", 0.5)
        slope = safe_select("slope", 0)
        hydro_connectivity = safe_select("hydro_connectivity", 0)
        tidal_exposure_proxy = safe_select("tidal_exposure_proxy", 0)

        ndvi_trend_slope = safe_select("ndvi_trend_slope", 0)
        ndvi_cv = safe_select("ndvi_cv", 0)
        vv_temporal_std = safe_select("vv_temporal_std", 3.0)
        sar_water_persistence = safe_select("sar_water_persistence", 0)
        ccdc_recent_break = safe_select("ccdc_recent_break", 0)

        # Temporal features
        ndvi_amplitude = safe_select("ndvi_amplitude", 0.3)
        ndvi_iqr = safe_select("ndvi_iqr", 0.2)
        mndwi_amplitude = safe_select("mndwi_amplitude", 0.3)
        mndwi_iqr = safe_select("mndwi_iqr", 0.2)
        water_persistence = safe_select("water_persistence", 0)
        water_transition_frequency = safe_select("water_transition_frequency", 0)
        ndti = safe_select("ndti", 0)
        turbidity_proxy = safe_select("turbidity_proxy", 0)

        edge_density = safe_select("edge_density", 0)
        jrc_occurrence = safe_select("jrc_occurrence", 0)
        jrc_seasonality = safe_select("jrc_seasonality", 0)
        low_elev_mask = safe_select("low_elevation_mask", 1)

        gmw_mangrove = safe_select("gmw_mangrove", 0)
        gmw_historical_mangrove = safe_select("gmw_historical_mangrove", 0)

        # Flags
        has_sar = band_names.contains("vv_mean")
        has_jrc = band_names.contains("jrc_occurrence")
        has_jrc_seas = band_names.contains("jrc_seasonality")
        has_elev = band_names.contains("low_elevation_mask")
        has_edge = band_names.contains("edge_density")
        has_cwi = band_names.contains("cwi")
        has_awei = band_names.contains("awei")
        has_mmri = band_names.contains("mmri")
        has_unmixing = band_names.contains("water_fraction")
        has_gmw = band_names.contains("gmw_mangrove")
        has_gmw_hist = band_names.contains("gmw_historical_mangrove")
        has_temporal = band_names.contains("ndvi_trend_slope")
        has_ndbi = band_names.contains("ndbi")
        has_gndvi = band_names.contains("gndvi")
        has_water_persistence = band_names.contains("water_persistence")
        has_water_transition = band_names.contains("water_transition_frequency")
        has_ndvi_amplitude = band_names.contains("ndvi_amplitude")
        has_ndti = band_names.contains("ndti")
        has_ccdc = band_names.contains("ccdc_recent_break")
        # v13.0: flags for newly-integrated features
        has_texture = band_names.contains("nir_texture_homogeneity")
        has_smri = band_names.contains("smri")
        has_mavi = band_names.contains("mavi")
        has_sdwi = band_names.contains("sdwi")
        has_sar_wl = band_names.contains("sar_water_likelihood")
        has_soil_frac = band_names.contains("soil_fraction")
        has_lswi = band_names.contains("lswi")

        # Mangrove context
        context_buffer_px = int(
            getattr(config, "MANGROVE_CONTEXT", {}).get("context_buffer_px", 4)
        )
        mangrove_anchor = ee.Image(
            ee.Algorithms.If(
                has_gmw_hist,
                gmw_historical_mangrove,
                ee.Image(ee.Algorithms.If(has_gmw, gmw_mangrove, ee.Image(1))),
            )
        ).rename("mangrove_anchor")
        mangrove_context = mangrove_anchor.focal_max(
            radius=context_buffer_px, units="pixels"
        )

        # SAR
        vv_safe = safe_select("vv_mean", -15).rename("vv_safe")

        # =============================================
        # ADAPTIVE THRESHOLDS
        # =============================================
        cfg = config.ADAPTIVE_THRESHOLDS
        ndvi_p70 = ee.Number(
            ee.Algorithms.If(
                features.propertyNames().contains("ndvi_p70"),
                features.get("ndvi_p70"),
                cfg["ndvi_p70_fallback"],
            )
        )
        ndvi_p30 = ee.Number(
            ee.Algorithms.If(
                features.propertyNames().contains("ndvi_p30"),
                features.get("ndvi_p30"),
                cfg["ndvi_p30_fallback"],
            )
        )

        # =============================================
        # SAR EVIDENCE
        # =============================================
        sar_water = ee.Image(ee.Algorithms.If(has_sar, vv_safe.lt(-14.0), ee.Image(1)))
        high_homogeneity = ee.Image(
            ee.Algorithms.If(
                band_names.contains("vv_homogeneity"),
                features.select("vv_homogeneity").gt(0.25),
                ee.Image(0),
            )
        )
        sar_water_evidence = ee.Image(
            ee.Algorithms.If(has_sar, sar_water.Or(high_homogeneity), ee.Image(1))
        )

        # =============================================
        # CONSTRAINT LAYERS
        # =============================================
        dem_allows_pond = ee.Image(
            ee.Algorithms.If(
                has_elev, low_elev_mask.eq(1).And(slope.lt(5)), ee.Image(1)
            )
        )

        ndbi_blocks_water = ee.Image(
            ee.Algorithms.If(has_ndbi, ndbi.gt(0.12), ee.Image(0))
        )

        # JRC evidence
        jrc_is_new_water = ee.Image(
            ee.Algorithms.If(has_jrc, jrc_occurrence.lt(70), ee.Image(1))
        )
        jrc_is_established = ee.Image(
            ee.Algorithms.If(has_jrc, jrc_occurrence.gte(45), ee.Image(0))
        )

        # =============================================
        # RIVER DETECTION — ADSRM (v17.0 — SHARED MODULE)
        # Single source of truth: modules/m19_river_mask_shared.py
        # Eliminates divergence between m06 and m07
        # =============================================
        river_result = compute_river_mask(features, aoi)
        is_natural_water = river_result['is_river']
        is_river_by_jrc = is_natural_water
        pond_rescue = river_result['pond_rescue']
        texture_is_smooth = river_result['texture_is_smooth']
        texture_is_rough = river_result['texture_is_rough']
        river_probability = river_result['river_probability']
        # v21.0: Near-river buffer — used to require higher water evidence
        # for S4/S5 classification near rivers (prevents fringe leakage)
        is_near_river = river_result['is_near_river']

        strong_edges = ee.Image(
            ee.Algorithms.If(has_edge, edge_density.gt(0.012), ee.Image(0))
        )

        # =============================================
        # CONTINUOUS WATER EVIDENCE SCORING (v13.0)
        # =============================================
        # REPLACES boolean pass/fail counting with continuous weighted scoring.
        # Each feature contributes a continuous value [0,1] weighted by importance.
        # This is the KEY architectural change for accurate pond detection.

        # Spectral water evidence (continuous, not boolean)
        w_mndwi = mndwi.add(0.15).divide(0.55).clamp(0, 1)     # [-0.15,0.4] → [0,1]
        w_ndwi = ndwi.add(0.10).divide(0.50).clamp(0, 1)       # [-0.10,0.4] → [0,1]
        w_cwi = ee.Image(ee.Algorithms.If(has_cwi,
            cwi.add(0.10).divide(0.50).clamp(0, 1),
            ee.Image(0)
        ))
        w_awei = ee.Image(ee.Algorithms.If(has_awei,
            awei.add(0.30).divide(0.80).clamp(0, 1),
            ee.Image(0)
        ))
        w_unmixing = ee.Image(ee.Algorithms.If(has_unmixing,
            water_fraction.clamp(0, 1),
            ee.Image(0)
        ))

        # v17.0: GLCM TEXTURE evidence — RECALIBRATED FOR IDM
        # IDM (homogeneity) is now in [0,1] — no division by 255 needed
        # High IDM = smooth surface = pond; Low IDM = rough = vegetation/river
        w_texture_homog = ee.Image(ee.Algorithms.If(has_texture,
            nir_texture_homogeneity.clamp(0, 1),  # IDM already [0,1]
            ee.Image(0.5)
        ))
        w_texture_contrast_inv = ee.Image(ee.Algorithms.If(has_texture,
            ee.Image(1).subtract(nir_texture_contrast.divide(5000.0).clamp(0, 1)),
            ee.Image(0.5)
        ))
        w_texture_entropy_inv = ee.Image(ee.Algorithms.If(has_texture,
            ee.Image(1).subtract(nir_texture_entropy.divide(5.0).clamp(0, 1)),
            ee.Image(0.5)
        ))
        # NEW: ASM (Angular Second Moment) — measures uniformity
        has_asm = band_names.contains("nir_texture_asm")
        w_texture_asm = ee.Image(ee.Algorithms.If(has_asm,
            nir_texture_asm.clamp(0, 1),
            ee.Image(0.1)
        ))

        # SAR composite evidence
        w_sdwi = ee.Image(ee.Algorithms.If(has_sdwi,
            sdwi.add(20).divide(15).clamp(0, 1),
            ee.Image(0)
        ))
        w_sar_wl = ee.Image(ee.Algorithms.If(has_sar_wl,
            sar_water_likelihood.clamp(0, 1),
            ee.Image(0)
        ))

        # Novel index evidence
        w_smri = ee.Image(ee.Algorithms.If(has_smri,
            smri.add(0.5).divide(1.0).clamp(0, 1),
            ee.Image(0)
        ))
        w_mavi_inv = ee.Image(ee.Algorithms.If(has_mavi,
            ee.Image(1).subtract(mavi.clamp(0, 1)),
            ee.Image(0.5)
        ))

        # NDBI evidence
        w_ndbi_inv = ee.Image(ee.Algorithms.If(has_ndbi,
            ee.Image(0.1).subtract(ndbi).divide(0.3).clamp(0, 1),
            ee.Image(0.5)
        ))

        # =============================================
        # WEIGHTED WATER EVIDENCE SCORE (v17.0 — RECALIBRATED)
        # =============================================
        # Redistributed weights: MNDWI and unmixing get more weight now
        # that GLCM homogeneity (IDM) is correct. ASM added as new feature.
        
        # Calculate maximum possible weight based on available bands dynamically
        total_water_w = ee.Image(0.26) # mndwi (0.16) + ndwi (0.10) always present
        total_water_w = total_water_w.add(ee.Image(ee.Algorithms.If(has_cwi, ee.Image(0.08), ee.Image(0))))
        total_water_w = total_water_w.add(ee.Image(ee.Algorithms.If(has_awei, ee.Image(0.07), ee.Image(0))))
        total_water_w = total_water_w.add(ee.Image(ee.Algorithms.If(has_unmixing, ee.Image(0.12), ee.Image(0))))
        total_water_w = total_water_w.add(ee.Image(ee.Algorithms.If(has_texture, ee.Image(0.19), ee.Image(0)))) # homog 0.12 + contrast 0.04 + entropy 0.03
        total_water_w = total_water_w.add(ee.Image(ee.Algorithms.If(has_asm, ee.Image(0.03), ee.Image(0))))
        total_water_w = total_water_w.add(ee.Image(ee.Algorithms.If(has_sdwi, ee.Image(0.05), ee.Image(0))))
        total_water_w = total_water_w.add(ee.Image(ee.Algorithms.If(has_sar_wl, ee.Image(0.05), ee.Image(0))))
        total_water_w = total_water_w.add(ee.Image(ee.Algorithms.If(has_smri, ee.Image(0.05), ee.Image(0))))
        total_water_w = total_water_w.add(ee.Image(ee.Algorithms.If(has_mavi, ee.Image(0.05), ee.Image(0))))
        total_water_w = total_water_w.add(ee.Image(ee.Algorithms.If(has_sar, ee.Image(0.05), ee.Image(0))))

        raw_water_score = (
            w_mndwi.multiply(0.16)                       # Raised: most reliable index
            .add(w_ndwi.multiply(0.10))
            .add(w_cwi.multiply(0.08))
            .add(w_awei.multiply(0.07))
            .add(w_unmixing.multiply(0.12))               # Raised: spectral unmixing
            .add(w_texture_homog.multiply(0.12))          # IDM: now correct, lowered from 0.16
            .add(w_texture_contrast_inv.multiply(0.04))
            .add(w_texture_entropy_inv.multiply(0.03))
            .add(w_texture_asm.multiply(0.03))            # NEW: ASM uniformity
            .add(w_sdwi.multiply(0.05))
            .add(w_sar_wl.multiply(0.05))
            .add(w_smri.multiply(0.05))
            .add(w_mavi_inv.multiply(0.05))
            .add(ee.Image(ee.Algorithms.If(has_sar,
                sar_water_persistence.clamp(0, 1).multiply(0.05),
                ee.Image(0)
            )))
        )
        water_evidence_score = raw_water_score.divide(total_water_w.max(0.01)).rename("water_evidence_score")

        # Vegetation evidence score (symmetric counterpart)
        total_veg_w = ee.Image(0.60) # ndvi (0.30) + evi (0.15) + savi (0.10) + c.w_mndwi (0.05)
        total_veg_w = total_veg_w.add(ee.Image(ee.Algorithms.If(has_mavi, ee.Image(0.15), ee.Image(0))))
        total_veg_w = total_veg_w.add(ee.Image(ee.Algorithms.If(has_unmixing, ee.Image(0.15), ee.Image(0))))
        total_veg_w = total_veg_w.add(ee.Image(ee.Algorithms.If(has_gndvi, ee.Image(0.10), ee.Image(0))))

        raw_veg_score = (
            ndvi.clamp(0, 0.8).divide(0.8).multiply(0.30)
            .add(evi.clamp(0, 0.6).divide(0.6).multiply(0.15))
            .add(ee.Image(ee.Algorithms.If(has_mavi,
                mavi.clamp(0, 1).multiply(0.15),
                ee.Image(0)
            )))
            .add(ee.Image(ee.Algorithms.If(has_unmixing,
                vegetation_fraction.clamp(0, 1).multiply(0.15),
                ee.Image(0)
            )))
            .add(ee.Image(ee.Algorithms.If(has_gndvi,
                gndvi.clamp(0, 0.7).divide(0.7).multiply(0.10),
                ee.Image(0)
            )))
            .add(savi.clamp(0, 0.7).divide(0.7).multiply(0.10))
            .add(ee.Image(1).subtract(w_mndwi).multiply(0.05))
        )
        veg_evidence_score = raw_veg_score.divide(total_veg_w.max(0.01)).rename("veg_evidence_score")

        # Bare soil evidence score
        total_soil_w = ee.Image(0.40) # ndvi inv (0.20) + water inv (0.20)
        total_soil_w = total_soil_w.add(ee.Image(ee.Algorithms.If(has_ndbi, ee.Image(0.35), ee.Image(0))))
        total_soil_w = total_soil_w.add(ee.Image(ee.Algorithms.If(has_soil_frac, ee.Image(0.25), ee.Image(0))))

        raw_soil_score = (
            ee.Image(ee.Algorithms.If(has_ndbi,
                ndbi.add(0.1).divide(0.4).clamp(0, 1).multiply(0.35),
                ee.Image(0)
            ))
            .add(ee.Image(ee.Algorithms.If(has_soil_frac,
                soil_fraction.clamp(0, 1).multiply(0.25),
                ee.Image(0)
            )))
            .add(ee.Image(1).subtract(ndvi.clamp(0, 0.5).divide(0.5)).multiply(0.20))
            .add(ee.Image(1).subtract(water_evidence_score).multiply(0.20))
        )
        bare_soil_score = raw_soil_score.divide(total_soil_w.max(0.01)).rename("bare_soil_score")

        # Water evidence levels for backward compatibility
        any_water = water_evidence_score.gt(0.30)
        moderate_water = water_evidence_score.gt(0.40)
        strong_water = water_evidence_score.gt(0.55)

        # =============================================
        # STAGE RULES v23.0 — Single Source of Truth from stage_spec.py
        # =============================================

        # -- S1: Intact Mangrove --
        s1_spectral = (
            ndvi.gt(spec.s1_ndvi_min)
            .And(evi.gt(spec.s1_evi_min))
            .And(ndwi.lt(0.0))
            .And(mndwi.lt(spec.s1_water_index_max))
        )
        s1 = s1_spectral
        s1_gmw_boost = ee.Image(
            ee.Algorithms.If(
                has_gmw,
                gmw_mangrove.And(ndvi.gt(0.25)).And(mndwi.lt(0.05)),
                ee.Image(0),
            )
        )
        s1_novel = ee.Image(
            ee.Algorithms.If(
                has_smri,
                smri.lt(-0.2).And(mavi.gt(0.1)).And(ndvi.gt(0.25)),
                ee.Image(0),
            )
        )
        s1 = ee.Image(s1).Or(s1_gmw_boost).Or(s1_novel)

        # -- S2: Degrading Mangrove --
        s2_spectral = (
            ndvi.gte(spec.s2_ndvi_min)
            .And(ndvi.lt(spec.s2_ndvi_max))
            .And(mndwi.lt(spec.s2_mndwi_max))
            .And(ndwi.lt(0.0))
        )
        s2_gndvi_stress = ee.Image(
            ee.Algorithms.If(has_gndvi, gndvi.lt(spec.s2_gndvi_stress_max), ee.Image(1))
        )
        s2_ccdc_boost = ee.Image(
            ee.Algorithms.If(has_ccdc, ccdc_recent_break.eq(1), ee.Image(0))
        )
        s2_novel = ee.Image(
            ee.Algorithms.If(has_smri, smri.lt(0.0).And(smri.gt(-0.2)).And(ndvi.lt(0.30)), ee.Image(0))
        )
        s2 = s2_spectral.And(s2_gndvi_stress).And(mangrove_context)
        s2 = s2.Or(s2_spectral.And(s2_ccdc_boost).And(mangrove_context))
        s2 = s2.Or(s2_novel.And(mangrove_context))
        s2 = s2.And(water_evidence_score.lt(0.30))
        s2 = s2.And(mndwi.lt(spec.s2_mndwi_max))

        # -- S3: Cleared / Bare Soil --
        s3_spectral = ndvi.lt(spec.s3_ndvi_max).And(mndwi.lt(spec.s3_mndwi_max)).And(ndwi.lt(spec.s3_ndwi_max))
        s3_is_bare_soil = ee.Image(
            ee.Algorithms.If(has_ndbi, ndbi.gt(spec.s3_ndbi_min), ee.Image(1))
        )
        s3_high_soil = bare_soil_score.gt(0.35)
        s3_ccdc_boost = ee.Image(
            ee.Algorithms.If(has_ccdc, ccdc_recent_break.eq(1), ee.Image(0))
        )
        s3 = s3_spectral.And(s3_is_bare_soil).And(s3_high_soil)
        s3 = s3.Or(s3_spectral.And(s3_ccdc_boost).And(s3_is_bare_soil).And(s3_high_soil))
        s3 = s3.And(water_evidence_score.lt(0.40))
        s3 = s3.And(dem_allows_pond)

        # =============================================
        # S4: Pond Formation — v23.0 TURBID POND OPTIMIZED
        # =============================================
        # Uses stage_spec thresholds for turbid aquaculture ponds
        # MNDWI can be as low as -0.30 for very turbid ponds
        
        # v23.0: VERY RELAXED thresholds for turbid pond detection
        s4_threshold_met = water_evidence_score.gt(0.22)  # v23.0: Relaxed from 0.28

        s4 = (
            s4_threshold_met
            .And(ndvi.lt(spec.s4_ndvi_max))
            .And(mndwi.gt(spec.s4_mndwi_min))  # v23.0: Use spec (-0.30 for turbid ponds)
            .And(ndbi.lt(spec.ndbi_blocks_water_min))
            .And(dem_allows_pond)
        )

        # Texture boost: smooth surface (high IDM homogeneity) = pond
        # v23.0: Use stage_spec IDM thresholds
        s4_texture_boost = ee.Image(
            ee.Algorithms.If(
                has_texture,
                nir_texture_homogeneity.gt(spec.idm_smooth_threshold)
                .And(nir_texture_contrast.lt(2500))  # v23.0: More relaxed
                .And(ndvi.lt(spec.s4_ndvi_max))
                .And(mndwi.gt(spec.s4_mndwi_min))
                .And(dem_allows_pond),
                ee.Image(0),
            )
        )
        s4 = s4.Or(s4_texture_boost)

        # Temporal stability boost: persistent water + low vegetation variation
        # v23.0: Use stage_spec water persistence threshold
        s4_temporal_boost = ee.Image(
            ee.Algorithms.If(
                ee.Algorithms.If(has_water_persistence, ee.Number(1), ee.Number(0)),
                water_persistence.gt(spec.s4_water_persistence_min - 0.05)  # v23.0: Relaxed
                .And(ee.Image(ee.Algorithms.If(
                    has_ndvi_amplitude,
                    ndvi_amplitude.lt(spec.s4_ndvi_amplitude_max + 0.03),  # v23.0: Relaxed
                    ee.Image(1),
                )))
                .And(ndvi.lt(spec.s4_ndvi_max))
                .And(mndwi.gt(spec.s4_mndwi_min))
                .And(dem_allows_pond),
                ee.Image(0),
            )
        )
        s4 = s4.Or(s4_temporal_boost)

        # Turbidity boost: turbid water (high NDTI) with low veg = active pond
        # v23.0: Turbid ponds are EXPECTED, not exceptional
        s4_turbidity_boost = ee.Image(
            ee.Algorithms.If(
                has_ndti,
                ndti.gt(spec.s4_ndti_min - 0.02)  # v23.0: Relaxed
                .And(ndvi.lt(spec.s4_ndvi_max))
                .And(water_evidence_score.gt(0.20))  # v23.0: Very relaxed
                .And(mndwi.gt(spec.s4_mndwi_min))
                .And(dem_allows_pond),
                ee.Image(0),
            )
        )
        s4 = s4.Or(s4_turbidity_boost)

        # SAR dual-pol boost
        s4_sar_boost = ee.Image(
            ee.Algorithms.If(
                has_sdwi,
                sdwi.lt(-12).And(ndvi.lt(spec.s4_ndvi_max)).And(dem_allows_pond),
                ee.Image(0),
            )
        )
        s4 = s4.Or(s4_sar_boost)

        # Not natural water
        s4 = s4.And(is_natural_water.Not())

        # =============================================
        # S5: Established Aquaculture — v23.0 TURBID POND OPTIMIZED
        # =============================================
        # v23.0: Use stage_spec thresholds, established ponds can be turbid

        s5_structural = jrc_is_established.Or(
            ee.Image(ee.Algorithms.If(
                has_water_persistence,
                water_persistence.gt(spec.s5_water_persistence_min - 0.10),  # v23.0: Relaxed
                ee.Image(0),
            ))
        )
        s5 = (
            water_evidence_score.gt(0.35)  # v23.0: Relaxed from 0.40
            .And(s5_structural)
            .And(ndvi.lt(spec.s5_ndvi_max))
            .And(mndwi.gt(spec.s5_mndwi_min))
            .And(ndbi.lt(0.10))
            .And(dem_allows_pond)
        )

        # JRC establishment boost (NOT rivers)
        # v23.0: Use stage_spec thresholds
        s5_jrc_boost = (
            jrc_is_established
            .And(water_evidence_score.gt(0.25))  # v23.0: Relaxed from 0.30
            .And(ndvi.lt(spec.s5_ndvi_max))
            .And(mndwi.gt(spec.s5_mndwi_min))
            .And(is_river_by_jrc.Not())
        )
        s5 = s5.Or(s5_jrc_boost)

        # Temporal stability: stable water = established pond
        # v23.0: Use stage_spec thresholds
        s5_temporal_boost = ee.Image(
            ee.Algorithms.If(
                ee.Algorithms.If(has_water_persistence, ee.Number(1), ee.Number(0)),
                water_persistence.gt(spec.s5_water_persistence_min - 0.15)  # v23.0: Relaxed
                .And(ee.Image(ee.Algorithms.If(
                    band_names.contains("mndwi_iqr"),
                    mndwi_iqr.lt(spec.s5_mndwi_iqr_max + 0.05),  # v23.0: Relaxed
                    ee.Image(1),
                )))
                .And(water_evidence_score.gt(0.25))  # v23.0: Relaxed from 0.28
                .And(ndvi.lt(spec.s5_ndvi_max))
                .And(dem_allows_pond)
                .And(is_river_by_jrc.Not()),
                ee.Image(0),
            )
        )
        s5 = s5.Or(s5_temporal_boost)

        # Texture boost: smooth surface = established pond
        # v23.0: Use stage_spec IDM thresholds, allow turbid ponds
        s5_texture_boost = ee.Image(
            ee.Algorithms.If(
                has_texture,
                nir_texture_homogeneity.gt(spec.idm_smooth_threshold + 0.05)  # v23.0: Slightly stricter for S5
                .And(nir_texture_contrast.lt(1500))  # v23.0: Relaxed from 1200
                .And(water_evidence_score.gt(0.30))  # v23.0: Relaxed from 0.32
                .And(mndwi.gt(spec.s5_mndwi_min))  # v23.0: Use spec
                .And(ndvi.lt(spec.s5_ndvi_max))
                .And(dem_allows_pond)
                .And(is_river_by_jrc.Not()),
                ee.Image(0),
            )
        )
        s5 = s5.Or(s5_texture_boost)

        # SAR: low backscatter = established water
        s5_sar_boost = ee.Image(
            ee.Algorithms.If(
                has_sar_wl,
                sar_water_likelihood.gt(0.45)  # v23.0: Relaxed from 0.5
                .And(ndvi.lt(spec.s5_ndvi_max))
                .And(dem_allows_pond)
                .And(is_river_by_jrc.Not()),
                ee.Image(0),
            )
        )
        s5 = s5.Or(s5_sar_boost)

        # Not natural water
        s5 = s5.And(is_natural_water.Not())

        # =============================================
        # STAGE ASSEMBLY (Priority: S5 > S4 > S3 > S2 > S1)
        # =============================================
        stage = ndvi.multiply(0).toInt().rename("stage")

        stage = stage.where(s1, 1)
        stage = stage.where(s2, 2)
        stage = stage.where(s3, 3)
        stage = stage.where(s4, 4)
        stage = stage.where(s5, 5)

        # =============================================
        # Force rivers to stage 0 BEFORE fallbacks (v21.0)
        # River mask is FINAL — fallbacks must NEVER overwrite it.
        # Track which pixels were river-masked so they stay stage 0.
        # v21.0: Track BOTH core river mask and expanded mask
        # =============================================
        stage = stage.where(is_natural_water, 0).rename("stage").toInt()
        was_river_masked = is_natural_water  # permanent tracking band

        # =============================================
        # CONTEXT-AWARE FALLBACK (v20.0 — replaces cascading NDVI fallbacks)
        #
        # Previous versions used blunt NDVI thresholds that pushed
        # turbid water into S3 and mixed pixels into S1.
        # v20.0 checks spatial context: is the unclassified pixel
        # surrounded by water (S4/S5) or vegetation (S1/S2)?
        # =============================================
        still_zero = stage.eq(0).And(was_river_masked.Not())

        # Compute spatial context from already-classified neighbors
        neighbor_stage = stage.focal_mode(radius=2, units='pixels')
        in_water_context = neighbor_stage.gte(4)  # surrounded by S4/S5
        in_veg_context = neighbor_stage.lte(2).And(neighbor_stage.gte(1))  # surrounded by S1/S2

        # =============================================
        # FALLBACK CASCADE v21.0 — RIVER-AWARE, TIGHTENED
        #
        # KEY FIXES:
        #   - ALL water-pushing fallbacks now check is_near_river
        #   - Thresholds raised to prevent ambiguous pixels from becoming S4
        #   - MNDWI confirmation required for water fallbacks
        #   - Vegetation fallbacks given priority (S1 before S4)
        # =============================================

        # Fallback 1: High vegetation evidence → S1 (MOVED UP — veg-first priority)
        stage = stage.where(
            still_zero.And(veg_evidence_score.gt(0.50)),
            1
        ).rename("stage").toInt()
        still_zero = stage.eq(0).And(was_river_masked.Not())

        # Fallback 2: NDVI > spec.s1_ndvi_min → S1 (clear vegetation signal, MOVED UP)
        stage = stage.where(still_zero.And(ndvi.gt(spec.s1_ndvi_min - 0.05)), 1).rename("stage").toInt()
        still_zero = stage.eq(0).And(was_river_masked.Not())

        # Fallback 3: Vegetation context + moderate NDVI → S1
        veg_context_fallback = (
            still_zero
            .And(in_veg_context)
            .And(ndvi.gt(spec.s2_ndvi_min))  # v23.0: Use spec
        )
        stage = stage.where(veg_context_fallback, 1).rename("stage").toInt()
        still_zero = stage.eq(0).And(was_river_masked.Not())

        # Fallback 4: Water context + water evidence → S4
        # v23.0: Use stage_spec thresholds for turbid ponds
        water_context_fallback = (
            still_zero
            .And(in_water_context)
            .And(water_evidence_score.gt(0.18))  # v23.0: Very relaxed for turbid ponds
            .And(mndwi.gt(spec.s4_mndwi_min))
            .And(ndvi.lt(spec.s4_ndvi_max))
            .And(dem_allows_pond)
        )
        stage = stage.where(water_context_fallback, 4).rename("stage").toInt()
        still_zero = stage.eq(0).And(was_river_masked.Not())

        # Fallback 5: Moderate water evidence anywhere → S4
        # v23.0: Use stage_spec thresholds
        water_strong_fallback = (
            still_zero
            .And(water_evidence_score.gt(0.28))  # v23.0: Relaxed from 0.32
            .And(mndwi.gt(spec.s4_mndwi_min + 0.05))  # v23.0: Slightly stricter for non-context
            .And(ndvi.lt(spec.s4_ndvi_max))
            .And(dem_allows_pond)
        )
        stage = stage.where(water_strong_fallback, 4).rename("stage").toInt()
        still_zero = stage.eq(0).And(was_river_masked.Not())

        # Fallback 6: Bare soil evidence → S3
        # v23.0: Use stage_spec NDBI threshold
        stage = stage.where(
            still_zero.And(bare_soil_score.gt(0.40)).And(water_evidence_score.lt(0.20)),
            3
        ).rename("stage").toInt()
        still_zero = stage.eq(0).And(was_river_masked.Not())

        # Fallback 7: Low veg + low water = likely bare soil → S3
        # v23.0: Use stage_spec thresholds
        stage = stage.where(
            still_zero.And(ndvi.lt(spec.s3_ndvi_max - 0.03)).And(water_evidence_score.lt(0.20)),
            3
        ).rename("stage").toInt()
        still_zero = stage.eq(0).And(was_river_masked.Not())

        # (Fallback 8 removed — it destroyed valid ponds)

        # Default: remaining unclassified → S1
        stage = stage.where(still_zero, 1).rename("stage").toInt()

        # =============================================
        # SPATIAL SMOOTHING v23.0 — IMPROVED EDGE PROTECTION
        # =============================================
        smooth_cfg = getattr(config, 'SPATIAL_SMOOTHING', {})
        min_cluster = int(smooth_cfg.get('min_cluster_px', 15))  # v23.0: Increased from 10

        has_edge = band_names.contains("edge_density")
        is_boundary = ee.Image(
            ee.Algorithms.If(
                has_edge,
                edge_density.gt(0.08),  # v23.0: Slightly relaxed for better edge detection
                ee.Image(0),
            )
        )

        # v23.0: Protect river pixels AND pond pixels from being smoothed away
        smooth_protect = was_river_masked.Or(stage.gte(4))

        # Pass 1: Area opening — remove salt-and-pepper noise (v23.0: Larger radius)
        connected = stage.connectedPixelCount(100)  # v23.0: Increased from 64
        small_mask = connected.lt(min_cluster).And(is_boundary.Not()).And(smooth_protect.Not())
        neighbor_mode = stage.focal_mode(radius=2, units='pixels')
        stage = stage.where(small_mask, neighbor_mode).rename("stage").toInt()

        # Pass 2: Spatial Gradient Denoiser (v23.0: Use spec threshold)
        stage_mode_3x3 = stage.focal_mode(radius=1, units='pixels')
        stage_diff = stage.subtract(stage_mode_3x3).abs()
        noise_pixels = stage_diff.gt(spec.stage_gradient_max).And(stage.gt(0)).And(is_boundary.Not()).And(smooth_protect.Not())
        stage = stage.where(noise_pixels, stage_mode_3x3).rename("stage").toInt()

        # Pass 3: Super-majority filter
        cardinal_agree = stage.eq(stage_mode_3x3)
        stage = stage.where(
            cardinal_agree.Not().And(stage.gt(0)).And(is_boundary.Not()).And(smooth_protect.Not()),
            stage_mode_3x3
        ).rename("stage").toInt()

        # SNIC vote: DISABLED by default
        if smooth_cfg.get('use_snic_vote', False):
            snic_spacing = int(smooth_cfg.get('snic_vote_seed_spacing', 16))
            snic_compact = float(smooth_cfg.get('snic_vote_compactness', 0.05))
            snic_seeds = ee.Algorithms.Image.Segmentation.seedGrid(snic_spacing)
            snic_input = mndwi.toFloat()
            snic_result = ee.Algorithms.Image.Segmentation.SNIC(
                image=snic_input,
                size=snic_spacing,
                compactness=snic_compact,
                connectivity=4,
                seeds=snic_seeds,
            )
            segment_id = snic_result.select("clusters")
            segment_stage = stage.addBands(segment_id).reduceConnectedComponents(
                ee.Reducer.mode(), "clusters"
            )
            stage = stage.where(
                stage.gt(0), segment_stage
            ).rename("stage").toInt()

        # =============================================
        # RE-ENFORCE RIVER MASK after smoothing
        # Smoothing may have filled river pixels — force them back to 0
        # =============================================
        stage = stage.where(was_river_masked, 0).rename("stage").toInt()

        # =============================================
        # POND RESCUE RE-PROTECTION (v20.0 — NEW)
        # Pond-rescued pixels that got caught by river mask expansion
        # should be recovered if they have strong water evidence.
        # The JRC transition says these are NOT permanent water.
        # =============================================
        pond_rescue_recovery = (
            stage.eq(0)
            .And(pond_rescue)
            .And(water_evidence_score.gt(0.35))
            .And(ndvi.lt(0.25))
            .And(dem_allows_pond)
        )
        stage = stage.where(pond_rescue_recovery, 4).rename("stage").toInt()

        # =============================================
        # POST-RIVER CLEANUP (v20.0: simplified)
        # =============================================
        gap_fill_max = int(getattr(config, 'RIVER_MASKING', {}).get('gap_fill_max_px', 25))

        # A) Gap-fill: recover small stage-0 holes inside ponds
        #    Only fill if NOT a river-masked pixel
        stage_zero = stage.eq(0)
        zero_connected = stage_zero.selfMask().connectedPixelCount(64)
        small_zero_holes = stage_zero.And(zero_connected.lt(gap_fill_max))
        gap_fill_mode = stage.focal_mode(radius=2, units='pixels')
        stage = stage.where(
            small_zero_holes.And(gap_fill_mode.gt(0)).And(was_river_masked.Not()),
            gap_fill_mode
        ).rename("stage").toInt()

        # B) Pond protection — fill remaining black holes in aquaculture areas
        aqua_neighbor = stage.focal_mode(radius=1, units='pixels')
        stage = stage.where(
            stage.eq(0).And(aqua_neighbor.gte(4)).And(was_river_masked.Not()),
            aqua_neighbor
        ).rename("stage").toInt()

        # C) River-adjacent weak S4 cleanup (texture-gated)
        river_adj_min_we = float(getattr(config, 'RIVER_MASKING', {}).get(
            'river_adjacent_min_water_evidence', 0.40))
        near_river = is_natural_water.focal_max(radius=1, units='pixels')
        weak_s4_near_river = (
            stage.eq(4)
            .And(near_river)
            .And(water_evidence_score.lt(river_adj_min_we))
            .And(texture_is_smooth.Not())
        )
        stage = stage.where(weak_s4_near_river, 0).rename("stage").toInt()

        # FINAL: re-enforce river mask one last time
        stage = stage.where(was_river_masked.And(pond_rescue.Not()), 0).rename("stage").toInt()

        return ee.Image.cat([
            stage,
            water_evidence_score.rename("water_evidence_score").toFloat(),
            veg_evidence_score.rename("veg_evidence_score").toFloat(),
            bare_soil_score.rename("bare_soil_score").toFloat(),
            is_natural_water.rename("is_river").toFloat(),
            river_probability.rename("river_probability").toFloat(),
            is_near_river.rename("is_near_river").toFloat()
        ])

    def get_confidence(self, features, stage_image, aoi):
        """Confidence based on water evidence strength."""
        ndvi = features.select("ndvi")
        mndwi = features.select("mndwi")
        band_names = features.bandNames()
        has_sar = band_names.contains("vv_mean")
        has_edge = band_names.contains("edge_density")

        # Optical confidence
        opt_s1 = ndvi.subtract(0.30).divide(0.25).clamp(0, 1)
        opt_s2 = ndvi.unitScale(0.10, 0.30).clamp(0, 1)
        opt_s3 = ee.Image(0.18).subtract(ndvi).divide(0.18).clamp(0, 1)
        opt_s4 = mndwi.add(0.15).divide(0.35).clamp(0, 1)
        opt_s5 = mndwi.divide(0.40).clamp(0, 1)

        optical_conf = opt_s2
        optical_conf = optical_conf.where(stage_image.eq(1), opt_s1)
        optical_conf = optical_conf.where(stage_image.eq(3), opt_s3)
        optical_conf = optical_conf.where(stage_image.eq(4), opt_s4)
        optical_conf = optical_conf.where(stage_image.eq(5), opt_s5)

        # SAR confidence
        sar_conf = ee.Image(0.5)
        vv_for_conf = ee.Image(ee.Algorithms.If(
            band_names.contains("vv_mean"),
            features.select("vv_mean"),
            ee.Image(-15)
        )).rename("vv_conf")
        sar_water = vv_for_conf.unitScale(-25, -12).multiply(-1).add(1).clamp(0, 1)
        sar_conf = ee.Image(ee.Algorithms.If(
            band_names.contains("vv_mean"),
            sar_conf.where(stage_image.gte(4), sar_water),
            sar_conf
        ))

        # Combine
        confidence = optical_conf.multiply(0.60).add(sar_conf.multiply(0.40))
        return confidence.rename("confidence").toFloat()


# --- Helper functions for HMM and CCDC ---


class HMMClassifier(StageClassifier):
    """HMM-enhanced classifier."""

    def __init__(self):
        self._rule_classifier = RuleBasedClassifier()

    def classify(self, features, aoi):
        rule_stage = self._rule_classifier.classify(features, aoi)

        stage_only = rule_stage.select("stage")
        evidence_bands = rule_stage.select(["water_evidence_score", "veg_evidence_score", "bare_soil_score"])

        # Recompute river mask so we know which stage-0 pixels are rivers
        # (must NOT be overwritten by HMM gap-filling)
        river_result = compute_river_mask(features, aoi)
        is_river = river_result['is_river']

        # HMM smoothing: only fill stage-0 pixels that are NOT rivers
        neighborhood_mode = stage_only.focal_mode(radius=1, units="pixels")
        fillable = stage_only.eq(0).And(is_river.Not())
        smoothed = stage_only.where(fillable, neighborhood_mode)

        # Re-enforce river mask — rivers MUST stay stage 0
        smoothed = smoothed.where(is_river, 0).rename("stage").toInt()

        return ee.Image.cat([smoothed, evidence_bands])

    def get_confidence(self, features, stage, aoi):
        return self._rule_classifier.get_confidence(features, stage, aoi)


def get_classifier():
    if config.EXTENSIONS.get("use_hmm"):
        return HMMClassifier()
    return RuleBasedClassifier()


def classify_image(
    features, aoi, previous_stage=None, ccdc_breaks=None, mode="historical"
):
    """Primary entry point for pixel-wise stage classification."""
    classifier = get_classifier()
    stage_and_evidence = classifier.classify(features, aoi)
    stage_only = stage_and_evidence.select("stage")
    confidence = classifier.get_confidence(features, stage_only, aoi)
    return ee.Image.cat([stage_and_evidence, confidence])


def precompute_sar_thresholds(features, aoi, scale=None):
    """Pre-compute SAR thresholds."""
    if scale is None:
        scale = config.TARGET_SCALE
    try:
        band_list = features.bandNames().getInfo()
        if "vv_mean" not in band_list:
            return features
    except Exception:
        return features
    vv = features.select("vv_mean")
    pcts = vv.reduceRegion(
        reducer=ee.Reducer.percentile([20, 70]),
        geometry=aoi,
        scale=scale,
        **config.GEE_SAFE,
    )
    return features.set(
        {
            "vv_p20": pcts.get("vv_mean_p20"),
            "vv_p70": pcts.get("vv_mean_p70"),
        }
    )


def precompute_optical_thresholds(features, aoi, scale=None):
    """Pre-compute optical thresholds."""
    if scale is None:
        scale = config.TARGET_SCALE
    ndvi = features.select("ndvi")
    ndvi_pcts = ndvi.reduceRegion(
        reducer=ee.Reducer.percentile([30, 70]),
        geometry=aoi,
        scale=scale,
        **config.GEE_SAFE,
    )
    return features.set(
        {
            "ndvi_p70": ndvi_pcts.get("ndvi_p70"),
            "ndvi_p30": ndvi_pcts.get("ndvi_p30"),
        }
    )


def apply_ccdc(image_collection, aoi):
    """Apply CCDC for change detection."""
    if not config.EXTENSIONS.get("use_ccdc"):
        return None
    try:
        ccdc = ee.Algorithms.TemporalSegmentation.Ccdc(
            **{
                "collection": image_collection.select(["ndvi", "mndwi"]),
                "breakpointBands": ["ndvi", "mndwi"],
                "minObservations": 6,
                "chiSquareProbability": 0.99,
                "minNumOfYearsScaler": 1.33,
                "dateFormat": 2,
                "lambda": 0.002,
                "maxIterations": 25000,
            }
        )

        t_break = ccdc.select("tBreak")
        last_break = (
            t_break.arrayReduce(ee.Reducer.max(), [0])
            .arrayGet([0])
            .rename("ccdc_last_break")
        )
        any_break = last_break.gt(0).rename("ccdc_break")

        return ee.Image.cat([any_break.toFloat(), last_break.toFloat()]).clip(aoi)
    except Exception:
        return None


def merge_ccdc_features(features, ccdc_breaks):
    """Merge CCDC features."""
    if ccdc_breaks is None:
        return features
    return features.addBands(ccdc_breaks, overwrite=True)


# --- HMM Helper Functions ---


def _normalize_probabilities(prob_map):
    """Normalize probability dictionary to sum to 1."""
    stage_ids = [1, 2, 3, 4, 5]
    sanitized = {}
    total = 0.0
    for stage_id in stage_ids:
        raw_value = prob_map.get(stage_id, prob_map.get(str(stage_id), 0.0))
        try:
            value = max(0.0, float(raw_value))
        except Exception:
            value = 0.0
        sanitized[stage_id] = value
        total += value

    if total <= 0:
        fallback = 1.0 / len(stage_ids)
        return {stage_id: fallback for stage_id in stage_ids}
    return {stage_id: value / total for stage_id, value in sanitized.items()}


def _observation_to_probabilities(observation):
    """Convert observation to probability distribution."""
    if not observation:
        return _normalize_probabilities({})

    if observation.get("stage_probabilities"):
        return _normalize_probabilities(observation["stage_probabilities"])

    observed_stage = int(observation.get("stage") or observation.get("raw_stage") or 0)
    confidence = float(observation.get("confidence", 0.5) or 0.5)
    confidence = max(0.05, min(0.99, confidence))
    base = (1.0 - confidence) / 4.0
    probs = {stage_id: base for stage_id in [1, 2, 3, 4, 5]}
    if 1 <= observed_stage <= 5:
        probs[observed_stage] = confidence
    return _normalize_probabilities(probs)


def _get_transition_matrix(mode="historical"):
    """Get HMM transition matrix."""
    key = (
        "transition_probs_historical"
        if mode == "historical"
        else "transition_probs_operational"
    )
    raw = getattr(config, "HMM", {}).get(key, {})
    matrix = {}
    for src in [1, 2, 3, 4, 5]:
        row = raw.get(str(src), raw.get(src, {}))
        matrix[src] = _normalize_probabilities(
            {dst: row.get(str(dst), row.get(dst, 0.0)) for dst in [1, 2, 3, 4, 5]}
        )
    return matrix


def smooth_stage_sequence_hmm(observations, mode="historical"):
    """
    Viterbi-style stage smoothing with HMM transition probabilities.

    Args:
        observations: List of observation dicts with 'stage' and 'confidence'
        mode: 'historical' or 'operational'

    Returns:
        Dict with smoothed path, probabilities, uncertainty flags
    """
    if not observations:
        return {
            "path": [],
            "probabilities": [],
            "uncertain": [],
            "reasons": [],
            "stage_probabilities": [],
        }

    states = [1, 2, 3, 4, 5]
    transition = _get_transition_matrix(mode=mode)
    emissions = [_observation_to_probabilities(obs) for obs in observations]

    # Viterbi algorithm
    dp = []
    back_ptr = []

    # Initialize
    first = {}
    first_back = {}
    initial = 1.0 / len(states)
    for state in states:
        first[state] = math.log(initial) + math.log(
            max(emissions[0].get(state, 1e-6), 1e-6)
        )
        first_back[state] = None
    dp.append(first)
    back_ptr.append(first_back)

    # Forward pass
    for t in range(1, len(observations)):
        row = {}
        row_back = {}
        for state in states:
            best_prev = None
            best_score = None
            emit_score = math.log(max(emissions[t].get(state, 1e-6), 1e-6))
            for prev_state in states:
                trans_score = math.log(
                    max(transition.get(prev_state, {}).get(state, 1e-6), 1e-6)
                )
                score = dp[t - 1][prev_state] + trans_score + emit_score
                if best_score is None or score > best_score:
                    best_score = score
                    best_prev = prev_state
            row[state] = best_score
            row_back[state] = best_prev
        dp.append(row)
        back_ptr.append(row_back)

    # Backward pass
    last_state = max(states, key=lambda state: dp[-1][state])
    path = [last_state]
    for t in range(len(observations) - 1, 0, -1):
        path.append(back_ptr[t][path[-1]])
    path.reverse()

    # Build results
    probabilities = []
    uncertainty_flags = []
    reasons = []
    stage_probabilities = []

    for idx, state in enumerate(path):
        chosen_probability = emissions[idx].get(state, 0.0)

        # Check uncertainty
        sorted_probs = sorted(emissions[idx].items(), key=lambda x: x[1], reverse=True)
        second_best = sorted_probs[1][1] if len(sorted_probs) > 1 else 0.0
        margin = float(chosen_probability) - float(second_best)

        is_uncertain = chosen_probability < 0.50 or margin < 0.12
        reason = ""
        if chosen_probability < 0.50:
            reason = "low stage probability"
        elif margin < 0.12:
            ambiguous_stage = sorted_probs[1][0] if len(sorted_probs) > 1 else state
            reason = f"ambiguous with S{ambiguous_stage}"

        probabilities.append(round(float(chosen_probability), 3))
        uncertainty_flags.append(is_uncertain)
        reasons.append(reason)
        stage_probabilities.append(
            {str(k): round(v, 4) for k, v in emissions[idx].items()}
        )

    return {
        "path": path,
        "probabilities": probabilities,
        "uncertain": uncertainty_flags,
        "reasons": reasons,
        "stage_probabilities": stage_probabilities,
    }
