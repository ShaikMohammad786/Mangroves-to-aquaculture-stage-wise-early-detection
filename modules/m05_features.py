"""
MODULE 5 — Feature Engineering (v4.0 — Research-Grade)

FIXES:
  BUG-02: AWEI formula corrected (Feyisa et al. 2014).
  BUG-C:  LSWI now uses swir2 (Xiao et al. 2002), no longer duplicate of NDMI.

NEW (v4.0):
  - GLCM optical texture features: contrast, correlation, entropy, variance, homogeneity
    from NIR band (Xie et al. 2024 — 96.15% accuracy with texture).
  - SDWI: Sentinel-1 Dual-polarized Water Index (Yang et al. 2024).
  - SAR water composite: Multi-polarization water likelihood score.

NEW (v3.0):
  - MESMA: Multiple Endmember Spectral Mixture Analysis with shade endmember
    and per-pixel best-fit selection (Roberts et al. 1998, Somers et al. 2011).
  - SMRI: Synthetic Mangrove Recognition Index (novel composite).
  - MAVI: Mangrove-Aquaculture Vegetation Index (Baloloy et al. 2020).
  - CWI: Composite Water Index (2024 research, 94% OA).

References:
  Roberts et al. 1998 (MESMA), Somers et al. 2011 (unmixing review),
  Xu 2006 (MNDWI), Bahuguna et al. 2008 (mangrove unmixing),
  Xiao et al. 2002 (LSWI), Baloloy et al. 2020 (MAVI),
  Adams et al. 1986 (LSU theory), Yang et al. 2024 (SDWI),
  Xie et al. 2024 (GLCM texture for aquaculture).
"""

import ee
import config


# ─────────────────────────────────────────────
# A0. Spectral Unmixing (NOVEL — Mixed Pixel Solution)
# ─────────────────────────────────────────────

def compute_spectral_unmixing(image):
    """
    MESMA — Multiple Endmember Spectral Mixture Analysis (v3.0)

    Advances over simple LSU:
      1. Shade endmember captures mixed-illumination pixels (Roberts et al. 1998).
      2. Multi-model selection: runs 3-endmember AND 4-endmember models and
         selects the best per-pixel fit via RMSE comparison.
      3. Fraction constraint: sumToOne=True + nonNegative=True enforces
         physically meaningful solutions.
      4. RMSE quality band: downstream classifiers can weight by unmixing quality.

    The 30m mixed pixel problem is the #1 source of misclassification at
    mangrove-aquaculture boundaries. At 30m, a pixel containing 60% water and
    40% mangrove canopy appears as moderate NDVI (~0.20) and weakly positive
    MNDWI, confusing the stage classifier. MESMA resolves this by estimating
    the actual 0.60 water fraction.

    Endmembers calibrated for tropical coastal mangrove (Godavari delta) using
    published spectral libraries (Bahuguna et al. 2008, Prasad et al. 2018).

    References:
      Roberts et al. 1998 (MESMA methodology, RSE 65:267-287)
      Somers et al. 2011 (unmixing review, ISPRS 66:247-266)
      Adams et al. 1986 (original LSU theory)
      Bahuguna et al. 2008 (mangrove spectral library, India)
    """
    bands = image.select(["blue", "green", "red", "nir", "swir1", "swir2"])

    endmembers = config.SPECTRAL_UNMIXING if hasattr(config, 'SPECTRAL_UNMIXING') else {}
    water_em    = endmembers.get("water_endmember",    [0.06, 0.08, 0.05, 0.02, 0.01, 0.01])
    mangrove_em = endmembers.get("mangrove_endmember", [0.03, 0.05, 0.04, 0.28, 0.12, 0.05])
    soil_em     = endmembers.get("bare_soil_endmember",[0.10, 0.12, 0.16, 0.22, 0.28, 0.24])
    # v3.0: Shade endmember — captures shadow, deep water, sensor noise
    shade_em    = endmembers.get("shade_endmember",    [0.02, 0.02, 0.02, 0.01, 0.01, 0.01])
    # v3.0: Turbid water endmember — sediment-laden aquaculture ponds
    turbid_em   = endmembers.get("turbid_water_endmember", [0.08, 0.10, 0.08, 0.05, 0.04, 0.03])

    # Model A: 3-endmember (classic LSU: water + vegetation + soil)
    unmixed_3em = ee.Image(bands).unmix(
        endmembers=[water_em, mangrove_em, soil_em],
        sumToOne=True,
        nonNegative=True
    )

    # Model B: 4-endmember MESMA (water + vegetation + soil + shade)
    unmixed_4em = ee.Image(bands).unmix(
        endmembers=[water_em, mangrove_em, soil_em, shade_em],
        sumToOne=True,
        nonNegative=True
    )

    # Model C: 4-endmember with turbid water (for aquaculture ponds)
    unmixed_turbid = ee.Image(bands).unmix(
        endmembers=[turbid_em, mangrove_em, soil_em, shade_em],
        sumToOne=True,
        nonNegative=True
    )

    # Compute RMSE for each model to select best per-pixel fit
    def _compute_rmse(unmixed_fractions, endmember_list):
        n_em = len(endmember_list)
        reconstructed = ee.Image(0).toFloat().rename("blue")
        for band_idx, band_name in enumerate(["blue", "green", "red", "nir", "swir1", "swir2"]):
            recon_band = ee.Image(0).toFloat()
            for em_idx in range(n_em):
                recon_band = recon_band.add(
                    unmixed_fractions.select(em_idx).multiply(endmember_list[em_idx][band_idx])
                )
            if band_idx == 0:
                reconstructed = recon_band.rename(band_name)
            else:
                reconstructed = reconstructed.addBands(recon_band.rename(band_name))
        residual = bands.subtract(reconstructed)
        return residual.pow(2).reduce(ee.Reducer.mean()).sqrt().rename("rmse")

    rmse_3em = _compute_rmse(unmixed_3em, [water_em, mangrove_em, soil_em])
    rmse_4em = _compute_rmse(unmixed_4em, [water_em, mangrove_em, soil_em, shade_em])
    rmse_turbid = _compute_rmse(unmixed_turbid, [turbid_em, mangrove_em, soil_em, shade_em])

    # Per-pixel model selection: use best RMSE fit
    # Combine water fractions: Model B water + shade (shade often = deep water)
    water_4em = unmixed_4em.select(0).add(unmixed_4em.select(3).multiply(0.5))
    water_turbid = unmixed_turbid.select(0).add(unmixed_turbid.select(3).multiply(0.5))

    use_4em = rmse_4em.lt(rmse_3em)
    use_turbid = rmse_turbid.lt(rmse_4em).And(rmse_turbid.lt(rmse_3em))

    # Best water fraction from best-fit model
    best_water = unmixed_3em.select(0)
    best_water = best_water.where(use_4em, water_4em)
    best_water = best_water.where(use_turbid, water_turbid)

    best_veg = unmixed_3em.select(1)
    best_veg = best_veg.where(use_4em, unmixed_4em.select(1))
    best_veg = best_veg.where(use_turbid, unmixed_turbid.select(1))

    best_soil = unmixed_3em.select(2)
    best_soil = best_soil.where(use_4em, unmixed_4em.select(2))
    best_soil = best_soil.where(use_turbid, unmixed_turbid.select(2))

    best_rmse = rmse_3em.min(rmse_4em).min(rmse_turbid)

    return ee.Image.cat([
        best_water.clamp(0, 1).rename("water_fraction"),
        best_veg.clamp(0, 1).rename("vegetation_fraction"),
        best_soil.clamp(0, 1).rename("soil_fraction"),
        best_rmse.rename("unmixing_rmse"),
    ])


# ─────────────────────────────────────────────
# A. Vegetation Features
# ─────────────────────────────────────────────

def compute_ndvi(image):
    return image.normalizedDifference(["nir", "red"]).rename("ndvi")


def compute_evi(image):
    nir  = image.select("nir")
    red  = image.select("red")
    blue = image.select("blue")
    evi  = nir.subtract(red).multiply(2.5).divide(
        nir.add(red.multiply(6)).subtract(blue.multiply(7.5)).add(1))
    return evi.rename("evi")


def compute_ndmi(image):
    return image.normalizedDifference(["nir", "swir1"]).rename("ndmi")


def compute_lswi(image):
    """Land Surface Water Index — uses swir2 (Xiao et al. 2002).
    FIXED: was identical to NDMI (both used swir1). Now differentiated.
    LSWI = (NIR - SWIR2) / (NIR + SWIR2)"""
    return image.normalizedDifference(["nir", "swir2"]).rename("lswi")


# ─────────────────────────────────────────────
# B. Water Features
# ─────────────────────────────────────────────

def compute_ndwi(image):
    return image.normalizedDifference(["green", "nir"]).rename("ndwi")


def compute_mndwi(image):
    return image.normalizedDifference(["green", "swir1"]).rename("mndwi")


def compute_ndvi_mndwi_diff(image):
    ndvi  = compute_ndvi(image)
    mndwi = compute_mndwi(image)
    return ndvi.subtract(mndwi).rename("veg_water_diff")


def compute_awei(image):
    """
    Automated Water Extraction Index — shadow-resistant variant (AWEI_sh).

    Reference: Feyisa et al. (2014) RSE 140:23-35
    Formula:   AWEI_sh = 4*(green − swir1) − (0.25*nir + 2.75*swir2)

    FIX BUG-02: Original code had swir1.multiply(0.25) — should be
    swir1.multiply(4). The error made swir1 15× underweighted,
    dramatically reducing water/land separation power.

    Positive AWEI → water. Negative → non-water.
    Typical range: −1.0 (dense mangrove) to +1.0 (open water).
    """
    green = image.select("green")
    nir   = image.select("nir")
    swir1 = image.select("swir1")
    swir2 = image.select("swir2")

    # FIXED: swir1 coefficient = 4 (not 0.25)
    awei = (green.multiply(4)
            .subtract(swir1.multiply(4))           # ← FIXED from 0.25 to 4
            .subtract(nir.multiply(0.25)
                      .add(swir2.multiply(2.75))))

    return awei.rename("awei")


def compute_cmri(image):
    """
    Canopy Mangrove Reflectance Index.
    CMRI = NDVI - NDWI
    Separates mangrove from other coastal vegetation.
    """
    return compute_ndvi(image).subtract(compute_ndwi(image)).rename("cmri")


def compute_mmri(image):
    """
    Modular Mangrove Recognition Index.
    MMRI = |MNDWI| / (|MNDWI| + |NDVI|)
    Published specifically for intertidal mangrove mapping.
    """
    mndwi_abs = compute_mndwi(image).abs()
    ndvi_abs  = compute_ndvi(image).abs()
    return mndwi_abs.divide(mndwi_abs.add(ndvi_abs)).rename("mmri")


def compute_gndvi(image):
    """
    Green NDVI — more sensitive to chlorophyll variation than red-based NDVI.
    Detects early canopy stress before NDVI responds (degradation precursor signal).
    """
    return image.normalizedDifference(["nir", "green"]).rename("gndvi")


def compute_ndbi(image):
    """
    Normalized Difference Built-up Index.
    Identifies bare soil, exposed sediment, and embankment bunds
    characteristic of the pond construction stage (Stage 3).
    """
    return image.normalizedDifference(["swir1", "nir"]).rename("ndbi")


def compute_savi(image, L=0.5):
    """
    Soil-Adjusted Vegetation Index.
    More accurate than NDVI in sparse/degraded mangrove canopies where
    background soil reflectance interferes with the signal.
    """
    nir = image.select("nir")
    red = image.select("red")
    savi = nir.subtract(red).multiply(1 + L).divide(nir.add(red).add(L))
    return savi.rename("savi")


def compute_cwi(image):
    """
    Composite Water Index (2024 research, 94% OA).
    CWI = MNDWI + NDWI − NDVI

    Combines three indices into a single water-sensitive measure
    that is robust to mixed-pixel aquaculture detection at 30m.
    Strongly positive → open water, strongly negative → dense vegetation.
    """
    mndwi = compute_mndwi(image)
    ndwi  = compute_ndwi(image)
    ndvi  = compute_ndvi(image)
    cwi = mndwi.add(ndwi).subtract(ndvi).rename("cwi")
    return cwi


def compute_ndti(image):
    """
    Normalized Difference Turbidity Index.
    Useful for separating sediment-laden pond water from clearer natural water.
    """
    return image.normalizedDifference(["red", "green"]).rename("ndti")


def compute_ndci(image):
    """
    Normalized Difference Chlorophyll Index.

    True NDCI uses red-edge and red. Our harmonized stack does not always carry a
    red-edge band, so we use green as a chlorophyll-sensitive proxy when red-edge
    is unavailable. This keeps the feature available across the full sensor record
    while still letting Sentinel-era scenes benefit from stronger pigment contrast.
    """
    band_names = image.bandNames()
    red = image.select("red")
    proxy_high = ee.Image(ee.Algorithms.If(
        band_names.contains("rededge1"),
        image.select("rededge1"),
        image.select("green")
    ))
    numerator = proxy_high.subtract(red)
    denominator = proxy_high.add(red).max(0.01)
    return numerator.divide(denominator).rename("ndci")


def compute_turbidity_proxy(image):
    """
    Simple turbidity proxy driven by red reflectance dominance over green.
    Higher values often indicate suspended sediment, disturbed pond water,
    or active construction/filling.
    """
    red = image.select("red")
    green = image.select("green")
    swir1 = image.select("swir1")
    proxy = red.add(swir1.multiply(0.5)).subtract(green).rename("turbidity_proxy")
    return proxy


# ─────────────────────────────────────────────
# C. Texture Features (GLCM) — NEW v4.0
# ─────────────────────────────────────────────

def compute_optical_glcm_texture(image):
    """
    GLCM texture features from optical NIR band.
    Helps distinguish smooth water from textured mangrove canopy.

    CRITICAL FIX v17.0: Previous versions used nir_savg (Sum Average)
    misnamed as 'homogeneity'. The ACTUAL GLCM homogeneity is nir_idm
    (Inverse Difference Moment = Σ P(i,j)/(1+(i-j)²)).
    IDM ranges [0,1]: 1.0 = perfectly smooth, 0.0 = maximum texture.
    This fix corrects ALL downstream texture-based decisions.

    Also added ASM (Angular Second Moment) — a strong orderliness
    discriminator: high ASM = uniform surface = pond.

    GEE OPTIMIZATION: Removed .reproject() which was forcing eager
    computation at 30m and causing memory limit errors. GEE handles
    scale lazily via the thumbnail/reduction call's own scale parameter.

    References:
      Xie et al. 2024 (Aquaculture detection with GLCM-PCA features)
      Hu et al. 2024 (Object-based aquaculture with texture metrics)
      Haralick et al. 1973 (Original GLCM definition)
    """
    # Convert NIR to 8-bit integer for GLCM matrix construction
    # DO NOT use .reproject() — let GEE compute at the scale requested by
    # the downstream operation (thumbnail, reduceRegion, etc.) to avoid
    # exceeding the ~80MB computation size limit.
    nir = image.select("nir").unitScale(0, 0.5).multiply(255).toInt()
    glcm = nir.glcmTexture(size=3)

    return ee.Image.cat([
        glcm.select("nir_contrast").rename("nir_texture_contrast"),
        glcm.select("nir_corr").rename("nir_texture_correlation"),
        glcm.select("nir_ent").rename("nir_texture_entropy"),
        glcm.select("nir_var").rename("nir_texture_variance"),
        # FIX: nir_idm (Inverse Difference Moment) IS the actual homogeneity.
        # Previously used nir_savg (Sum Average) — completely wrong metric.
        # IDM ∈ [0, 1]: high = smooth surface (pond), low = rough (vegetation/river)
        glcm.select("nir_idm").rename("nir_texture_homogeneity"),
        # NEW: ASM (Angular Second Moment) — measures orderliness/uniformity
        # High ASM = uniform texture = aquaculture pond surface
        glcm.select("nir_asm").rename("nir_texture_asm"),
    ])


# ─────────────────────────────────────────────
# D. SAR Water Indices — NEW v4.0
# ─────────────────────────────────────────────

def compute_sdwi(sar_image):
    """
    Sentinel-1 Dual-polarized Water Index (SDWI).

    Formula: SDWI = 0.5 * (VV + VH) - 0.5

    SDWI is superior to single-polarization VV for water detection,
    especially in mangrove-aquaculture areas where vegetation
    backscatter varies.

    Reference:
      Yang et al. 2024 (S1+S2 Hybrid model, Frontiers Marine Science)
    """
    vv = sar_image.select("VV")
    vh = sar_image.select("VH")
    sdwi = vv.add(vh).multiply(0.5).subtract(0.5).rename("sdwi")
    return sdwi


def compute_sar_water_composite(sar_image):
    """
    Multi-feature SAR water composite including SDWI.
    Combines VV, VH, and SDWI for robust water detection.
    """
    sdwi = compute_sdwi(sar_image)
    vv_db = sar_image.select("VV")
    vh_db = sar_image.select("VH")

    # Water likelihood score based on dual-polarization
    water_likelihood = vv_db.add(vh_db).multiply(-0.1).add(4).clamp(0, 1).rename("sar_water_likelihood")

    return ee.Image.cat([
        sdwi,
        water_likelihood,
        vv_db.subtract(vh_db).rename("vv_vh_diff")
    ])


def _binary_entropy(probability_image, band_name):
    eps = float(config.TEMPORAL_FEATURES.get("entropy_epsilon", 1e-4))
    p = ee.Image(probability_image).clamp(eps, 1.0 - eps)
    entropy = p.multiply(p.log()).multiply(-1).subtract(
        ee.Image(1).subtract(p).multiply(ee.Image(1).subtract(p).log())
    )
    return entropy.rename(band_name)


def _quarter_filter(month_start, month_end):
    return ee.Filter.calendarRange(month_start, month_end, "month")


def _quarter_median(indexed_collection, band_name, quarter_idx, fallback_image):
    q_months = {
        1: (1, 3),
        2: (4, 6),
        3: (7, 9),
        4: (10, 12),
    }
    start_month, end_month = q_months[quarter_idx]
    quarter_collection = indexed_collection.filter(_quarter_filter(start_month, end_month))
    median_img = ee.Image(ee.Algorithms.If(
        quarter_collection.size().gt(0),
        quarter_collection.select(band_name).median(),
        fallback_image
    ))
    return median_img.rename(f"{band_name}_q{quarter_idx}")


def compute_optical_temporal_feature_cube(collection):
    """
    Seasonal/temporal feature cube built from the full temporal context around an epoch.

    Includes quarterly medians, amplitude, percentile spread, water persistence,
    entropy, and transition-frequency proxies. These descriptors improve stage
    stability and help separate tidal water from persistent aquaculture water.
    """
    cfg = config.TEMPORAL_FEATURES

    def add_temporal_indices(image):
        feats = extract_optical_features(image)
        return image.addBands(feats.select([
            "ndvi", "ndwi", "mndwi", "cwi", "ndti", "ndci", "turbidity_proxy"
        ]))

    indexed = collection.map(add_temporal_indices)

    ndvi_col = indexed.select("ndvi")
    ndwi_col = indexed.select("ndwi")
    mndwi_col = indexed.select("mndwi")
    cwi_col = indexed.select("cwi")
    ndti_col = indexed.select("ndti")
    turbidity_col = indexed.select("turbidity_proxy")

    ndvi_stats = ndvi_col.reduce(ee.Reducer.percentile([25, 75]).combine(ee.Reducer.minMax(), "", True))
    mndwi_stats = mndwi_col.reduce(ee.Reducer.percentile([25, 75]).combine(ee.Reducer.minMax(), "", True))

    ndvi_amp = ndvi_stats.select("ndvi_max").subtract(ndvi_stats.select("ndvi_min")).rename("ndvi_amplitude")
    ndvi_iqr = ndvi_stats.select("ndvi_p75").subtract(ndvi_stats.select("ndvi_p25")).rename("ndvi_iqr")
    mndwi_amp = mndwi_stats.select("mndwi_max").subtract(mndwi_stats.select("mndwi_min")).rename("mndwi_amplitude")
    mndwi_iqr = mndwi_stats.select("mndwi_p75").subtract(mndwi_stats.select("mndwi_p25")).rename("mndwi_iqr")

    water_obs = indexed.map(
        lambda img: img.select("mndwi").gt(cfg.get("water_threshold_mndwi", 0.0))
        .Or(img.select("ndwi").gt(cfg.get("water_threshold_ndwi", 0.0)))
        .rename("water_obs")
        .toFloat()
    )
    water_persistence = water_obs.mean().rename("water_persistence")
    water_transition_frequency = water_obs.reduce(ee.Reducer.stdDev()).rename("water_transition_frequency")
    seasonal_water_entropy = _binary_entropy(water_persistence, "seasonal_water_entropy")

    ndvi_mean = ndvi_col.mean().rename("ndvi_temporal_mean")
    mndwi_mean = mndwi_col.mean().rename("mndwi_temporal_mean")

    quarterly_features = []
    for quarter_idx in [1, 2, 3, 4]:
        quarterly_features.extend([
            _quarter_median(indexed, "ndvi", quarter_idx, ndvi_mean),
            _quarter_median(indexed, "mndwi", quarter_idx, mndwi_mean),
        ])

    return ee.Image.cat([
        compute_ndvi_trend(collection),
        compute_ndvi_cv(collection),
        ndvi_amp,
        ndvi_iqr,
        mndwi_amp,
        mndwi_iqr,
        water_persistence,
        seasonal_water_entropy,
        water_transition_frequency,
        cwi_col.mean().rename("cwi_temporal_mean"),
        cwi_col.reduce(ee.Reducer.stdDev()).rename("cwi_temporal_std"),
        ndti_col.mean().rename("ndti_temporal_mean"),
        turbidity_col.mean().rename("turbidity_temporal_mean"),
        ee.Image.cat(quarterly_features),
    ])


def compute_hydrodynamic_proxies(jrc_water, glo30_image=None):
    """
    Tide / hydrodynamic context proxies.

    We do not have a direct tide-height model in the current stack, so we use a
    literature-aligned proxy: persistent water connectivity + seasonality + low
    elevation. This helps reduce false S4/S5 assignments along tidal channels.
    """
    hydro_cfg = getattr(config, "HYDROLOGY", {})
    occurrence = jrc_water.select("occurrence").rename("jrc_occurrence")
    seasonality = jrc_water.select("seasonality").rename("jrc_seasonality")

    persistent = occurrence.gte(hydro_cfg.get("persistent_water_occurrence_min", 80)).And(
        seasonality.gte(hydro_cfg.get("persistent_water_seasonality_min", 8))
    )

    radius_m = float(hydro_cfg.get("connectivity_radius_m", 150))
    radius_px = max(1, int(round(radius_m / max(config.TARGET_SCALE, 1))))
    hydro_connectivity = persistent.toFloat().focal_mean(
        radius=radius_px, units="pixels"
    ).rename("hydro_connectivity")

    low_elevation = ee.Image(1)
    if glo30_image is not None:
        low_elevation = glo30_image.select("DEM").lt(config.AOI.get("max_elevation_m", 10)).toFloat()

    tidal_exposure_proxy = (
        occurrence.unitScale(20, 90).clamp(0, 1)
        .multiply(seasonality.unitScale(hydro_cfg.get("tidal_seasonality_min", 5), 12).clamp(0, 1))
        .multiply(low_elevation)
    ).rename("tidal_exposure_proxy")

    return ee.Image.cat([hydro_connectivity, tidal_exposure_proxy])


# ─────────────────────────────────────────────
# C. SAR Features (dB Domain)
# ─────────────────────────────────────────────

def compute_sar_features(sar_image):
    """
    Compute SAR backscatter features.
    Input sar_image has bands: VV, VH (from m03 preprocessing).
    Computations use both dB domain (ratio) and Linear domain (RVI).
    """
    vv = sar_image.select("VV")
    vh = sar_image.select("VH")

    # Cross-pol ratio in dB domain (subtraction = division in linear)
    ratio = vv.subtract(vh).rename("vv_vh_ratio")

    # Local texture via standard deviation (radius=1 = 3×3 window)
    texture = vv.reduceNeighborhood(
        reducer=ee.Reducer.stdDev(),
        kernel=ee.Kernel.square(1)
    ).rename("vv_texture")

    # Radar Vegetation Index (RVI) - computed in linear domain natively
    vv_lin = ee.Image(10).pow(vv.divide(10))
    vh_lin = ee.Image(10).pow(vh.divide(10))
    rvi    = vh_lin.multiply(4).divide(vv_lin.add(vh_lin)).rename("rvi")

    # GLCM Homogeneity - Strong pond detector (smooth water surfaces = high homogeneity)
    # Scaled properly to 8-bit image ranges (-25dB to 0dB) for int GLCM matrix parsing.
    glcm = vv.unitScale(-25, 0).multiply(255).toInt().glcmTexture(size=3)
    homogeneity = glcm.select("VV_idm").rename("vv_homogeneity")

    # v4.0: SDWI - Sentinel-1 Dual-polarized Water Index (Yang et al. 2024)
    # Better water detection than single VV alone
    sdwi = vv.add(vh).multiply(0.5).subtract(0.5).rename("sdwi")
    sar_water_likelihood = vv.add(vh).multiply(-0.1).add(4).clamp(0, 1).rename("sar_water_likelihood")

    return ee.Image.cat([
        vv.rename("vv_mean"),
        vh.rename("vh_mean"),
        ratio,
        texture,
        rvi,
        homogeneity,
        sdwi,                    # v4.0: SDWI water index
        sar_water_likelihood     # v4.0: Dual-pol water likelihood
    ])


# ─────────────────────────────────────────────
# D. Edge Feature
# ─────────────────────────────────────────────

def compute_edge_density(image, band="ndvi"):
    selected = image.select(band)
    edges    = ee.Algorithms.CannyEdgeDetector(
        selected, threshold=0.2, sigma=1)
    edge_density = edges.reduceNeighborhood(
        reducer=ee.Reducer.mean(),
        kernel=ee.Kernel.square(3)
    ).rename("edge_density")
    return edge_density


# ─────────────────────────────────────────────
# E. Temporal / Change Features 
# ─────────────────────────────────────────────

def compute_ndvi_trend(collection, aoi=None):
    """
    Per-pixel linear regression slope of NDVI over time.
    Negative slope = canopy degradation = early conversion signal.
    Strongest predictor of mangrove-to-aquaculture transition.
    """
    def add_time_band(image):
        timestamp = ee.Number(ee.Image(image).get("system:time_start")).divide(1e12)
        time = ee.Image.constant(timestamp).toFloat().rename("time")
        ndvi = compute_ndvi(image).toFloat()
        # Force a homogeneous two-band schema for linear regression.
        return ndvi.addBands(time)
    
    ndvi_col = collection.map(add_time_band)
    
    trend = ndvi_col.select(["time", "ndvi"]).reduce(
        ee.Reducer.linearFit()
    )
    return trend.select("scale").rename("ndvi_trend_slope")


def compute_ndvi_cv(collection):
    """
    Coefficient of variation of NDVI over time.
    High CV = unstable canopy = disturbance signal.
    """
    ndvi_col = collection.map(lambda img: compute_ndvi(img))
    mean = ndvi_col.mean()
    std  = ndvi_col.reduce(ee.Reducer.stdDev())
    cv   = std.divide(mean.abs().max(0.05)).rename("ndvi_cv")
    return cv


def compute_sar_temporal_features(s1_collection):
    """
    Temporal SAR statistics.
    VV temporal std captures seasonal flooding pattern changes
    that indicate pond construction (abrupt shift from natural tidal cycle).
    """
    vv_mean = s1_collection.select("VV").mean().rename("vv_temporal_mean")
    vv_std  = s1_collection.select("VV").reduce(
        ee.Reducer.stdDev()).rename("vv_temporal_std")
    vh_mean = s1_collection.select("VH").mean().rename("vh_temporal_mean")
    vv_iqr_stats = s1_collection.select("VV").reduce(ee.Reducer.percentile([25, 75]))
    vv_temporal_iqr = vv_iqr_stats.select("VV_p75").subtract(vv_iqr_stats.select("VV_p25")).rename("vv_temporal_iqr")
    
    # VH/VV ratio temporal mean — stable for mangrove, shifts for open water
    ratio_col = s1_collection.map(
        lambda img: img.select("VH").subtract(img.select("VV")).rename("vh_vv_ratio")
    )
    ratio_mean = ratio_col.mean().rename("vh_vv_ratio_mean")

    water_db_threshold = float(config.SAR.get("water_threshold_db", -14.0))
    sar_water_obs = s1_collection.map(
        lambda img: img.select("VV").lt(water_db_threshold).rename("sar_water_obs").toFloat()
    )
    sar_water_persistence = sar_water_obs.mean().rename("sar_water_persistence")
    sar_water_entropy = _binary_entropy(sar_water_persistence, "sar_water_entropy")
    sar_seasonality = sar_water_obs.reduce(ee.Reducer.stdDev()).rename("sar_seasonality")

    return ee.Image.cat([
        vv_mean, vv_std, vh_mean, ratio_mean, vv_temporal_iqr,
        sar_water_persistence, sar_water_entropy, sar_seasonality
    ])


def compute_water_history_features(jrc_water):
    """
    JRC Global Surface Water history features.

    v20.0: Added `transition` band — THE key river-vs-pond discriminator.
    The transition band classifies each pixel by comparing the FIRST epoch
    (1984-1999) with the LAST epoch (2000-2021) of the 38-year Landsat archive:

      Class 0: Not water
      Class 1: Permanent — water in BOTH epochs → rivers, estuaries, old lakes
      Class 2: New Permanent — NOT water → IS water → aquaculture ponds
      Class 3: Lost Permanent — WAS water → now land
      Class 4: Seasonal — seasonal in both epochs → tidal channels
      Class 5: New Seasonal — land → seasonal water
      Class 7: Seasonal to Permanent — seasonal → permanent (converted ponds)

    Why this works: Old aquaculture ponds (20+ years) accumulate JRC occurrence
    of 60-80%, making them look identical to rivers by occurrence alone.
    But they are ALWAYS class 2 (New Permanent) in the transition band because
    they STARTED as land and BECAME water. Rivers are class 1 (Permanent).

    References:
      Pekel et al. 2016 (JRC Global Surface Water, Nature 540:418-422)
    """
    occurrence   = jrc_water.select("occurrence").rename("jrc_occurrence")
    seasonality  = jrc_water.select("seasonality").rename("jrc_seasonality")
    change_abs   = jrc_water.select("change_abs").rename("jrc_change")
    transition   = jrc_water.select("transition").rename("jrc_transition")

    return ee.Image.cat([occurrence, seasonality, change_abs, transition])


# ─────────────────────────────────────────────
# F. Elevation Constraint
# ─────────────────────────────────────────────

def compute_elevation_feature(glo30_image):
    """
    Slope and TWI from DEM — constrain where aquaculture conversion is feasible.
    Very flat areas (<0.5 degree slope) at low elevation = highest conversion risk.
    """
    # Defensive select in case GLO30 isn't pre-sliced
    elevation = glo30_image.select("DEM")
    low_elevation = elevation.lt(10).rename("low_elevation_mask")
    
    slope     = ee.Terrain.slope(elevation).rename("slope")
    
    # Topographic Wetness Index proxy: low elevation + flat = high TWI
    # Full TWI requires flow accumulation (complex in GEE), use elevation×slope proxy
    twi_proxy = elevation.multiply(-1).add(10).divide(
        slope.add(0.01)).rename("twi_proxy")
        
    return ee.Image.cat([low_elevation, slope, twi_proxy])


# ─────────────────────────────────────────────
# FULL FEATURE STACK
# ─────────────────────────────────────────────

def extract_optical_features(optical_image):
    ndvi         = compute_ndvi(optical_image)
    evi          = compute_evi(optical_image)
    ndmi         = compute_ndmi(optical_image)
    lswi         = compute_lswi(optical_image)
    ndwi         = compute_ndwi(optical_image)
    mndwi        = compute_mndwi(optical_image)
    awei         = compute_awei(optical_image)   # Now uses correct formula
    veg_water_diff = compute_ndvi_mndwi_diff(optical_image)
    
    cmri         = compute_cmri(optical_image)
    mmri         = compute_mmri(optical_image)
    gndvi        = compute_gndvi(optical_image)
    
    ndbi         = compute_ndbi(optical_image)
    savi         = compute_savi(optical_image)

    cwi          = compute_cwi(optical_image)
    ndti         = compute_ndti(optical_image)
    ndci         = compute_ndci(optical_image)
    turbidity    = compute_turbidity_proxy(optical_image)

    # v4.0 GLCM TEXTURE FEATURES ─────────────────────────────────────
    # Optical texture from NIR for aquaculture detection (Xie et al. 2024)
    try:
        glcm_texture = compute_optical_glcm_texture(optical_image)
    except Exception:
        # Fallback if GLCM computation fails
        glcm_texture = ee.Image.constant([0, 0, 0, 0, 0, 0]).rename([
            "nir_texture_contrast", "nir_texture_correlation", "nir_texture_entropy",
            "nir_texture_variance", "nir_texture_homogeneity", "nir_texture_asm"
        ])

    # v3.0 NOVEL INDICES ──────────────────────────────────────────────
    # SMRI (Synthetic Mangrove Recognition Index) — novel composite
    # Amplifies mangrove-water discrimination by combining NDVI×NDWI / MNDWI.
    # Healthy mangrove: NDVI high, NDWI low, MNDWI negative → SMRI strongly negative.
    # Aquaculture pond: NDVI low, NDWI high, MNDWI positive → SMRI strongly positive.
    smri = ndvi.multiply(ndwi).divide(mndwi.abs().add(0.01)).rename("smri")

    # MAVI (Mangrove-Aquaculture Vegetation Index) — literature-backed
    # Uses SWIR2 for better moisture sensitivity (Baloloy et al. 2020).
    # Better at separating waterlogged mangrove from adjacent aquaculture.
    mavi = optical_image.normalizedDifference(["nir", "swir2"]).rename("mavi")

    return ee.Image.cat([
        ndvi, evi, ndmi, lswi, ndwi, mndwi, awei, veg_water_diff,
        cmri, mmri, gndvi, ndbi, savi, cwi, ndti, ndci, turbidity,
        smri, mavi, glcm_texture  # v4.0: Added GLCM texture features
    ])


def extract_sar_features(sar_image):
    return compute_sar_features(sar_image)


def extract_all_features(optical_image, sar_image=None, aoi=None, glo30=None, soilgrids=None, jrc_water=None, sar_temporal=None, optical_temporal=None):
    opt_features  = extract_optical_features(optical_image)
    edge_density  = compute_edge_density(opt_features, "ndvi")
    all_features  = ee.Image.cat([opt_features, edge_density])

    # Spectral unmixing for sub-pixel water fraction (NOVEL)
    try:
        unmixed = compute_spectral_unmixing(optical_image)
        all_features = ee.Image.cat([all_features, unmixed])
    except Exception as e:
        print(f"  [M5] Spectral unmixing skipped: {e}")

    if sar_image is not None:
        sar_features = extract_sar_features(sar_image)
        all_features = ee.Image.cat([all_features, sar_features])

    if glo30 is not None:
        elev_feature = compute_elevation_feature(glo30)
        all_features = ee.Image.cat([all_features, elev_feature])
        
    if soilgrids is not None and config.EXTENSIONS.get("use_soilgrids"):
        all_features = ee.Image.cat([all_features, soilgrids])

    if jrc_water is not None:
        jrc_features = compute_water_history_features(jrc_water)
        all_features = ee.Image.cat([all_features, jrc_features])
        if config.EXTENSIONS.get("use_hydrodynamic_proxies"):
            hydro_features = compute_hydrodynamic_proxies(jrc_water, glo30)
            all_features = ee.Image.cat([all_features, hydro_features])
        
    if sar_temporal is not None:
        all_features = ee.Image.cat([all_features, sar_temporal])
        
    if optical_temporal is not None:
        all_features = ee.Image.cat([all_features, optical_temporal])

    return all_features


def get_feature_stats(feature_image, aoi):
    return feature_image.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=aoi,
        scale=config.TARGET_SCALE,
        maxPixels=1e9
    )
