"""
MODULE 3 — SAR Preprocessing (Research-Grade v2.0)

Sentinel-1:
  ✔ Linear → dB conversion
  ✔ Border noise masking
  ✔ Adaptive Lee-style speckle filtering
  ✔ Smaller window (5x5 default)
  ✔ Fully server-side safe
"""

import ee
import config


# ─────────────────────────────────────────────────────────────
# Speckle Filtering (Enhanced Lee Sigma)
# ─────────────────────────────────────────────────────────────

def enhanced_lee_sigma_filter(image):
    """
    Enhanced Lee Sigma speckle filter.
    Operates in the linear power domain (before dB conversion)
    to better preserve pond edges and reduce noise.
    """
    image = ee.Image(image)
    orig_bands = image.bandNames()
    
    # Equivalent Number of Looks for S1 GRD IW is ~4.4
    enl = 4.4
    radius = config.SAR.get("speckle_window", 5) // 2
    kernel = ee.Kernel.square(radius, "pixels")


    # Local statistics
    # ee.Reducer.mean() appends _mean to band names. We rename back immediately
    # so math operations match exactly with the original image bands.
    mean = image.reduceNeighborhood(ee.Reducer.mean(), kernel).rename(orig_bands)
    variance = image.reduceNeighborhood(ee.Reducer.variance(), kernel).rename(orig_bands)

    # Speckle variance estimation (V_speckle = mean^2 / ENL)
    speckle_var = mean.pow(2).divide(enl)

    # Weight (W) = 1 - (V_speckle / V_local)
    weight = ee.Image().expression(
        "1.0 - (speckle_var / local_var)",
        {
            "speckle_var": speckle_var,
            "local_var": variance.max(1e-10) # avoid division by zero
        }
    ).clamp(0, 1)

    filtered = mean.add(weight.multiply(image.subtract(mean)))

    return filtered.rename(orig_bands)


# ─────────────────────────────────────────────────────────────
# Sentinel-1 Image Preprocessing
# ─────────────────────────────────────────────────────────────

def db_to_linear(image):
    """
    Converts decibels (dB) to linear scale.
    Formula: linear = 10 ^ (dB / 10)
    """
    image = ee.Image(image)
    return ee.Image(10.0).pow(image.divide(10.0))

def to_gamma0(image):
    """
    Converts GRD linear Sigma0 to Gamma0 to remove incidence angle bias.
    Crucial for temporal stability over flat delta regions.
    """
    import math
    image = ee.Image(image)
    theta = image.select("angle").multiply(math.pi / 180.0)
    cos_theta = theta.cos()
    gamma0 = image.select(config.SAR["polarization"]).divide(cos_theta)
    return gamma0

def preprocess_s1_image(image):
    """
    Sentinel-1 preprocessing:
      1. Mask border noise using angle band and low intensity threshold (in dB).
      2. Convert native dB → linear Sigma0.
      3. Convert linear Sigma0 → linear Gamma0.
      4. Apply enhanced Lee Sigma filter in the linear domain.
      5. Convert linear Gamma0 → dB.
    """
    image = ee.Image(image)

    # Select polarizations (linear)
    sar = image.select(config.SAR["polarization"])

    # 1. Mask border noise
    # The 'angle' band helps mask out the image edges which are noisy.
    # Sentinel-1 IW swath usually spans 30-45 degrees incidence angle.
    angle = image.select('angle')
    invalid_angle = angle.lt(29.0).Or(angle.gt(46.0))
    # Note: sar is natively in dB. -40 dB is ~ 1e-4 linear.
    low_vv = sar.select("VV").lt(-40.0)
    low_vh = sar.select("VH").lt(-40.0)
    low_values = low_vv.Or(low_vh)
    
    # Mask out invalid angles and extremely low values (border tracking)
    sar = sar.updateMask(invalid_angle.Not().And(low_values.Not()))
    native_mask = image.mask().reduce(ee.Reducer.min())
    sar = sar.updateMask(native_mask)
    # 2. Convert dB to linear Sigma0 BEFORE any arithmetic filtering
    sar_linear = db_to_linear(sar)

    # Preserve angle for Gamma0 conversion by adding it back temporarily
    sar_with_angle = sar_linear.addBands(angle)

    # 3. Convert to Gamma0
    sar_gamma0 = to_gamma0(sar_with_angle)

    # 4. Apply speckle filter in the linear domain
    sar_filtered = ee.Image(enhanced_lee_sigma_filter(sar_gamma0))

    # 5. Convert back to dB
    sar_db = sar_filtered.log10().multiply(10.0)
    
    # Force rename back to core polarizations just in case
    sar_db = sar_db.rename(config.SAR["polarization"])

    return ee.Image(sar_db.copyProperties(image, image.propertyNames()))


# ─────────────────────────────────────────────────────────────
# Collection Processing
# ─────────────────────────────────────────────────────────────

def preprocess_s1_collection(collection, aoi=None):
    """
    Preprocess entire Sentinel-1 collection.
    """
    preprocessed = collection.map(preprocess_s1_image)

    # Tidal normalization hook: Filter out high-tide images
    if config.EXTENSIONS.get("use_tidal_normalization") and aoi is not None:
        print("  [SAR] Applying tidal normalization (water fraction filtering)...")
        
        # Load the adaptive water threshold from config (defaulting to -14.0dB for mangrove deltas)
        water_thresh = config.SAR.get("water_threshold_db", -14.0)
        
        def calculate_water_fraction(img):
            # Threshold VV band to detect exposed water
            water = img.select("VV").lt(water_thresh)
            
            # Using Reducer.mean() on a boolean mask mathematically returns 
            # the fraction of valid pixels that are flooded (water_count / total_valid_pixels).
            # This fundamentally corrects for variable border-noise masking per image.
            stats = water.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=aoi,
                scale=120,   # Coarse scale to prevent GEE limits
                maxPixels=1e9,
                bestEffort=True,
                tileScale=8
            )
            
            # Extract fraction explicitly handling empty/masked images with a 0 default
            fraction = ee.Number(stats.get("VV", 0))
            return img.set("water_fraction", fraction)
            
        preprocessed_with_frac = preprocessed.map(calculate_water_fraction)
        
        # Calculate the 50th percentile of water fraction across the time series
        stats = preprocessed_with_frac.reduceColumns(
            reducer=ee.Reducer.percentile([50]),
            selectors=["water_fraction"]
        )
        p50 = ee.Number(stats.get("p50"))
        
        # Retain only images where water fraction is less than or equal to median (low tide)
        preprocessed = preprocessed_with_frac.filter(ee.Filter.lte("water_fraction", p50))

    return preprocessed


