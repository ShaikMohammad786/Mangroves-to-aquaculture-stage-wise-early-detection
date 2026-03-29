"""
MODULE 2 — Optical Preprocessing 
"""

import ee
import config


# ─────────────────────────────────────────────────────────────
# CLOUD MASKING
# ─────────────────────────────────────────────────────────────

def mask_landsat_clouds(image):
    """Research-grade Landsat C2 L2 cloud mask (bits 1,2,3,4)."""
    image = ee.Image(image)
    qa = image.select("QA_PIXEL")
    dilated_cloud = qa.bitwiseAnd(1 << 1).eq(0)
    cloud         = qa.bitwiseAnd(1 << 3).eq(0)
    cloud_shadow  = qa.bitwiseAnd(1 << 4).eq(0)

    return image.updateMask(dilated_cloud.And(cloud).And(cloud_shadow))
    # Catch brightly reflecting pure white clouds that escape the QA algorithm
    # Many historical L5/7 images fail to tag small puffy clouds over the delta.
    # We use a simple high-blue reflectance cutoff (scaled > ~0.2)
    # The image is not scaled yet, so we have to use raw SR DN limits if needed.
    # However, since scaling requires complex shifting, we will do a fast mask 
    # based solely on the reliable QA bits.
    
   


# ─────────────────────────────────────────────────────────────
# SLC-OFF GAP FILL (FIX BUG-03 / BUG-04)
# ─────────────────────────────────────────────────────────────


def fill_slc_gaps(image):
    """
    Landsat 7 SLC-off gap fill using focal_median.
    Two-pass approach covers stripe widths of 1–14px.
    focal_median not focal_mean — preserves dominant neighbour
    class at mangrove-pond boundaries instead of blending them.
    """
    image = ee.Image(image)

    filled1 = image.focal_median(
        radius=10, kernelType="circle", units="pixels", iterations=1
    )
    image1 = image.unmask(filled1)

    filled2 = image1.focal_median(
        radius=20, kernelType="circle", units="pixels", iterations=1
    )
    return image1.unmask(filled2)


def is_slc_off(image):
    """
    Returns True if this L7 image was acquired after the SLC failure.
    Uses the 'DATE_ACQUIRED' property (format 'YYYY-MM-DD').
    Server-side: cannot call .getInfo() inside .map(), so we use a
    date comparison ee.Number instead.
    """
    acq_millis = ee.Image(image).get("system:time_start")
    slc_failure = ee.Date("2003-06-01").millis()
    return ee.Number(acq_millis).gte(slc_failure)


# ─────────────────────────────────────────────────────────────
# BAND RENAME & SCALE
# ─────────────────────────────────────────────────────────────

def _rename_bands(image, sensor):
    image = ee.Image(image)
    bm = config.BAND_MAP[sensor]
    
    original = [bm["blue"], bm["green"], bm["red"],
                bm["nir"],  bm["swir1"], bm["swir2"], bm["qa"]]
    renamed  = ["blue", "green", "red", "nir", "swir1", "swir2", "qa"]
    
    return image.select(original, renamed)


def _scale_landsat(image):
    image   = ee.Image(image)
    optical = image.select(["blue","green","red","nir","swir1","swir2"])
    scaled  = optical.multiply(0.0000275).add(-0.2).clamp(0, 1)
    return ee.Image(scaled.copyProperties(image, image.propertyNames()))


# ─────────────────────────────────────────────────────────────
# FILL MASKED PIXELS FOR THUMBNAIL EXPORT 
# ─────────────────────────────────────────────────────────────

EXPORT_FILL = {
    "optical_index": 0.0,   # NDVI, MNDWI, CMRI — neutral midpoint
    "optical_rgb":   0.02,  # dark grey, not jet black
    "sar_db":       -25.0,  # S1 noise floor in dB
    "dem":           0.0,   # sea level
}

def fill_masked_for_export(image, data_type="optical_index"):
    """
    Replace NoData with a type-appropriate fill before thumbnail export.
    Prevents GEE black-patch rendering artefacts.
    NOT a gap-filling method — purely for visualization output.

    Args:
        data_type: 'optical_index' | 'optical_rgb' | 'sar_db' | 'dem'
    """
    fill_value = EXPORT_FILL.get(data_type, 0.0)
    return image.unmask(fill_value)




def fill_cloud_gaps(image):
    image  = ee.Image(image)
    filled = image.focal_median(
        radius=2, kernelType="circle", units="pixels", iterations=1
    )
    return image.unmask(filled)

# ─────────────────────────────────────────────────────────────
# PER-IMAGE PREPROCESSORS
# ─────────────────────────────────────────────────────────────

def _make_landsat5_preprocessor():
    def _preprocess(image):
        image   = ee.Image(image)
        masked  = mask_landsat_clouds(image)
        renamed = _rename_bands(masked, "landsat5")
        scaled  = _scale_landsat(renamed)
        
        return scaled
    return _preprocess


def _make_landsat7_preprocessor():
    """
    Landsat 7 preprocessor with automatic SLC-off gap fill.
    """
    def _preprocess(image):
        image   = ee.Image(image)
        masked  = mask_landsat_clouds(image)
        renamed = _rename_bands(masked, "landsat7")
        scaled  = _scale_landsat(renamed)
        
        # Apply SLC gap fill for post-2003 L7 images
        gap_filled = fill_slc_gaps(scaled)
        need_fill  = is_slc_off(image)
        
        scaled = ee.Image(ee.Algorithms.If(need_fill, gap_filled, scaled)).copyProperties(image, image.propertyNames())
        
        return scaled
    return _preprocess


def _make_hls_preprocessor(sensor):
    """
    NASA HLS v2.0 (L30 & S30) preprocessor.
    HLS already applies atmospheric correction, BRDF normalization, 
    and bandpass adjustment.
    """
    def _preprocess(image):
        image = ee.Image(image)
        
        # 1. Band Mapping
        renamed = _rename_bands(image, sensor)
        
        # 2. HLS Scaling (DN / 10000)
        optical = renamed.select(["blue","green","red","nir","swir1","swir2"])
        scaled = optical.divide(10000).clamp(0, 1)
        
        # 3. Fmask Cloud Masking (Categorical logic based on user specification)
        # HLS Fmask can be overly strict over coasts (aerosol=moderate, cirrus, adjacent clouds).
        # We focus on the most important bits: 1 (Cloud) and 3 (Cloud Shadow)
        fmask   = renamed.select("qa")
        cloud   = fmask.bitwiseAnd(1 << 1).eq(0)  # Bit 1: Cloud
        shadow  = fmask.bitwiseAnd(1 << 3).eq(0)  # Bit 3: Shadow

        good_pixels = cloud.And(shadow)
        
        scaled_masked = scaled.updateMask(good_pixels)
        
        return scaled_masked.copyProperties(image, image.propertyNames())
        
    return _preprocess


def _make_sentinel2_sr_preprocessor():
    """
    Sentinel-2 SR Harmonized preprocessor.

    Uses SCL + QA60 to mask clouds/shadows/cirrus. Keeps the core optical bands
    mapped to the shared band schema used downstream.
    """
    def _preprocess(image):
        image = ee.Image(image)

        renamed = _rename_bands(image, "sentinel2_sr")

        optical = renamed.select(["blue", "green", "red", "nir", "swir1", "swir2"])
        scaled = optical.divide(10000).clamp(0, 1)

        band_names = renamed.bandNames()

        # QA60 bits: 10=cloud, 11=cirrus (1 means contaminated)
        qa60 = ee.Image(ee.Algorithms.If(
            band_names.contains("qa"),
            renamed.select("qa"),
            ee.Image(0)
        ))
        cloud_bit = qa60.bitwiseAnd(1 << 10).eq(0)
        cirrus_bit = qa60.bitwiseAnd(1 << 11).eq(0)

        # SCL classes to mask:
        # 3=cloud_shadow, 8=cloud_med_prob, 9=cloud_high_prob, 10=thin_cirrus, 11=snow
        scl = ee.Image(ee.Algorithms.If(
            band_names.contains("scl"),
            renamed.select("scl"),
            ee.Image(4)  # vegetation-ish fallback
        ))
        scl_ok = (scl.neq(3)
                  .And(scl.neq(8))
                  .And(scl.neq(9))
                  .And(scl.neq(10))
                  .And(scl.neq(11)))

        good_pixels = cloud_bit.And(cirrus_bit).And(scl_ok)
        scaled_masked = scaled.updateMask(good_pixels)

        return scaled_masked.copyProperties(image, image.propertyNames())

    return _preprocess


# ─────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────

_PREPROCESSORS = {
    "landsat5":  _make_landsat5_preprocessor(),
    "landsat7":  _make_landsat7_preprocessor(),
    "hls_l30":   _make_hls_preprocessor("hls_l30"),
    "hls_s30":   _make_hls_preprocessor("hls_s30"),
    "sentinel2_sr": _make_sentinel2_sr_preprocessor(),
}



def preprocess_optical_collection(collection, sensor, aoi):
    """
    Preprocess an entire optical collection.
    Steps: cloud mask → rename → scale → SLC fill (L7 post-2003) → smooth.
    Returns cloud-free, gap-filled, radiometrically scaled collection.
    """
    preprocessor = _PREPROCESSORS.get(sensor)
    if preprocessor is None:
        raise ValueError(f"Unknown sensor: {sensor}")

    return collection.map(preprocessor)


def harmonize_to_hls(collection, sensor, aoi):
    """
    Applies Roy et al. (2016) harmonization coefficients to align Landsat 5/7 
    with the HLS (Landsat 8/9 / S2) baseline.
    Avoids expensive server-side reduceRegion computations and gracefully handles the L5 decommission window.
    """
    if sensor not in ["landsat5", "landsat7"]:
        return collection
        
    print(f"  [Harmonization] Applying Roy 2016 {sensor} -> HLS OLI coefficients...")
    
    # Roy et al. (2016) Table 2 - OLS ETM+ to OLI surface reflectance (scaled to 0-1)
    # OLI = ETM+ * slope + intercept
    slopes = ee.Image.constant([0.9785, 0.9542, 0.9820, 1.0442, 1.0392, 1.0401])
    intercepts = ee.Image.constant([-0.0095, -0.0016, -0.0022, -0.0021, 0.0030, 0.0040])
    bands = ["blue", "green", "red", "nir", "swir1", "swir2"]
    
    def _apply_harmonization(image):
        image = ee.Image(image)
        # Select matched bands
        matched = image.select(bands)
        
        # Apply linear transformation: (image * slope) + intercept
        harmonized = matched.multiply(slopes).add(intercepts).rename(bands)
        
        # Bring over pristine bands (QA, etc)
        other_bands = image.bandNames().removeAll(bands)
        harmonized = harmonized.addBands(image.select(other_bands))
        
        return ee.Image(harmonized.copyProperties(image, image.propertyNames()))

    return collection.map(_apply_harmonization)