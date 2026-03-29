"""
RAW IMAGE DOWNLOADER — Diagnostic Script

Downloads raw (unprocessed) satellite thumbnails directly from GEE 
into a 'raw_images/' folder. Bypasses all preprocessing to confirm 
the raw GEE data contains actual satellite imagery.

Usage:
    python download_raw_images.py
"""

import ee
import os
import urllib.request
import socket
import logging

ee.Initialize(project="dip-temporal-satellite-image")

import config

# ── Logging Setup ──
RAW_DIR = os.path.join(config.BASE_DIR, "raw_images")
os.makedirs(RAW_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [RAW-DL] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(RAW_DIR, "download_log.txt"), mode="w"),
    ]
)
log = logging.getLogger("raw_download")

RAW_DIR = os.path.join(config.BASE_DIR, "raw_images")
os.makedirs(RAW_DIR, exist_ok=True)

THUMB_SIZE = 1024


def _download_thumbnail(image, aoi_rect, vis_params, filepath, label):
    """Download a single GEE thumbnail with logging."""
    try:
        image = ee.Image(image).clip(aoi_rect)

        # Log band names
        band_names = image.bandNames().getInfo()
        log.info(f"  [{label}] Bands: {band_names}")

        # Log pixel stats (mean values over AOI)
        stats = image.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=aoi_rect,
            scale=60,
            maxPixels=1e9,
            bestEffort=True,
        ).getInfo()
        log.info(f"  [{label}] Mean pixel values: {stats}")

        # Log min/max
        min_stats = image.reduceRegion(
            reducer=ee.Reducer.minMax(),
            geometry=aoi_rect,
            scale=60,
            maxPixels=1e9,
            bestEffort=True,
        ).getInfo()
        log.info(f"  [{label}] Min/Max pixel values: {min_stats}")

        vis_image = image.visualize(**vis_params)

        region_coords = [
            [config.AOI["lon_min"], config.AOI["lat_min"]],
            [config.AOI["lon_max"], config.AOI["lat_min"]],
            [config.AOI["lon_max"], config.AOI["lat_max"]],
            [config.AOI["lon_min"], config.AOI["lat_max"]],
        ]

        params = {"region": region_coords, "dimensions": str(THUMB_SIZE), "format": "png"}
        url = vis_image.getThumbURL(params)
        log.info(f"  [{label}] Thumbnail URL: {url[:120]}...")

        req = urllib.request.urlopen(url, timeout=180)
        data = req.read()
        with open(filepath, "wb") as f:
            f.write(data)

        file_size = os.path.getsize(filepath)
        log.info(f"  [{label}] SAVED: {os.path.basename(filepath)} ({file_size:,} bytes)")
        return True

    except Exception as e:
        log.error(f"  [{label}] FAILED: {e}")
        return False


def download_raw_landsat5(aoi):
    """Download raw Landsat 5 image (1988 epoch)."""
    log.info("=" * 60)
    log.info("Downloading RAW Landsat 5 (1988 epoch)")
    log.info("=" * 60)

    col = (ee.ImageCollection(config.GEE_DATASETS["landsat5_sr"])
           .filterBounds(aoi)
           .filterDate("1987-01-01", "1989-12-31")
           .filter(ee.Filter.calendarRange(11, 3, "month"))
           .sort("CLOUD_COVER")
           .limit(10))

    size = col.size().getInfo()
    log.info(f"  Collection size: {size} images")

    if size == 0:
        log.warning("  No Landsat 5 images found!")
        return

    # Raw median composite (NO cloud masking, NO scaling, NO normalization)
    raw_median = col.median()

    bm = config.BAND_MAP["landsat5"]

    # 1. Raw DN (surface reflectance scaled values)
    vis_raw = {
        "bands": [bm["red"], bm["green"], bm["blue"]],
        "min": 7000, "max": 12000,  # L5 C2 L2 SR typical range
    }
    _download_thumbnail(raw_median, aoi, vis_raw,
                        os.path.join(RAW_DIR, "raw_L5_1988_DN.png"),
                        "L5-RAW-DN")

    # 2. Scaled reflectance (apply scale factor only, no other processing)
    bm = config.BAND_MAP["landsat5"]
    raw_optical = raw_median.select(
        [bm["blue"], bm["green"], bm["red"], bm["nir"], bm["swir1"], bm["swir2"]],
        ["blue", "green", "red", "nir", "swir1", "swir2"]
    )
    scaled = raw_optical.multiply(0.0000275).add(-0.2).clamp(0, 1)

    vis_scaled = {
        "bands": ["red", "green", "blue"],
        "min": 0.0, "max": 0.3, "gamma": 1.4,
    }
    _download_thumbnail(scaled, aoi, vis_scaled,
                        os.path.join(RAW_DIR, "raw_L5_1988_scaled.png"),
                        "L5-SCALED")


def download_raw_hls(aoi, year=2022):
    """Download raw HLS image — tries S30, falls back to L30, then Landsat 8 C2."""
    log.info("=" * 60)
    log.info(f"Downloading RAW HLS ({year} epoch)")
    log.info("=" * 60)

    # Try HLS S30 first (widest possible window, no seasonal filter for diagnostic)
    attempts = [
        ("hls_s30", f"{year-2}-01-01", f"{year+2}-12-31", "HLS S30"),
        ("hls_l30", f"{year-2}-01-01", f"{year+2}-12-31", "HLS L30"),
    ]

    for dataset_key, start, end, label in attempts:
        col = (ee.ImageCollection(config.GEE_DATASETS[dataset_key])
               .filterBounds(aoi)
               .filterDate(start, end))

        size = col.size().getInfo()
        log.info(f"  {label} collection ({start} to {end}): {size} images")

        if size == 0:
            log.warning(f"  {label}: No images found, trying next...")
            continue

        # Use only first 50 images to avoid timeouts
        col = col.limit(50)

        raw_median = col.median()
        bm = config.BAND_MAP[dataset_key]

        # 1. Raw DN
        vis_raw = {
            "bands": [bm["red"], bm["green"], bm["blue"]],
            "min": 0, "max": 3000,  # HLS typical DN range (wider for safety)
        }
        _download_thumbnail(raw_median, aoi, vis_raw,
                            os.path.join(RAW_DIR, f"raw_{label.replace(' ', '_')}_{year}_DN.png"),
                            f"{label}-RAW-DN-{year}")

        # 2. Scaled
        raw_optical = raw_median.select(
            [bm["blue"], bm["green"], bm["red"], bm["nir"], bm["swir1"], bm["swir2"]],
            ["blue", "green", "red", "nir", "swir1", "swir2"]
        )
        scaled = raw_optical.divide(10000).clamp(0, 1)

        vis_scaled = {
            "bands": ["red", "green", "blue"],
            "min": 0.0, "max": 0.3, "gamma": 1.4,
        }
        _download_thumbnail(scaled, aoi, vis_scaled,
                            os.path.join(RAW_DIR, f"raw_{label.replace(' ', '_')}_{year}_scaled.png"),
                            f"{label}-SCALED-{year}")
        return  # Success — don't try fallback

    log.error(f"  ALL HLS sources empty for {year}! No raw HLS image downloaded.")


def download_raw_sar(aoi, year=2022):
    """Download raw SAR image."""
    log.info("=" * 60)
    log.info(f"Downloading RAW Sentinel-1 ({year})")
    log.info("=" * 60)

    col = (ee.ImageCollection(config.GEE_DATASETS["sentinel1_grd"])
           .filterBounds(aoi)
           .filterDate(f"{year}-01-01", f"{year}-12-31")
           .filter(ee.Filter.eq("instrumentMode", "IW"))
           .filter(ee.Filter.eq("orbitProperties_pass", "DESCENDING"))
           .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
           .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
           .limit(20))

    size = col.size().getInfo()
    log.info(f"  Collection size: {size} images")

    if size == 0:
        log.warning("  No S1 images found!")
        return

    raw_median = col.select(["VV", "VH"]).median()

    vis_sar = {
        "bands": ["VV"],
        "min": -25, "max": -5,
        "palette": ["000000", "333333", "666666", "999999", "cccccc", "ffffff"],
    }
    _download_thumbnail(raw_median, aoi, vis_sar,
                        os.path.join(RAW_DIR, f"raw_SAR_{year}_VV.png"),
                        f"SAR-VV-{year}")


def main():
    log.info("=" * 60)
    log.info("RAW IMAGE DOWNLOADER — Diagnostic")
    log.info(f"AOI: {config.AOI['name']}")
    log.info(f"Bounds: {config.AOI['lon_min']}–{config.AOI['lon_max']}°E, "
             f"{config.AOI['lat_min']}–{config.AOI['lat_max']}°N")
    log.info(f"Output: {RAW_DIR}")
    log.info("=" * 60)

    aoi = ee.Geometry.Rectangle([
        config.AOI["lon_min"], config.AOI["lat_min"],
        config.AOI["lon_max"], config.AOI["lat_max"],
    ])

    download_raw_landsat5(aoi)
    download_raw_hls(aoi, year=2017)
    download_raw_hls(aoi, year=2022)
    download_raw_sar(aoi, year=2022)

    log.info("")
    log.info("=" * 60)
    log.info("DONE — Check raw_images/ folder for results")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
