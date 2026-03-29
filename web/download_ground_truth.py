"""
Ground Truth High-Resolution Image Downloader

Downloads high-resolution satellite basemap tiles for the AOI from
publicly available tile services for visual validation of aquaculture
pond detection results.

Sources:
  - ESRI World Imagery (~0.5m resolution in most areas)
  - Google Satellite tiles (for comparison)

Usage:
    python download_ground_truth.py
"""

import os
import math
import urllib.request
import ssl
import time
import logging

# Try importing image processing libraries
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


# -- Configuration --
import config

GROUND_TRUTH_DIR = os.path.join(config.OUTPUT_DIR, "ground_truth")
os.makedirs(GROUND_TRUTH_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [GT-DL] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(GROUND_TRUTH_DIR, "download_log.txt"), mode="w"),
    ]
)
log = logging.getLogger("ground_truth")

# ESRI World Imagery tile server (public, no API key needed)
ESRI_TILE_URL = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/"
    "World_Imagery/MapServer/tile/{z}/{y}/{x}"
)

# Google Satellite tiles (public)
GOOGLE_TILE_URL = (
    "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}"
)

# Tile size in pixels
TILE_SIZE = 256

# Zoom levels to download
ZOOM_LEVELS = [14, 16, 17]  # 14=~10m, 16=~2m, 17=~1m


def lat_lon_to_tile(lat, lon, zoom):
    """Convert lat/lon to tile x, y coordinates."""
    lat_rad = math.radians(lat)
    n = 2.0 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * n)
    return x, y


def download_tile(url, filepath, retries=3):
    """Download a single tile with retry logic."""
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            response = urllib.request.urlopen(req, timeout=30, context=ctx)
            data = response.read()

            with open(filepath, "wb") as f:
                f.write(data)
            return True

        except Exception as e:
            if attempt < retries - 1:
                time.sleep(1)
            else:
                log.warning(f"Failed to download {url}: {e}")
                return False

    return False


def stitch_tiles(tile_dir, output_path, x_range, y_range):
    """Stitch downloaded tiles into a single image."""
    if not HAS_PIL:
        log.warning("Pillow not installed, cannot stitch tiles")
        return False

    x_min, x_max = x_range
    y_min, y_max = y_range
    width = (x_max - x_min + 1) * TILE_SIZE
    height = (y_max - y_min + 1) * TILE_SIZE

    result = Image.new("RGB", (width, height))

    for x in range(x_min, x_max + 1):
        for y in range(y_min, y_max + 1):
            tile_path = os.path.join(tile_dir, f"tile_{x}_{y}.png")
            if os.path.exists(tile_path):
                try:
                    tile = Image.open(tile_path)
                    px = (x - x_min) * TILE_SIZE
                    py = (y - y_min) * TILE_SIZE
                    result.paste(tile, (px, py))
                except Exception as e:
                    log.warning(f"Error loading tile {tile_path}: {e}")

    result.save(output_path, "PNG")
    log.info(f"Stitched image saved: {output_path} ({width}x{height}px)")
    return True


def download_basemap(source_name, tile_url_template, zoom):
    """Download all tiles for the AOI at a given zoom level from a basemap source."""
    aoi = config.AOI
    lat_min, lat_max = aoi["lat_min"], aoi["lat_max"]
    lon_min, lon_max = aoi["lon_min"], aoi["lon_max"]

    # Get tile range
    x_min, y_max = lat_lon_to_tile(lat_min, lon_min, zoom)
    x_max, y_min = lat_lon_to_tile(lat_max, lon_max, zoom)

    total_tiles = (x_max - x_min + 1) * (y_max - y_min + 1)
    log.info(f"[{source_name}] Zoom {zoom}: {total_tiles} tiles "
             f"({x_min}-{x_max} x {y_min}-{y_max})")

    # Create tile directory
    tile_dir = os.path.join(GROUND_TRUTH_DIR, f"{source_name}_z{zoom}_tiles")
    os.makedirs(tile_dir, exist_ok=True)

    downloaded = 0
    for x in range(x_min, x_max + 1):
        for y in range(y_min, y_max + 1):
            tile_path = os.path.join(tile_dir, f"tile_{x}_{y}.png")

            if os.path.exists(tile_path) and os.path.getsize(tile_path) > 100:
                downloaded += 1
                continue

            url = tile_url_template.format(x=x, y=y, z=zoom)
            if download_tile(url, tile_path):
                downloaded += 1

            # Rate limiting
            time.sleep(0.1)

    log.info(f"[{source_name}] Downloaded {downloaded}/{total_tiles} tiles")

    # Stitch tiles
    output_path = os.path.join(
        GROUND_TRUTH_DIR,
        f"{source_name}_z{zoom}_{aoi['name'].replace(' ', '_')}.png"
    )
    stitch_tiles(tile_dir, output_path, (x_min, x_max), (y_min, y_max))

    return output_path


def main():
    aoi = config.AOI
    log.info("=" * 60)
    log.info("GROUND TRUTH HIGH-RES IMAGE DOWNLOADER")
    log.info(f"AOI: {aoi['name']}")
    log.info(f"Bounds: {aoi['lon_min']}-{aoi['lon_max']}E, "
             f"{aoi['lat_min']}-{aoi['lat_max']}N")
    log.info(f"Output: {GROUND_TRUTH_DIR}")
    log.info("=" * 60)

    # Download ESRI World Imagery at multiple zoom levels
    for zoom in ZOOM_LEVELS:
        log.info(f"\n--- ESRI World Imagery (Zoom {zoom}) ---")
        try:
            path = download_basemap("esri", ESRI_TILE_URL, zoom)
            log.info(f"Saved: {path}")
        except Exception as e:
            log.error(f"ESRI zoom {zoom} failed: {e}")

    # Download Google Satellite at one zoom level
    try:
        log.info("\n--- Google Satellite (Zoom 16) ---")
        path = download_basemap("google", GOOGLE_TILE_URL, 16)
        log.info(f"Saved: {path}")
    except Exception as e:
        log.error(f"Google satellite failed: {e}")

    log.info("\n" + "=" * 60)
    log.info("DONE — Check outputs/ground_truth/ folder")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
