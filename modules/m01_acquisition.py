"""
MODULE 1 — Data Acquisition 

"""

import ee
import config


# ─────────────────────────────────────────────────────────────
# Dry-Season Filter Helper (single source of truth)
# ─────────────────────────────────────────────────────────────
def get_dry_season_filter():
    """
    Returns an ee.Filter for dry-season months, sourced from config.DRY_SEASON_MONTHS.

    Using a single source of truth here prevents inconsistencies between modules
    (historical stream, acquisition, and operational paths).
    """
    months = getattr(config, "DRY_SEASON_MONTHS", None) or [11, 12, 1, 2, 3]
    # Deduplicate while keeping stable order
    seen = set()
    ordered = []
    for m in months:
        try:
            mi = int(m)
        except Exception:
            continue
        if 1 <= mi <= 12 and mi not in seen:
            seen.add(mi)
            ordered.append(mi)

    # Fallback if misconfigured
    if not ordered:
        ordered = [11, 12, 1, 2, 3]

    filt = None
    for mi in ordered:
        mf = ee.Filter.calendarRange(mi, mi, "month")
        filt = mf if filt is None else ee.Filter.Or(filt, mf)
    return filt

# ─────────────────────────────────────────────────────────────
# Landsat Collections
# ─────────────────────────────────────────────────────────────

def _get_landsat(dataset_id, aoi, start, end, name):
    col = (ee.ImageCollection(dataset_id)
           .filterBounds(aoi)
           .filterDate(start, end)
           .filter(get_dry_season_filter())
           .sort("system:time_start"))
    print(f"[M1] {name} collection loaded")
    return col


def get_landsat5(aoi, start=None, end=None):
    return _get_landsat(
        config.GEE_DATASETS["landsat5_sr"], aoi,
        start or config.DATE_RANGE["start"],
        end   or "2012-05-01",
        "Landsat 5")


def get_landsat7(aoi, start=None, end=None):
    """
    Note: L7 SLC failed May 2003. Use L7 only for pre-2003 epochs.
    Post-2003 L7 imagery is handled in m02 with gap fill, but
    quality is still degraded. Prefer L8/L9/S2 for 2013+.
    """
    return _get_landsat(
        config.GEE_DATASETS["landsat7_sr"], aoi,
        start or config.DATE_RANGE["start"],
        end   or config.DATE_RANGE["end"],
        "Landsat 7")


# ─────────────────────────────────────────────────────────────
# NASA HLS v2.0 (Harmonized Landsat and Sentinel-2)
# ─────────────────────────────────────────────────────────────

def get_hls_l30(aoi, start=None, end=None):
    col = (ee.ImageCollection(config.GEE_DATASETS["hls_l30"])
           .filterBounds(aoi)
           .filterDate(start or "2013-04-11", end or config.DATE_RANGE["end"])
           .filter(get_dry_season_filter())
           .sort("system:time_start"))
    print("[M1] HLS L30 (Landsat) collection loaded")
    return col


def get_hls_s30(aoi, start=None, end=None):
    col = (ee.ImageCollection(config.GEE_DATASETS["hls_s30"])
           .filterBounds(aoi)
           .filterDate(start or "2015-11-28", end or config.DATE_RANGE["end"])
           .filter(get_dry_season_filter())
           .sort("system:time_start"))
    print("[M1] HLS S30 (Sentinel-2) collection loaded")
    return col


def get_hls_operational(aoi, days_back=60):
    """Operational mode: no seasonal filter, recent HLS S30 imagery."""
    from datetime import datetime
    today = ee.Date(datetime.utcnow().isoformat())
    start = today.advance(-days_back, "day")
    col   = (ee.ImageCollection(config.GEE_DATASETS["hls_s30"])
             .filterBounds(aoi)
             .filterDate(start, today)
             .filter(ee.Filter.lt("ext_cloud_coverage", 30)) # HLS S30 cloud cover property approx.
             .sort("system:time_start"))
    return col


# ─────────────────────────────────────────────────────────────
# Sentinel-2 SR (10m/20m) — High resolution optical
# ─────────────────────────────────────────────────────────────

def get_sentinel2_sr(aoi, start=None, end=None):
    """
    Sentinel-2 Surface Reflectance (harmonized) collection.
    Used to reduce mixed-pixel errors and improve pond geometry at 10m.
    """
    col = (ee.ImageCollection(config.GEE_DATASETS["sentinel2_sr"])
           .filterBounds(aoi)
           .filterDate(start or "2017-01-01", end or config.DATE_RANGE["end"])
           .filter(get_dry_season_filter())
           .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", config.CLOUD.get("sentinel2_max_cloud_pct", 20)))
           .sort("system:time_start"))
    print("[M1] Sentinel-2 SR Harmonized collection loaded")
    return col

                
# ─────────────────────────────────────────────────────────────
# Sentinel-1
# ─────────────────────────────────────────────────────────────

def get_sentinel1(aoi, start=None, end=None):
    col = (ee.ImageCollection(config.GEE_DATASETS["sentinel1_grd"])
           .filterBounds(aoi)
           .filterDate(start or "2014-10-01", end or config.DATE_RANGE["end"])
           .filter(ee.Filter.eq("instrumentMode",   config.SAR["instrument_mode"]))
           .filter(ee.Filter.eq("orbitProperties_pass", "DESCENDING"))
           .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
           .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
           .filter(ee.Filter.eq("resolution_meters", 10))
           .sort("system:time_start"))
    print("[M1] Sentinel-1 collection loaded")
    return col


# ─────────────────────────────────────────────────────────────
# Ancillary Layers
# ─────────────────────────────────────────────────────────────

def get_jrc_water(aoi):
    water = ee.Image(config.GEE_DATASETS["jrc_water"]).clip(aoi)
    print("[M1] JRC Global Surface Water loaded")
    return water


def get_jrc_monthly(aoi, start=None, end=None):
    col = (ee.ImageCollection(config.GEE_DATASETS["jrc_monthly"])
           .filterBounds(aoi)
           .filterDate(start or config.DATE_RANGE["start"],
                       end   or config.DATE_RANGE["end"])
           .sort("system:time_start"))
    print("[M1] JRC Monthly collection loaded")
    return col



def get_glo30(aoi):
    dem = (ee.ImageCollection(config.GEE_DATASETS["glo30"])
           .filterBounds(aoi)
           .select("DEM")
           .mosaic()
           .clip(aoi))
    print("[M1] Copernicus GLO30 DEM loaded")
    return dem


# def get_tide_data(aoi):
#     try:
#         tide = (ee.ImageCollection(config.GEE_DATASETS["tide_data"])
#                 .filterBounds(aoi)
#                 .sort("system:time_start"))
#         print("[M1] Tide dataset loaded")
#         return tide
#     except Exception as e:
#         print(f"[M1] Tide dataset not available: {e}")
#         return None


def get_mangrove_baseline(aoi):
    try:
        mangrove = (ee.ImageCollection(config.GEE_DATASETS["mangrove_watch"])
                    .filterBounds(aoi)
                    .sort("system:time_start"))
        print("[M1] Mangrove Watch collection loaded")
        return mangrove
    except Exception as e:
        print(f"[M1] Mangrove Watch not available: {e}")
        return None

def get_soilgrids(aoi):
    if not config.EXTENSIONS.get("use_soilgrids"):
        return None
    try:
        sand = ee.Image("projects/soilgrids-isric/sand_mean").select("sand_0-5cm_mean").clip(aoi)
        soc = ee.Image("projects/soilgrids-isric/soc_mean").select("soc_0-5cm_mean").clip(aoi)
        print("[M1] SoilGrids loaded")
        return ee.Image.cat([sand, soc]).rename(["sand_mean", "soc_mean"])
    except Exception as e:
        print(f"[M1] SoilGrids error: {e}")
        return None



# ─────────────────────────────────────────────────────────────
# Acquire All
# ─────────────────────────────────────────────────────────────

def acquire_all(aoi):
    print("\n" + "=" * 60)
    print("[M1] DATA ACQUISITION")
    print("=" * 60)

    return {
        "landsat5":         get_landsat5(aoi),
        "landsat7":         get_landsat7(aoi),
        "hls_l30":          get_hls_l30(aoi),
        "hls_s30":          get_hls_s30(aoi),
        "sentinel2_sr":     get_sentinel2_sr(aoi),
        "sentinel1":        get_sentinel1(aoi),
        "jrc_water":        get_jrc_water(aoi),
        "jrc_monthly":      get_jrc_monthly(aoi),
        "glo30":            get_glo30(aoi),
        # "tide_data":        get_tide_data(aoi),
        "mangrove_baseline": get_mangrove_baseline(aoi),
        "soilgrids":        get_soilgrids(aoi),
    }