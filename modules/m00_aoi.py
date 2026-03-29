"""
MODULE 0 — AOI Initialization
Define the study area polygon and mask out irrelevant terrain.
"""

import ee
import json
import os
import config


def create_aoi_rectangle():
    """Create bounding box from config coordinates."""
    return ee.Geometry.Rectangle([
        config.AOI["lon_min"], config.AOI["lat_min"],
        config.AOI["lon_max"], config.AOI["lat_max"]
    ])


def buffer_aoi(geometry):
    """Buffer the AOI by configured distance."""
    buffer_m = config.AOI["buffer_km"] * 1000
    return geometry.buffer(buffer_m)


def get_elevation_mask(aoi):
    """Create mask where elevation <= max_elevation_m (coastal lowlands only)."""
    dem = ee.Image(config.GEE_DATASETS["glo30"])
    elevation = dem.select("DEM")
    mask = elevation.lte(config.AOI["max_elevation_m"])
    return mask.clip(aoi)


def initialize_aoi():
    """
    Full AOI initialization pipeline.
    Returns: dict with 'geometry' (ee.Geometry), 'elevation_mask' (ee.Image), 'bbox' (list)
    """
    # Step 1: Create rectangle
    rectangle = create_aoi_rectangle()

    # Step 2: Buffer
    buffered = buffer_aoi(rectangle)

    # Step 3: Elevation mask
    elev_mask = get_elevation_mask(buffered)

    aoi_info = {
        "geometry": buffered,
        "elevation_mask": elev_mask,
        "bbox": [
            config.AOI["lon_min"], config.AOI["lat_min"],
            config.AOI["lon_max"], config.AOI["lat_max"]
        ],
        "crs": config.AOI["crs"],
        "resolution_m": config.AOI["resolution_m"],
    }

    print(f"[M0] AOI initialized: {config.AOI['name']}")
    print(f"     Bounds: {aoi_info['bbox']}")
    print(f"     CRS: {aoi_info['crs']}, Resolution: {aoi_info['resolution_m']}m")
    print(f"     Max elevation mask: {config.AOI['max_elevation_m']}m")

    return aoi_info


def export_aoi_geojson(aoi_info):
    """Export AOI boundary as GeoJSON file."""
    geojson = {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "properties": {
                "name": config.AOI["name"],
                "crs": config.AOI["crs"],
                "max_elevation_m": config.AOI["max_elevation_m"],
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [config.AOI["lon_min"], config.AOI["lat_min"]],
                    [config.AOI["lon_max"], config.AOI["lat_min"]],
                    [config.AOI["lon_max"], config.AOI["lat_max"]],
                    [config.AOI["lon_min"], config.AOI["lat_max"]],
                    [config.AOI["lon_min"], config.AOI["lat_min"]],
                ]]
            }
        }]
    }

    path = os.path.join(config.POLYGON_DIR, "aoi.geojson")
    with open(path, "w") as f:
        json.dump(geojson, f, indent=2)
    print(f"[M0] AOI exported to {path}")
    return path
