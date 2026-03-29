"""
MODULE 10 — Web Export (Fixed v3.4)

FIXES THIS ROUND:
  RGB dark images: min=0.02, max=0.45 caused all images to appear very dark.
    Landsat surface reflectance over coastal mangroves: 0.02–0.18 typical max.
    Fixed: max=0.18, gamma=1.8.
  Stage/outline HTTP 400: deep computation graph rejected by getThumbURL.
    Fixed: force_scale=30 calls .reproject() before thumbnail to pre-evaluate.
  Thumbnail size: reduced 768→512 to reduce memory load on GEE side.
"""

import ee
import json
import os
import shutil
import time
import urllib.request
import socket
import logging
import math
import config

export_log = logging.getLogger("EXPORT")

THUMB_SIZE = max(512, int(config.WEB.get("thumbnail_width", 1024)))  # v23.0: Reduced from 1536
_THUMB_CRS = (
    "EPSG:3857"  # Web Mercator so scale = meters (Not EPSG:4326 which = degrees)
)
_THUMB_SCALE = 30  # 30m native Landsat/HLS resolution

# v15.0: Research-standard HIGH-CONTRAST stage colors
# S1: Forest Green, S2: Golden Yellow, S3: Fire Red, S4: Ocean Blue, S5: Deep Indigo
_STAGE_PALETTE = ["1b7837", "fee08b", "d73027", "4575b4", "313695"]
STAGE_LABELS = {
    1: "S1: Dense Mangrove",
    2: "S2: Degradation",
    3: "S3: Clearing",
    4: "S4: Water Filling",
    5: "S5: Operational Pond",
}


STAGE_COLORS_RGB = {
    1: (27, 120, 55),     # forest green - S1
    2: (254, 224, 139),   # golden yellow - S2
    3: (215, 48, 39),     # fire red - S3
    4: (69, 117, 180),    # ocean blue - S4
    5: (49, 54, 149),     # deep indigo - S5
}
STAGE_COLORS_HEX = {
    stage_id: "#{:02x}{:02x}{:02x}".format(*rgb)
    for stage_id, rgb in STAGE_COLORS_RGB.items()
}


def stage_label(stage_id):
    try:
        stage_key = int(stage_id)
    except Exception:
        return "Unclassified"
    if stage_key <= 0:
        return "Unclassified"
    return STAGE_LABELS.get(stage_key, f"S{stage_key}")


def stage_color_hex(stage_id):
    try:
        stage_key = int(stage_id)
    except Exception:
        return "#666666"
    return STAGE_COLORS_HEX.get(stage_key, "#666666")


def _annotate_stage_legend(filepath):
    try:
        from PIL import Image, ImageDraw, ImageFont

        img = Image.open(filepath).convert("RGBA")
        draw = ImageDraw.Draw(img)
        width, _ = img.size

        try:
            font = ImageFont.truetype("arial.ttf", max(14, int(width * 0.016)))
        except Exception:
            font = ImageFont.load_default()

        x = 14
        y = 14
        pad = 6
        box_w = max(18, int(width * 0.02))
        box_h = box_w

        legend_h = (box_h + 8) * 5 + 20
        legend_w = max(220, int(width * 0.20))
        draw.rectangle([x, y, x + legend_w, y + legend_h], fill=(0, 0, 0, 175))
        draw.text((x + pad, y + pad), "Stage Legend", fill=(255, 255, 255), font=font)
        y += 28

        for stage_id in range(1, 6):
            color = STAGE_COLORS_RGB[stage_id]
            label = STAGE_LABELS.get(stage_id, f"S{stage_id}")
            draw.rectangle(
                [x + pad, y, x + pad + box_w, y + box_h],
                fill=color,
                outline=(255, 255, 255, 180),
            )
            draw.text((x + pad + box_w + 10, y), label, fill=(255, 255, 255), font=font)
            y += box_h + 8

        img.save(filepath, "PNG")
    except Exception as e:
        print(f"[M10] Warning drawing stage legend: {e}")


def _lonlat_to_pixel(lon, lat, width, height):
    lon_min = config.AOI["lon_min"]
    lon_max = config.AOI["lon_max"]
    lat_min = config.AOI["lat_min"]
    lat_max = config.AOI["lat_max"]
    x = (lon - lon_min) / max(lon_max - lon_min, 1e-9) * width
    y = height - ((lat - lat_min) / max(lat_max - lat_min, 1e-9) * height)
    return x, y


def _iter_geometry_rings(geom_dict):
    if not isinstance(geom_dict, dict):
        return
    gtype = geom_dict.get("type")
    coords = geom_dict.get("coordinates", [])
    if gtype == "Polygon":
        for ring in coords:
            yield ring
    elif gtype == "MultiPolygon":
        for polygon in coords:
            for ring in polygon:
                yield ring


def _geometry_to_pixel_rings(geom_dict, width, height):
    rings = []
    for ring in _iter_geometry_rings(geom_dict):
        px_ring = []
        for coord in ring:
            if not isinstance(coord, (list, tuple)) or len(coord) < 2:
                continue
            x, y = _lonlat_to_pixel(coord[0], coord[1], width, height)
            px_ring.append((x, y))
        if len(px_ring) >= 3:
            rings.append(px_ring)
    return rings


def _draw_text_with_halo(draw, position, text, color, font, halo_width):
    halo_width = max(1, int(halo_width))
    try:
        draw.text(
            position,
            text,
            fill=color,
            font=font,
            stroke_width=halo_width,
            stroke_fill=(0, 0, 0, 255),
        )
    except TypeError:
        x, y = position
        for dx in range(-halo_width, halo_width + 1):
            for dy in range(-halo_width, halo_width + 1):
                if dx == 0 and dy == 0:
                    continue
                draw.text((x + dx, y + dy), text, fill=(0, 0, 0, 255), font=font)
        draw.text(position, text, fill=color, font=font)


def _boxes_overlap(box_a, box_b, padding=0):
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b
    return not (
        (ax1 + padding) < bx0
        or (bx1 + padding) < ax0
        or (ay1 + padding) < by0
        or (by1 + padding) < ay0
    )


def _annotate_image_with_text(
    filepath, polygon_list, fill_polygons=False, allow_stage_less=False
):
    try:
        from PIL import Image, ImageDraw, ImageFont

        img = Image.open(filepath).convert("RGBA")
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        width, height = img.size

        try:
            font = ImageFont.truetype("arial.ttf", max(16, int(width * 0.025)))
        except:
            font = ImageFont.load_default()

        outline_w = max(4, int(width * 0.004))  # v12.0: Thicker outlines
        halo_w = outline_w + 4
        label_halo_w = max(2, outline_w // 2)
        marker_r = max(8, int(width * 0.009))
        render_cfg = getattr(config, "EXPORT_RENDER", {})
        min_display_area_m2 = float(render_cfg.get("min_display_area_m2", 100))
        max_labels = max(0, int(render_cfg.get("max_overlay_labels", 36)))
        collision_padding = max(0, int(render_cfg.get("label_collision_padding_px", 6)))
        label_min_conf = float(render_cfg.get("label_min_confidence", 0.0) or 0.0)
        detection_fill_alpha = int(render_cfg.get("detection_fill_alpha", 92))
        ground_truth_fill_alpha = int(render_cfg.get("ground_truth_fill_alpha", 104))
        placed_labels = []

        sorted_polygons = sorted(
            polygon_list,
            key=lambda p: (
                float(p.get("confidence", 0) or 0.0),
                float(p.get("area_m2", 0) or 0.0),
            ),
            reverse=True,
        )

        for p in sorted_polygons:
            # v12.0: Skip tiny fragments below minimum display area
            p_area = float(p.get("area_m2", 0) or 0)
            if p_area > 0 and p_area < min_display_area_m2:
                continue
            lat = p.get("lat")
            lon = p.get("lon")
            geom = p.get("geometry")
            stage = p.get("confirmed_stage") or p.get("raw_stage")
            if stage:
                color_rgb = STAGE_COLORS_RGB.get(stage, (255, 0, 0))
                text = f"S{stage}"
                if p.get("confidence") is not None:
                    try:
                        text = f"{text} {float(p.get('confidence')):.2f}"
                    except Exception:
                        pass
                fill_alpha = detection_fill_alpha
            elif allow_stage_less:
                color_rgb = tuple(p.get("color_rgb") or (230, 68, 54))
                text = str(p.get("label") or "GT")
                fill_alpha = ground_truth_fill_alpha
            else:
                continue

            color = (*color_rgb, 255)
            if (lat is None or lon is None) and geom:
                rings = _geometry_to_pixel_rings(geom, width, height)
                if rings:
                    xs = [pt[0] for ring in rings for pt in ring]
                    ys = [pt[1] for ring in rings for pt in ring]
                    lon = lon if lon is not None else 0
                    lat = lat if lat is not None else 0
                    x = sum(xs) / len(xs)
                    y = sum(ys) / len(ys)
                else:
                    continue
            else:
                if lat is None or lon is None:
                    continue
                x, y = _lonlat_to_pixel(lon, lat, width, height)

            rings = _geometry_to_pixel_rings(geom, width, height)
            if rings:
                all_x = []
                all_y = []
                for ring in rings:
                    if fill_polygons:
                        draw.polygon(ring, fill=(*color_rgb, fill_alpha))
                    line = ring + [ring[0]]
                    draw.line(line, fill=(0, 0, 0, 255), width=halo_w, joint="curve")
                    draw.line(line, fill=color, width=outline_w, joint="curve")
                    all_x.extend(pt[0] for pt in ring)
                    all_y.extend(pt[1] for pt in ring)
                if all_x and all_y:
                    min_x = max(0, int(min(all_x)))
                    min_y = max(0, int(min(all_y)))
                    label_x = min_x
                    label_y = max(0, min_y - marker_r - 8)
                else:
                    label_x = x
                    label_y = y
            else:
                draw.ellipse(
                    [x - marker_r, y - marker_r, x + marker_r, y + marker_r],
                    outline=(0, 0, 0, 255),
                    width=halo_w,
                )
                draw.ellipse(
                    [x - marker_r, y - marker_r, x + marker_r, y + marker_r],
                    outline=color,
                    width=outline_w,
                )
                label_x = x + marker_r + 6
                label_y = y - marker_r

            bbox = draw.textbbox((label_x, label_y), text, font=font)
            pad = max(4, outline_w)
            label_box = [bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad]
            if p.get("show_label", True) is False:
                continue
            try:
                current_conf = float(
                    p.get("confidence", 1 if allow_stage_less and not stage else 0)
                    or 0.0
                )
            except Exception:
                current_conf = 1.0 if allow_stage_less and not stage else 0.0
            if current_conf < label_min_conf:
                continue
            if len(placed_labels) >= max_labels:
                continue
            if any(
                _boxes_overlap(label_box, existing, collision_padding)
                for existing in placed_labels
            ):
                continue
            _draw_text_with_halo(
                draw, (label_x, label_y), text, color, font, label_halo_w
            )
            placed_labels.append(label_box)

        img = Image.alpha_composite(img, overlay)
        img.save(filepath, "PNG")
    except Exception as e:
        print(f"[M10] Warning drawing text: {e}")


def _thumb(
    image,
    aoi,
    vis_params,
    filepath,
    retries=3,
    fill_value=None,
    force_scale=None,
    polygons_fc=None,
    polygon_list=None,
    dimensions=None,
    scale_m=None,
    crs=None,
):
    """
    Download a GEE thumbnail.
    """
    image = ee.Image(image).clip(aoi)

    if fill_value is not None:
        image = image.unmask(fill_value)

    # Removed forced reproject() here. It causes the solid bounding
    # box / solid color artifacts. Let GEE compute it at native scale
    # or the scale requested by thumbnail dimensions.

    if vis_params:
        vis_image = image.visualize(**vis_params)
    else:
        vis_image = image

    # Avoid server-side vector styling when we already draw polygons locally.
    # This keeps the thumbnail graph lighter and reduces HTTP 400/timeouts.
    if polygons_fc is not None and not polygon_list:
        try:
            styled = polygons_fc.style(color="FF0000", width=2, fillColor="00000000")
            vis_image = vis_image.blend(styled)
        except Exception as e:
            print(f"[M10] Warning styling polygons: {e}")

    region_coords = [
        [config.AOI["lon_min"], config.AOI["lat_min"]],
        [config.AOI["lon_max"], config.AOI["lat_min"]],
        [config.AOI["lon_max"], config.AOI["lat_max"]],
        [config.AOI["lon_min"], config.AOI["lat_max"]],
    ]

    params = {"region": region_coords, "format": "png"}
    # GEE constraint: you cannot specify both dimensions and scale.
    if scale_m is not None:
        params["scale"] = float(scale_m)
    else:
        params["dimensions"] = str(dimensions or THUMB_SIZE)
    if crs is not None:
        params["crs"] = crs

    try:
        url = vis_image.getThumbURL(params)
    except Exception as e:
        export_log.error(f"URL error ({os.path.basename(filepath)}): {e}")
        print(f"[M10] URL error ({os.path.basename(filepath)}): {e}")
        return False

    for attempt in range(retries):
        try:
            req = urllib.request.urlopen(url, timeout=300)
            raw_data = req.read()
            with open(filepath, "wb") as f:
                f.write(raw_data)

            # ── PNG integrity check — detect corrupt/truncated downloads ──
            try:
                from PIL import Image as PILImage

                test_img = PILImage.open(filepath)
                test_img.load()  # Force full decode to catch CRC errors
                test_img.close()
            except Exception as png_err:
                print(
                    f"[M10] Corrupt PNG ({os.path.basename(filepath)}): {png_err}, retrying..."
                )
                os.remove(filepath)
                if attempt < retries - 1:
                    time.sleep(3**attempt)
                    continue
                else:
                    return False

            if polygon_list and len(polygon_list) > 0:
                _annotate_image_with_text(filepath, polygon_list)

            print(f"[M10] Saved: {os.path.basename(filepath)}")
            file_size = os.path.getsize(filepath)
            export_log.info(
                f"  Saved {os.path.basename(filepath)} ({file_size:,} bytes)"
            )

            if config.EXTENSIONS.get("use_dip_enhancement", False):
                try:
                    import dip_pipeline

                    fname = os.path.basename(filepath).lower()
                    render_cfg = getattr(config, "EXPORT_RENDER", {})
                    is_stage = "stage_" in fname
                    is_rgb = "rgb_" in fname
                    is_feature = any(
                        k in fname
                        for k in [
                            "ndwi",
                            "mndwi",
                            "awei",
                            "ndvi",
                            "savi",
                            "ndbi",
                            "sar_",
                            "rvi_",
                        ]
                    )

                    do_enhance = (
                        (is_stage and render_cfg.get("apply_dip_to_stage", False))
                        or (is_rgb and render_cfg.get("apply_dip_to_rgb", False))
                        or (
                            is_feature and render_cfg.get("apply_dip_to_features", True)
                        )
                    )

                    if do_enhance:
                        if any(
                            k in fname
                            for k in ["ndwi", "mndwi", "awei", "sar_", "rvi_"]
                        ):
                            dip_mode = "water"
                        elif is_rgb:
                            dip_mode = "full"
                        elif is_stage:
                            dip_mode = "stage"
                        else:
                            dip_mode = "standard"

                        if dip_pipeline.enhance_image(filepath, mode=dip_mode):
                            print(
                                f"[M10] DIP Enhanced ({dip_mode}): {os.path.basename(filepath)}"
                            )

                        if is_rgb:
                            dip_pipeline.generate_false_color_water(filepath)
                except Exception as e:
                    print(f"[M10] DIP Error on {os.path.basename(filepath)}: {e}")

            return True
        except urllib.error.HTTPError as e:
            err_body = e.read().decode() if hasattr(e, "read") else ""
            print(
                f"[M10] Failed ({os.path.basename(filepath)}): HTTP {e.code} | Body: {err_body}"
            )
            if e.code == 400:
                return False
            if attempt < retries - 1:
                time.sleep(3**attempt)
            else:
                return False
        except (urllib.error.URLError, socket.timeout, ConnectionResetError) as e:
            if attempt < retries - 1:
                time.sleep(3**attempt)
            else:
                print(f"[M10] Failed ({os.path.basename(filepath)}): {e}")
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(3**attempt)
            else:
                print(f"[M10] Unexpected Error ({os.path.basename(filepath)}): {e}")

    return False


# ── RGB / Satellite Image ────────────────────────────────────────────


def export_rgb_thumbnail(
    image,
    aoi,
    filename,
    polygons_fc=None,
    polygon_list=None,
    thumb_size=None,
    scale_m=None,
):
    """
    True-color RGB satellite image with adaptive percentile stretch.
    Uses actual red/green/blue bands from the composite for realistic
    satellite imagery. Computes per-image 2nd/98th percentile to
    dynamically adjust min/max, handling sensor and scene variation.
    Falls back to fixed stretch if percentile computation fails.
    """
    image = ee.Image(image)

    # Select actual RGB bands from the composite
    rgb = image.select(["red", "green", "blue"])

    # Compute per-band percentiles for adaptive stretching
    try:
        stats = rgb.reduceRegion(
            reducer=ee.Reducer.percentile([2, 98]),
            geometry=aoi,
            scale=60,  # coarse scale to save computation
            maxPixels=1e8,
            bestEffort=True,
            tileScale=16,
        ).getInfo()

        r_min = stats.get("red_p2", 0.0)
        r_max = stats.get("red_p98", 0.3)
        g_min = stats.get("green_p2", 0.0)
        g_max = stats.get("green_p98", 0.3)
        b_min = stats.get("blue_p2", 0.0)
        b_max = stats.get("blue_p98", 0.3)

        # Sanity check: if all values are None or identical, use fixed fallback
        if (
            r_min is None
            or r_max is None
            or r_min >= r_max
            or g_min is None
            or g_max is None
            or g_min >= g_max
            or b_min is None
            or b_max is None
            or b_min >= b_max
        ):
            raise ValueError("Invalid percentile stats")

        vis = {
            "bands": ["red", "green", "blue"],
            "min": [r_min, g_min, b_min],
            "max": [r_max, g_max, b_max],
            "gamma": 1.3,
        }
    except Exception:
        # Fixed fallback for surface reflectance (0-1 range typical for HLS/Landsat)
        vis = {"bands": ["red", "green", "blue"], "min": 0.0, "max": 0.25, "gamma": 1.4}

    path = os.path.join(config.IMAGE_DIR, f"rgb_{filename}")
    os.makedirs(config.IMAGE_DIR, exist_ok=True)
    return _thumb(
        rgb,
        aoi,
        vis,
        path,
        force_scale=_THUMB_SCALE,
        polygons_fc=polygons_fc,
        polygon_list=polygon_list,
        dimensions=thumb_size,
        scale_m=scale_m,
        crs=_THUMB_CRS,
    )


def export_detection_overlay_thumbnail(
    image, aoi, filename, polygon_list=None, thumb_size=None, scale_m=None
):
    """Detection overlay = RGB satellite image + filled polygon borders."""
    path = os.path.join(config.IMAGE_DIR, f"detection_{filename}")
    os.makedirs(config.IMAGE_DIR, exist_ok=True)

    # v22.0: Reuse existing RGB thumbnail instead of re-computing from GEE.
    # export_rgb_thumbnail() is always called first in main.py, so the file
    # should already exist. This eliminates a redundant reduceRegion + download.
    rgb_path = os.path.join(config.IMAGE_DIR, f"rgb_{filename}")
    if os.path.exists(rgb_path):
        import shutil
        shutil.copy2(rgb_path, path)
    else:
        # Fallback: compute from GEE if RGB not available
        image = ee.Image(image)
        rgb = image.select(["red", "green", "blue"])
        try:
            stats = rgb.reduceRegion(
                reducer=ee.Reducer.percentile([2, 98]),
                geometry=aoi,
                scale=60,
                maxPixels=1e8,
                bestEffort=True,
                tileScale=16,
            ).getInfo()
            r_min = stats.get("red_p2", 0.0)
            r_max = stats.get("red_p98", 0.3)
            g_min = stats.get("green_p2", 0.0)
            g_max = stats.get("green_p98", 0.3)
            b_min = stats.get("blue_p2", 0.0)
            b_max = stats.get("blue_p98", 0.3)
            if (
                r_min is None or r_max is None or r_min >= r_max
                or g_min is None or g_max is None or g_min >= g_max
                or b_min is None or b_max is None or b_min >= b_max
            ):
                raise ValueError("Invalid percentile stats")
            vis = {
                "bands": ["red", "green", "blue"],
                "min": [r_min, g_min, b_min],
                "max": [r_max, g_max, b_max],
                "gamma": 1.3,
            }
        except Exception:
            vis = {"bands": ["red", "green", "blue"], "min": 0.0, "max": 0.25, "gamma": 1.4}

        _thumb(
            rgb, aoi, vis, path,
            force_scale=_THUMB_SCALE,
            polygons_fc=None, polygon_list=None,
            dimensions=thumb_size, scale_m=scale_m, crs=_THUMB_CRS,
        )

    if not os.path.exists(path):
        return False
    if polygon_list:
        _annotate_image_with_text(path, polygon_list, fill_polygons=True)
    return True


def export_ground_truth_overlay_thumbnail(
    image, aoi, filename, gt_polygon_list=None, thumb_size=None, scale_m=None
):
    if not gt_polygon_list:
        return False

    raw_path = os.path.join(config.IMAGE_DIR, f"rgb_{filename}")
    path = os.path.join(config.IMAGE_DIR, f"ground_truth_{filename}")
    if not os.path.exists(raw_path):
        export_rgb_thumbnail(
            image, aoi, filename, None, None, thumb_size=thumb_size, scale_m=scale_m
        )
    if not os.path.exists(raw_path):
        return False
    shutil.copyfile(raw_path, path)
    if os.path.exists(path):
        _annotate_image_with_text(
            path, gt_polygon_list, fill_polygons=True, allow_stage_less=True
        )
        return True
    return False


# ── RGB with Pond Boundaries ─────────────────────────────────────────


def _draw_pond_boundaries_on_rgb(
    rgb_path, output_path, polygon_list, edge_color=(255, 255, 0), edge_width=3
):
    """
    Draw bright, thick polygon boundaries on an RGB image.
    """
    try:
        from PIL import Image, ImageDraw

        img = Image.open(rgb_path).convert("RGBA")
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        width, height = img.size

        # Get AOI bounds for coordinate conversion
        lon_min = config.AOI["lon_min"]
        lon_max = config.AOI["lon_max"]
        lat_min = config.AOI["lat_min"]
        lat_max = config.AOI["lat_max"]

        for p in polygon_list:
            geom = p.get("geometry")
            if not geom:
                continue

            # Convert geometry to pixel coordinates
            rings = []
            coords = geom.get("coordinates", [])
            gtype = geom.get("type")

            if gtype == "Polygon":
                rings = coords
            elif gtype == "MultiPolygon":
                for poly in coords:
                    rings.extend(poly)

            for ring in rings:
                px_points = []
                for coord in ring:
                    if isinstance(coord, (list, tuple)) and len(coord) >= 2:
                        lon, lat = coord[0], coord[1]
                        x = (lon - lon_min) / max(lon_max - lon_min, 1e-9) * width
                        y = height - (
                            (lat - lat_min) / max(lat_max - lat_min, 1e-9) * height
                        )
                        px_points.append((x, y))

                if len(px_points) >= 3:
                    # Draw thick edge with black outline for visibility
                    line = px_points + [px_points[0]]
                    draw.line(
                        line, fill=(0, 0, 0, 255), width=edge_width + 2
                    )  # Black outline
                    draw.line(
                        line, fill=(*edge_color, 255), width=edge_width
                    )  # Colored edge

        img = Image.alpha_composite(img, overlay)
        img.save(output_path, "PNG")
        return True
    except Exception as e:
        print(f"[M10] Error drawing pond boundaries: {e}")
        return False


def export_rgb_with_pond_boundaries(
    image,
    aoi,
    filename,
    polygon_list=None,
    thumb_size=None,
    scale_m=None,
    edge_color=(255, 255, 0),
):
    """
    Export RGB image with clear aquaculture pond boundaries marked.
    Saves both plain RGB and marked RGB versions.
    """
    # First export plain RGB
    rgb_path = os.path.join(config.IMAGE_DIR, f"rgb_{filename}")
    if not export_rgb_thumbnail(
        image, aoi, filename, None, None, thumb_size=thumb_size, scale_m=scale_m
    ):
        return False

    if not polygon_list or len(polygon_list) == 0:
        return os.path.exists(rgb_path)

    # Create marked version with pond boundaries
    marked_path = os.path.join(config.IMAGE_DIR, f"rgb_marked_{filename}")
    if _draw_pond_boundaries_on_rgb(
        rgb_path, marked_path, polygon_list, edge_color=edge_color, edge_width=4
    ):
        print(f"[M10] Saved marked RGB: {os.path.basename(marked_path)}")
        return True
    return False


# ── Spectral Index Thumbnails ─────────────────────────────────────────


def export_ndvi_thumbnail(
    feature_image,
    aoi,
    filename,
    polygons_fc=None,
    polygon_list=None,
    thumb_size=None,
    scale_m=None,
):
    vis = {
        "min": -0.2,
        "max": 0.8,
        "palette": ["d73027", "fc8d59", "fee08b", "d9ef8b", "91cf60", "1a9850"],
    }
    path = os.path.join(config.FEATURE_DIR, f"ndvi_{filename}")
    os.makedirs(config.FEATURE_DIR, exist_ok=True)
    return _thumb(
        ee.Image(feature_image).select("ndvi"),
        aoi,
        vis,
        path,
        fill_value=None,
        force_scale=_THUMB_SCALE,
        polygons_fc=polygons_fc,
        polygon_list=polygon_list,
        dimensions=thumb_size,
        scale_m=scale_m,
        crs=_THUMB_CRS,
    )


def export_ndwi_thumbnail(
    feature_image,
    aoi,
    filename,
    polygons_fc=None,
    polygon_list=None,
    thumb_size=None,
    scale_m=None,
):
    vis = {
        "min": -0.4,
        "max": 0.6,
        "palette": ["ffffcc", "a1dab4", "41b6c4", "2c7fb8", "253494"],
    }
    path = os.path.join(config.FEATURE_DIR, f"ndwi_{filename}")
    os.makedirs(config.FEATURE_DIR, exist_ok=True)
    return _thumb(
        ee.Image(feature_image).select("ndwi"),
        aoi,
        vis,
        path,
        fill_value=None,
        force_scale=_THUMB_SCALE,
        polygons_fc=polygons_fc,
        polygon_list=polygon_list,
        dimensions=thumb_size,
        scale_m=scale_m,
        crs=_THUMB_CRS,
    )


def export_awei_thumbnail(
    feature_image,
    aoi,
    filename,
    polygons_fc=None,
    polygon_list=None,
    thumb_size=None,
    scale_m=None,
):
    vis = {
        "min": -1.0,
        "max": 1.0,
        "palette": [
            "543005",
            "8c510a",
            "bf812d",
            "dfc27d",
            "80cdc1",
            "35978f",
            "01665e",
        ],
    }
    path = os.path.join(config.FEATURE_DIR, f"awei_{filename}")
    os.makedirs(config.FEATURE_DIR, exist_ok=True)
    return _thumb(
        ee.Image(feature_image).select("awei"),
        aoi,
        vis,
        path,
        fill_value=None,
        force_scale=_THUMB_SCALE,
        polygons_fc=polygons_fc,
        polygon_list=polygon_list,
        dimensions=thumb_size,
        scale_m=scale_m,
        crs=_THUMB_CRS,
    )


def export_savi_thumbnail(
    feature_image,
    aoi,
    filename,
    polygons_fc=None,
    polygon_list=None,
    thumb_size=None,
    scale_m=None,
):
    vis = {
        "min": -0.2,
        "max": 0.8,
        "palette": ["d73027", "fc8d59", "fee08b", "d9ef8b", "91cf60", "1a9850"],
    }
    path = os.path.join(config.FEATURE_DIR, f"savi_{filename}")
    os.makedirs(config.FEATURE_DIR, exist_ok=True)
    return _thumb(
        ee.Image(feature_image).select("savi"),
        aoi,
        vis,
        path,
        fill_value=None,
        force_scale=_THUMB_SCALE,
        polygons_fc=polygons_fc,
        polygon_list=polygon_list,
        dimensions=thumb_size,
        scale_m=scale_m,
        crs=_THUMB_CRS,
    )


def export_mndwi_thumbnail(
    feature_image,
    aoi,
    filename,
    polygons_fc=None,
    polygon_list=None,
    thumb_size=None,
    scale_m=None,
):
    vis = {
        "min": -0.4,
        "max": 0.6,
        "palette": ["8c510a", "d8b365", "f6e8c3", "c7eae5", "5ab4ac", "01665e"],
    }
    path = os.path.join(config.FEATURE_DIR, f"mndwi_{filename}")
    os.makedirs(config.FEATURE_DIR, exist_ok=True)
    return _thumb(
        ee.Image(feature_image).select("mndwi"),
        aoi,
        vis,
        path,
        fill_value=None,
        force_scale=_THUMB_SCALE,
        polygons_fc=polygons_fc,
        polygon_list=polygon_list,
        dimensions=thumb_size,
        scale_m=scale_m,
        crs=_THUMB_CRS,
    )


def export_ndbi_thumbnail(
    feature_image,
    aoi,
    filename,
    polygons_fc=None,
    polygon_list=None,
    thumb_size=None,
    scale_m=None,
):
    vis = {
        "min": -0.5,
        "max": 0.5,
        "palette": ["1a9850", "91cf60", "d9ef8b", "fee08b", "fc8d59", "d73027"],
    }
    path = os.path.join(config.FEATURE_DIR, f"ndbi_{filename}")
    os.makedirs(config.FEATURE_DIR, exist_ok=True)
    return _thumb(
        ee.Image(feature_image).select("ndbi"),
        aoi,
        vis,
        path,
        fill_value=None,
        force_scale=_THUMB_SCALE,
        polygons_fc=polygons_fc,
        polygon_list=polygon_list,
        dimensions=thumb_size,
        scale_m=scale_m,
        crs=_THUMB_CRS,
    )


def export_sar_thumbnail(
    sar_image,
    aoi,
    filename,
    polygons_fc=None,
    polygon_list=None,
    thumb_size=None,
    scale_m=None,
):
    vis = {
        "min": -25,
        "max": -5,
        "palette": ["000000", "333333", "666666", "999999", "cccccc", "ffffff"],
    }
    path = os.path.join(config.FEATURE_DIR, f"sar_{filename}")
    os.makedirs(config.FEATURE_DIR, exist_ok=True)
    return _thumb(
        ee.Image(sar_image).select("VV"),
        aoi,
        vis,
        path,
        fill_value=-25,
        force_scale=_THUMB_SCALE,
        polygons_fc=polygons_fc,
        polygon_list=polygon_list,
        dimensions=thumb_size,
        scale_m=scale_m,
        crs=_THUMB_CRS,
    )


def export_rvi_thumbnail(
    feature_image,
    aoi,
    filename,
    polygons_fc=None,
    polygon_list=None,
    thumb_size=None,
    scale_m=None,
):
    vis = {
        "min": 0,
        "max": 1.2,
        "palette": ["000000", "333333", "666666", "999999", "cccccc", "ffffff"],
    }
    path = os.path.join(config.FEATURE_DIR, f"rvi_{filename}")
    os.makedirs(config.FEATURE_DIR, exist_ok=True)
    return _thumb(
        ee.Image(feature_image).select("rvi"),
        aoi,
        vis,
        path,
        fill_value=0,
        force_scale=_THUMB_SCALE,
        polygons_fc=polygons_fc,
        polygon_list=polygon_list,
        dimensions=thumb_size,
        scale_m=scale_m,
        crs=_THUMB_CRS,
    )


# ── Static Ancillary Layers (export once) ────────────────────────────

_STATIC_EXPORTED = {"jrc": False, "glo30": False}


def export_jrc_thumbnail(jrc_water, aoi, filename="static.png"):
    if jrc_water is None:
        return False
    if _STATIC_EXPORTED["jrc"]:
        print("[M10] JRC already exported (static — skipping)")
        return True
    vis = {
        "min": 0,
        "max": 100,
        "palette": ["ffffff", "d4e7f7", "89c4e1", "3690c0", "02818a", "014636"],
    }
    path = os.path.join(config.FEATURE_DIR, "jrc_static.png")
    os.makedirs(config.FEATURE_DIR, exist_ok=True)
    result = _thumb(
        jrc_water.select("occurrence"),
        aoi,
        vis,
        path,
        fill_value=0,
        force_scale=_THUMB_SCALE,
    )
    if result:
        _STATIC_EXPORTED["jrc"] = True
    return result


def export_glo30_thumbnail(glo30, aoi, filename="static.png"):
    if glo30 is None:
        return False
    if _STATIC_EXPORTED["glo30"]:
        print("[M10] GLO30 DEM already exported (static — skipping)")
        return True
    vis = {
        "min": 0,
        "max": 20,
        "palette": [
            "023858",
            "045a8d",
            "0570b0",
            "3690c0",
            "74a9cf",
            "a6bddb",
            "d0d1e6",
            "f1eef6",
        ],
    }
    path = os.path.join(config.FEATURE_DIR, "glo30_static.png")
    os.makedirs(config.FEATURE_DIR, exist_ok=True)
    result = _thumb(
        glo30.select("DEM"), aoi, vis, path, fill_value=0, force_scale=_THUMB_SCALE
    )
    if result:
        _STATIC_EXPORTED["glo30"] = True
    return result


def reset_static_export_flags():
    _STATIC_EXPORTED["jrc"] = False
    _STATIC_EXPORTED["glo30"] = False


# ── Stage Map Thumbnails (FIX: HTTP 400) ─────────────────────────────


def _add_stage_boundary_grid(filepath):
    """Draw 1px black boundary lines between differently-colored stage pixels.
    
    v21.0: Creates a 'grid' effect that makes individual stage pixels
    distinguishable. Detects color transitions between adjacent pixels
    and draws thin black lines at those boundaries.
    
    Runs entirely on the downloaded PNG — zero GEE computation cost.
    """
    try:
        import cv2
        import numpy as np
        
        img = cv2.imread(filepath)
        if img is None:
            return
        
        h, w = img.shape[:2]
        # Detect horizontal boundaries (color change between pixel[y] and pixel[y+1])
        diff_v = np.any(img[:-1, :, :] != img[1:, :, :], axis=2)
        # Detect vertical boundaries (color change between pixel[x] and pixel[x+1])
        diff_h = np.any(img[:, :-1, :] != img[:, 1:, :], axis=2)
        
        # Create boundary mask (1px lines)
        boundary = np.zeros((h, w), dtype=bool)
        boundary[:-1, :] |= diff_v
        boundary[1:, :] |= diff_v
        boundary[:, :-1] |= diff_h
        boundary[:, 1:] |= diff_h
        
        # Only darken pixels that are at actual stage boundaries,
        # not within the same-colored region. Filter out noise by
        # only applying where the color difference is significant.
        # Compute color distance for horizontal and vertical neighbors
        color_dist = np.zeros((h, w), dtype=np.float32)
        # Vertical neighbor distance
        vd = np.sqrt(np.sum((img[:-1, :, :].astype(np.float32) - img[1:, :, :].astype(np.float32)) ** 2, axis=2))
        color_dist[:-1, :] = np.maximum(color_dist[:-1, :], vd)
        color_dist[1:, :] = np.maximum(color_dist[1:, :], vd)
        # Horizontal neighbor distance
        hd = np.sqrt(np.sum((img[:, :-1, :].astype(np.float32) - img[:, 1:, :].astype(np.float32)) ** 2, axis=2))
        color_dist[:, :-1] = np.maximum(color_dist[:, :-1], hd)
        color_dist[:, 1:] = np.maximum(color_dist[:, 1:], hd)
        
        # Only draw boundary where color distance is significant (> 30)
        # This prevents boundaries within smoothly-varying same-stage regions
        significant = color_dist > 30
        boundary = boundary & significant
        
        # Draw 1px black lines at boundaries
        img[boundary] = [0, 0, 0]
        
        cv2.imwrite(filepath, img)
    except Exception as e:
        print(f"[M10] Stage boundary grid warning: {e}")


def _enhance_stage_contrast(filepath):
    """Apply edge-preserving enhancement to stage classification map.
    
    v21.0 FIXES:
      - Replaced medianBlur (smears edges) with bilateralFilter (preserves edges)
      - Removed MORPH_CLOSE (merges stage colors at boundaries)
      - Added Laplacian edge sharpening for crisp stage boundaries
      - Increased PIL contrast/saturation/sharpness for vivid colors
    
    Runs entirely on the downloaded PNG — zero GEE computation cost.
    """
    try:
        import cv2
        import numpy as np
        
        img = cv2.imread(filepath)
        if img is not None:
            # v21.0: Bilateral filter — smooths WITHIN color regions while
            # preserving sharp edges between different stages.
            # d=5, sigmaColor=50: only average pixels with similar color
            # sigmaSpace=50: spatial locality
            img = cv2.bilateralFilter(img, d=5, sigmaColor=50, sigmaSpace=50)
            
            # v21.0: Laplacian edge sharpening — detect edges and blend back
            # to emphasize boundaries between stages
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
            laplacian = np.abs(laplacian)
            laplacian = (laplacian / laplacian.max() * 255).astype(np.uint8) if laplacian.max() > 0 else laplacian.astype(np.uint8)
            # Create dark edge overlay (darken edges for clear boundaries)
            edge_mask = laplacian > 30  # threshold for significant edges
            for c in range(3):
                channel = img[:, :, c]
                # Darken edge pixels slightly for crisp boundary lines
                channel[edge_mask] = np.clip(channel[edge_mask].astype(np.int16) - 40, 0, 255).astype(np.uint8)
                img[:, :, c] = channel
            
            cv2.imwrite(filepath, img)
    except Exception as e:
        print(f"[M10] Stage cv2 cleanup warning: {e}")
    
    try:
        from PIL import Image, ImageEnhance
        img = Image.open(filepath)
        # v21.0: Stronger contrast for crisp stage boundaries
        img = ImageEnhance.Contrast(img).enhance(1.6)
        # v21.0: Stronger saturation for distinct stage colors
        img = ImageEnhance.Color(img).enhance(1.5)
        # v21.0: Strong sharpening for publication quality
        img = ImageEnhance.Sharpness(img).enhance(2.5)
        img.save(filepath, "PNG")
    except Exception as e:
        print(f"[M10] Stage PIL enhancement warning: {e}")


def export_stage_thumbnail(
    stage_image,
    aoi,
    filename,
    polygons_fc=None,
    polygon_list=None,
    thumb_size=None,
    scale_m=None,
):
    """
    Stage classification map with polygon boundaries colored by stage.

    v11.0: Now draws polygon boundaries with stage colors for clear S4/S5 visualization.
    """
    path = os.path.join(config.IMAGE_DIR, f"stage_{filename}")
    os.makedirs(config.IMAGE_DIR, exist_ok=True)

    def _build_stage_visual(export_scale=None):
        # v23.0: Explicitly select ONLY the stage band to reduce memory footprint
        # The stage_image may have multiple bands (stage, water_evidence, etc.)
        stage_band = ee.Image(stage_image).select("stage").toByte()
        stage_band = stage_band.updateMask(stage_band.gt(0))
        # v23.0: Simplify computation graph for export by explicitly reprojecting
        if export_scale is not None:
            # Force computation at target scale to reduce tile count
            stage_band = stage_band.reproject(crs=_THUMB_CRS, scale=float(export_scale))
        else:
            # Default to 60m to keep under memory limits for large AOIs
            stage_band = stage_band.reproject(crs=_THUMB_CRS, scale=60)
        stage_vis = stage_band.visualize(min=1, max=5, palette=_STAGE_PALETTE)
        return stage_vis

    primary = _thumb(
        _build_stage_visual(scale_m),
        aoi,
        None,
        path,
        fill_value=None,
        force_scale=_THUMB_SCALE,
        polygons_fc=None,
        polygon_list=None,
        dimensions=thumb_size,
        scale_m=scale_m,
        crs=_THUMB_CRS,
    )

    # v11.0: Draw polygon boundaries colored by stage
    if primary and polygon_list:
        try:
            _draw_stage_colored_polygons(path, polygon_list)
        except Exception as e:
            print(f"[M10] Warning: Could not draw stage-colored polygons: {e}")

    if primary:
        _enhance_stage_contrast(path)
        _add_stage_boundary_grid(path)
        _annotate_stage_legend(path)
        return True

    # Historical full-scene stage thumbnails can exceed GEE's thumbnail graph
    # size limit when exported at fine/native resolution. Retry progressively
    # coarser scale-based exports so the pipeline continues instead of failing
    # the whole epoch visualization.
    # v23.0: Extended fallback scales for larger AOIs
    fallback_scales = []
    base_scale = max(int(getattr(config, "TARGET_SCALE", 30)), 30)
    if scale_m is None:
        fallback_scales.extend([base_scale, 45, 60, 90, 120])
    else:
        fallback_scales.extend(
            [max(int(scale_m), base_scale), max(int(scale_m * 1.5), 45), 60, 90, 120]
        )

    tried = set()
    for fallback_scale in fallback_scales:
        fallback_scale = int(max(10, fallback_scale))
        if fallback_scale in tried:
            continue
        tried.add(fallback_scale)
        print(f"[M10] Retrying stage export at coarser scale: {fallback_scale} m")
        if _thumb(
            _build_stage_visual(fallback_scale),
            aoi,
            None,
            path,
            fill_value=None,
            force_scale=_THUMB_SCALE,
            polygons_fc=None,
            polygon_list=None,
            dimensions=None,
            scale_m=fallback_scale,
            crs=_THUMB_CRS,
        ):
            # Draw polygons on fallback too
            if polygon_list:
                try:
                    _draw_stage_colored_polygons(path, polygon_list)
                except Exception:
                    pass
            return True

    return False


def _draw_stage_colored_polygons(filepath, polygon_list):
    """
    Draw ALL polygon boundaries colored by their stage (S1-S5).

    v11.2: ALL stages, THIN (1px) sharp boundaries, DISTINCT colors.
    """
    try:
        from PIL import Image, ImageDraw

        img = Image.open(filepath).convert("RGBA")
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        width, height = img.size

        # AOI bounds
        lon_min = config.AOI["lon_min"]
        lon_max = config.AOI["lon_max"]
        lat_min = config.AOI["lat_min"]
        lat_max = config.AOI["lat_max"]

        # STAGE COLORS - DISTINCT and BRIGHT
        STAGE_COLORS = {
            1: (34, 139, 34),  # Forest Green - S1
            2: (255, 215, 0),  # Gold - S2
            3: (255, 140, 0),  # Dark Orange - S3
            4: (0, 255, 255),  # Cyan - S4
            5: (0, 0, 255),  # Blue - S5
        }

        # Draw ALL stages (filter out unclassified/stage 0)
        valid_polygons = [
            p
            for p in polygon_list
            if (p.get("confirmed_stage") or p.get("raw_stage") or 0) in [1, 2, 3, 4, 5]
        ]

        # Sort by stage (S1 first, S5 last - S5 drawn on top)
        sorted_polygons = sorted(
            valid_polygons,
            key=lambda p: p.get("confirmed_stage") or p.get("raw_stage") or 0,
        )

        drawn_count = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}

        for p in sorted_polygons:
            stage = p.get("confirmed_stage") or p.get("raw_stage") or 0
            geom = p.get("geometry")
            if not geom or stage not in STAGE_COLORS:
                continue

            color_rgb = STAGE_COLORS[stage]
            drawn_count[stage] += 1

            # Convert geometry
            coords = geom.get("coordinates", [])
            gtype = geom.get("type")

            rings = []
            if gtype == "Polygon":
                rings = coords
            elif gtype == "MultiPolygon":
                for poly in coords:
                    rings.extend(poly)

            for ring in rings:
                px_points = []
                for coord in ring:
                    if isinstance(coord, (list, tuple)) and len(coord) >= 2:
                        lon, lat = coord[0], coord[1]
                        x = (lon - lon_min) / max(lon_max - lon_min, 1e-9) * width
                        y = height - (
                            (lat - lat_min) / max(lat_max - lat_min, 1e-9) * height
                        )
                        px_points.append((x, y))

                if len(px_points) >= 3:
                    line = px_points + [px_points[0]]
                    # v14.0: Thicker outlines for clearer boundaries
                    draw.line(line, fill=(255, 255, 255, 200), width=5)  # White halo
                    draw.line(line, fill=(*color_rgb, 255), width=2)   # Colored line

                    # v14.0: Stronger fill for visible stage regions (alpha 25→80)
                    draw.polygon(px_points, fill=(*color_rgb, 80))

        img = Image.alpha_composite(img, overlay)
        img.save(filepath, "PNG")
        total_drawn = sum(drawn_count.values())

        # Debug: Print centroid distribution
        if sorted_polygons:
            lons = [
                p.get("centroid_lon", 0)
                for p in sorted_polygons
                if p.get("centroid_lon")
            ]
            lats = [
                p.get("centroid_lat", 0)
                for p in sorted_polygons
                if p.get("centroid_lat")
            ]
            if lons and lats:
                print(
                    f"[M10] Pond centroid range: Lon {min(lons):.3f}-{max(lons):.3f}, Lat {min(lats):.3f}-{max(lats):.3f}"
                )
                print(
                    f"[M10] AOI bounds: Lon {lon_min}-{lon_max}, Lat {lat_min}-{lat_max}"
                )

        print(
            f"[M10] Drew {total_drawn} polygons: S1={drawn_count[1]}, S2={drawn_count[2]}, S3={drawn_count[3]}, S4={drawn_count[4]}, S5={drawn_count[5]}"
        )
        return True
    except Exception as e:
        print(f"[M10] Error drawing stage-colored polygons: {e}")
        import traceback

        traceback.print_exc()
        return False


# ── JSON Exports ─────────────────────────────────────────────────────


def generate_timeline_data(stage_history):
    # Deprecated. Handled directly in main.py _export_dashboard_data via PondRegistry stats.
    pass


def _build_image_assets(year):
    asset_specs = {
        "rgb": os.path.join(config.IMAGE_DIR, f"rgb_{year}.png"),
        "detection": os.path.join(config.IMAGE_DIR, f"detection_{year}.png"),
        "ground_truth": os.path.join(config.IMAGE_DIR, f"ground_truth_{year}.png"),
        "stage": os.path.join(config.IMAGE_DIR, f"stage_{year}.png"),
        "ndvi": os.path.join(config.FEATURE_DIR, f"ndvi_{year}.png"),
        "mndwi": os.path.join(config.FEATURE_DIR, f"mndwi_{year}.png"),
        "ndwi": os.path.join(config.FEATURE_DIR, f"ndwi_{year}.png"),
        "savi": os.path.join(config.FEATURE_DIR, f"savi_{year}.png"),
        "ndbi": os.path.join(config.FEATURE_DIR, f"ndbi_{year}.png"),
        "awei": os.path.join(config.FEATURE_DIR, f"awei_{year}.png"),
        "sar": os.path.join(config.FEATURE_DIR, f"sar_{year}.png"),
        "rvi": os.path.join(config.FEATURE_DIR, f"rvi_{year}.png"),
    }
    assets = {}
    for key, abs_path in asset_specs.items():
        if os.path.exists(abs_path):
            rel = os.path.relpath(abs_path, config.BASE_DIR).replace("\\", "/")
            assets[key] = f"../{rel}"
    return assets


def generate_stats_data(all_results):
    records = []
    for r in all_results:
        year = r.get("year")
        record = {
            "date": r.get("date"),
            "year": r.get("year"),
            "sensor": r.get("sensor"),
            "stage": r.get("stage"),
            "stage_name": stage_label(r.get("stage", 0)),
            "stage_label": stage_label(r.get("stage", 0)),
            "confirmed_stage": r.get("confirmed_stage"),
            "confirmed_stage_name": stage_label(
                r.get("confirmed_stage", r.get("stage", 0))
            ),
            "confidence": r.get("confidence"),
            "stage_probability": r.get("stage_probability"),
            "stage_probabilities": r.get("stage_probabilities", {}),
            "uncertain": r.get("uncertain", False),
            "uncertainty_reason": r.get("uncertainty_reason", ""),
            "validation_score": r.get("validation_score"),
            "mangrove_score": r.get("mangrove_score"),
            "water_score": r.get("water_score"),
            "elevation_score": r.get("elevation_score"),
            "sar_score": r.get("sar_score"),
            "ndvi_mean": r.get("ndvi_mean"),
            "mndwi_mean": r.get("mndwi_mean"),
            "cwi_mean": r.get("cwi_mean"),
            "water_fraction_mean": r.get("water_fraction_mean"),
            "alert_triggered": r.get("alert_triggered"),
            "polygon_count": r.get("polygon_count", 0),
            "stage_distribution": r.get("stage_distribution", {}),
            "gt_validation": r.get("gt_validation", {}),
            "gmw_validation": r.get("gmw_validation", {}),
            "assets": _build_image_assets(year) if year is not None else {},
        }
        records.append(record)
    stats = {
        "image_stats": records,
        "yearly_stats": records,
        "available_years": sorted(
            {r.get("year") for r in records if r.get("year") is not None}
        ),
        "stage_names": {str(k): v for k, v in STAGE_LABELS.items()},
        "stage_colors": {str(k): v for k, v in STAGE_COLORS_HEX.items()},
        "aoi": {
            "name": config.AOI["name"],
            "bbox": [
                config.AOI["lon_min"],
                config.AOI["lat_min"],
                config.AOI["lon_max"],
                config.AOI["lat_max"],
            ],
            "center_lat": config.WEB.get(
                "map_center_lat", (config.AOI["lat_min"] + config.AOI["lat_max"]) / 2
            ),
            "center_lon": config.WEB.get(
                "map_center_lon", (config.AOI["lon_min"] + config.AOI["lon_max"]) / 2
            ),
            "zoom": config.WEB.get("map_zoom", 13),
        },
    }
    path = os.path.join(config.WEB_DATA_DIR, "stats.json")
    os.makedirs(config.WEB_DATA_DIR, exist_ok=True)
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[M10] stats.json saved ({len(all_results)} records)")


def export_all_web_data(dummy_history, all_results):
    print("\n" + "=" * 60)
    print("[M10] WEB DATA EXPORT")
    print("=" * 60)
    generate_stats_data(all_results)
    print("[M10] Web export complete.")
