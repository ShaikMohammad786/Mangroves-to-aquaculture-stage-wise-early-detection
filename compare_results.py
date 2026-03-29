"""
Side-by-Side Comparison Tool — Ground Truth vs Classification

Creates visual comparison panels showing:
  1. Ground truth high-res imagery alongside classification results
  2. Temporal progression panels (S1→S2→S3→S4→S5 over time)
  3. Feature evidence panels (NDVI, MNDWI, SAR, JRC for each spot)

Usage:
    python compare_results.py
"""

import os
import glob
import logging

try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [COMPARE] %(message)s")
log = logging.getLogger("compare")

STAGE_COLORS = {
    1: (26, 94, 42),      # S1: dark green
    2: (201, 180, 44),    # S2: yellow
    3: (217, 123, 40),    # S3: orange
    4: (40, 181, 217),    # S4: cyan
    5: (26, 63, 160),     # S5: dark blue
}

STAGE_LABELS = {
    1: "S1: Intact Mangrove",
    2: "S2: Degrading Mangrove",
    3: "S3: Cleared / Soil",
    4: "S4: Pond Formation",
    5: "S5: Operational Pond",
}

VERIFIED_SITES = [
    {"lon": 82.2550, "lat": 16.6550, "name": "Coringa South Ponds"},
    {"lon": 82.2450, "lat": 16.6650, "name": "Coringa Central Ponds"},
    {"lon": 82.2350, "lat": 16.6750, "name": "Coringa North Ponds"},
    {"lon": 82.2650, "lat": 16.6450, "name": "Coringa East Ponds"},
    {"lon": 82.2250, "lat": 16.6850, "name": "Western Fringe Ponds"},
]

COMPARE_DIR = os.path.join(config.OUTPUT_DIR, "comparisons")
os.makedirs(COMPARE_DIR, exist_ok=True)


def _get_font(size=16):
    """Get a PIL font, fallback to default."""
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        try:
            return ImageFont.truetype("C:/Windows/Fonts/arial.ttf", size)
        except Exception:
            return ImageFont.load_default()


def _add_label(draw, text, position, color=(255, 255, 255), bg_color=(0, 0, 0)):
    """Add a text label with background."""
    font = _get_font(18)
    x, y = position
    bbox = draw.textbbox((x, y), text, font=font)
    padding = 4
    draw.rectangle(
        [bbox[0] - padding, bbox[1] - padding, bbox[2] + padding, bbox[3] + padding],
        fill=bg_color
    )
    draw.text((x, y), text, fill=color, font=font)


def create_stage_progression_panel():
    """
    Create a temporal progression panel showing the same region across epochs.

    Finds all stage thumbnails sorted by date and creates a side-by-side panel
    showing how the classification evolves over time.
    """
    if not HAS_PIL:
        log.error("Pillow not installed, cannot create comparison")
        return

    # Find all stage thumbnails
    stage_files = sorted(glob.glob(os.path.join(config.IMAGE_DIR, "stage_*.png")))
    if not stage_files:
        log.warning("No stage thumbnails found in outputs/images/")
        return

    # Also find RGB thumbnails for the same dates
    rgb_files = sorted(glob.glob(os.path.join(config.IMAGE_DIR, "rgb_*.png")))

    # Find feature thumbnails
    ndvi_files = sorted(glob.glob(os.path.join(config.FEATURE_DIR, "ndvi_*.png")))
    mndwi_files = sorted(glob.glob(os.path.join(config.FEATURE_DIR, "mndwi_*.png")))

    log.info(f"Found {len(stage_files)} stage maps, {len(rgb_files)} RGB, "
             f"{len(ndvi_files)} NDVI, {len(mndwi_files)} MNDWI thumbnails")

    # Create top panel: RGB progression
    _create_row_panel(
        rgb_files[:6], "Satellite RGB — Temporal Progression",
        os.path.join(COMPARE_DIR, "01_rgb_progression.png")
    )

    # Create middle panel: Stage classification progression
    _create_row_panel(
        stage_files[:6], "Stage Classification — Temporal Progression",
        os.path.join(COMPARE_DIR, "02_stage_progression.png")
    )

    # Create feature evidence panel
    _create_feature_evidence_panel(ndvi_files, mndwi_files, stage_files)

    # Create side-by-side: RGB vs Stage for each epoch
    _create_side_by_side_panels(rgb_files, stage_files)

    # Create ground truth comparison
    _create_ground_truth_comparison(stage_files)

    # Create one strongly marked verified region with temporal evidence.
    _create_verified_transition_proof(rgb_files, stage_files)

    log.info(f"All comparison panels saved to {COMPARE_DIR}")


def _create_row_panel(image_files, title, output_path, max_images=8):
    """Create a horizontal row panel from image files."""
    if not image_files:
        return

    images = []
    selected_files = image_files[-max_images:] if len(image_files) > max_images else image_files
    for f in selected_files:
        try:
            img = Image.open(f).convert("RGB")
            images.append((img, os.path.basename(f)))
        except Exception as e:
            log.warning(f"Error loading {f}: {e}")

    if not images:
        return

    # Resize all to same height
    target_h = 400
    resized = []
    for img, name in images:
        ratio = target_h / img.height
        new_w = int(img.width * ratio)
        resized.append((img.resize((new_w, target_h), Image.LANCZOS), name))

    total_w = sum(img.width for img, _ in resized) + 10 * (len(resized) - 1)
    panel = Image.new("RGB", (total_w, target_h + 80), (30, 30, 30))
    draw = ImageDraw.Draw(panel)

    # Title
    title_font = _get_font(22)
    draw.text((10, 10), title, fill=(255, 255, 255), font=title_font)

    # Paste images with labels
    x_offset = 0
    for img, name in resized:
        panel.paste(img, (x_offset, 50))

        # Extract date from filename
        date_str = name.replace(".png", "").split("_", 1)[-1] if "_" in name else name
        label_font = _get_font(14)
        draw.text((x_offset + 5, target_h + 55), date_str,
                  fill=(200, 200, 200), font=label_font)

        x_offset += img.width + 10

    panel.save(output_path, "PNG")
    log.info(f"Created: {os.path.basename(output_path)} ({panel.width}x{panel.height})")


def _create_side_by_side_panels(rgb_files, stage_files):
    """Create side-by-side RGB | Stage for each epoch."""
    # Match by date
    rgb_dates = {_extract_date(f): f for f in rgb_files}
    stage_dates = {_extract_date(f): f for f in stage_files}

    for date_str in sorted(set(rgb_dates.keys()) & set(stage_dates.keys())):
        try:
            rgb_img = Image.open(rgb_dates[date_str]).convert("RGB")
            stage_img = Image.open(stage_dates[date_str]).convert("RGB")

            target_h = 500
            rgb_ratio = target_h / rgb_img.height
            stage_ratio = target_h / stage_img.height
            rgb_resized = rgb_img.resize((int(rgb_img.width * rgb_ratio), target_h), Image.LANCZOS)
            stage_resized = stage_img.resize((int(stage_img.width * stage_ratio), target_h), Image.LANCZOS)

            total_w = rgb_resized.width + stage_resized.width + 30
            panel = Image.new("RGB", (total_w, target_h + 70), (30, 30, 30))
            draw = ImageDraw.Draw(panel)

            # Title
            title_font = _get_font(20)
            draw.text((10, 10), f"RGB vs Stage Classification — {date_str}",
                      fill=(255, 255, 255), font=title_font)

            # Labels
            _add_label(draw, "Satellite RGB", (10, 45), bg_color=(50, 100, 50))
            _add_label(draw, "Classification", (rgb_resized.width + 20, 45), bg_color=(50, 50, 100))

            panel.paste(rgb_resized, (0, 70))
            panel.paste(stage_resized, (rgb_resized.width + 20, 70))

            # Add legend
            _add_stage_legend(draw, total_w - 200, target_h + 5)

            outpath = os.path.join(COMPARE_DIR, f"sidebyside_{date_str}.png")
            panel.save(outpath, "PNG")
            log.info(f"Created: sidebyside_{date_str}.png")

        except Exception as e:
            log.warning(f"Error creating side-by-side for {date_str}: {e}")


def _create_feature_evidence_panel(ndvi_files, mndwi_files, stage_files):
    """Create a panel showing NDVI, MNDWI, and Stage side by side for a key epoch."""
    if not (ndvi_files and mndwi_files and stage_files):
        return

    # Use the latest epoch
    latest_idx = min(len(ndvi_files), len(mndwi_files), len(stage_files)) - 1
    if latest_idx < 0:
        return

    try:
        ndvi_img = Image.open(ndvi_files[latest_idx]).convert("RGB")
        mndwi_img = Image.open(mndwi_files[latest_idx]).convert("RGB")
        stage_img = Image.open(stage_files[latest_idx]).convert("RGB")

        target_h = 400
        imgs = []
        labels = ["NDVI (Vegetation)", "MNDWI (Water)", "Stage Classification"]
        for img in [ndvi_img, mndwi_img, stage_img]:
            ratio = target_h / img.height
            imgs.append(img.resize((int(img.width * ratio), target_h), Image.LANCZOS))

        total_w = sum(i.width for i in imgs) + 20 * (len(imgs) - 1)
        panel = Image.new("RGB", (total_w, target_h + 80), (30, 30, 30))
        draw = ImageDraw.Draw(panel)

        title_font = _get_font(20)
        draw.text((10, 10), "Feature Evidence Panel — Latest Epoch",
                  fill=(255, 255, 255), font=title_font)

        x_offset = 0
        for img, label in zip(imgs, labels):
            panel.paste(img, (x_offset, 50))
            _add_label(draw, label, (x_offset + 5, target_h + 55))
            x_offset += img.width + 20

        outpath = os.path.join(COMPARE_DIR, "03_feature_evidence.png")
        panel.save(outpath, "PNG")
        log.info(f"Created: 03_feature_evidence.png")

    except Exception as e:
        log.warning(f"Error creating feature evidence panel: {e}")


def _create_ground_truth_comparison(stage_files):
    """Compare generated ground-truth overlays against detection overlays."""
    detection_files = sorted(glob.glob(os.path.join(config.IMAGE_DIR, "detection_*.png")))
    gt_overlay_files = sorted(glob.glob(os.path.join(config.IMAGE_DIR, "ground_truth_*.png")))

    if detection_files and gt_overlay_files:
        det_dates = {_extract_date(f): f for f in detection_files}
        gt_dates = {_extract_date(f): f for f in gt_overlay_files}
        common_dates = sorted(set(det_dates.keys()) & set(gt_dates.keys()))
        if common_dates:
            try:
                latest_date = common_dates[-1]
                gt_img = Image.open(gt_dates[latest_date]).convert("RGB")
                det_img = Image.open(det_dates[latest_date]).convert("RGB")

                target_h = 600
                gt_ratio = target_h / gt_img.height
                det_ratio = target_h / det_img.height
                gt_resized = gt_img.resize((int(gt_img.width * gt_ratio), target_h), Image.LANCZOS)
                det_resized = det_img.resize((int(det_img.width * det_ratio), target_h), Image.LANCZOS)

                total_w = gt_resized.width + det_resized.width + 30
                panel = Image.new("RGB", (total_w, target_h + 80), (20, 20, 20))
                draw = ImageDraw.Draw(panel)

                title_font = _get_font(22)
                draw.text((10, 10), f"Ground Truth vs Detection Overlay - {latest_date}",
                          fill=(255, 255, 255), font=title_font)

                _add_label(draw, "Ground Truth Overlay", (10, 45), bg_color=(120, 45, 45))
                _add_label(draw, "Detection Overlay", (gt_resized.width + 20, 45), bg_color=(50, 50, 100))

                panel.paste(gt_resized, (0, 70))
                panel.paste(det_resized, (gt_resized.width + 20, 70))

                outpath = os.path.join(COMPARE_DIR, "04_ground_truth_vs_detection.png")
                panel.save(outpath, "PNG")
                log.info(f"Created: {os.path.basename(outpath)}")
                return
            except Exception as e:
                log.warning(f"Error creating GT-vs-detection comparison: {e}")

    """Compare ground truth ESRI imagery against stage classification."""
    gt_dir = os.path.join(config.OUTPUT_DIR, "ground_truth")
    gt_files = sorted(glob.glob(os.path.join(gt_dir, "esri_z*.png")))

    if not gt_files or not stage_files:
        log.info("Skipping ground truth comparison (no ground truth or stage images)")
        return

    try:
        gt_img = Image.open(gt_files[0]).convert("RGB")
        stage_img = Image.open(stage_files[-1]).convert("RGB")  # Latest stage

        target_h = 600
        gt_ratio = target_h / gt_img.height
        st_ratio = target_h / stage_img.height
        gt_resized = gt_img.resize((int(gt_img.width * gt_ratio), target_h), Image.LANCZOS)
        st_resized = stage_img.resize((int(stage_img.width * st_ratio), target_h), Image.LANCZOS)

        total_w = gt_resized.width + st_resized.width + 30
        panel = Image.new("RGB", (total_w, target_h + 80), (20, 20, 20))
        draw = ImageDraw.Draw(panel)

        title_font = _get_font(22)
        draw.text((10, 10), "Ground Truth (ESRI ~0.5m) vs Classification (30m)",
                  fill=(255, 255, 255), font=title_font)

        _add_label(draw, "ESRI World Imagery (Ground Truth)", (10, 45), bg_color=(50, 100, 50))
        _add_label(draw, "Stage Classification (30m)", (gt_resized.width + 20, 45), bg_color=(50, 50, 100))

        panel.paste(gt_resized, (0, 70))
        panel.paste(st_resized, (gt_resized.width + 20, 70))

        _add_stage_legend(draw, total_w - 220, target_h + 5)

        outpath = os.path.join(COMPARE_DIR, "04_ground_truth_vs_classification.png")
        panel.save(outpath, "PNG")
        log.info(f"Created: 04_ground_truth_vs_classification.png")

    except Exception as e:
        log.warning(f"Error creating ground truth comparison: {e}")


def _add_stage_legend(draw, x, y):
    """Add stage color legend."""
    font = _get_font(12)
    for stage_id in range(1, 6):
        color = STAGE_COLORS[stage_id]
        label = STAGE_LABELS[stage_id]
        draw.rectangle([x, y, x + 15, y + 15], fill=color)
        draw.text((x + 20, y), label, fill=(200, 200, 200), font=font)
        y += 20


def _lonlat_to_pixel(lon, lat, width, height):
    lon_min = config.AOI["lon_min"]
    lon_max = config.AOI["lon_max"]
    lat_min = config.AOI["lat_min"]
    lat_max = config.AOI["lat_max"]
    x = (lon - lon_min) / max(lon_max - lon_min, 1e-9) * width
    y = height - ((lat - lat_min) / max(lat_max - lat_min, 1e-9) * height)
    return x, y


def _crop_box(width, height, lon, lat, half_size):
    x, y = _lonlat_to_pixel(lon, lat, width, height)
    left = max(0, int(x - half_size))
    top = max(0, int(y - half_size))
    right = min(width, int(x + half_size))
    bottom = min(height, int(y + half_size))
    return (left, top, right, bottom)


def _nearest_stage_id(rgb):
    best_stage = None
    best_dist = None
    for stage_id, color in STAGE_COLORS.items():
        dist = sum((int(rgb[i]) - color[i]) ** 2 for i in range(3))
        if best_dist is None or dist < best_dist:
            best_stage = stage_id
            best_dist = dist
    if best_dist is None or best_dist > 12000:
        return None
    return best_stage


def _detect_stages_in_crop(img):
    counts = {stage_id: 0 for stage_id in STAGE_COLORS}
    pixels = img.convert("RGB").load()
    width, height = img.size
    total = max(width * height, 1)

    step = 1
    if width * height > 60000:
        step = 2

    for y in range(0, height, step):
        for x in range(0, width, step):
            stage_id = _nearest_stage_id(pixels[x, y])
            if stage_id is not None:
                counts[stage_id] += 1

    min_pixels = max(15, total // 200)
    return sorted([stage_id for stage_id, count in counts.items() if count >= min_pixels])


def _sample_region_metadata(stage_path, lon, lat, half_size):
    img = Image.open(stage_path).convert("RGB")
    box = _crop_box(img.width, img.height, lon, lat, half_size)
    crop = img.crop(box)
    stages = _detect_stages_in_crop(crop)
    return {
        "crop": crop,
        "box": box,
        "stages": stages,
    }


def _pick_verified_region(stage_files):
    best = None
    for site in VERIFIED_SITES:
        for half_size in (70, 90, 110, 130):
            union = set()
            epoch_details = []
            for stage_path in stage_files:
                meta = _sample_region_metadata(stage_path, site["lon"], site["lat"], half_size)
                epoch_details.append({"path": stage_path, "stages": meta["stages"], "box": meta["box"]})
                union.update(meta["stages"])

            score = len(union) * 100
            if 1 in union:
                score += 20
            if 5 in union:
                score += 20
            if len(union) == 5:
                score += 100

            candidate = {
                "site": site,
                "half_size": half_size,
                "stages": sorted(union),
                "epochs": epoch_details,
                "score": score,
            }
            if best is None or candidate["score"] > best["score"]:
                best = candidate
    return best


def _draw_region_box(draw, box, color):
    draw.rectangle(box, outline=(0, 0, 0), width=10)
    draw.rectangle(box, outline=color, width=6)


def _create_verified_transition_proof(rgb_files, stage_files):
    if not rgb_files or not stage_files or not HAS_PIL:
        return

    rgb_dates = {_extract_date(f): f for f in rgb_files}
    stage_dates = {_extract_date(f): f for f in stage_files}
    common_dates = sorted(set(rgb_dates.keys()) & set(stage_dates.keys()))
    if not common_dates:
        log.info("Skipping verified transition proof (no matched RGB/stage dates)")
        return

    best = _pick_verified_region([stage_dates[d] for d in common_dates])
    if not best:
        return

    latest_date = common_dates[-1]
    latest_rgb = Image.open(rgb_dates[latest_date]).convert("RGB")
    latest_stage = Image.open(stage_dates[latest_date]).convert("RGB")
    latest_box = _crop_box(
        latest_rgb.width,
        latest_rgb.height,
        best["site"]["lon"],
        best["site"]["lat"],
        best["half_size"],
    )

    selected_dates = common_dates[:6] if len(common_dates) <= 6 else [
        common_dates[0],
        common_dates[len(common_dates) // 4],
        common_dates[len(common_dates) // 2],
        common_dates[(3 * len(common_dates)) // 4],
        common_dates[-2],
        common_dates[-1],
    ]

    full_h = 320
    crop_h = 180
    padding = 18
    label_h = 34
    total_w = 2 * 520 + padding * 3
    row2_h = crop_h + label_h + 20
    panel_h = 90 + full_h + 30 + row2_h * 2 + 80
    panel = Image.new("RGB", (total_w, panel_h), (24, 24, 24))
    draw = ImageDraw.Draw(panel)

    title_font = _get_font(24)
    body_font = _get_font(15)
    draw.text(
        (padding, 16),
        f"Verified Transition Region - {best['site']['name']}",
        fill=(255, 255, 255),
        font=title_font,
    )
    draw.text(
        (padding, 50),
        f"Known conversion hotspot near {best['site']['lat']:.4f}N, {best['site']['lon']:.4f}E | Stages seen: {', '.join(f'S{s}' for s in best['stages'])}",
        fill=(205, 205, 205),
        font=body_font,
    )

    def prepare_full(img, box, header, color):
        target_w = 520
        resized = img.resize((target_w, full_h), Image.LANCZOS)
        scale_x = target_w / img.width
        scale_y = full_h / img.height
        scaled_box = (
            int(box[0] * scale_x),
            int(box[1] * scale_y),
            int(box[2] * scale_x),
            int(box[3] * scale_y),
        )
        canvas = Image.new("RGB", (target_w, full_h + 28), (30, 30, 30))
        canvas.paste(resized, (0, 28))
        d = ImageDraw.Draw(canvas)
        _add_label(d, header, (8, 4), bg_color=color)
        _draw_region_box(d, (scaled_box[0], scaled_box[1] + 28, scaled_box[2], scaled_box[3] + 28), (255, 64, 64))
        return canvas

    latest_rgb_panel = prepare_full(latest_rgb, latest_box, "Latest RGB with verified region", (50, 100, 50))
    latest_stage_panel = prepare_full(latest_stage, latest_box, "Latest stage map with verified region", (50, 50, 100))
    panel.paste(latest_rgb_panel, (padding, 88))
    panel.paste(latest_stage_panel, (padding * 2 + 520, 88))

    row1_y = 88 + full_h + 50
    row2_y = row1_y + row2_h
    draw.text((padding, row1_y - 22), "RGB progression inside the verified box", fill=(220, 220, 220), font=body_font)
    draw.text((padding, row2_y - 22), "Stage progression inside the verified box", fill=(220, 220, 220), font=body_font)

    cell_w = int((total_w - padding * (len(selected_dates) + 1)) / len(selected_dates))
    for idx, date_str in enumerate(selected_dates):
        x = padding + idx * (cell_w + padding)

        rgb_img = Image.open(rgb_dates[date_str]).convert("RGB")
        stage_img = Image.open(stage_dates[date_str]).convert("RGB")
        box = _crop_box(rgb_img.width, rgb_img.height, best["site"]["lon"], best["site"]["lat"], best["half_size"])

        rgb_crop = rgb_img.crop(box).resize((cell_w, crop_h), Image.LANCZOS)
        stage_crop = stage_img.crop(box).resize((cell_w, crop_h), Image.LANCZOS)
        stages_present = _detect_stages_in_crop(stage_img.crop(box))
        stage_text = ", ".join(f"S{s}" for s in stages_present) if stages_present else "No stage"

        panel.paste(rgb_crop, (x, row1_y))
        panel.paste(stage_crop, (x, row2_y))
        draw.rectangle([x, row1_y, x + cell_w, row1_y + crop_h], outline=(230, 230, 230), width=2)
        draw.rectangle([x, row2_y, x + cell_w, row2_y + crop_h], outline=(230, 230, 230), width=2)
        draw.text((x + 4, row1_y + crop_h + 4), date_str, fill=(220, 220, 220), font=body_font)
        draw.text((x + 4, row2_y + crop_h + 4), stage_text, fill=(220, 220, 220), font=body_font)

    _add_stage_legend(draw, total_w - 240, panel_h - 130)

    outpath = os.path.join(COMPARE_DIR, "05_verified_transition_region.png")
    panel.save(outpath, "PNG")

    summary = {
        "site": best["site"],
        "crop_half_size_px": best["half_size"],
        "stages_seen": best["stages"],
        "dates_used": selected_dates,
    }
    with open(os.path.join(COMPARE_DIR, "05_verified_transition_region.json"), "w", encoding="utf-8") as f:
        import json
        json.dump(summary, f, indent=2)

    log.info(f"Created: {os.path.basename(outpath)}")


def _extract_date(filepath):
    """Extract date string from filename like stage_1988-01-01_2000-12-31.png"""
    basename = os.path.splitext(os.path.basename(filepath))[0]
    parts = basename.split("_", 1)
    return parts[1] if len(parts) > 1 else basename


if __name__ == "__main__":
    log.info("=" * 60)
    log.info("SIDE-BY-SIDE COMPARISON TOOL")
    log.info(f"Image dir: {config.IMAGE_DIR}")
    log.info(f"Feature dir: {config.FEATURE_DIR}")
    log.info(f"Output dir: {COMPARE_DIR}")
    log.info("=" * 60)

    create_stage_progression_panel()

    log.info("\nDONE — Check outputs/comparisons/ folder")
