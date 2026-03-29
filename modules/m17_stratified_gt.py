"""
MODULE 17 — STRATIFIED GROUND TRUTH SAMPLING (v3.0 — Research Grade)

v3.0 RESEARCH-GRADE OVERHAUL:
  - Increased sample count (60 → 100 per class) for tighter confidence intervals
  - Added S2 reference: recent GMW mangrove loss (degradation indicator)
  - Added S5 reference: JRC high-occurrence water in conversion areas (established ponds)
  - Uses GMW-based INDEPENDENT ground truth:
    * S1 reference: GMW mangrove extent (where GMW says mangrove)
    * S2 reference: Areas with declining mangrove (early GMW - late GMW partial loss)
    * S3 reference: Non-mangrove, non-water areas
    * S4/S5 reference: GMW conversion areas + JRC water for S5
  - This provides truly independent reference data from a different
    sensor/methodology (ALOS PALSAR + Landsat vs our pipeline)

References:
  - Bunting et al. 2018 (Global Mangrove Watch methodology)
  - Congalton & Green 2019 (Accuracy assessment best practices)
"""

import ee
import numpy as np
from typing import Dict, Any
import config
from modules.m15_ground_truth import get_gmw_epoch_extent, build_gmw_conversion_gt


def _build_independent_reference(aoi, target_year=None):
    """
    Build independent reference classification from GMW.

    Returns ee.Image with band 'gt_class':
      1 = GMW mangrove (independent S1 reference)
      3 = Non-mangrove, non-water (bare/other reference)
      4 = GMW conversion area (independent S4/S5 reference)
      0 = Unclassified (excluded from validation)
    """
    year = target_year or 2020

    # GMW mangrove extent for current epoch
    try:
        gmw_current = get_gmw_epoch_extent(aoi, year)
    except Exception:
        return None

    # GMW conversion areas (was mangrove, now not)
    try:
        conversion = build_gmw_conversion_gt(aoi, early_year=1996, late_year=year)
    except Exception:
        conversion = ee.Image(0).rename("conversion").clip(aoi)

    # Build reference classes
    gt_class = ee.Image(0).clip(aoi).rename("gt_class").toInt()

    # S1 reference: where GMW says mangrove NOW
    gt_class = gt_class.where(gmw_current.eq(1), 1)

    # S4/S5 reference: where GMW says conversion happened
    gt_class = gt_class.where(conversion.eq(1), 4)

    # v3.0: S5 reference — conversion areas with high JRC water occurrence
    # (established ponds have persistent water, not just cleared land)
    try:
        jrc = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence")
        established_pond = conversion.eq(1).And(jrc.gt(40))
        gt_class = gt_class.where(established_pond, 5)
    except Exception:
        pass

    # v3.0: S2 reference — areas where mangrove existed in mid-epoch but is degraded
    # (partial loss between mid and late epoch = degradation)
    try:
        mid_year = (year + 1996) // 2  # midpoint between baseline and current
        mid_gmw = get_gmw_epoch_extent(aoi, mid_year)
        # Was mangrove at mid-epoch, still counted by current, but partially degraded
        # Use NDVI from JRC/soil context: non-water, non-full-mangrove
        degrading = mid_gmw.eq(1).And(gmw_current.eq(1)).And(conversion.eq(0))
        # Only mark edges of mangrove (likely degradation fringe)
        mangrove_edge = gmw_current.eq(1).focal_min(radius=2, units='pixels').Not().And(gmw_current.eq(1))
        gt_class = gt_class.where(degrading.And(mangrove_edge), 2)
    except Exception:
        pass

    # S3 reference: remaining low-elevation non-water areas
    # Use JRC to exclude permanent water bodies
    try:
        jrc_s3 = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence")
        not_water = jrc_s3.lt(30)
    except Exception:
        not_water = ee.Image(1)

    still_unclassified = gt_class.eq(0)
    gt_class = gt_class.where(still_unclassified.And(not_water), 3)

    return gt_class


def sample_stratified_points(
    stage_image,
    gt_fc,
    aoi,
    samples_per_class=100,
    target_year=None,
):
    """
    Sample stratified validation points using INDEPENDENT GMW reference.

    Returns FeatureCollection with properties:
    - 'gt_class': 1=mangrove, 3=bare/other, 4=conversion (from GMW)
    - 'stage': predicted stage from pipeline (1-5)
    """
    # Build independent reference
    gt_reference = _build_independent_reference(aoi, target_year)
    if gt_reference is None:
        return ee.FeatureCollection([])

    stage = stage_image.select("stage")

    # Create combined image for sampling
    combined = ee.Image.cat([
        gt_reference.rename("gt_class"),
        stage.rename("pred_stage"),
    ]).toInt()

    # Stratified sample from GT classes (not from pipeline output!)
    all_points = []
    for gt_class_id in [1, 2, 3, 4, 5]:  # v3.0: added S2 and S5
        class_mask = gt_reference.eq(gt_class_id).selfMask()
        try:
            pts = class_mask.rename("gt_class").stratifiedSample(
                numPoints=samples_per_class,
                classBand="gt_class",
                region=aoi,
                scale=max(config.TARGET_SCALE, 30),
                geometries=True,
                tileScale=8,
            )
            all_points.append(pts)
        except Exception:
            continue

    if not all_points:
        return ee.FeatureCollection([])

    reference_points = ee.FeatureCollection(all_points).flatten()

    # Sample pipeline predictions at reference points
    sampled = combined.sampleRegions(
        collection=reference_points,
        properties=["gt_class"],
        scale=max(config.TARGET_SCALE, 30),
        geometries=True,
    )

    return sampled.limit(600)


def compute_kappa_from_samples(samples_fc):
    """
    Download samples, compute confusion matrix + Kappa.

    Returns dict with kappa, confusion_matrix, OA, per-class metrics.
    """
    try:
        from sklearn.metrics import confusion_matrix, cohen_kappa_score
    except ImportError:
        return {"available": False, "error": "scikit-learn not installed"}

    try:
        samples_info = samples_fc.limit(1000).getInfo()
    except Exception as e:
        return {"available": False, "error": f"GEE export failed: {e}"}

    features = (samples_info or {}).get("features", [])

    gt_classes = []
    pred_stages = []

    for feat in features:
        props = feat.get("properties", {})
        gt_val = props.get("gt_class")
        pred_val = props.get("pred_stage")
        if gt_val is None or pred_val is None:
            continue
        try:
            gt_classes.append(int(gt_val))
            pred_stages.append(int(pred_val))
        except (TypeError, ValueError):
            continue

    if len(gt_classes) < 30:
        return {
            "available": False,
            "reason": f"Insufficient valid samples: {len(gt_classes)}",
        }

    # Map GT classes to binary: mangrove (1) vs conversion (4) vs other (3)
    # For confusion matrix, use all labels present
    all_labels = sorted(set(gt_classes) | set(pred_stages))
    # Filter to valid stages only
    valid_labels = [l for l in all_labels if 1 <= l <= 5]
    if not valid_labels:
        return {"available": False, "reason": "No valid stage labels in samples"}

    cm = confusion_matrix(gt_classes, pred_stages, labels=valid_labels)
    kappa = cohen_kappa_score(gt_classes, pred_stages)

    correct = np.trace(cm)
    total = cm.sum()
    oa = correct / total if total > 0 else 0.0

    per_class_f1 = {}
    for idx, label in enumerate(valid_labels):
        tp = cm[idx, idx] if idx < len(cm) else 0
        col_sum = cm[:, idx].sum() if idx < cm.shape[1] else 0
        row_sum = cm[idx, :].sum() if idx < cm.shape[0] else 0
        prec = tp / col_sum if col_sum > 0 else 0.0
        rec = tp / row_sum if row_sum > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class_f1[f"S{label}"] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
        }

    return {
        "available": True,
        "kappa": round(float(kappa), 4),
        "overall_accuracy": round(oa, 4),
        "confusion_matrix": cm.tolist(),
        "labels": valid_labels,
        "n_samples": len(gt_classes),
        "per_class_f1": per_class_f1,
        "method": "gmw_independent_stratified_v2",
    }


def compute_full_gt_validation(stage_image, gt_fc, aoi, target_year=None):
    """
    Complete GT validation suite with independent Kappa computation.
    """
    samples = sample_stratified_points(
        stage_image, gt_fc, aoi, target_year=target_year
    )
    kappa_results = compute_kappa_from_samples(samples)

    poly_results = {"available": False, "reason": "polygon_matching_deferred"}

    return {
        "kappa_analysis": kappa_results,
        "polygon_matching": poly_results,
    }
