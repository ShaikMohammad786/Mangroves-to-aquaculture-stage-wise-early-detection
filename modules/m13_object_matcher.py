"""
MODULE 13 - Object Matcher

Fast spatially indexed pond matching with temporal-cluster context.

Key improvements over centroid-only matching:
  - Spatial hashing reduces the number of pairwise distance checks.
  - Matching uses distance + area similarity + shape similarity + bbox overlap.
  - Each candidate receives nearby historical cluster support before classification.
"""

from math import radians, cos, sin, asin, sqrt
import config


def haversine(lon1, lat1, lon2, lat2):
    """Great-circle distance in meters."""
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371000
    return c * r


def _safe_float(value, default=0.0):
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _approx_xy_m(lon, lat, ref_lat):
    meters_per_deg_lat = 111320.0
    meters_per_deg_lon = meters_per_deg_lat * cos(radians(ref_lat))
    return lon * meters_per_deg_lon, lat * meters_per_deg_lat


def _grid_key(lon, lat, bin_m, ref_lat):
    x, y = _approx_xy_m(lon, lat, ref_lat)
    return (int(x // max(bin_m, 1.0)), int(y // max(bin_m, 1.0)))


def _iter_rings(geom_dict):
    if not isinstance(geom_dict, dict):
        return
    gtype = geom_dict.get("type")
    coords = geom_dict.get("coordinates") or []
    if gtype == "Polygon":
        for ring in coords:
            yield ring
    elif gtype == "MultiPolygon":
        for polygon in coords:
            for ring in polygon:
                yield ring


def _geom_bbox(geom_dict):
    xs = []
    ys = []
    for ring in _iter_rings(geom_dict):
        for coord in ring:
            if isinstance(coord, (list, tuple)) and len(coord) >= 2:
                xs.append(_safe_float(coord[0], None))
                ys.append(_safe_float(coord[1], None))
    if not xs or not ys:
        return None
    return (min(xs), min(ys), max(xs), max(ys))


def _bbox_area(bbox):
    if not bbox:
        return 0.0
    width = max(0.0, bbox[2] - bbox[0])
    height = max(0.0, bbox[3] - bbox[1])
    return width * height


def _bbox_iou(bbox_a, bbox_b):
    if not bbox_a or not bbox_b:
        return 0.0
    ix0 = max(bbox_a[0], bbox_b[0])
    iy0 = max(bbox_a[1], bbox_b[1])
    ix1 = min(bbox_a[2], bbox_b[2])
    iy1 = min(bbox_a[3], bbox_b[3])
    inter = _bbox_area((ix0, iy0, ix1, iy1))
    if inter <= 0:
        return 0.0
    union = _bbox_area(bbox_a) + _bbox_area(bbox_b) - inter
    if union <= 0:
        return 0.0
    return inter / union


def _quick_bbox_distance_check(bbox_a, bbox_b, threshold_m, ref_lat=16.0):
    """
    Quick bounding box distance check before expensive haversine.
    Returns True if centroids could possibly be within threshold.
    """
    if not bbox_a or not bbox_b:
        return True  # Allow if bbox missing

    # Get bbox centers
    center_lon_a = (bbox_a[0] + bbox_a[2]) / 2
    center_lat_a = (bbox_a[1] + bbox_a[3]) / 2
    center_lon_b = (bbox_b[0] + bbox_b[2]) / 2
    center_lat_b = (bbox_b[1] + bbox_b[3]) / 2

    # Rough distance estimate using approximation
    meters_per_deg_lat = 111320.0
    meters_per_deg_lon = meters_per_deg_lat * cos(radians(ref_lat))

    dx = abs(center_lon_a - center_lon_b) * meters_per_deg_lon
    dy = abs(center_lat_a - center_lat_b) * meters_per_deg_lat

    # Quick Euclidean estimate
    rough_dist = sqrt(dx * dx + dy * dy)

    # Allow if within 1.5x threshold (generous for diagonal cases)
    return rough_dist <= threshold_m * 1.5


def _estimate_polygon_overlap(cand, record, dist_m):
    """
    Estimate polygon overlap score based on centroid distance and area similarity.
    Returns 0-1 score where 1 = high overlap, 0 = no overlap.

    v4.0: Uses bbox overlap ratio + area similarity as proxy for true polygon IoU.
    """
    cand_bbox = _geom_bbox(cand.get("geometry"))
    record_bbox = record.get("bbox")

    if not cand_bbox or not record_bbox:
        return 0.5  # Neutral if no bbox

    # Bbox IoU as proxy for polygon overlap
    bbox_overlap = _bbox_iou(cand_bbox, record_bbox)

    # Area similarity
    cand_area = _safe_float(cand.get("area_m2"), 0.0)
    record_area = _safe_float(record.get("area_m2"), 0.0)

    if cand_area <= 0 or record_area <= 0:
        area_sim = 0.5
    else:
        area_sim = min(cand_area, record_area) / max(cand_area, record_area)

    # Distance factor: closer = more likely to overlap
    # Typical pond ~70m diameter, so distance > 50m means unlikely overlap
    dist_factor = max(0, 1 - dist_m / 100.0)

    # Combined score: bbox_overlap is best proxy for polygon overlap
    # Weight: bbox_iou (0.5) + area_sim (0.3) + dist_factor (0.2)
    overlap_score = bbox_overlap * 0.5 + area_sim * 0.3 + dist_factor * 0.2

    return min(1.0, max(0.0, overlap_score))


def _latest_area(pond_data):
    area_history = pond_data.get("area_history") or []
    if area_history:
        return _safe_float(area_history[-1].get("area_m2"), 0.0)
    return _safe_float(pond_data.get("area_m2"), 0.0)


def _latest_rectangularity(pond_data):
    return _safe_float(pond_data.get("rectangularity"), 0.0)


def _latest_stage_probability(pond_data):
    history = pond_data.get("stage_history") or []
    if history:
        latest = history[-1]
        value = latest.get("stage_probability")
        if value is not None:
            return _safe_float(value, 0.0)
        return _safe_float(latest.get("confidence"), 0.0)
    return _safe_float(pond_data.get("confidence"), 0.0)


def _latest_confirmed_stage(pond_data):
    stage = pond_data.get("confirmed_stage")
    if stage is not None:
        try:
            return int(stage)
        except Exception:
            return 0
    history = pond_data.get("stage_history") or []
    if history:
        try:
            return int(history[-1].get("confirmed_stage") or history[-1].get("raw_stage") or 0)
        except Exception:
            return 0
    return 0


def _neighbor_cells(key, radius_cells=1):
    gx, gy = key
    for dx in range(-radius_cells, radius_cells + 1):
        for dy in range(-radius_cells, radius_cells + 1):
            yield (gx + dx, gy + dy)


def _build_registry_records(registry_ponds, ref_lat, bin_m):
    records = []
    index = {}
    for pond_id, pond_data in registry_ponds.items():
        centroid = pond_data.get("centroid") or []
        if len(centroid) < 2:
            continue
        lat = _safe_float(centroid[0], None)
        lon = _safe_float(centroid[1], None)
        if lat is None or lon is None:
            continue
        bbox = pond_data.get("bbox") or _geom_bbox(pond_data.get("geometry"))
        record = {
            "pond_id": pond_id,
            "lat": lat,
            "lon": lon,
            "area_m2": _latest_area(pond_data),
            "rectangularity": _latest_rectangularity(pond_data),
            "bbox": bbox,
            "confirmed_stage": _latest_confirmed_stage(pond_data),
            "stage_probability": _latest_stage_probability(pond_data),
        }
        records.append(record)
        key = _grid_key(lon, lat, bin_m, ref_lat)
        index.setdefault(key, []).append(record)
    return records, index


def _area_similarity(area_a, area_b):
    if area_a <= 0 or area_b <= 0:
        return 0.5
    return max(0.0, min(area_a, area_b) / max(area_a, area_b))


def _shape_similarity(rect_a, rect_b):
    if rect_a <= 0 or rect_b <= 0:
        return 0.5
    return max(0.0, 1.0 - abs(rect_a - rect_b))


def _cluster_summary(nearby_records):
    if not nearby_records:
        return {
            "cluster_neighbor_count": 0,
            "cluster_recent_water_ratio": 0.0,
            "cluster_recent_conversion_ratio": 0.0,
            "cluster_modal_stage": 0,
        }

    stages = [_latest_confirmed_stage(record) if isinstance(record, dict) and "pond_id" not in record else int(record.get("confirmed_stage", 0) or 0) for record in nearby_records]
    stages = [stage for stage in stages if stage > 0]
    if not stages:
        return {
            "cluster_neighbor_count": len(nearby_records),
            "cluster_recent_water_ratio": 0.0,
            "cluster_recent_conversion_ratio": 0.0,
            "cluster_modal_stage": 0,
        }

    hist = {}
    for stage in stages:
        hist[stage] = hist.get(stage, 0) + 1

    water_count = sum(1 for stage in stages if stage >= 4)
    conversion_count = sum(1 for stage in stages if stage >= 2)
    modal_stage = max(hist, key=hist.get)

    return {
        "cluster_neighbor_count": len(nearby_records),
        "cluster_recent_water_ratio": round(water_count / len(stages), 4),
        "cluster_recent_conversion_ratio": round(conversion_count / len(stages), 4),
        "cluster_modal_stage": int(modal_stage),
    }


def match_polygons_to_registry(candidates, registry_ponds, distance_threshold_m=50.0):
    """
    Match current candidates to registry ponds using a fast spatial index and
    a multi-criterion similarity score.
    """
    cfg = getattr(config, "OBJECT_MATCHING", {})
    distance_threshold_m = _safe_float(distance_threshold_m, 0.0) or _safe_float(cfg.get("distance_threshold_m"), 120.0)
    bin_m = _safe_float(cfg.get("spatial_bin_m"), max(distance_threshold_m * 1.5, 180.0))
    cluster_radius_m = _safe_float(cfg.get("cluster_radius_m"), max(distance_threshold_m * 2.0, 240.0))
    max_area_ratio = max(1.0, _safe_float(cfg.get("max_area_ratio"), 6.0))
    distance_weight = _safe_float(cfg.get("distance_weight"), 0.45)
    area_weight = _safe_float(cfg.get("area_weight"), 0.20)
    shape_weight = _safe_float(cfg.get("shape_weight"), 0.15)
    bbox_weight = _safe_float(cfg.get("bbox_weight"), 0.20)

    ref_lat = _safe_float(config.AOI.get("lat_min"), 0.0) + (
        _safe_float(config.AOI.get("lat_max"), 0.0) - _safe_float(config.AOI.get("lat_min"), 0.0)
    ) / 2.0

    _, registry_index = _build_registry_records(registry_ponds, ref_lat, bin_m)

    candidate_matches = []
    nearby_matches_by_candidate = {}
    nearby_candidates_by_pond = {}
    unmatched = []

    for idx, cand in enumerate(candidates):
        cand_lon = cand.get("centroid_lon")
        cand_lat = cand.get("centroid_lat")
        if cand_lon is None or cand_lat is None:
            unmatched.append(cand)
            continue

        cand_lon = _safe_float(cand_lon, None)
        cand_lat = _safe_float(cand_lat, None)
        if cand_lon is None or cand_lat is None:
            unmatched.append(cand)
            continue

        cand_area = _safe_float(cand.get("area_m2"), 0.0)
        cand_rect = _safe_float(cand.get("rectangularity"), 0.0)
        cand_bbox = _geom_bbox(cand.get("geometry"))
        key = _grid_key(cand_lon, cand_lat, bin_m, ref_lat)
        cluster_neighbors = []
        local_registry = []

        radius_cells = max(1, int(cluster_radius_m // max(bin_m, 1.0)) + 1)
        for n_key in _neighbor_cells(key, radius_cells):
            for record in registry_index.get(n_key, []):
                # v4.0: Quick bbox pre-filter before expensive haversine
                if not _quick_bbox_distance_check(cand_bbox, record.get("bbox"), distance_threshold_m, ref_lat):
                    continue  # Skip expensive haversine calculation

                dist = haversine(cand_lon, cand_lat, record["lon"], record["lat"])
                if dist <= cluster_radius_m:
                    cluster_neighbors.append(record)
                if dist <= distance_threshold_m:
                    local_registry.append((record, dist))

        cand.update(_cluster_summary(cluster_neighbors))

        if not local_registry:
            unmatched.append(cand)
            continue

        found_any = False
        for record, dist in local_registry:
            area_ratio = _area_similarity(cand_area, record["area_m2"])
            shape_score = _shape_similarity(cand_rect, record["rectangularity"])
            bbox_score = _bbox_iou(cand_bbox, record["bbox"])

            if area_ratio < (1.0 / max_area_ratio):
                continue

            # v4.0: Add estimated polygon overlap score
            polygon_overlap = _estimate_polygon_overlap(cand, record, dist)

            dist_score = max(0.0, 1.0 - dist / max(distance_threshold_m, 1.0))

            # v4.0: Updated weights - include polygon overlap
            # Redistribute: distance 35%, area 20%, shape 15%, bbox 10%, polygon 20%
            match_score = (
                dist_score * 0.35
                + area_ratio * 0.20
                + shape_score * 0.15
                + bbox_score * 0.10
                + polygon_overlap * 0.20
            )
            if match_score <= 0.05:
                continue

            found_any = True
            candidate_matches.append((match_score, -dist, idx, record["pond_id"]))
            nearby_matches_by_candidate.setdefault(idx, []).append((match_score, dist, record))
            nearby_candidates_by_pond.setdefault(record["pond_id"], []).append((match_score, dist, idx))

        if not found_any and cand not in unmatched:
            unmatched.append(cand)

    used_candidate_indices = set()
    used_pond_ids = set()
    matched_by_index = {}

    for match_score, neg_dist, idx, pond_id in sorted(candidate_matches, reverse=True):
        if idx in used_candidate_indices or pond_id in used_pond_ids:
            continue
        cand = candidates[idx]
        best_record = None
        for score, dist, record in sorted(nearby_matches_by_candidate.get(idx, []), key=lambda item: (-item[0], item[1])):
            if record["pond_id"] == pond_id:
                best_record = (score, dist, record)
                break
        if best_record is None:
            continue
        score, dist, record = best_record
        cand["pond_id"] = pond_id
        cand["match_distance_m"] = round(dist, 2)
        cand["match_score"] = round(score, 4)
        cand["previous_confirmed_stage"] = int(record.get("confirmed_stage", 0) or 0)
        cand["previous_stage_probability"] = _safe_float(record.get("stage_probability"), 0.0)
        cand["previous_area_m2"] = _safe_float(record.get("area_m2"), 0.0)
        merge_sources = [rec["pond_id"] for _, _, rec in sorted(nearby_matches_by_candidate.get(idx, []), key=lambda item: (-item[0], item[1]))]
        if len(merge_sources) > 1:
            cand["merge_from_ids"] = merge_sources
        matched_by_index[idx] = cand
        used_candidate_indices.add(idx)
        used_pond_ids.add(pond_id)

    matched = [matched_by_index[idx] for idx in sorted(matched_by_index)]

    for pond_id, candidate_refs in nearby_candidates_by_pond.items():
        if len(candidate_refs) <= 1:
            continue
        ordered_indices = [idx for _, _, idx in sorted(candidate_refs, key=lambda item: (-item[0], item[1]))]
        primary_idx = ordered_indices[0]
        for idx in ordered_indices[1:]:
            cand = candidates[idx]
            cand["split_from_id"] = pond_id
            cand["split_group_size"] = len(ordered_indices)
            if idx in matched_by_index:
                cand.setdefault("split_from_id", pond_id)

        if primary_idx in matched_by_index:
            matched_by_index[primary_idx].setdefault("split_group_size", len(ordered_indices))

    for idx, cand in enumerate(candidates):
        if idx not in used_candidate_indices and cand not in unmatched:
            unmatched.append(cand)

    return matched, unmatched
