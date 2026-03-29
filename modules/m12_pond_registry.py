"""
MODULE 12 — Pond Registry

Stores persistent objects (ponds) across time and enforces 
lifecycle transition constraints and per-object persistence.
"""

import os
import json
import uuid
import config
from modules.m06_stage_engine import smooth_stage_sequence_hmm

class PondRegistry:
    def __init__(self, mode="historical", reset=False):
        self.registry_file = os.path.join(config.STATS_DIR, "PondTemporalHistory.json")
        os.makedirs(os.path.dirname(self.registry_file), exist_ok=True)
        self.ponds = {} if reset else self._load()
        self.mode = mode
        self.required_persists = config.PERSISTENCE_REQUIRED_COUNT.get(mode, 1)

    def _load(self):
        if os.path.exists(self.registry_file):
            try:
                with open(self.registry_file, "r") as f:
                    return json.load(f).get("ponds", {})
            except Exception:
                return {}
        return {}

    def save(self):
        os.makedirs(os.path.dirname(self.registry_file), exist_ok=True)
        with open(self.registry_file, "w") as f:
            json.dump({"ponds": self.ponds}, f, indent=2)

    def _is_valid_transition(self, old_stage, new_stage):
        if old_stage is None or old_stage == 0:
            return True
        old_name = f"S{old_stage}"
        new_name = f"S{new_stage}"
        # Historical mode allows jump-transitions (decade-scale epoch gaps)
        if self.mode == "historical":
            allowed = config.HISTORICAL_ALLOWED_TRANSITIONS.get(old_name, [])
        else:
            allowed = config.ALLOWED_TRANSITIONS.get(old_name, [])
        return new_name in allowed

    def register_or_update(self, pond_id, date_str, features):
        """
        features is a dict from reduceRegion with values plus 'centroid', 'area_m2'.
        stage, confidence passed implicitly or separately? Let's assume stage and confidence 
        are already calculated and added to `features`.
        """
        stage = int(features.get("stage", 0) or 0)
        confidence = float(features.get("confidence") or 0.0)
        validation_score = float(features.get("validation_score") or confidence or 0.0)
        stage_probabilities = features.get("stage_probabilities") or {}
        uncertain = bool(features.get("uncertain", False))
        uncertainty_reason = features.get("uncertainty_reason", "")
        merge_from_ids = list(features.get("merge_from_ids") or [])
        split_from_id = features.get("split_from_id")
        
        if pond_id is None:
            # Generate new pond ID
            pond_id = f"P_{uuid.uuid4().hex[:6].upper()}"
            self.ponds[pond_id] = {
                "pond_id": pond_id,
                "first_detected": date_str,
                "last_detected": date_str,
                "centroid": [
                    features.get("centroid_lat", 0),
                    features.get("centroid_lon", 0)
                ],
                "stage_history": [],
                "area_history": [],
                "confidence_history": [],
                "alert_history": [],
                "lineage": {
                    "merge_from_ids": [],
                    "split_from_id": None,
                },
                "_candidate_stage": None,
                "_candidate_count": 0,
                "confirmed_stage": None
            }

        pond = self.ponds[pond_id]
        pond["last_detected"] = date_str
        if merge_from_ids:
            lineage_merges = pond.setdefault("lineage", {}).setdefault("merge_from_ids", [])
            for source_id in merge_from_ids:
                if source_id not in lineage_merges:
                    lineage_merges.append(source_id)
        if split_from_id and not pond.setdefault("lineage", {}).get("split_from_id"):
            pond["lineage"]["split_from_id"] = split_from_id
        
        # Update centroid from latest detection (tracks drifting pond boundaries)
        if features.get("centroid_lat") and features.get("centroid_lon"):
            pond["centroid"] = [
                features.get("centroid_lat"),
                features.get("centroid_lon")
            ]
        
        previous_confirmed_stage = pond.get("confirmed_stage")
        alert_triggered = False

        pond["stage_history"].append({
            "date": date_str,
            "raw_stage": stage,
            "confirmed_stage": previous_confirmed_stage,
            "confidence": confidence,
            "validation_score": validation_score,
            "stage_probabilities": stage_probabilities,
            "uncertain": uncertain,
            "uncertainty_reason": uncertainty_reason,
            "hydro_connectivity": features.get("hydro_connectivity", 0.0),
            "tidal_exposure_proxy": features.get("tidal_exposure_proxy", 0.0),
        })
        
        if "area_m2" in features:
            pond["area_history"].append({
                "date": date_str,
                "area_m2": features["area_m2"]
            })
            
        pond["confidence_history"].append({
            "date": date_str,
            "confidence": confidence
        })

        if getattr(config, "HMM", {}).get("enabled") and getattr(config, "HMM", {}).get("pond_enabled", True):
            smoothed = smooth_stage_sequence_hmm(pond["stage_history"], mode=self.mode)
            for idx, entry in enumerate(pond["stage_history"]):
                entry["confirmed_stage"] = smoothed["path"][idx]
                entry["stage_probability"] = smoothed["probabilities"][idx]
                entry["uncertain"] = smoothed["uncertain"][idx]
                entry["uncertainty_reason"] = smoothed["reasons"][idx]
                entry["stage_probabilities"] = smoothed["stage_probabilities"][idx]

            confirmed_stage = smoothed["path"][-1]
            pond["confirmed_stage"] = confirmed_stage
            valid_score = confidence > config.VALIDATION["confidence_threshold"]
            if (
                previous_confirmed_stage is not None
                and confirmed_stage != previous_confirmed_stage
                and valid_score
                and self._is_valid_transition(previous_confirmed_stage, confirmed_stage)
            ):
                alert_triggered = True
                pond["alert_history"].append({
                    "date": date_str,
                    "from_stage": previous_confirmed_stage,
                    "to_stage": confirmed_stage,
                    "confidence": confidence,
                    "stage_probability": smoothed["probabilities"][-1],
                    "uncertain": smoothed["uncertain"][-1],
                    "uncertainty_reason": smoothed["reasons"][-1],
                })
        else:
            confirmed_stage = previous_confirmed_stage
            valid_score = confidence > config.VALIDATION["confidence_threshold"]
            if confirmed_stage is None:
                pond["confirmed_stage"] = stage
                confirmed_stage = stage
            elif stage == confirmed_stage:
                pond["_candidate_stage"] = None
                pond["_candidate_count"] = 0
            else:
                if stage == pond["_candidate_stage"]:
                    pond["_candidate_count"] += 1
                else:
                    pond["_candidate_stage"] = stage
                    pond["_candidate_count"] = 1

                if (pond["_candidate_count"] >= self.required_persists
                    and valid_score
                    and self._is_valid_transition(confirmed_stage, stage)):

                    alert_triggered = True
                    pond["alert_history"].append({
                        "date": date_str,
                        "from_stage": confirmed_stage,
                        "to_stage": stage,
                        "confidence": confidence
                    })

                    pond["confirmed_stage"] = stage
                    confirmed_stage = stage
                    pond["_candidate_stage"] = None
                    pond["_candidate_count"] = 0
            pond["stage_history"][-1]["confirmed_stage"] = confirmed_stage

        return pond_id, confirmed_stage, alert_triggered
        
    def get_all(self):
        return [pond for _, pond in self.ponds.items()]
