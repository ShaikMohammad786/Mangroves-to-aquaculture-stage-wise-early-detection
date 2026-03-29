"""
MODULE 9 — Alert Engine (Research-Grade v3.1)

Hybrid Persistence:
  ✔ Image-count based for historical (2 consecutive images)
  ✔ Day-based for operational mode
  ✔ Validation score gate
  ✔ Biologically valid transitions
"""

import json
import os
from datetime import datetime
import uuid
import config
from modules.m06_stage_engine import smooth_stage_sequence_hmm


# ─────────────────────────────────────────────
# ALERT STORAGE
# ─────────────────────────────────────────────

class AlertStore:

    def __init__(self, reset=False):
        self.alert_file = os.path.join(config.ALERT_DIR, "alerts.json")
        os.makedirs(config.ALERT_DIR, exist_ok=True)
        self.alerts = [] if reset else self._load()

    def _load(self):
        if os.path.exists(self.alert_file):
            with open(self.alert_file, "r") as f:
                return json.load(f)
        return []

    def save(self):
        os.makedirs(os.path.dirname(self.alert_file), exist_ok=True)
        with open(self.alert_file, "w") as f:
            json.dump(self.alerts, f, indent=2)

    def add_alert(self, alert):
        self.alerts.append(alert)
        self.save()

    def get_all(self):
        return self.alerts


# ─────────────────────────────────────────────
# STAGE HISTORY
# ─────────────────────────────────────────────

class StageHistory:

    def __init__(self):
        self.history_file = os.path.join(config.STATS_DIR, "stage_history.json")
        os.makedirs(config.STATS_DIR, exist_ok=True)
        self.history = self._load()

    def _load(self):
        if os.path.exists(self.history_file):
            with open(self.history_file, "r") as f:
                return json.load(f)
        return {}

    def save(self):
        os.makedirs(os.path.dirname(self.history_file), exist_ok=True)
        with open(self.history_file, "w") as f:
            json.dump(self.history, f, indent=2)

    def get_entries(self, region_id="aoi"):
        return self.history.get(region_id, [])

    def get_last_stage(self, region_id="aoi"):
        entries = self.get_entries(region_id)
        if entries:
            return entries[-1]
        return None

    def add_entry(self, region_id, entry):
        if region_id not in self.history:
            self.history[region_id] = []
        self.history[region_id].append(entry)
        self.save()


# ─────────────────────────────────────────────
# STAGE UTIL
# ─────────────────────────────────────────────

def _stage_name(stage_num):
    names = {
        1: "S1 Dense Mangrove",
        2: "S2 Degradation",
        3: "S3 Clearing",
        4: "S4 Water Filling",
        5: "S5 Operational Pond",
    }
    return names.get(stage_num, f"Unknown ({stage_num})")


# ─────────────────────────────────────────────
# PERSISTENCE ENGINE
# ─────────────────────────────────────────────

class PersistenceEngine:
    """
    Image-count based persistence.

    For historical backtest: confirm after 2 consecutive images.
    For operational: confirm after 2 consecutive images.

    Simple, robust, no day-based calculation that breaks
    with irregular epoch spacing.
    """

    def __init__(self, required_count=2, mode="historical"):
        self.confirmed_stage = None
        self.candidate_stage = None
        self.candidate_count = 0
        self.required_count = required_count
        self.mode = mode
        self.observations = []

    def _is_valid_transition(self, old_stage, new_stage):
        if old_stage is None:
            return True
        old_name = f"S{old_stage}"
        new_name = f"S{new_stage}"
        if self.mode == "historical":
            allowed = config.HISTORICAL_ALLOWED_TRANSITIONS.get(old_name, [])
        else:
            allowed = config.ALLOWED_TRANSITIONS.get(old_name, [])
        return new_name in allowed

    def process(self, image_date_str, dominant_stage,
                validation_score, confidence,
                stage_distribution=None, stage_probabilities=None):
        """
        Process one image. Returns result dict.
        """
        threshold = config.VALIDATION["confidence_threshold"]
        self.observations.append({
            "date": image_date_str,
            "stage": dominant_stage,
            "confidence": confidence,
            "validation_score": validation_score,
            "stage_distribution": stage_distribution or {},
            "stage_probabilities": stage_probabilities or {},
        })

        if getattr(config, "HMM", {}).get("enabled") and getattr(config, "HMM", {}).get("scene_enabled", True):
            smoothed = smooth_stage_sequence_hmm(self.observations, mode=self.mode)
            latest_stage = smoothed["path"][-1]
            latest_prob = smoothed["probabilities"][-1]
            latest_uncertain = smoothed["uncertain"][-1]
            latest_reason = smoothed["reasons"][-1]
            previous_stage = smoothed["path"][-2] if len(smoothed["path"]) > 1 else None

            alert = (
                previous_stage is not None
                and previous_stage != latest_stage
                and validation_score >= threshold
                and self._is_valid_transition(previous_stage, latest_stage)
            )
            self.confirmed_stage = latest_stage

            return {
                "confirmed": True,
                "stage": latest_stage,
                "candidate": None,
                "images_remaining": 0,
                "alert": alert,
                "old_stage": previous_stage,
                "stage_probability": latest_prob,
                "uncertain": latest_uncertain,
                "uncertainty_reason": latest_reason,
                "stage_probabilities": smoothed["stage_probabilities"][-1],
            }

        # First image
        if self.confirmed_stage is None:
            self.confirmed_stage = dominant_stage
            return {
                "confirmed": True,
                "stage": dominant_stage,
                "candidate": None,
                "images_remaining": 0,
                "alert": False,
                "stage_probability": confidence,
                "uncertain": False,
                "uncertainty_reason": "",
                "stage_probabilities": stage_probabilities or {},
            }

        # Same as confirmed — stable
        if dominant_stage == self.confirmed_stage:
            self.candidate_stage = None
            self.candidate_count = 0
            return {
                "confirmed": False,
                "stage": self.confirmed_stage,
                "candidate": None,
                "images_remaining": 0,
                "alert": False,
                "stage_probability": confidence,
                "uncertain": False,
                "uncertainty_reason": "",
                "stage_probabilities": stage_probabilities or {},
            }

        # Different from confirmed — candidate tracking
        if dominant_stage == self.candidate_stage:
            # Same candidate persisting
            self.candidate_count += 1
        else:
            # New candidate
            self.candidate_stage = dominant_stage
            self.candidate_count = 1

        # Check confirmation
        if (self.candidate_count >= self.required_count
                and validation_score >= threshold
                and self._is_valid_transition(
                    self.confirmed_stage, dominant_stage)):

            old = self.confirmed_stage
            self.confirmed_stage = dominant_stage
            self.candidate_stage = None
            self.candidate_count = 0

            print(f"  [PERSIST] ✓ CONFIRMED: S{old} → S{dominant_stage} "
                  f"(after {self.required_count} images)")

            return {
                "confirmed": True,
                "stage": dominant_stage,
                "candidate": None,
                "images_remaining": 0,
                "alert": True,
                "old_stage": old,
                "stage_probability": confidence,
                "uncertain": False,
                "uncertainty_reason": "",
                "stage_probabilities": stage_probabilities or {},
            }

        remaining = max(0, self.required_count - self.candidate_count)
        print(f"  [PERSIST] Candidate S{dominant_stage} "
              f"({self.candidate_count}/{self.required_count}) "
              f"— {remaining} images to confirm")

        return {
            "confirmed": False,
            "stage": self.confirmed_stage,
            "candidate": dominant_stage,
            "images_remaining": remaining,
            "alert": False,
            "stage_probability": confidence,
            "uncertain": False,
            "uncertainty_reason": "",
            "stage_probabilities": stage_probabilities or {},
        }


# ─────────────────────────────────────────────
# ALERT CREATION
# ─────────────────────────────────────────────

def create_alert(old_stage, new_stage, image_date,
                 validation_score, confidence,
                 stage_probability=None,
                 uncertain=False,
                 uncertainty_reason=""):
    return {
        "alert_id": f"alert_{uuid.uuid4().hex[:8]}",
        "date": str(image_date),
        "old_stage": old_stage,
        "old_stage_name": _stage_name(old_stage) if old_stage else "None",
        "new_stage": new_stage,
        "new_stage_name": _stage_name(new_stage),
        "validation_score": round(validation_score, 3),
        "confidence": round(confidence, 3),
        "stage_probability": round(float(stage_probability if stage_probability is not None else confidence), 3),
        "uncertain": bool(uncertain),
        "uncertainty_reason": uncertainty_reason or "",
        "created_at": datetime.now().isoformat(),
    }


# ─────────────────────────────────────────────
# RECORD HISTORY
# ─────────────────────────────────────────────

def record_image_event(image_date, dominant_stage,
                       validation_score, confidence,
                       validation_scores=None,
                       polygon_count=0,
                       region_id="aoi",
                       stage_probability=None,
                       uncertain=False,
                       uncertainty_reason="",
                       stage_probabilities=None):
    history = StageHistory()

    entry = {
        "date": str(image_date),
        "stage": dominant_stage,
        "stage_name": _stage_name(dominant_stage),
        "validation_score": round(validation_score, 3),
        "confidence": round(confidence, 3),
        "polygon_count": polygon_count,
        "scores": validation_scores or {},
        "stage_probability": round(float(stage_probability if stage_probability is not None else confidence), 3),
        "uncertain": bool(uncertain),
        "uncertainty_reason": uncertainty_reason or "",
        "stage_probabilities": stage_probabilities or {},
    }

    history.add_entry(region_id, entry)
    return entry
