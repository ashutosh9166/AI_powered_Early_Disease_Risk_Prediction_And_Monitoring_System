"""
Personalized Baseline Engine
Learns individual physiological baseline ranges per patient (rolling mean, standard deviation, and range)
instead of relying solely on generic population thresholds.
"""

from typing import Dict, Any, Tuple


POPULATION_THRESHOLDS = {
    "glucose": {
        "normal_max": 99.0,
        "prediabetic_max": 125.0,
        "diabetic_min": 126.0,
        "critical_urgent": 180.0
    },
    "blood_pressure_systolic": {
        "normal_max": 119.0,
        "elevated_max": 129.0,
        "hypertensive_min": 130.0,
        "critical_urgent": 160.0
    },
    "blood_pressure_diastolic": {
        "normal_max": 79.0,
        "hypertensive_min": 80.0,
        "critical_urgent": 100.0
    },
    "heart_rate": {
        "bradycardia_max": 59.0,
        "normal_min": 60.0,
        "normal_max": 90.0,
        "tachycardia_min": 100.0
    },
    "spo2": {
        "normal_min": 95.0,
        "mild_hypoxia": 93.0,
        "critical_urgent": 91.0
    },
    "bmi": {
        "underweight": 18.5,
        "normal_max": 24.9,
        "overweight_max": 29.9,
        "obese_min": 30.0
    }
}


class PersonalizedBaselineEngine:
    def __init__(self, smoothing_factor: float = 0.15):
        """
        smoothing_factor (alpha): Weight given to newest reading when updating baseline exponential moving average.
        """
        self.alpha = smoothing_factor

    def update_baseline(self, current_baseline: Dict[str, float], new_reading: Dict[str, float]) -> Dict[str, float]:
        """
        Updates patient's personalized baseline using Exponential Moving Average (EMA).
        """
        updated = dict(current_baseline)
        for key, new_val in new_reading.items():
            if key in ["patient_id", "timestamp", "age", "outcome"]:
                continue
            if isinstance(new_val, (int, float)):
                if key in updated:
                    # Update rolling EMA
                    updated[key] = round(self.alpha * float(new_val) + (1 - self.alpha) * updated[key], 2)
                else:
                    updated[key] = round(float(new_val), 2)
        return updated

    def evaluate_vital_against_baseline(
        self,
        feature_name: str,
        current_value: float,
        baseline_value: float
    ) -> Dict[str, Any]:
        """
        Compares a current vital reading against the patient's individual baseline.
        Returns percentage delta, directional trend, and baseline anomaly severity.
        """
        if baseline_value is None or baseline_value == 0:
            return {
                "current": current_value,
                "baseline": None,
                "delta_percent": 0.0,
                "is_anomaly": False,
                "message": "No baseline established yet"
            }

        delta_percent = ((current_value - baseline_value) / baseline_value) * 100.0
        abs_delta = abs(delta_percent)

        is_anomaly = False
        severity_tag = "normal"

        # Vitals where an increase indicates risk
        if feature_name in ["glucose", "glucose_variability", "blood_pressure_systolic", "blood_pressure_diastolic", "heart_rate"]:
            if delta_percent >= 30.0:
                is_anomaly = True
                severity_tag = "critical_spike"
            elif delta_percent >= 15.0:
                is_anomaly = True
                severity_tag = "moderate_elevation"
        # Vitals where a drop indicates risk (e.g. SpO2 or physical activity)
        elif feature_name in ["spo2"]:
            if delta_percent <= -4.0 or current_value < 94.0:
                is_anomaly = True
                severity_tag = "critical_drop"
        elif feature_name in ["activity_steps"]:
            if delta_percent <= -45.0:
                is_anomaly = True
                severity_tag = "marked_inactivity"

        return {
            "feature": feature_name,
            "current": round(current_value, 1),
            "baseline": round(baseline_value, 1),
            "delta_percent": round(delta_percent, 1),
            "is_anomaly": is_anomaly,
            "severity_tag": severity_tag
        }

    def generate_comprehensive_comparison(
        self,
        current_vitals: Dict[str, float],
        baseline_vitals: Dict[str, float]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Produces baseline vs current comparative evaluations across all tracked vitals.
        """
        comparison = {}
        for feature, curr_val in current_vitals.items():
            if feature in ["patient_id", "timestamp", "age", "outcome"]:
                continue
            if isinstance(curr_val, (int, float)):
                base_val = baseline_vitals.get(feature)
                comparison[feature] = self.evaluate_vital_against_baseline(feature, curr_val, base_val)
        return comparison
