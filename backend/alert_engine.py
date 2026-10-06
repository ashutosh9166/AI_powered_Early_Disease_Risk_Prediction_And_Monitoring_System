"""
Alert & Notification Engine
Implements the 5-step Alert & Notification Flow:
1. Risk scoring
2. Threshold + personalized baseline check
3. Severity tiering (Informational, Moderate, Urgent)
4. Multi-party routing (Doctor Dashboard, Patient App, Caregiver Notification)
5. Audit logging
"""

import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional
from backend.models import AlertRecord


class AlertEngine:
    def __init__(self):
        pass

    def evaluate_and_route_alert(
        self,
        patient_id: str,
        patient_name: str,
        risk_probability: float,
        current_vitals: Dict[str, Any],
        baseline_comparison: Dict[str, Dict[str, Any]],
        shap_explanation: Dict[str, Any],
        caregiver_name: Optional[str] = None,
        caregiver_contact: Optional[str] = None
    ) -> AlertRecord:
        """
        Evaluates risk score and baseline anomalies to determine tier, generate routed messages,
        and format the complete alert record.
        """
        glucose_curr = current_vitals.get("glucose", 0.0)
        sys_bp_curr = current_vitals.get("blood_pressure_systolic", 0.0)
        spo2_curr = current_vitals.get("spo2", 98.0)
        
        glucose_baseline_comp = baseline_comparison.get("glucose", {})
        glucose_delta_pct = glucose_baseline_comp.get("delta_percent", 0.0)
        
        # Determine Severity Tier based on BOTH ML Score and Clinical / Baseline checks
        is_urgent = (
            risk_probability >= 0.75 or
            glucose_curr >= 180.0 or
            sys_bp_curr >= 160.0 or
            spo2_curr <= 92.0 or
            (glucose_delta_pct >= 35.0 and risk_probability >= 0.50)
        )
        
        is_moderate = (
            not is_urgent and (
                risk_probability >= 0.30 or
                glucose_curr >= 110.0 or
                sys_bp_curr >= 132.0 or
                glucose_delta_pct >= 12.0
            )
        )

        if is_urgent:
            severity = "Urgent"
        elif is_moderate:
            severity = "Moderate"
        else:
            severity = "Informational"

        # Format Clinician Summary (Technical, data-rich)
        top_factors = shap_explanation.get("top_contributing_factors", [])
        factors_summary = "; ".join(top_factors) if top_factors else "Vitals stable within threshold"
        
        doctor_summary = (
            f"[{severity.upper()}] Risk: {risk_probability*100:.1f}%. "
            f"Glucose: {glucose_curr:.1f} mg/dL ({'+' if glucose_delta_pct > 0 else ''}{glucose_delta_pct:.1f}% vs baseline). "
            f"Top Factors: {factors_summary}."
        )

        # Format Patient-Friendly Message (Empathetic, clear, actionable, non-alarmist)
        if severity == "Urgent":
            patient_message = (
                f"Your glucose reading ({glucose_curr:.0f} mg/dL) is noticeably higher than your usual baseline "
                f"today. Our clinical team has been notified. Please drink a glass of water, rest, and check in with your doctor or caregiver."
            )
        elif severity == "Moderate":
            patient_message = (
                f"Your metabolic numbers show a moderate rise ({glucose_curr:.0f} mg/dL). "
                f"We recommend a 15-minute light walk after your meal, staying well hydrated, and monitoring your evening readings."
            )
        else:
            patient_message = (
                f"Your vitals are steady and aligned with your personal baseline. Great job maintaining your healthy routine!"
            )

        # Caregiver Notification (Triggered on Urgent alerts)
        caregiver_notified = False
        caregiver_message = None
        if severity == "Urgent" and caregiver_name and caregiver_contact:
            caregiver_notified = True
            caregiver_message = (
                f"RPM Alert for {patient_name}: Health vitals indicated an urgent metabolic spike "
                f"(Glucose {glucose_curr:.0f} mg/dL). The care team has been alerted. "
                f"Please verify {patient_name.split()[0]} is comfortable and following hydration guidelines."
            )

        alert_record = AlertRecord(
            alert_id=f"ALT-{uuid.uuid4().hex[:8].upper()}",
            patient_id=patient_id,
            patient_name=patient_name,
            severity=severity,
            risk_score=round(risk_probability, 3),
            timestamp=datetime.utcnow().isoformat() + "Z",
            summary_message=doctor_summary,
            patient_friendly_message=patient_message,
            contributing_factors=top_factors,
            raw_vitals_snapshot=current_vitals,
            baseline_comparison=baseline_comparison,
            caregiver_notified=caregiver_notified,
            caregiver_message=caregiver_message,
            acknowledged_by_clinician=False,
            acknowledged_at=None
        )

        return alert_record
