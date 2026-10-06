"""
ML Training, Comparison, Evaluation and Explainability (SHAP) Pipeline
Trains and compares Logistic Regression, Random Forest, and XGBoost on wearable diabetes data.
Evaluates Accuracy, Precision, Recall, F1, and ROC-AUC.
Exports best model, scaler, and SHAP explainer for the FastAPI backend and dashboard.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
import shap


FEATURE_COLUMNS = [
    "glucose",
    "glucose_variability",
    "blood_pressure_systolic",
    "blood_pressure_diastolic",
    "heart_rate",
    "spo2",
    "bmi",
    "activity_steps",
    "sleep_hours",
    "age"
]

TARGET_COLUMN = "outcome"


def train_and_evaluate_models(data_path: str = "data/diabetes_wearable_dataset.csv", output_dir: str = "ml/models"):
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}. Run data/generate_dataset.py first.")
        
    print(f"Loading dataset from {data_path}...")
    df = pd.read_csv(data_path)
    
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define candidate models
    models = {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "Random Forest": RandomForestClassifier(
            n_estimators=150, max_depth=7, min_samples_split=4, random_state=42
        ),
        "XGBoost": XGBClassifier(
            n_estimators=120, max_depth=4, learning_rate=0.07, eval_metric="logloss", random_state=42
        )
    }
    
    results = {}
    fitted_models = {}
    
    print("\n--- Model Training & Comparison ---")
    for name, model in models.items():
        if name == "Logistic Regression":
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred).tolist()
        
        results[name] = {
            "accuracy": round(float(acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score": round(float(f1), 4),
            "roc_auc": round(float(auc), 4),
            "confusion_matrix": cm
        }
        fitted_models[name] = model
        
        print(f"\nModel: {name}")
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print(f"  ROC-AUC:   {auc:.4f}")

    # Select best ensemble tree model (Random Forest vs XGBoost) for non-linear risk modeling & TreeExplainer
    tree_candidates = ["Random Forest", "XGBoost"]
    best_name = max(tree_candidates, key=lambda k: results[k]["roc_auc"] + results[k]["f1_score"])
    print(f"\n>>> Best Performing Ensemble Model: {best_name} <<<")
    best_model = fitted_models[best_name]
    
    # Fit SHAP TreeExplainer directly on the chosen best model
    print(f"\nFitting SHAP TreeExplainer on {best_name} for Explainable AI (XAI)...")
    explainer = shap.TreeExplainer(best_model)
        
    # Calculate feature importances
    if hasattr(best_model, "feature_importances_"):
        importances = dict(zip(FEATURE_COLUMNS, [round(float(v), 4) for v in best_model.feature_importances_]))
    else:
        importances = dict(zip(FEATURE_COLUMNS, [round(float(abs(v)), 4) for v in best_model.coef_[0]]))
    # Sort feature importances descending
    importances = dict(sorted(importances.items(), key=lambda item: item[1], reverse=True))
    
    # Save Model Artifacts
    joblib.dump(best_model, os.path.join(output_dir, "best_diabetes_model.joblib"))
    joblib.dump(scaler, os.path.join(output_dir, "scaler.joblib"))
    joblib.dump(explainer, os.path.join(output_dir, "shap_explainer.joblib"))
    
    metadata = {
        "best_model_name": best_name,
        "features": FEATURE_COLUMNS,
        "feature_importances": importances,
        "performance_metrics": results[best_name],
        "all_models_comparison": results,
        "test_sample_size": len(y_test),
        "train_sample_size": len(y_train),
        "clinical_reference_thresholds": {
            "glucose": {"normal_max": 99.0, "prediabetic_max": 125.0, "diabetic_min": 126.0, "critical_urgent": 180.0},
            "blood_pressure_systolic": {"normal_max": 119, "elevated_max": 129, "hypertensive_min": 130, "critical_urgent": 160},
            "blood_pressure_diastolic": {"normal_max": 79, "hypertensive_min": 80, "critical_urgent": 100},
            "heart_rate": {"normal_min": 60, "normal_max": 90, "tachycardia_min": 100},
            "spo2": {"normal_min": 95.0, "hypoxia_urgent": 92.0},
            "bmi": {"healthy_max": 24.9, "overweight_max": 29.9, "obese_min": 30.0},
            "activity_steps": {"sedentary_max": 4999, "active_min": 8000},
            "sleep_hours": {"insufficient_max": 5.9, "optimal_min": 7.0}
        }
    }
    
    with open(os.path.join(output_dir, "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
        
    print(f"\nArtifacts successfully saved to {output_dir}:")
    print(f" - best_diabetes_model.joblib ({best_name})")
    print(f" - scaler.joblib")
    print(f" - shap_explainer.joblib")
    print(f" - model_metadata.json")
    
    return results, metadata


def generate_plain_language_explanation(
    vitals: Dict[str, float],
    risk_prob: float,
    shap_contributions: Dict[str, float],
    personal_baseline: Dict[str, float] = None
) -> Dict[str, Any]:
    """
    Generates a natural-language clinical explanation combining SHAP feature attributions
    with personalized baseline deviations and clinical thresholds.
    """
    factors = []
    
    # Sort features by positive SHAP impact (risk drivers)
    sorted_factors = sorted(shap_contributions.items(), key=lambda x: x[1], reverse=True)
    
    for feature, shap_val in sorted_factors:
        curr_val = vitals.get(feature)
        if curr_val is None:
            continue
            
        base_val = personal_baseline.get(feature) if personal_baseline else None
        
        # Calculate deviation from personal baseline if available
        base_diff_str = ""
        if base_val is not None and base_val > 0:
            pct_diff = ((curr_val - base_val) / base_val) * 100
            if abs(pct_diff) >= 5:
                sign = "+" if pct_diff > 0 else ""
                base_diff_str = f" ({sign}{pct_diff:.1f}% vs your baseline of {base_val:.1f})"
                
        if feature == "glucose" and (shap_val > 0.05 or curr_val > 105):
            factors.append(f"Blood glucose elevated at {curr_val:.1f} mg/dL{base_diff_str}")
        elif feature == "glucose_variability" and shap_val > 0.04:
            factors.append(f"High glucose fluctuation (±{curr_val:.1f} mg/dL){base_diff_str}")
        elif feature == "bmi" and (shap_val > 0.03 or curr_val >= 28):
            factors.append(f"Elevated BMI at {curr_val:.1f}")
        elif feature == "blood_pressure_systolic" and (shap_val > 0.03 or curr_val >= 130):
            dia = vitals.get("blood_pressure_diastolic", 80)
            factors.append(f"Elevated blood pressure ({curr_val:.0f}/{dia:.0f} mmHg){base_diff_str}")
        elif feature == "activity_steps" and (shap_val > 0.03 or curr_val < 4500):
            factors.append(f"Low daily physical activity ({int(curr_val):,} steps){base_diff_str}")
        elif feature == "sleep_hours" and (shap_val > 0.02 or curr_val < 6.0):
            factors.append(f"Suboptimal sleep duration ({curr_val:.1f} hrs){base_diff_str}")
        elif feature == "heart_rate" and (shap_val > 0.03 or curr_val > 90):
            factors.append(f"Elevated resting heart rate ({int(curr_val)} bpm){base_diff_str}")
        elif feature == "age" and shap_val > 0.03:
            factors.append(f"Age metabolic demographic factor ({int(curr_val)} yrs)")

    # Build plain english summary
    if risk_prob >= 0.70:
        level_str = "High"
    elif risk_prob >= 0.35:
        level_str = "Moderate"
    else:
        level_str = "Low"
        
    top_3 = factors[:3] if factors else ["All monitored vitals are within normal range"]
    summary_text = f"Risk is {level_str.lower()} mainly due to: " + ", ".join(top_3) + "."
    
    return {
        "risk_level": level_str,
        "risk_probability": round(risk_prob, 3),
        "top_contributing_factors": factors[:4],
        "plain_language_summary": summary_text,
        "shap_values": {k: round(v, 4) for k, v in sorted_factors[:6]}
    }


if __name__ == "__main__":
    train_and_evaluate_models()
