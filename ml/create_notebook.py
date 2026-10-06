"""
Generates the ml/eda_and_training.ipynb notebook file with formatted cells and markdown.
"""

import json
import os

def create_notebook():
    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# AI-Powered Early Disease Risk Prediction & Remote Patient Monitoring\n",
                    "## Phase I: Exploratory Data Analysis, ML Model Benchmarking & Explainable AI (SHAP)\n",
                    "\n",
                    "**Objective:** Predict diabetes and pre-diabetic metabolic risk from continuous wearable vitals and lifestyle metrics to enable proactive clinical intervention."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import os\n",
                    "import numpy as np\n",
                    "import pandas as pd\n",
                    "import matplotlib.pyplot as plt\n",
                    "import seaborn as sns\n",
                    "\n",
                    "# Load Dataset\n",
                    "data_path = '../data/diabetes_wearable_dataset.csv' if os.path.exists('../data/diabetes_wearable_dataset.csv') else 'data/diabetes_wearable_dataset.csv'\n",
                    "df = pd.read_csv(data_path)\n",
                    "print(f'Dataset Shape: {df.shape}')\n",
                    "df.head()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 1. Exploratory Data Analysis (EDA)\n",
                    "Let's examine descriptive statistics and class distribution."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "display(df.describe().T[['mean', 'std', 'min', '50%', 'max']])\n",
                    "print('\\nOutcome Distribution (0 = Normal/Low Risk, 1 = Pre-diabetic/High Risk):')\n",
                    "print(df['outcome'].value_counts(normalize=True))"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "fig, axes = plt.subplots(2, 3, figsize=(15, 9))\n",
                    "features_to_plot = ['glucose', 'blood_pressure_systolic', 'heart_rate', 'bmi', 'activity_steps', 'sleep_hours']\n",
                    "for i, feat in enumerate(features_to_plot):\n",
                    "    ax = axes[i // 3, i % 3]\n",
                    "    sns.kdeplot(data=df, x=feat, hue='outcome', fill=True, common_norm=False, alpha=0.4, ax=ax, palette=['#10b981', '#ef4444'])\n",
                    "    ax.set_title(f'Distribution of {feat} by Risk Outcome')\n",
                    "plt.tight_layout()\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 2. Feature Correlation Matrix"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "plt.figure(figsize=(10, 8))\n",
                    "corr = df.corr()\n",
                    "sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', cbar=True)\n",
                    "plt.title('Correlation Matrix of Wearable Vitals & Lifestyle Metrics')\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 3. Model Training & Benchmarking\n",
                    "We benchmark Logistic Regression, Random Forest, and XGBoost classifiers."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from sklearn.model_selection import train_test_split\n",
                    "from sklearn.preprocessing import StandardScaler\n",
                    "from sklearn.linear_model import LogisticRegression\n",
                    "from sklearn.ensemble import RandomForestClassifier\n",
                    "from xgboost import XGBClassifier\n",
                    "from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score\n",
                    "\n",
                    "features = [c for c in df.columns if c != 'outcome']\n",
                    "X = df[features]\n",
                    "y = df['outcome']\n",
                    "\n",
                    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)\n",
                    "\n",
                    "scaler = StandardScaler()\n",
                    "X_train_scaled = scaler.fit_transform(X_train)\n",
                    "X_test_scaled = scaler.transform(X_test)\n",
                    "\n",
                    "models = {\n",
                    "    'Logistic Regression': (LogisticRegression(max_iter=1000), True),\n",
                    "    'Random Forest': (RandomForestClassifier(n_estimators=150, max_depth=7, random_state=42), False),\n",
                    "    'XGBoost': (XGBClassifier(n_estimators=120, max_depth=4, learning_rate=0.07, eval_metric='logloss', random_state=42), False)\n",
                    "}\n",
                    "\n",
                    "results = []\n",
                    "for name, (model, use_scaled) in models.items():\n",
                    "    xtr = X_train_scaled if use_scaled else X_train\n",
                    "    xte = X_test_scaled if use_scaled else X_test\n",
                    "    model.fit(xtr, y_train)\n",
                    "    preds = model.predict(xte)\n",
                    "    probs = model.predict_proba(xte)[:, 1]\n",
                    "    results.append({\n",
                    "        'Model': name,\n",
                    "        'Accuracy': accuracy_score(y_test, preds),\n",
                    "        'Precision': precision_score(y_test, preds),\n",
                    "        'Recall': recall_score(y_test, preds),\n",
                    "        'F1-Score': f1_score(y_test, preds),\n",
                    "        'ROC-AUC': roc_auc_score(y_test, probs)\n",
                    "    })\n",
                    "\n",
                    "res_df = pd.DataFrame(results)\n",
                    "res_df"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 4. Explainable AI (XAI) with SHAP\n",
                    "Extract feature attributions to provide clinicians and patients with transparent, interpretable risk factors."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import shap\n",
                    "rf_model = models['Random Forest'][0]\n",
                    "explainer = shap.TreeExplainer(rf_model)\n",
                    "shap_values = explainer.shap_values(X_test)\n",
                    "\n",
                    "# Summary Plot\n",
                    "shap.summary_plot(shap_values[1] if isinstance(shap_values, list) else shap_values, X_test, feature_names=features)"
                ]
            }
        ],
        "metadata": {
            "language_info": {
                "name": "python",
                "version": "3.11"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    os.makedirs("ml", exist_ok=True)
    with open(os.path.join("ml", "eda_and_training.ipynb"), "w") as f:
        json.dump(nb, f, indent=2)
    print("Created ml/eda_and_training.ipynb")

if __name__ == "__main__":
    create_notebook()
