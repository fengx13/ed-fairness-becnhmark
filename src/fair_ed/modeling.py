"""Shared feature specification and base-model preparation.

Centralizes the structured triage feature set used across the modeling and
bias-mitigation notebooks so they stay consistent with one another and with the
manuscript (demographics, ED/hospital/ICU history counts, triage vitals, chief
complaints, and Charlson/Elixhauser comorbidity flags).
"""
from __future__ import annotations

from sklearn.preprocessing import LabelEncoder, StandardScaler

# Predictor columns (the manuscript's structured triage feature set).
FEATURE_COLUMNS = [
    "age", "gender",
    "n_ed_30d", "n_ed_90d", "n_ed_365d",
    "n_hosp_30d", "n_hosp_90d", "n_hosp_365d",
    "n_icu_30d", "n_icu_90d", "n_icu_365d",
    "triage_temperature", "triage_heartrate", "triage_resprate",
    "triage_o2sat", "triage_sbp", "triage_dbp", "triage_acuity",
    "chiefcom_chest_pain", "chiefcom_abdominal_pain", "chiefcom_headache",
    "chiefcom_shortness_of_breath", "chiefcom_back_pain", "chiefcom_cough",
    "chiefcom_nausea_vomiting", "chiefcom_fever_chills", "chiefcom_syncope",
    "chiefcom_dizziness",
    "cci_MI", "cci_CHF", "cci_PVD", "cci_Stroke", "cci_Dementia",
    "cci_Pulmonary", "cci_Rheumatic", "cci_PUD", "cci_Liver1", "cci_DM1",
    "cci_DM2", "cci_Paralysis", "cci_Renal", "cci_Cancer1", "cci_Liver2",
    "cci_Cancer2", "cci_HIV",
    "eci_Arrhythmia", "eci_Valvular", "eci_PHTN", "eci_HTN1", "eci_HTN2",
    "eci_NeuroOther", "eci_Hypothyroid", "eci_Lymphoma", "eci_Coagulopathy",
    "eci_Obesity", "eci_WeightLoss", "eci_FluidsLytes", "eci_BloodLoss",
    "eci_Anemia", "eci_Alcohol", "eci_Drugs", "eci_Psychoses", "eci_Depression",
]

# Continuous columns standardized before fitting (e.g. Logistic Regression).
STANDARDIZE_COLUMNS = [
    "age",
    "n_ed_30d", "n_ed_90d", "n_ed_365d",
    "n_hosp_30d", "n_hosp_90d", "n_hosp_365d",
    "n_icu_30d", "n_icu_90d", "n_icu_365d",
    "triage_temperature", "triage_heartrate",
    "triage_resprate", "triage_o2sat",
    "triage_sbp", "triage_dbp",
]


def prepare_xy(df_train, df_test, outcome, features=FEATURE_COLUMNS):
    """Slice train/test feature matrices and label vectors for an outcome.

    ``gender`` is label-encoded with the encoder fit on the training split.
    """
    X_train = df_train[features].copy()
    X_test = df_test[features].copy()
    if "gender" in features:
        encoder = LabelEncoder()
        X_train["gender"] = encoder.fit_transform(X_train["gender"].astype(str))
        X_test["gender"] = encoder.transform(X_test["gender"].astype(str))
    y_train = df_train[outcome].copy()
    y_test = df_test[outcome].copy()
    return X_train, X_test, y_train, y_test


def standardize(X_train, X_test, columns=STANDARDIZE_COLUMNS):
    """Standardize the continuous columns; returns copies and the fitted scaler."""
    scaler = StandardScaler()
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train[columns] = scaler.fit_transform(X_train[columns])
    X_test[columns] = scaler.transform(X_test[columns])
    return X_train, X_test, scaler
