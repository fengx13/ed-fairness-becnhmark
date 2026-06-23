"""Subgroup fairness evaluation for the FairED benchmark.

Implements the standardized subgroup analysis described in the manuscript:
per-group performance metrics with bootstrapped 95% confidence intervals,
oriented toward the two fairness notions reported in the paper -- Equal
Opportunity (equal sensitivity/TPR across groups) and Equalized Odds (equal TPR
and FPR). Sensitivity is the primary indicator and FPR is co-reported.

Also provides the canonical subgroup definitions used across both cohorts so the
modeling and mitigation notebooks share one source of truth:

* gender: male, female
* age (years): 18-24, 25-34, 35-44, 45-54, 55-64, 65-74, >=75
* race/ethnicity: Asian, Black/African American, Hispanic/Latino, White, Other
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from fairlearn.metrics import MetricFrame, selection_rate
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.utils import resample

# --- Canonical subgroup definitions (match the manuscript) ---------------------

# pd.cut with right=True: (17, 24] -> "18-24", ... , (74, 150] -> "75+".
AGE_BINS = [17, 24, 34, 44, 54, 64, 74, 150]
AGE_LABELS = ["18-24", "25-34", "35-44", "45-54", "55-64", "65-74", "75+"]

# Values treated as missing/unusable for race and ethnicity.
INVALID_GROUP_VALUES = {"", "nan", "unknown", "unable to obtain", "declines to state"}


# --- Confusion-matrix-derived metrics -----------------------------------------

def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else np.nan


def false_positive_rate(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return fp / (fp + tn) if (fp + tn) > 0 else np.nan


def false_negative_rate(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return fn / (fn + tp) if (fn + tp) > 0 else np.nan


# --- Subgroup assignment ------------------------------------------------------

def group_by_age(age: "pd.Series") -> "pd.Series":
    """Bin a numeric age series into the manuscript's seven adult age groups."""
    return pd.cut(age, bins=AGE_BINS, labels=AGE_LABELS, right=True)


def map_race_ethnicity(race_val, ethnicity_val) -> str:
    """Collapse raw race + ethnicity into the five reported groups.

    Hispanic/Latino ethnicity takes precedence over race, matching the paper.
    """
    race_str = str(race_val).strip()
    ethnicity_str = str(ethnicity_val).strip()
    if "Hispanic/Latino" in ethnicity_str:
        return "Hispanic/Latino"
    elif "White" in race_str:
        return "White"
    elif "Asian" in race_str:
        return "Asian"
    elif "Black or African American" in race_str:
        return "Black or African American"
    else:
        return "Other"


# --- Bootstrapped subgroup metrics --------------------------------------------

def bootstrap_fairness_metric_ci(
    df, y_true_col, y_pred_col, y_score_col, group_col, B=1000, random_seed=42
):
    """Per-subgroup performance metrics with bootstrap 95% confidence intervals.

    Returns a tidy DataFrame with columns ``Metric, Group, Mean, CI_lower,
    CI_upper`` for accuracy, precision, recall (sensitivity/TPR), F1,
    specificity, FPR, FNR, selection rate, and AUC.
    """
    group_vals = df[group_col].unique()

    metric_funcs = {
        "accuracy": accuracy_score,
        "precision": precision_score,
        "recall": recall_score,
        "f1": f1_score,
        "specificity": specificity_score,
        "fpr": false_positive_rate,
        "fnr": false_negative_rate,
        "selection_rate": selection_rate,
    }

    results = {metric: {g: [] for g in group_vals} for metric in metric_funcs}
    results["auc"] = {g: [] for g in group_vals}

    for b in range(B):
        sample = resample(df, replace=True, random_state=random_seed + b)

        mf = MetricFrame(
            metrics=metric_funcs,
            y_true=sample[y_true_col],
            y_pred=sample[y_pred_col],
            sensitive_features=sample[group_col],
        )

        for metric in metric_funcs:
            for g in group_vals:
                results[metric][g].append(mf.by_group[metric][g])

        for g in group_vals:
            mask = sample[group_col] == g
            auc_val = roc_auc_score(sample.loc[mask, y_true_col], sample.loc[mask, y_score_col])
            results["auc"][g].append(auc_val)

    rows = []
    for metric in results:
        for g in group_vals:
            values = results[metric][g]
            rows.append({
                "Metric": metric,
                "Group": g,
                "Mean": np.mean(values),
                "CI_lower": np.percentile(values, 2.5),
                "CI_upper": np.percentile(values, 97.5),
            })

    return pd.DataFrame(rows)


def add_error_columns(ci_df, mean_col="Mean", lower_col="CI_lower", upper_col="CI_upper"):
    """Add asymmetric error-bar columns for plotting bootstrap CIs."""
    ci_df = ci_df.copy()
    ci_df["Error Lower"] = ci_df[mean_col] - ci_df[lower_col]
    ci_df["Error Upper"] = ci_df[upper_col] - ci_df[mean_col]
    return ci_df
