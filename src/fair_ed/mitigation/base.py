"""Shared utilities for the bias-mitigation wrappers.

Provides the common plumbing every mitigation method needs: loading the
processed train/test splits, constructing AIF360 ``BinaryLabelDataset`` objects
for a given outcome and binary protected attribute, turning model predictions
into AIF360 datasets (for post-processing), and evaluating mitigated predictions
with the same bootstrapped subgroup metrics used elsewhere in FairED.

Third-party fairness libraries are imported lazily so that importing this module
(or ``fair_ed``) does not require aif360 to be installed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fair_ed.config import load_paths
from fair_ed.evaluation.fairness import bootstrap_fairness_metric_ci


def load_splits(config_path: str | None = None):
    """Load the processed Stanford train/test CSVs from the path config."""
    paths = load_paths(config_path)
    train = pd.read_csv(paths["train_path"])
    test = pd.read_csv(paths["test_path"])
    return train, test


def group_dicts(protected_col: str, privileged_value=1, unprivileged_value=0):
    """Return ``(privileged_groups, unprivileged_groups)`` in AIF360 form."""
    return (
        [{protected_col: privileged_value}],
        [{protected_col: unprivileged_value}],
    )


def to_aif360_dataset(
    df: "pd.DataFrame",
    feature_cols: list[str],
    outcome_col: str,
    protected_col: str,
    favorable_label: float = 1.0,
):
    """Build an AIF360 ``BinaryLabelDataset`` from a tidy DataFrame.

    ``protected_col`` must be a binary (0/1) column. Imported lazily; requires
    the ``mitigation`` extra (``pip install -e .[mitigation]``).
    """
    from aif360.datasets import BinaryLabelDataset

    cols = list(dict.fromkeys([*feature_cols, protected_col, outcome_col]))
    frame = df[cols].dropna().copy()
    return BinaryLabelDataset(
        df=frame,
        label_names=[outcome_col],
        protected_attribute_names=[protected_col],
        favorable_label=favorable_label,
        unfavorable_label=1.0 - favorable_label,
    )


def dataset_from_predictions(template, y_pred, y_score=None):
    """Copy an AIF360 dataset, overwriting labels (and optionally scores).

    Used to build the ``*_pred`` datasets that post-processing methods consume.
    """
    out = template.copy(deepcopy=True)
    out.labels = np.asarray(y_pred, dtype=float).reshape(-1, 1)
    if y_score is not None:
        out.scores = np.asarray(y_score, dtype=float).reshape(-1, 1)
    return out


def evaluate_subgroups(y_true, y_pred, y_score, groups, B: int = 1000, random_seed: int = 42):
    """Bootstrap subgroup metrics for one model's predictions.

    Thin convenience wrapper over
    :func:`fair_ed.evaluation.fairness.bootstrap_fairness_metric_ci`.
    """
    frame = pd.DataFrame(
        {"y_true": np.asarray(y_true), "y_pred": np.asarray(y_pred),
         "y_score": np.asarray(y_score), "group": np.asarray(groups)}
    )
    return bootstrap_fairness_metric_ci(
        frame, "y_true", "y_pred", "y_score", "group", B=B, random_seed=random_seed
    )
