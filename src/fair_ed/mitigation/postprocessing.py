"""Post-processing bias-mitigation methods.

These adjust a trained model's outputs without retraining. Each fits on a
validation split -- a ``*_true`` dataset (ground-truth labels) paired with a
``*_pred`` dataset (the model's predicted labels/scores) -- and then transforms
the predictions on the test set.

* **Reject Option Classification (ROC)** (Kamiran et al., 2012) --
  ``aif360.algorithms.postprocessing.RejectOptionClassification``.
* **Equalized Odds Post-processing (EOP)** (Hardt et al., 2016) --
  ``aif360.algorithms.postprocessing.EqOddsPostprocessing``.
* **Calibrated Equalized Odds (CEOP)** (Pleiss et al., 2017) --
  ``aif360.algorithms.postprocessing.CalibratedEqOddsPostprocessing``.

Libraries are imported lazily; install with ``pip install -e .[mitigation]``.
"""
from __future__ import annotations


def reject_option_classification(
    valid_true,
    valid_pred,
    test_pred,
    privileged_groups,
    unprivileged_groups,
    metric_name: str = "Statistical parity difference",
):
    """Fit Reject Option Classification on validation data; transform test predictions."""
    from aif360.algorithms.postprocessing import RejectOptionClassification

    roc = RejectOptionClassification(
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups,
        low_class_thresh=0.01,
        high_class_thresh=0.99,
        num_class_thresh=100,
        num_ROC_margin=50,
        metric_name=metric_name,
    )
    roc.fit(valid_true, valid_pred)
    return roc.predict(test_pred)


def equalized_odds_postprocessing(
    valid_true, valid_pred, test_pred, privileged_groups, unprivileged_groups, seed: int = 42
):
    """Fit Equalized Odds post-processing on validation data; transform test predictions."""
    from aif360.algorithms.postprocessing import EqOddsPostprocessing

    eop = EqOddsPostprocessing(
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups,
        seed=seed,
    )
    eop.fit(valid_true, valid_pred)
    return eop.predict(test_pred)


def calibrated_equalized_odds(
    valid_true,
    valid_pred,
    test_pred,
    privileged_groups,
    unprivileged_groups,
    cost_constraint: str = "weighted",
    seed: int = 42,
):
    """Fit Calibrated Equalized Odds on validation data; transform test predictions.

    ``cost_constraint`` is one of ``"fpr"``, ``"fnr"``, or ``"weighted"``.
    """
    from aif360.algorithms.postprocessing import CalibratedEqOddsPostprocessing

    ceop = CalibratedEqOddsPostprocessing(
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups,
        cost_constraint=cost_constraint,
        seed=seed,
    )
    ceop.fit(valid_true, valid_pred)
    return ceop.predict(test_pred)
