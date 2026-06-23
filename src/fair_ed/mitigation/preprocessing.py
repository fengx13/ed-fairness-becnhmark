"""Pre-processing bias-mitigation methods.

These alter the training distribution before a model is fit:

* **Reweighing** (Kamiran & Calders, 2012) -- reweights instances so the
  protected attribute and label become statistically independent.
  ``aif360.algorithms.preprocessing.Reweighing``.
* **SMOTE** (Chawla et al., 2002) -- synthesizes minority-class samples.
  ``imblearn.over_sampling.SMOTE``.
* **Disparate Impact Remover (DIR)** (Feldman et al., 2015) -- monotonic
  feature transform that reduces correlation with the protected attribute.
  ``aif360.algorithms.preprocessing.DisparateImpactRemover``.

Libraries are imported lazily; install with ``pip install -e .[mitigation]``.
"""
from __future__ import annotations


def reweigh(dataset, privileged_groups, unprivileged_groups):
    """Apply AIF360 Reweighing.

    Returns the reweighed ``BinaryLabelDataset``; its ``instance_weights`` can be
    passed as ``sample_weight`` when fitting a classifier.
    """
    from aif360.algorithms.preprocessing import Reweighing

    rw = Reweighing(
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups,
    )
    return rw.fit_transform(dataset)


def smote_resample(X, y, random_state: int = 42):
    """Oversample the minority class with SMOTE.

    Returns ``(X_resampled, y_resampled)``. The full grid-searched pipeline used
    in the paper is in ``notebooks/03_mitigation_pre_smote.ipynb``.
    """
    from imblearn.over_sampling import SMOTE

    return SMOTE(random_state=random_state).fit_resample(X, y)


def disparate_impact_remover(dataset, repair_level: float = 1.0):
    """Apply AIF360 Disparate Impact Remover and return the repaired dataset.

    The protected attribute should be dropped from the feature matrix before
    training on the repaired data.
    """
    from aif360.algorithms.preprocessing import DisparateImpactRemover

    return DisparateImpactRemover(repair_level=repair_level).fit_transform(dataset)
