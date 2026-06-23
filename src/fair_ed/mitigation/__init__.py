"""Bias-mitigation methods for the FairED benchmark.

Wrappers around the seven AIF360 / imbalanced-learn mitigation algorithms
evaluated in the manuscript, organized by the stage at which they intervene:

================  ===========================================================
Stage             Methods
================  ===========================================================
pre-processing    Reweighing (RW), SMOTE, Disparate Impact Remover (DIR)
in-processing     Adversarial Debiasing (AD)
post-processing   Reject Option Classification (ROC), Equalized Odds
                  Post-processing (EOP), Calibrated Equalized Odds (CEOP)
================  ===========================================================

Each method has a thin functional wrapper here and a worked notebook under
``notebooks/`` (``03_*`` pre, ``04_*`` in, ``05_*`` post). Shared dataset
construction and subgroup evaluation live in :mod:`fair_ed.mitigation.base`.

All third-party fairness libraries (aif360, imblearn, tensorflow) are imported
lazily inside the functions so that ``import fair_ed`` stays lightweight.
"""

STAGES = ("preprocessing", "inprocessing", "postprocessing")
