"""In-processing bias-mitigation methods.

* **Adversarial Debiasing (AD)** (Zhang et al., 2018) -- jointly trains a
  predictor and an adversary that tries to infer the protected attribute,
  discouraging the predictor from encoding it.
  ``aif360.algorithms.inprocessing.AdversarialDebiasing`` (TensorFlow backend).

Requires the ``mitigation`` and ``deep`` extras
(``pip install -e .[mitigation,deep]``); TensorFlow is imported lazily.
"""
from __future__ import annotations


def adversarial_debiasing(
    train_dataset,
    test_dataset,
    privileged_groups,
    unprivileged_groups,
    scope_name: str = "adversarial_debiasing",
    num_epochs: int = 50,
    debias: bool = True,
    seed: int = 42,
):
    """Train Adversarial Debiasing on ``train_dataset`` and predict on ``test_dataset``.

    Returns the predicted AIF360 dataset (``.labels`` / ``.scores`` hold the
    debiased predictions). Runs under a TensorFlow v1 session, which is opened
    and closed within the call.
    """
    import tensorflow.compat.v1 as tf
    from aif360.algorithms.inprocessing import AdversarialDebiasing

    tf.disable_eager_execution()
    tf.reset_default_graph()
    sess = tf.Session()
    try:
        model = AdversarialDebiasing(
            privileged_groups=privileged_groups,
            unprivileged_groups=unprivileged_groups,
            scope_name=scope_name,
            num_epochs=num_epochs,
            debias=debias,
            sess=sess,
            seed=seed,
        )
        model.fit(train_dataset)
        return model.predict(test_dataset)
    finally:
        sess.close()
