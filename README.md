# FairED

**An open benchmarking framework for reproducible evaluation of algorithmic fairness in emergency care across clinical sites.**

FairED is a transparent, reproducible **multicenter benchmark** for evaluating the fairness of
machine-learning models used for **emergency department (ED) risk stratification**. Using two large,
publicly available datasets from distinct health systems, it assesses predictive performance and
subgroup fairness across clinical outcomes and acute diagnoses, and evaluates bias-mitigation
strategies spanning pre-, in-, and post-processing.

This repository accompanies the manuscript:

> Mai X, Liu Y, Ning Y, Wacker DA, Lin M, Liu N, Puskarich MA, Xie F.
> *FairED: An Open Benchmarking Framework for Reproducible Evaluation of Algorithmic Fairness in
> Emergency Care Across Clinical Sites.* Research Square (preprint), 2026.
> doi:[10.21203/rs.3.rs-9161309/v1](https://doi.org/10.21203/rs.3.rs-9161309/v1)

---

## Datasets

FairED is built on two PhysioNet cohorts. The raw data are **not** included here and are available
under PhysioNet's credentialed data use agreement.

| Cohort | Source | Site | Period | PhysioNet |
|---|---|---|---|---|
| **BIDMC** | MIMIC-IV-ED | Beth Israel Deaconess Medical Center (MIT) | 2011–2019 | [10.13026/5ntk-km72](https://doi.org/10.13026/5ntk-km72) |
| **Stanford** | MC-MED | Stanford Health Care | 2020–2022 | [10.13026/jz99-4j81](https://doi.org/10.13026/jz99-4j81) |

The pipeline harmonizes both sources to a shared encounter-level schema so models and fairness
metrics can be compared head-to-head across institutions.

## Benchmark tasks

- **Utilization outcomes:** hospitalization (admission to a general ward or ICU); a composite
  *critical* outcome (ICU admission, ED death, or admission with ICU transfer within 12 hours).
- **Acute disease-specific outcomes:** sepsis, pulmonary embolism (PE), acute coronary syndrome
  (ACS), acute heart failure (AHF), and stroke identified from structured ICD-9/10 codes
  (see `src/fair_ed/data/disease_defs.py`) and ED disposition.

## Models

Logistic Regression, Random Forest, XGBoost, and a Multilayer Perceptron (MLP), with the
**National Early Warning Score (NEWS)** as a rule-based clinical comparator. The main analyses
focus on LR, RF, and NEWS; XGBoost and MLP are reported in the appendix.

## Fairness evaluation

Subgroup performance is evaluated across:

- **Gender:** male, female
- **Age (years):** 18–24, 25–34, 35–44, 45–54, 55–64, 65–74, ≥75
- **Race/ethnicity:** Asian, Black/African American, Hispanic/Latino, White, Other

Reported through two notions: **Equal Opportunity** (equal sensitivity across groups) and
**Equalized Odds** (equal TPR and FPR) with 95% bootstrap confidence intervals. Sensitivity is
the primary indicator and FPR is co-reported. See `src/fair_ed/evaluation/fairness.py`.

## Bias mitigation

Seven mitigation algorithms (AIF360 / imbalanced-learn) are organized by intervention stage, each
with a wrapper in `src/fair_ed/mitigation/` and a worked notebook.

| Stage | Method | Library class |
|---|---|---|
| Pre | Reweighing (RW) | `aif360…preprocessing.Reweighing` |
| Pre | SMOTE | `imblearn.over_sampling.SMOTE` |
| Pre | Disparate Impact Remover (DIR) | `aif360…preprocessing.DisparateImpactRemover` |
| In | Adversarial Debiasing (AD) | `aif360…inprocessing.AdversarialDebiasing` |
| Post | Reject Option Classification (ROC) | `aif360…postprocessing.RejectOptionClassification` |
| Post | Equalized Odds Post-processing (EOP) | `aif360…postprocessing.EqOddsPostprocessing` |
| Post | Calibrated Equalized Odds (CEOP) | `aif360…postprocessing.CalibratedEqOddsPostprocessing` |

## Installation

Requires Python 3.10–3.13 (the paper used 3.13).

```bash
git clone https://github.com/fengx13/ed-fairness-becnhmark.git
cd ed-fairness-becnhmark
pip install -e .            # core
pip install -e ".[all]"    # + mitigation (AIF360), deep (TensorFlow), viz, jupyter
```

Then point FairED at your local data:

```bash
cp configs/paths.example.yaml configs/paths.yaml
# edit configs/paths.yaml with your MIMIC-IV-ED / MC-MED file locations
```

`configs/paths.yaml` is gitignored, so machine-specific paths never get committed.

## Workflow

The pipeline mirrors the manuscript's methods, stage by stage:

| Step | Notebook / module | What it does |
|---|---|---|
| 1. Data extraction & preprocessing | `python -m fair_ed.data.build_stanford` → `notebooks/01_data_extraction_and_preprocessing.ipynb` | Build the MC-MED master dataset, then filter (age ≥18, valid triage), remove outliers, split 80/20, impute, and save train/test. |
| 2. Model development & evaluation | `notebooks/02_model_development_and_evaluation.ipynb` | Train LR / RF / MLP and NEWS; evaluate subgroup performance and fairness with bootstrap CIs. |
| 3–5. Bias mitigation | `notebooks/03_*` (pre), `04_*` (in), `05_*` (post) | Apply and evaluate the mitigation methods (numbering self-sorts by stage). |

## Repository layout

```
src/fair_ed/
  config.py            # load data paths from configs/paths.yaml
  mappers/             # ICD-9/10 crosswalks, Charlson/Elixhauser code sets
  data/                # MIMIC-IV-ED & MC-MED extraction, features, disease defs, builder
  scores/              # rule-based clinical scores (NEWS, NEWS2, MEWS, REMS, CART, ...)
  evaluation/          # performance metrics, bootstrap CIs, subgroup fairness, LSTM generator
  mitigation/          # AIF360/Fairlearn wrappers by stage (pre/in/post)
configs/               # paths.example.yaml (template)
notebooks/             # 01 → 05, the end-to-end workflow
```

## Data & code availability

All datasets are publicly available and de-identified, distributed under PhysioNet's credentialed
data use agreement (links above). No additional patient-level data are collected or shared. All
analyses are in Python; the benchmarking pipeline uses scikit-learn, XGBoost, Fairlearn, AIF360,
pandas, numpy, matplotlib, and seaborn.

## Citation

```bibtex
@article{mai2026faired,
  title   = {FairED: An Open Benchmarking Framework for Reproducible Evaluation of
             Algorithmic Fairness in Emergency Care Across Clinical Sites},
  author  = {Mai, Xinnie and Liu, Yunqian and Ning, Yilin and Wacker, David A. and
             Lin, Mingquan and Liu, Nan and Puskarich, Michael A. and Xie, Feng},
  journal = {Research Square (preprint)},
  year    = {2026},
  doi     = {10.21203/rs.3.rs-9161309/v1}
}
```

## License

Released under the [MIT License](LICENSE). The accompanying manuscript is licensed CC BY 4.0.
