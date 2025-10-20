# Fairness-Aware Benchmarking of Predictive Models for Emergency Department Risk Stratification

This repository provides the implementation, data processing scripts, and evaluation framework for fairness benchmarking and bias mitigation in emergency department (ED) risk stratification models.  
The project integrates multicenter EHR data, machine learning models, and fairness metrics to assess subgroup disparities and evaluate bias mitigation strategies across sites.

Python workflow for generating benchmark datasets and machine learning models from the **MIMIC-IV-ED** and **MC-MED** databases.

---

## General Information

Making timely and equitable decisions in the emergency department (ED) is critical for improving patient outcomes and ensuring efficient use of limited healthcare resources.  
In recent years, machine learning (ML)-based prediction models have been increasingly adopted to support triage, early warning, and clinical decision-making.  
However, concerns regarding algorithmic bias and fairness have limited their trust and adoption in real-world ED settings.

To address this gap, we conduct a multicenter fairness benchmarking study using two large, publicly available ED datasets:

- **[MIMIC-IV-ED](https://physionet.org/content/mimic-iv-ed/2.2/)** — developed by the Massachusetts Institute of Technology (MIT) from the Beth Israel Deaconess Medical Center (BIDMC).  
  This dataset includes over 400,000 adult ED visits (2011–2019) with detailed demographics, triage assessments, and clinical outcomes.

- **[MC-MED](https://physionet.org/content/mc-med/1.0.0/)** — released by Stanford Health Care, comprising 118,000 ED visits (2020–2022) from 70,000+ unique patients.  
  It is the first publicly available ED dataset combining structured EHR variables, continuous vital signs, high-frequency waveforms, and free-text clinical documentation.

Together, these datasets enable robust evaluation of ML model performance, fairness, and generalizability across distinct institutional populations and time periods.

---

## Workflow Overview

This repository provides a standardized workflow for:

1. **Data Processing**  
   - Extracting, cleaning, and harmonizing data from MIMIC-IV-ED and MC-MED.  
   - Standardizing variable definitions and preparing train/test splits.

2. **Comprehensive Model Comparison and Evaluation**  
   - Training and evaluating baseline models (e.g., Logistic Regression, Random Forest, Gradient Boosting, MLP).  
   - Comparing model performance between BIDMC and Stanford datasets.  
   - Reporting AUROC, recall, FPR, and calibration across demographic subgroups.

3. **Bias Mitigation Methods**  
   - Implementing pre-, in-, and post-processing fairness mitigation strategies.  
   - Evaluating fairness metrics such as Equal Opportunity Difference, Demographic Parity, and Equalized Odds.

