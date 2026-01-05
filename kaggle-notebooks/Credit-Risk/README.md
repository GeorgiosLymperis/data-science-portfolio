# Credit Risk Modeling & Profit Optimization (Lending Club, Python)

This project develops a **production-style credit risk pipeline** for consumer loans, focusing on **probability of default (PD) modeling** and **profit-driven credit policy design**.  
The work mirrors real-world practices in **credit portfolio optimization**, emphasizing leakage control, time-aware validation, and economically meaningful decision rules.

## Overview
The goal is to build a **point-in-time PD model** for loan approval decisions and translate model outputs into **actionable credit policies**.  
Rather than optimizing purely for predictive metrics, the project evaluates **approval cutoffs** based on **expected portfolio profit** under explicit economic assumptions.

The analysis is performed on the Lending Club accepted loans dataset (2007–2018), using a strict out-of-time evaluation to simulate production deployment.

## Key Features
- **Target construction with censoring control**
  - Clear definition of default vs non-default outcomes
  - Removal of right-censored loans (current / in-progress)
- **Leakage-safe feature governance**
  - Automatic detection and removal of post-issuance variables
  - Exclusion of identifiers and free-text fields
- **Time-aware data splitting**
  - Train / validation / test splits by loan issue date
  - Realistic simulation of model deployment over time
- **Baseline statistical PD model**
  - Logistic regression with class weighting
  - Sparse one-hot encoding and robust preprocessing
- **Risk-oriented evaluation**
  - AUC and Brier score
  - Kolmogorov–Smirnov (KS) statistic
- **Credit policy simulation**
  - PD-based approval rules
  - Confusion matrices at multiple cutoffs
- **Profit optimization**
  - Expected profit maximization under explicit LGD assumptions
  - Out-of-time validation on a frozen policy

## Modeling Approach
- **Preprocessing**
  - Median imputation for numeric features
  - Most-frequent imputation for categorical features
  - Rare-category grouping via one-hot encoding
- **Model**
  - Logistic Regression (L2 regularization, saga solver)
  - Chosen for interpretability, stability, and calibration suitability
- **Decision Rule**
  - Approve loan if predicted `PD ≤ cutoff`
  - Reject otherwise

## Results
- The PD model achieves **stable ranking power** across vintages:
  - KS ≈ 0.30 on validation and test sets
- Approval policies exhibit a clear **risk–volume tradeoff**:
  - Lower cutoffs yield very low default rates but limited scale
  - Higher cutoffs increase volume at the cost of higher losses
- **Unconstrained profit optimization** favors highly conservative policies:
  - Profit-maximizing cutoffs approve <1% of applicants
  - Higher approval targets lead to negative expected profit under baseline assumptions
- Final conclusions are confirmed on a **strict out-of-time test set**, without retuning.

These outcomes reflect a realistic credit environment where **scale requires pricing, collections, or cross-subsidization**, not risk selection alone.

## Data
The dataset is provided by Lending Club via Kaggle and is not included in this repository.

Download instructions:
- URL: https://www.kaggle.com/datasets/wordsforthewise/lending-club

## Notes
This project intentionally prioritizes:
- Economic validity over metric chasing
- Decision-making over pure prediction
- Simplicity and interpretability over complex models

Also, I would love to hear your comments on my notebook in kaggle
[Credit Risk Analysis](https://www.kaggle.com/code/giorgoslyberis/credit-risk-analysis)
