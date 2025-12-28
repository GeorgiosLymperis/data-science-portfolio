# E-commerce A/B Testing Analysis

This project analyzes an e-commerce A/B experiment to evaluate whether a new product experience improves user conversion.

## Overview

The objective is to assess the causal impact of a new landing page on conversion rate using a randomized A/B test.
The analysis applies multiple complementary statistical frameworks to ensure robustness and consistency of conclusions.

The dataset contains user-level experiment assignments, conversion outcomes, timestamps, and country information, enabling both global and segment-level evaluation.

## Key Features

- End-to-end A/B testing workflow

- Sample Ratio Mismatch (SRM) check via chi-square test

- Frequentist inference:

    - one-sided z-test for proportions

    - covariate-adjusted logistic regression

- Non-parametric bootstrap for uncertainty estimation

- Bayesian A/B testing with Beta–Binomial conjugate model

- Segment-level analysis with interaction terms

- Translation of statistical results into business impact

## Statistical Methods

- **Hypothesis testing**

    - One-sided test for conversion uplift

    - Proper control of assumptions and interpretation

- **Bootstrap analysis**

    - Empirical confidence intervals for absolute lift

    - Focus on effect size and economic relevance

- **Logistic regression**

    - Treatment indicator

    - Country fixed effects

    - Treatment × country interactions

- **Bayesian inference**

    - Posterior sampling

    - Direct estimation of $P(p_{treatment}>p_{control})$

## Results

Across all analytical approaches, results are highly consistent:

- Estimated treatment effect is **small and negative**

- No statistically significant uplift in conversion

- Bootstrap confidence intervals are narrow and centered near zero

- Bayesian posterior assigns low probability to treatment superiority

- No reliable heterogeneous effects by country

Observed differences are not due to insufficient data, but reflect the absence of a meaningful treatment effect.

## Business Impact

Assuming realistic traffic volumes (e.g., 100,000 users per day), the expected effect corresponds to a **reduction in daily conversions** under treatment.

Even under optimistic assumptions, the potential upside is economically negligible relative to downside risk.

**Final recommendation**: do not deploy the treatment, either globally or in targeted segments.

## Data
Dataset is from Kaggle. Due to licensing restrictions, it is not included.  
Download instructions:
- URL: [Ecommerce AB Testing 2022](https://www.kaggle.com/datasets/putdejudomthai/ecommerce-ab-testing-2022-dataset1)

Also, I would love to hear your comments on my notebook in kaggle
[AB Testing](https://www.kaggle.com/code/giorgoslyberis/ab-testing)