# Car Sales Elasticity Modeling

## Overview
This project analyzes car sales using classical and nonlinear statistical models:
- OLS regression
- Negative Binomial GLM
- Poisson GAM with spline terms
- Price elasticity estimation via numerical differentiation

## Why this project matters
Understanding elasticity is crucial for pricing decisions in automotive markets. 
This analysis provides a quantitative framework for evaluating how consumers react 
to price changes across different market segments.


## Key Questions
- How do price, horsepower, fuel efficiency, and size affect sales volume?
- How does price elasticity vary across market segments?

## Techniques Used
- Statistical modeling (OLS, GLM, GAM)
- EDA and outlier detection
- Residual diagnostics and model validation
- Spline modeling and elasticity interpretation

## Results
- Sales are highly elastic between 30k–50k price range.
- GAM spline effects reveal nonlinear consumer sensitivity.
- Classical models and GAMs produce different elasticity profiles.

## Notebook
`car-sales-price-elasticity-with-regression.ipynb`

## Data
Dataset is from Kaggle. Due to licensing restrictions, it is not included.  
Download instructions:
- URL: [Car sales](https://www.kaggle.com/datasets/gagandeep16/car-sales)

Also, I would love to hear your comments on my notebook in kaggle
[car-sales-price-elasticity-with-regression](https://www.kaggle.com/code/giorgoslyberis/car-sales-price-elasticity-with-regression)
