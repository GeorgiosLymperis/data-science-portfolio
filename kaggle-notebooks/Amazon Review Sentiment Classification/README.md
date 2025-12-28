# Amazon Review Sentiment Classification

This project builds and compares sentiment classifiers for Amazon product reviews
using both classical NLP features (TF-IDF) and modern transformer-based embeddings
from DistilBERT, combined with logistic regression.

## Overview

The goal is to classify Amazon reviews into negative or positive sentiment.
The notebook explores how much performance can be gained by replacing traditional
bag-of-words representations with contextual embeddings from a pretrained
transformer model.

Rather than fine-tuning the transformer, DistilBERT is used as a fixed feature
extractor, enabling efficient training while still leveraging rich language
representations.

## Key Features
- End-to-end sentiment analysis pipeline:
    - data loading from compressed FastText format
    - text cleaning and preprocessing
    - exploratory analysis of review lengths
- Classical baseline:
    - TF-IDF (uni- and bi-grams)
    - Logistic regression classifier
- Transformer-based representations:
    - DistilBERT embeddings with:
        - mean pooling of the last hidden layer
        - average of the last N layers
        - weighted sum of the last layers
- Fair model comparison on a validation set
- Final evaluation with:
    - accuracy
    - classification report
    - confusion matrix
- Reproducible experiments with fixed random seeds

## Results
Validation performance shows that transformer embeddings provide consistent, though
modest, improvements over the strong TF-IDF baseline:

- DistilBERT Last Layer (mean pooling): ~0.882
- DistilBERT Weighted Sum: ~0.881
- TF-IDF + Logistic Regression: ~0.879
- DistilBERT Avg Last N Layers: ~0.878

Key observations:

- Mean pooling over the last DistilBERT layer yields the best results.
- TF-IDF remains highly competitive, highlighting the strength of classical NLP
for sentiment tasks.
- Gains from more complex layer aggregation are marginal for this dataset.

Final evaluation on a held-out test set reports accuracy, class-level precision/recall,
and a confusion matrix, showing balanced performance across both sentiment classes.

## Data
The dataset consists of Amazon product reviews in FastText format and is provided
via Kaggle. Due to licensing restrictions, it is not included in this repository.

Download from:
- URL: [Amazon Reviews for Sentiment Analysis](https://www.kaggle.com/datasets/bittlingmayer/amazonreviews)

Also, I would love to hear your comments on my notebook in kaggle
[Logistic Regr (DistilBERT embs) sentiment analysis](https://www.kaggle.com/code/giorgoslyberis/logistic-regr-distilbert-embs-sentiment-analysis)

