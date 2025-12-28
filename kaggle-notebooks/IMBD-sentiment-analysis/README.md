# IMDB Sentiment Analysis: Classical ML vs Transformers

This project compares multiple approaches for binary sentiment classification on
IMDB movie reviews, ranging from classical machine learning baselines to modern
transformer-based models.

## Overview

The goal is to classify movie reviews as positive or negative and to study
the trade-offs between:

- traditional NLP pipelines,

- zero-shot transfer with pretrained transformers,

- supervised fine-tuning,

- and prompt-based instruction following.

Rather than focusing on a single model, the notebook provides a comparative,
end-to-end analysis of performance, data efficiency, and practical usability
across paradigms.

## Key Features

- **Complete sentiment analysis pipeline:**
  - data loading and deduplication
  - text cleaning (HTML, URLs, normalization)
  - exploratory analysis of review lengths

- **Classical baseline:**
  - TF-IDF features (up to 5k terms)
  - Logistic Regression classifier

- **Zero-shot inference:**
  - DeBERTa model via Hugging Face pipeline
  - no task-specific training required

- **Supervised fine-tuning:**
  - `bert-base-uncased` fine-tuned on labeled IMDB data
  - evaluation with accuracy and F1 scores

- **Prompt-based inference:**
  - FLAN-T5 for instruction-driven sentiment classification
  - multiple prompt styles tested

- **Evaluation with:**
  - accuracy
  - precision, recall, F1-score
  - confusion matrices

- Fixed random seeds and controlled subsampling for reproducibility

## Models Compared

The project evaluates four sentiment classification paradigms:

1. **TF-IDF + Logistic Regression**  
   A strong classical NLP baseline using bag-of-words features and a linear classifier.

2. **Zero-Shot DeBERTa**  
   A pretrained DeBERTa model used in a zero-shot setting, mapping reviews to sentiment labels via natural language prompts.

3. **Fine-Tuned BERT**  
   A supervised transformer approach where `bert-base-uncased` is fine-tuned on a labeled subset of IMDB reviews using the Hugging Face Trainer API.

4. **Prompted FLAN-T5**  
   An instruction-tuned sequence-to-sequence model that infers sentiment purely from natural language prompts, without any task-specific training.

## Results

**FLAN-T5 achieves the highest accuracy and F1-score among all evaluated models**,
demonstrating the strong capability of instruction-tuned generative models for
sentiment classification without any task-specific training.
*Note:* Results are obtained on a smaller evaluation subset for computational
reasons, but the observed performance highlights the strong potential of prompt-
based large language models for zero-shot classification.