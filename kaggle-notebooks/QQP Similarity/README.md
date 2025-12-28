# Text Similarity with Siamese CNNs and RNNs

This project implements the paper
**“Text similarity based on two independent channels: Siamese Convolutional Neural Networks and Siamese Recurrent Neural Networks”** (Neurocomputing, 2025),
to detect whether two questions are semantically equivalent.

## Overview

The goal is to perform binary text similarity / duplicate detection on question pairs.
The model follows a dual-channel Siamese architecture:

- a **Siamese CNN (SCNN)** branch to capture local n-gram patterns, and

- a **Siamese RNN (SRNN)** branch to capture sequential and contextual semantics.

Each channel independently encodes both questions, and similarity is modeled through
element-wise squared distance. The two predictions are combined to produce the final
duplicate probability.

The project demonstrates how complementary neural encoders can be jointly trained for
robust semantic matching.

## Key Features

- **End-to-end pipeline:**
  - data loading and cleaning
  - tokenization and padding of question pairs
  - pretrained word embeddings initialization
  - dual-channel Siamese model training
- **SCNN encoder** with multiple convolution filters (3, 4, 5-grams)
- **SRNN encoder** with stacked bidirectional LSTMs
- **Pretrained GloVe 100d embeddings**, fine-tuned during training
- **Joint optimization** with weighted multi-output loss
- **Early stopping** based on validation loss
- **Evaluation with:**
  - accuracy
  - confusion matrix
  - precision, recall, and F1-score
- **Learning curves** for training diagnostics
- **Inference examples** for real-world usage

## Model Architecture

The model consists of two independent Siamese branches implemented in **TensorFlow**:

### Siamese CNN (SCNN):
- Embedding layer (GloVe initialization)
- Conv1D filters with kernel sizes: 3, 4, 5
- Global max pooling
- Dense projection to 128-d vector

### Siamese RNN (SRNN):
- Embedding layer (shared initialization)
- Stacked Bidirectional LSTMs:
  - 128 → 128 → 64 units
- Dense projection to 128-d vector

### For each channel:
- The two question embeddings are compared using **squared difference**.
- A sigmoid layer outputs the probability of duplication.

**Final prediction:** average of SCNN and SRNN probabilities.

## Results

On the held-out test set, the model achieves:

- **Accuracy:** ~0.83
- **Class 0 (not duplicate):**
  - Precision: 0.90, Recall: 0.83, F1: 0.86
- **Class 1 (duplicate):**
  - Precision: 0.74, Recall: 0.84, F1: 0.79

### Key observations:
- High recall for duplicates indicates strong ability to identify semantically equivalent questions.
- Precision for duplicates is lower, reflecting some false positives, which is common in semantic similarity tasks.
- The model performs better at rejecting non-duplicates, yielding robust filtering.

Inference examples show intuitive behavior, assigning high similarity to equivalent questions (e.g., password reset paraphrases) and near-zero probability to unrelated pairs.
