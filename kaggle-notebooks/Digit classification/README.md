# Sudoku Digit Classification (CNN, PyTorch)

This project builds a convolutional neural network (CNN) to classify digits extracted
from Sudoku board images.

## Overview
The goal is to classify grayscale 28×28 digit images (0–9). 
The dataset contains handwritten or printed Sudoku digits, including imperfect, 
erased, or low-quality samples that make classification challenging.

## Key Features
- Custom PyTorch `Dataset` with realistic augmentations:
  - rotation
  - brightness shifts
  - translation
  - scale variation
- CNN architecture with:
  - batch normalization
  - dropout regularization
  - three convolutional blocks
- Early stopping based on validation loss
- Confusion matrix and class-level error inspection
- Visualization of misclassified examples

## Model Architecture
- Conv(1→32) → BN → ReLU  
- Conv(32→64) → BN → ReLU → MaxPool  
- Conv(64→128) → BN → ReLU  
- Conv(128→128) → BN → ReLU → MaxPool  
- Linear → ReLU → Dropout → Linear(10)

## Results
The model achieves strong performance, with most errors coming from ambiguous digits, 
notably 0 and 8 due to erased or incomplete writing.

Evaluation includes:
- Accuracy on validation and test sets
- Confusion matrix
- Visual analysis of misclassified samples

## Data
Dataset is from Kaggle. Due to licensing restrictions, it is not included.  
Download instructions:
- URL: [Sudoku Digit Classification](https://www.kaggle.com/datasets/rohit369/sudoku-digit-classification)

Also, I would love to hear your comments on my notebook in kaggle
[car-sales-price-elasticity-with-regression](https://www.kaggle.com/code/giorgoslyberis/cnn-sudoku-digit-classifier)

