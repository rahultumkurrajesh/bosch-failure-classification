# Bosch Production Line — Predictive Failure Classification

## Overview
Binary classification project to predict equipment failures on Bosch production line sensor data.
Built as part of M.Sc. Data Science coursework at the University of Europe for Applied Sciences, Germany.

## Problem
The dataset has a severe class imbalance — only 10.5% of records are actual failures.
A naive model that predicts "no failure" every time would score 89.5% accuracy while being useless.
The goal was to build a model that genuinely detects failures, not just the majority class.

## Approach
- Applied **SMOTE oversampling** to balance the training set before fitting any model
- Benchmarked **four classifiers** using 5-fold cross-validated grid search:
  - Logistic Regression
  - K-Nearest Neighbours (KNN)
  - Random Forest
  - MLP Neural Network
- Evaluated using **ROC AUC** as the primary metric (more meaningful than accuracy on imbalanced data)
- Visualised results via ROC curves and confusion matrices

## Results

| Model | Accuracy | ROC AUC |
|---|---|---|
| Logistic Regression | 74% | 0.56 |
| KNN | 52% | 0.54 |
| **Random Forest** | **87%** | **0.64** |
| Neural Network (MLP) | 81% | 0.55 |

Random Forest was the best performer on both metrics.

## Tech Stack
Python, Scikit-learn, Pandas, NumPy, SMOTE (imbalanced-learn), Matplotlib, Seaborn

## Files
- `Improved_LightGBM_Notebook.ipynb` — full pipeline: data prep, balancing, model training, evaluation
