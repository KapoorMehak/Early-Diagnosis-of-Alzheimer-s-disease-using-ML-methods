# Early-Diagnosis-of-Alzheimer-s-disease-using-ML-methods (2021)

## Overview

This project implements multiple machine learning models to classify early-stage Alzheimer’s disease using structured feature data. 

The objective is to compare classical ML methods and evaluate their diagnostic performance using accuracy, sensitivity, specificity, confusion matrices, and ROC curves.

Models implemented:
- Support Vector Machine (SVM)
- Random Forest (RF)
- Artificial Neural Network (ANN / MLP)

---

## Dataset

The dataset used in this study contains structured features with a binary classification label (`Label`).

- Number of samples: 220
- Number of features: 8000
- Target variable: `Label` (0 = Control, 1 = Alzheimer's)

## Methodology

1. Data loading and preprocessing
2. Feature scaling using MinMaxScaler
3. Train-test split (80/20)
4. Model training:
   - SVM (RBF kernel)
   - Random Forest (20 estimators)
   - ANN (3 hidden layers: 90,90,90)
5. 5-fold Cross Validation
6. Evaluation:
   - Accuracy
   - Sensitivity
   - Specificity
   - Confusion Matrix
   - Classification Report
   - ROC Curve comparison
  
## ROC Curve- results

The ROC comparison shows performance differences among SVM, RF, and ANN classifiers.

## Requirements

Python 3.x

Libraries:
- numpy
- pandas
- matplotlib
- scikit-learn
