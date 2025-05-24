# Taiwanese Bankruptcy Prediction

**Team Members**: Hubert Witkos, Han Chen, Soujan Niroula  
**Institution**: University of Illinois Urbana-Champaign  
**Date**: May 1, 2025

## 📌 Project Overview

This project focuses on predicting bankruptcy of Taiwanese companies using machine learning models. The dataset—sourced from the UCI Machine Learning Repository—presents a highly imbalanced binary classification problem. We implemented SMOTE for oversampling and evaluated models using the F2 Score to prioritize sensitivity over precision due to the economic risk of false negatives.

## 📊 Dataset

- **Source**: [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Taiwanese+Bankruptcy+Prediction)
- **Period**: 1999–2009
- **Samples**: 6,820 companies
- **Features**: 95 financial indicators + 1 binary target (`Bankrupt.`)

### Preprocessing Highlights

- Removed highly correlated features (|r| > 0.95) → reduced predictors to 78
- Dropped categorical predictors with zero variance
- Applied SMOTE (k = 5, dup_size = 30) to address severe class imbalance (3.23% bankrupt)
- Standardized features before modeling

## ⚙️ Models Implemented

We implemented and compared four models:

### 1. 📈 Penalized Logistic Regression (Elastic Net)
- **Tool**: `glmnet` (R)
- **F2 Score**: 0.567
- **Sensitivity**: 0.736
- **Accuracy**: 0.922

### 2. 🔲 Support Vector Machines (SVM)
- **Kernel**: Radial
- **F2 Score**: 0.542
- **Sensitivity**: 0.679
- **Accuracy**: 0.922

### 3. ⚡ XGBoost
- **Objective**: `binary:logistic`
- **Best threshold**: 0.1
- **F2 Score**: 0.569
- **Sensitivity**: 0.623
- **Accuracy**: 0.952
- **Top Feature**: `Persistent.EPS.in.the.Last.Four.Seasons`

### 4. 🌲 Random Forest
- **Weak performance on minority class**
- **F2 Score**: 0.513
- **Sensitivity**: 0.509
- **Accuracy**: 0.960

## 📈 Model Comparison

| Model            | Sensitivity | Accuracy | F2 Score |
|------------------|-------------|----------|----------|
| Logistic Reg.    | 0.736       | 0.922    | 0.567    |
| SVM (Radial)     | 0.679       | 0.922    | 0.542    |
| XGBoost          | 0.623       | 0.952    | 0.569    |
| Random Forest    | 0.509       | 0.960    | 0.513    |

## 🔍 Key Takeaways

- Logistic regression achieved the best sensitivity and was most robust in identifying bankrupt firms.
- XGBoost had the highest F2 score, balancing sensitivity and precision effectively.
- SVM provided competitive performance, while Random Forest underperformed on the minority class.

## 🧪 Future Work

- Explore alternative resampling strategies (e.g., Edited Nearest Neighbors, Tomek Links)
- Experiment with Naive Bayes, QDA, and ensemble techniques
- Perform feature selection with domain knowledge integration

## 📚 References

1. Chawla et al. (2002). *SMOTE: Synthetic Minority Over-Sampling Technique*
2. Chen & Guestrin (2016). *XGBoost: A Scalable Tree Boosting System*
3. Wang & Liu (2021). *Undersampling Bankruptcy Prediction: Taiwan Bankruptcy Data*

## ✍️ Acknowledgment

This project was completed as part of our coursework at UIUC. Writing, modeling, and formatting support was enhanced with the help of ChatGPT.
