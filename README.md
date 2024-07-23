# Churn Prediction Project

This project aims to predict customer churn using various machine learning models. Customer churn refers to the loss of clients or customers. By predicting churn, businesses can take proactive measures to retain their customers.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Models Used](#models-used)
- [Results](#results)


## Introduction
Customer churn is a critical issue for many businesses. This project explores multiple machine learning models to predict churn and identify the key factors that influence customer retention.

## Dataset
The dataset used in this project is obtained from https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset 

## Data Preprocessing
- Initial Data Inspection
- Outlier Removal
- Encoding Categorical Variables

## Models Used
We implemented and compared the following machine learning models to predict customer churn:
1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Naive Bayes
5. K-Nearest Neighbors (KNN)
6. Multi-layer Perceptron (MLP)
7. LSTM

## Results
The performance of each model was evaluated using accuracy, precision, recall, and F1-score. The results are summarized below:

| Model                | Accuracy (Train) | Accuracy (Test) | Precision (Yes) | Recall (Yes) |
|----------------------|------------------|-----------------|-----------------|--------------|
| MLP                  | 77.36%           | 72.88%          | 0.40            | 0.34         |
| KNN                  | 81.32%           | 76.27%          | 0.00            | 0.00         |
| Logistic Regression  | 81.89%           | 82.49%          | 0.92            | 0.27         |
| Decision Tree        | 85.28%           | 81.36%          | 0.79            | 0.27         |
| Random Forest        | 85.47%           | 80.23%          | 0.80            | 0.20         |
| LSTM                 | 87.35%           | 77.4%           | 0.55            | 0.15         |
| Gaussian Naive Bayes | 81.32%           | 82.49%          | 0.63            | 0.59         |

Among these, Naive Bayes emerged as the best model for our objectives, particularly excelling in recall for the 'yes' class.

### Naive Bayes Implementation and Improvement
- Initial Implementation
- Data Transformation
- Hyperparameter Tuning
- Feature Importance and Reduction
- Retraining and Evaluation

The final model's performance metrics are:
- Accuracy Score on Train Data: 83.96%
- Accuracy Score on Test Data: 84.18%
- Precision (Yes): 0.65
- Recall (Yes): 0.68
- F1-score (Yes): 0.67

