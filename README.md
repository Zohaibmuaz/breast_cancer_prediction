# Breast Cancer Prediction using Logistic Regression
## Overview
This project implements a logistic regression model to predict breast cancer diagnoses based on various features from the dataset. The model is trained on a dataset containing information about tumors, and it aims to classify tumors as malignant or benign.
## Dataset
The dataset used in this project is the Breast Cancer dataset, which includes various features related to tumor characteristics. The target variable is the diagnosis, indicating whether the tumor is malignant (M) or benign (B).
## Workflow
### Data Loading: 
The dataset is loaded from a CSV file.
### Data Preprocessing:
Remove unnecessary columns (e.g., "Unnamed: 32").

Check for and handle missing values.

Split the data into features (X) and target (y).

Split the dataset into training and testing sets.

Standardize the feature values using StandardScaler.

## Model Training:
Train a logistic regression model on the normalized training data.
## Model Evaluation:
Predict the diagnoses on the test set.

Calculate and display the accuracy of the model.

Compare actual vs. predicted values for misclassifications.

## Requirements
Make sure to have the following libraries installed:

numpy

pandas

scikit-learn

You can install the required libraries using:
bash


pip install numpy pandas scikit-learn

How to Run
Download the Dataset: Ensure you have the breast cancer.csv file in the same directory as your script.

Run the Script:
bash


python breast_cancer_prediction.py

Results

The script will output the accuracy of the model and display a comparison of actual vs. predicted diagnoses for any misclassified instances.

Example Output

accuracy = 0.95
  Actual     Predicted
0 ) M       B
1 ) B       M
...
