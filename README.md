# Smart Workforce Analytics and Employee Attrition Prediction System

An end-to-end machine learning system that predicts employee attrition (Yes/No), identifies the key organizational drivers of turnover, and creates actionable per-employee risk profiles for HR teams.

## Project Overview
* **Dataset:** IBM HR Analytics Employee Attrition dataset (1,470 records, 35 raw features).
* **Target Variable:** `Attrition` (Imbalanced: ~84% No / 16% Yes).
* **Objective:** Clean data, explore underlying turnover drivers, engineer features, and evaluate multiple supervised classification algorithms to pinpoint the most reliable risk model.

## Core Implementation Pipeline
1. **Data Collection & Cleaning:** Loading records, checking for missing entries or duplicate profiles, and discarding non-informative identifiers (`EmployeeCount`, `StandardHours`, `Over18`, `EmployeeNumber`).
2. **Data Preprocessing:** Transforming nominal features via One-Hot Encoding, mapping binary outcomes into numerical representations, and standardizing numerical scales using `StandardScaler`.
3. **Exploratory Data Analysis (EDA):** Visualization of department structures, workload rates, compensation imbalances, and employee feedback distributions.
4. **Model Development:** Fine-tuning and executing a variety of classifiers including Logistic Regression, Decision Tree, Random Forest, and XGBoost Classifier using `StratifiedKFold` validation.
5. **Hyperparameter Optimization:** Tuning models using `GridSearchCV` to cross-validate candidate algorithms and elevate F1 scores or ROC-AUC scores.
6. **Risk Prediction Module:** Applying the final trained predictive model to compute clear risk probability thresholds per employee.
7. **Analytics & Insights:** Drawing structural conclusions from global feature importances to highlight the principal organizational factors driving attrition.

## Environment Dependencies
Ensure you have Python installed, then set up the following required packages:

```text
numpy
pandas
matplotlib
seaborn
scikit-learn
xgboost