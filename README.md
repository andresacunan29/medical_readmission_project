# Hospital Readmission Prediction

## Overview
This project builds a baseline machine learning model to predict whether a patient will be readmitted to the hospital based on demographic, clinical, and visit-related features.

Hospital readmission prediction is a challenging and imbalanced classification problem commonly encountered in healthcare analytics. The goal of this project is to establish a clean, interpretable baseline model and evaluate its performance using industry-standard practices.

---

## Objective
Develop an end-to-end baseline machine learning pipeline that:
- Handles mixed numerical and categorical clinical data
- Accounts for class imbalance
- Produces interpretable results
- Serves as a foundation for future model improvements

---

## Dataset
The dataset contains patient-level hospital visit information, including:
- Demographics (e.g. age)
- Length of hospital stay
- Number of procedures, medications, and emergency visits
- Medical specialty and diagnosis categories
- Readmission outcome (target variable)

The target variable is whether a patient was readmitted to the hospital.

---

## Project Structure
ds-readmissions/
├── notebooks/
│ ├── 01_eda_readmissions.ipynb
│ └── 02_model_baseline.ipynb
├── data/
│ └── raw/
├── models/
├── reports/
├── requirements.txt
└── README.md



---

## Exploratory Data Analysis
Initial exploratory analysis includes:
- Dataset shape and feature types
- Target variable distribution
- Identification of numerical and categorical features
- Basic data quality checks

This analysis is documented in `01_eda_readmissions.ipynb`.

---

## Modeling Approach

### Baseline Model
A logistic regression model was used as the baseline classifier.

### Preprocessing
An end-to-end preprocessing pipeline was built using `scikit-learn`:
- Numerical features: median imputation
- Categorical features: most frequent imputation + one-hot encoding
- Implemented using `ColumnTransformer` and `Pipeline`

### Train-Test Split
- 80/20 split
- Stratified by the target variable to preserve class distribution

### Class Imbalance
Because readmissions are less frequent than non-readmissions, class imbalance was addressed using:
- `class_weight="balanced"` in logistic regression

---

## Evaluation Metrics
Model performance was evaluated using:
- Classification report (precision, recall, F1-score)
- ROC AUC
- ROC curve visualization

### Results
- ROC AUC: approximately **0.64–0.65**
- The model shows moderate discriminative ability
- Recall for readmitted patients is lower than for non-readmitted patients, which is expected for a simple baseline model

---

## Model Interpretation
Logistic regression coefficients were analyzed to identify features most strongly associated with readmission risk.

This step improves model transparency and helps connect predictions to clinical and operational factors.

---

## Key Takeaways
- Predicting hospital readmissions from tabular clinical data is challenging
- A clean baseline model provides valuable benchmarking
- Proper preprocessing and evaluation are critical in healthcare ML tasks
- Interpretability is essential when working with clinical data

---

## Future Improvements
Potential next steps include:
- Cross-validation and hyperparameter tuning
- Tree-based models (Random Forest, Gradient Boosting)
- Feature engineering and domain-informed transformations
- Advanced interpretability techniques (e.g. SHAP)

---

## Tools & Technologies
- Python
- pandas, numpy
- scikit-learn
- matplotlib, seaborn
- Jupyter Notebook

