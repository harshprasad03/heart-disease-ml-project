# 🫀 Heart Disease Prediction using Machine Learning

## 📌 Project Overview

This project builds a complete end-to-end machine learning pipeline to predict the presence of heart disease using clinical and diagnostic features.

The objective is to develop a reliable predictive model while ensuring:

- Proper exploratory data analysis (EDA)
- Robust model evaluation
- Cross-validation
- Hyperparameter tuning
- Overfitting assessment
- Clinical interpretation of results

Given the medical context, model performance is evaluated primarily using **ROC-AUC and recall**, rather than accuracy alone.

---

## 🧠 Problem Statement

Heart disease is one of the leading causes of mortality worldwide. Early detection using machine learning can assist in risk stratification and clinical decision-making.

This project aims to:

- Predict whether a patient has heart disease (binary classification)
- Compare multiple ML algorithms
- Optimize the best-performing model
- Evaluate generalization performance

---

## 📊 Dataset

The dataset contains patient-level medical attributes including:

- Age
- Sex
- Resting Blood Pressure
- Cholesterol
- Maximum Heart Rate
- Exercise-Induced Angina
- ST Depression (Oldpeak)
- Chest Pain Type
- Fasting Blood Sugar
- Resting ECG results
- ST Slope

Target Variable:
- `HeartDisease` (0 = No, 1 = Yes)

---

## 🔎 Project Workflow

### 1️⃣ Exploratory Data Analysis (EDA)
- Data cleaning and preprocessing
- Statistical significance testing
- Correlation analysis
- Feature scaling
- PCA analysis (variance retention study)

### 2️⃣ Modeling Phase
Three classification models were trained and evaluated:

- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)

Evaluation metrics:
- 5-fold Cross-Validation (ROC-AUC)
- Test Accuracy
- Test ROC-AUC
- Classification Report
- ROC Curve Comparison

Random Forest achieved the strongest baseline performance.

---

### 3️⃣ Hyperparameter Tuning

GridSearchCV was used to optimize Random Forest.

Parameters tuned:
- `n_estimators`
- `max_depth`
- `min_samples_split`
- `min_samples_leaf`

Scoring metric: **ROC-AUC**

The tuned model showed:

- Slight improvement in ROC-AUC
- Improved generalization
- Minimal overfitting (CV vs Test AUC difference ≈ 0.0026)

---

## 🏆 Final Model Performance

| Model | Test Accuracy | Test ROC-AUC |
|-------|--------------|--------------|
| Baseline Random Forest | ~0.87 | ~0.924 |
| Tuned Random Forest | ~0.886 | ~0.926 |

The tuned Random Forest was selected as the final model due to:

- Highest ROC-AUC
- Strong recall for positive class
- Stable cross-validation performance
- Minimal overfitting

---

## 📈 Overfitting Assessment

To ensure robust generalization:

- Cross-validation ROC-AUC was compared to test ROC-AUC
- Difference ≈ 0.0026
- Indicates strong stability and minimal variance inflation

This confirms that the model generalizes well to unseen data.

---

## 🧪 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook

---

## 📁 Project Structure

```

heart_disease_project/
│
├── data/
│   ├── heart.csv
│   └── heart_cleaned.csv
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Modeling.ipynb
│   └── 03_Tuning.ipynb
│
├── models/
│   └── (model files excluded via .gitignore)
│
└── README.md

```

---

## 🎯 Key Takeaways

- Random Forest outperformed linear and margin-based classifiers.
- Hyperparameter tuning provided marginal but meaningful performance improvement.
- Cross-validation stability indicates strong model reliability.
- ROC-AUC is a more appropriate metric for medical classification tasks than accuracy alone.

---

## 🚀 Future Improvements

- Implement SHAP for model interpretability
- Deploy model using Streamlit
- Perform threshold optimization for improved recall
- Test additional ensemble methods (XGBoost, LightGBM)

---

## 👨‍💻 Author

Harsh Prasad  
MSc Data Science  
Student | VIT Vellore

---
