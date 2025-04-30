
# 🩺 Diabetes Prediction Using Machine Learning

This project implements a machine learning approach to predict whether a patient is diabetic or not based on medical diagnostic data. It uses **Random Forest** and **XGBoost** classifiers with appropriate data preprocessing, class imbalance handling, and hyperparameter tuning to ensure robust and accurate predictions.

---

## 📂 Project Structure

```
├── diabetes.csv                # Dataset used for training and evaluation
├── random_forest_model.pkl     # Saved Random Forest model
├── xgboost_model.pkl           # Saved XGBoost model
├── random_forest_script.py     # Code to train and evaluate Random Forest
├── xgboost_script.py           # Code to train and evaluate XGBoost
└── README.md                   # Project documentation
```

---

## 🧠 Problem Statement

The goal is to develop a supervised binary classification model to predict the presence of diabetes based on patient data. The model must address:
- **Missing/invalid data** (e.g., 0 values in medical features),
- **Class imbalance**, and
- **Model optimization** through **hyperparameter tuning**.

---

## 📊 Dataset

- Source: [Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- Target: `Outcome` (0 = non-diabetic, 1 = diabetic)
- Key features:
  - Glucose
  - BloodPressure
  - BMI
  - Insulin
  - Age
  - etc.

---

## ⚙️ Preprocessing

- Replaced invalid `0` values in critical columns (`Glucose`, `BloodPressure`, `BMI`, `SkinThickness`, `Insulin`) with median values.
- Handled class imbalance:
  - **Data-level**: Applied **SMOTE** (Synthetic Minority Oversampling Technique).
  - **Algorithm-level**: Used **`class_weight='balanced'`** in models.
- Split dataset into training (80%) and test (20%) sets.

---

## 🚀 Models Used

### 🔁 Random Forest
- Ensemble of decision trees using majority voting.
- Handles non-linearity and reduces overfitting through bagging.
- Suitable for small-to-medium-sized medical datasets.
- Hyperparameters tuned using **GridSearchCV**.

### ⚡ XGBoost
- Gradient boosting algorithm with regularization.
- Efficient and accurate with good handling of imbalanced classes.
- Also optimized using **GridSearchCV** and `scale_pos_weight`.

---

## 📈 Evaluation Metrics

Key metrics used to evaluate model performance:
- **Accuracy**
- **Precision**
- **Recall (Sensitivity)** – Crucial for detecting diabetic cases
- **F1-score**
- **Confusion Matrix**

### 🔬 Results Summary

| Metric             | Random Forest | XGBoost |
|--------------------|----------------|---------|
| Accuracy           | 75.97%         | 75.32%  |
| Recall (Class 1)   | 0.80           | 0.78    |
| F1-score (Class 1) | 0.70           | 0.69    |

> **Conclusion**: Random Forest slightly outperforms XGBoost in recall and F1-score, making it more suitable for medical use cases where identifying true positives is critical.

---

## 💾 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/diabetes-prediction.git
   cd diabetes-prediction
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the training script:
   ```bash
   python random_forest_script.py
   # or
   python xgboost_script.py
   ```

---

## 📚 Requirements

- Python 3.7+
- pandas, numpy
- scikit-learn
- seaborn, matplotlib
- imbalanced-learn
- xgboost
- joblib

Install all dependencies via:

```bash
pip install pandas numpy scikit-learn seaborn matplotlib imbalanced-learn xgboost joblib
```

---

## 📌 Future Work

- Integrate with a web UI for user-friendly prediction.
- Deploy as an API using Flask/FastAPI.
- Try deep learning techniques (e.g., MLP).

---

## 📄 License

This project is licensed under the MIT License.
