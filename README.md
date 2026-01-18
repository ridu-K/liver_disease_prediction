# 🩺 Liver Disease Prediction Using Machine Learning

This project implements a **machine learning–based liver disease prediction system** using the **Indian Liver Patient Dataset**. Multiple classification models are trained and evaluated, with **Random Forest** selected as the primary model for **severity-based prediction and recommendations**.

---

## 📌 Project Overview

* 📂 **Dataset**: Indian Liver Patient Dataset (`indian_liver_patient.csv`)
* 🎯 **Objective**: Predict the presence and severity of liver disease
* 🧠 **Models Used**:

  * Random Forest (with Hyperparameter Tuning)
  * K-Nearest Neighbors (KNN)
  * Logistic Regression
  * XGBoost
* ⚖️ **Imbalance Handling**: SMOTE (Synthetic Minority Oversampling Technique)
* 🧪 **Evaluation Metric**: Accuracy Score
* 🧑‍⚕️ **User Interaction**: CLI-based patient input with severity assessment

---

## 📁 Dataset Description

The dataset consists of patient medical records with the following features:

| Feature          | Description                               |
| ---------------- | ----------------------------------------- |
| Age              | Age of the patient                        |
| Gender           | Male / Female                             |
| Total Bilirubin  | Total bilirubin level                     |
| Direct Bilirubin | Direct bilirubin level                    |
| Alkphos          | Alkaline Phosphotase                      |
| SGPT             | Alanine Aminotransferase                  |
| SGOT             | Aspartate Aminotransferase                |
| Total Proteins   | Total protein level                       |
| Albumin          | Albumin level                             |
| A/G Ratio        | Albumin and Globulin ratio                |
| Dataset          | Target label (Liver disease / No disease) |

---

## 🛠️ Tech Stack

* **Python**
* **Pandas & NumPy**
* **Scikit-learn**
* **XGBoost**
* **Imbalanced-learn (SMOTE)**

---

## 🔄 Data Preprocessing

* Removed missing values
* Encoded categorical variable (`Gender`)
* Selected highly correlated features using correlation matrix
* Standardized features using **StandardScaler**
* Balanced classes using **SMOTE**

---

## 🧠 Model Training & Evaluation

### ✔ Models Trained

* **Random Forest (GridSearchCV optimized)**
* K-Nearest Neighbors
* Logistic Regression
* XGBoost Classifier

### 🔍 Hyperparameter Tuning

Random Forest was optimized using **GridSearchCV** with:

* Number of estimators
* Maximum depth
* Minimum samples split
* Minimum samples per leaf

### 📊 Evaluation Output

Accuracy scores are printed for all four models, allowing direct comparison.

---

## ⭐ Best Model: Random Forest Classifier

Random Forest is used for:

* Final disease prediction
* Probability estimation
* Severity classification

---

## 🚨 Severity Classification Logic

Based on the predicted probability:

| Probability Range | Severity | Recommendation              |
| ----------------- | -------- | --------------------------- |
| < 0.30            | Mild     | Regular check-ups           |
| 0.30 – 0.60       | Moderate | Consult a doctor            |
| 0.60 – 0.80       | Severe   | Immediate medical attention |
| > 0.80            | Critical | Urgent medical care         |

---

## 🧑‍💻 User Input & Prediction

The system accepts real-time patient data via command-line input:

* Age
* Gender
* Bilirubin levels
* Enzyme values
* Protein values

### Output:

* Probability of liver disease
* Severity category
* Medical recommendation

---

## 💾 Dataset Update

After prediction:

* User input and result are appended to a new CSV file:

  ```text
  liver_dataset.csv
  ```
* Enables future retraining and dataset expansion

---

## ▶️ How to Run the Project

1. Install dependencies:

   ```bash
   pip install pandas numpy scikit-learn xgboost imbalanced-learn
   ```
2. Place `indian_liver_patient.csv` in the project directory
3. Run the script:

   ```bash
   python liver (1).py
   ```
4. Enter patient details when prompted

---

## 📌 Applications

* Medical decision support systems
* Early liver disease screening
* Healthcare analytics
* Clinical risk assessment tools

---

## 🔮 Future Improvements

* Web-based interface (Flask / Streamlit)
* ROC-AUC and confusion matrix analysis
* Feature importance visualization
* Deep learning models (ANN)
* Real-time clinical deployment
