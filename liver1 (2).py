import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE

import warnings
warnings.filterwarnings('ignore')


# Load the dataset (assuming it's in a CSV file)
data = pd.read_csv('indian_liver_patient.csv')

data.dropna(inplace=True)

# Encode categorical variables (Gender)
label_encoder = LabelEncoder()
data['Gender'] = label_encoder.fit_transform(data['Gender'])

# Calculate the correlation matrix
correlation_matrix = data.corr()

# Select features with the highest correlation with the 'Result' column
selected_features = correlation_matrix['Dataset'].abs().sort_values(ascending=False).index[1:]

# Split the data into training and testing sets (70% training, 30% testing)
X = data[selected_features]
y = data['Dataset']

# Apply SMOTE to address class imbalance
#Synthetic Minority Over Sampling Technique
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Hyperparameter tuning for Random Forest Classifier using Grid Search
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf_classifier = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf_classifier, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_rf_classifier = grid_search.best_estimator_
best_rf_classifier.fit(X_train, y_train)

# Train a K-NN Classifier
knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_train, y_train)

# Train a Logistic Regression Classifier
logistic_regression_classifier = LogisticRegression(random_state=42)
logistic_regression_classifier.fit(X_train, y_train)

# Train an XGBoost Classifier
xgboost_classifier = XGBClassifier(random_state=42)
xgboost_classifier.fit(X_train, y_train)

# Make predictions on the test data for all classifiers
rf_predictions = best_rf_classifier.predict(X_test)
knn_predictions = knn_classifier.predict(X_test)
logistic_regression_predictions = logistic_regression_classifier.predict(X_test)
xgboost_predictions = xgboost_classifier.predict(X_test)

# Evaluate and print accuracy for all classifiers
rf_accuracy = accuracy_score(y_test, rf_predictions)
knn_accuracy = accuracy_score(y_test, knn_predictions)
logistic_regression_accuracy = accuracy_score(y_test, logistic_regression_predictions)
xgboost_accuracy = accuracy_score(y_test, xgboost_predictions)
print(f'Accuracy of 4 classifiers used for the Liver disease dataset:')
print()
print(f'Random Forest Classifier Accuracy: {rf_accuracy:.2f}')
print(f'K-NN Classifier Accuracy: {knn_accuracy:.2f}')
print(f'Logistic Regression Classifier Accuracy: {logistic_regression_accuracy:.2f}')
print(f'XGBoost Classifier Accuracy: {xgboost_accuracy:.2f}')
print()
print(f'USING RANDOM FOREST CLASSIFIER')
print()
# Take input from the user and predict the probability of liver disease using all classifiers
print("Enter patient data:")
age = float(input("Age (years): "))
gender = input("Gender (Male/Female): ")
total_bilirubin = float(input("Total Bilirubin (mg/dL): "))
direct_bilirubin = float(input("Direct Bilirubin (mg/dL): "))
alkphos = float(input("Alkphos Alkaline Phosphotase (U/L): "))
sgpt = float(input("Sgpt Alamine Aminotransferase (U/L): "))
sgot = float(input("Sgot Aspartate Aminotransferase (U/L): "))
total_proteins = float(input("Total Protiens (g/dL): "))
alb = float(input("ALB Albumin (g/dL): "))
ag_ratio = float(input("A/G Ratio Albumin and Globulin Ratio: "))
print("__________________________________________________")
print("__________________________________________________")
user_data = np.array([age, 0 if gender == 'Male' else 1, total_bilirubin, direct_bilirubin, alkphos, sgpt, sgot, total_proteins, alb, ag_ratio])
user_data[1] = label_encoder.transform([user_data[1]])[0]
user_data = user_data.reshape(1, -1)
user_data = scaler.transform(user_data)

threshold_mild = 0.3
threshold_moderate = 0.6
threshold_severe = 0.8

# Calculate the probability of liver disease using Random Forest Classifier
rf_probability = best_rf_classifier.predict_proba(user_data)[0][1]

# Categorize the severity based on thresholds
if rf_probability < threshold_mild:
    severity_category = "Mild"
    recommendations = "Low risk. Regular check-ups recommended."
elif rf_probability < threshold_moderate:
    severity_category = "Moderate"
    recommendations = "Moderate risk. Consult with a healthcare provider for further evaluation."
elif rf_probability < threshold_severe:
    severity_category = "Severe"
    recommendations = "High risk. Immediate medical attention is advised."
else:
    severity_category = "Critical"
    recommendations = "Critical risk. Urgently seek medical help."

# Print the severity category and recommendations
print(f'Severity Category: {severity_category}')
print(f'Recommended Steps: {recommendations}')

# Set the threshold for binary classification
threshold = 0.5
result = 1 if rf_probability >= threshold else 0
print("__________________________________________________")
print("__________________________________________________")
# Print the result and probability
print(f'Probability of Liver Disease (Random Forest): {rf_probability:.2f}')

# Append user input and result to the CSV file
user_data = user_data.tolist()[0]  # Convert to a list
user_data.append(result)
data.loc[len(data)] = user_data  # Append to the DataFrame
data.to_csv('liver_dataset.csv', index=False)  # Update the dataset file
