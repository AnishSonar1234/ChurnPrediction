# 📊 Customer Churn Prediction

This project builds a **Customer Churn Prediction Model** using Machine Learning techniques.  
The goal is to predict whether a customer will leave (churn) based on demographic and service usage data.
![Buinesss Idealogy](images/gemini-3-pro-image-preview-2k_a_From_this_image_crea.png)
---

## 🚀 Project Overview

Customer churn prediction helps businesses:

- 📉 Reduce customer loss  
- 📈 Improve retention strategies  
- 💰 Increase revenue  
- 🎯 Target high-risk customers  

This project includes:

- Data Cleaning
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Handling Class Imbalance
- Model Training
- Model Evaluation

---

## 📁 Project Structure
├── churn.ipynb # Data cleaning & preprocessing  
├── model.ipynb # Model training & evaluation  
├── README.md # Project documentation

---

# 🔍 1️⃣ Data Preprocessing (churn.ipynb)

### 📌 Step 1: Import Libraries

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
```

📌 Step 3: Data Cleaning

Converted TotalCharges to numeric

Handled missing values
Dropped unnecessary columns:
   a. customerID
   b. tenure (after feature creation
```python
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)
```
📌 Step 4: Feature Engineering
Created tenure groups:
```python
df['tenure_group'] = pd.cut(df['tenure'], bins=[0,12,24,48,60,72])
```
📌 Step 5: Encoding
Converted categorical features using One-Hot Encoding:
```python
df_dummies = pd.get_dummies(df, drop_first=True)
```
Converted target variable:
```python
df_dummies['Churn'] = df_dummies['Churn'].map({'Yes':1, 'No':0})
```
📌 Step 6: Save Processed Data
```python
df_dummies.to_csv("tel_churn.csv", index=False)
```
---
🤖 2️⃣ Model Building (model.ipynb)
---

📌 Step 1: Import Libraries
```python
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.combine import SMOTEENN
```
📌 Step 2: Train-Test Split
```python
df = pd.read_csv("tel_churn.csv")

X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```
🌳 Model 1: Decision Tree
```python
dt_model = DecisionTreeClassifier(
    criterion='gini',
    max_depth=7,
    min_samples_leaf=8
)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)
```
Evaluation
```python
print(classification_report(y_test, y_pred_dt))
print(confusion_matrix(y_test, y_pred_dt))
```
⚖ Handling Class Imbalance
Used SMOTEENN (Combination of SMOTE + Edited Nearest Neighbours):
```python
sm = SMOTEENN()
X_resampled, y_resampled = sm.fit_resample(X, y)
```
🌳 Model 2: Decision Tree with SMOTEENN
Trained Decision Tree on resampled dataset for better recall of churn class.

🌲 Final Model: Random Forest
```python
rf_model = RandomForestClassifier(
    n_estimators=100,
    criterion='entropy',
    max_depth=8,
    min_samples_leaf=6,
    random_state=42
)

rf_model.fit(X_resampled, y_resampled)
y_pred_rf = rf_model.predict(X_test)
```
📊 Model Evaluation
```python
print(classification_report(y_test, y_pred_rf))
print(confusion_matrix(y_test, y_pred_rf))
```
🎯 Focus Metrics
Precision
Recall
F1 Score
Confusion Matrix

🏆 Final Result

✅ Random Forest with SMOTEENN performed best
It improved:
   a. Recall for churn class
   b. Reduced false negatives
   c. Overall classification performance
