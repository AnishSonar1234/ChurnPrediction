# 📊 Customer Churn Prediction

This project builds a **Customer Churn Prediction Model** using Machine Learning techniques.  
The goal is to predict whether a customer will leave (churn) based on demographic and service usage data.

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
├── tel_churn.csv # Processed dataset
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

📌 Step &: Model Creation
