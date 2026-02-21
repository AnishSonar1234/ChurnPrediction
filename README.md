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

customerID

tenure (after feature creation)
