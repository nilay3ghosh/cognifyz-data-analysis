# -*- coding: utf-8 -*-
"""
Created on Fri Dec 12 23:34:15 2025

@author: USER
"""

# ============================================================
# TASK: PREDICTIVE MODELING
# ============================================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# ------------------------------------------------------------
# 1️⃣ Load Dataset
# ------------------------------------------------------------

df = pd.read_csv("D:\MACHINE_LEARNING___PYTHON\cognifyz_task\Dataset .csv")  
print("Dataset Loaded Successfully!\n")

# ------------------------------------------------------------
# 2️⃣ Preprocessing & Feature Selection
# ------------------------------------------------------------

# Remove unnecessary columns if present
drop_cols = ["Restaurant ID", "URL"]
for col in drop_cols:
    if col in df.columns:
        df.drop(col, axis=1, inplace=True)

# Drop rows where aggregate rating is missing
df = df[df["Aggregate rating"] > 0]

# Encode categorical columns
categorical_cols = df.select_dtypes(include=["object"]).columns

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# ------------------------------------------------------------
# 3️⃣ Define Input (X) and Target (y)
# ------------------------------------------------------------

X = df.drop("Aggregate rating", axis=1)
y = df["Aggregate rating"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Train/Test Split Completed!")
print(f"Training Samples: {len(X_train)}")
print(f"Testing Samples: {len(X_test)}\n")

# ------------------------------------------------------------
# 4️⃣ Function for Model Evaluation
# ------------------------------------------------------------

def evaluate_model(model, name):
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"\n📌 MODEL: {name}")
    print("---------------------------")
    print(f"MAE  : {mae:.4f}")
    print(f"MSE  : {mse:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R²    : {r2:.4f}")

    return [name, mae, mse, rmse, r2]

# ------------------------------------------------------------
# 5️⃣ Model Training & Evaluation
# ------------------------------------------------------------

results = []

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
results.append(evaluate_model(lr, "Linear Regression"))

# Decision Tree
dt = DecisionTreeRegressor(max_depth=10, random_state=42)
dt.fit(X_train, y_train)
results.append(evaluate_model(dt, "Decision Tree"))

# Random Forest
rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
results.append(evaluate_model(rf, "Random Forest"))

# ------------------------------------------------------------
# 6️⃣ Performance Comparison Table
# ------------------------------------------------------------

results_df = pd.DataFrame(
    results, columns=["Model", "MAE", "MSE", "RMSE", "R2 Score"]
)

print("\n\n==================== MODEL PERFORMANCE SUMMARY ====================")
print(results_df)

# Save results
results_df.to_csv("model_performance_results.csv", index=False)
print("\n📁 Results saved as 'model_performance_results.csv'")
