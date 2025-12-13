# -*- coding: utf-8 -*-
"""
Created on Fri Dec 12 22:48:48 2025

@author: USER
"""

# ============================================================
# TASK: DATA EXPLORATION & PREPROCESSING
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------
# 1️⃣ Load Dataset
# ------------------------------------------------------------

df = pd.read_csv("D:\MACHINE_LEARNING___PYTHON\cognifyz_task\Dataset .csv")     # <--- Replace with your dataset
print("Dataset Loaded Successfully!\n")

# Number of rows and columns
print("Shape of Dataset (rows, columns):", df.shape)

# Display top 5 rows
print("\nFirst 5 Rows:")
print(df.head())

# ------------------------------------------------------------
# 2️⃣ Check for Missing Values
# ------------------------------------------------------------

print("\nMissing Values in Each Column:")
print(df.isnull().sum())

# Optional: Visualize missing values
plt.figure(figsize=(10,5))
sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
plt.title("Missing Value Heatmap")
plt.show()

# Handling missing values: (Modify based on dataset characteristics)

# Example: Fill numeric columns with median
numeric_cols = df.select_dtypes(include=np.number).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# Example: Fill categorical columns with mode
categorical_cols = df.select_dtypes(include="object").columns
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

print("\nMissing values handled successfully!")

# ------------------------------------------------------------
# 3️⃣ Data Type Conversion (if necessary)
# ------------------------------------------------------------

print("\nData Types Before Conversion:")
print(df.dtypes)

# Example conversions:
# Convert numeric column stored as string to numeric
for col in df.columns:
    if df[col].dtype == "object":
        try:
            df[col] = pd.to_numeric(df[col])
        except:
            pass  # Ignore non-numeric text columns

# Print data types after conversion
print("\nData Types After Conversion:")
print(df.dtypes)

# ------------------------------------------------------------
# 4️⃣ Analyze Target Variable Distribution ("Aggregate rating")
# ------------------------------------------------------------

if "Aggregate rating" in df.columns:

    print("\nSummary of Aggregate Rating:")
    print(df["Aggregate rating"].describe())

    # Distribution plot
    plt.figure(figsize=(8,4))
    sns.histplot(df["Aggregate rating"], kde=True)
    plt.title("Distribution of Aggregate Rating")
    plt.xlabel("Aggregate Rating")
    plt.ylabel("Count")
    plt.show()

    # ------------------------------------------------------------
    # 5️⃣ Check Class Imbalance
    # ------------------------------------------------------------

    print("\nClass Counts for Aggregate Rating:")
    print(df["Aggregate rating"].value_counts())

    # Visualize class frequencies
    plt.figure(figsize=(10,5))
    sns.countplot(x=df["Aggregate rating"])
    plt.title("Class Distribution of Aggregate Rating")
    plt.xlabel("Aggregate Rating")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45)
    plt.show()

else:
    print("\n❌ Column 'Aggregate rating' NOT found in dataset!")
