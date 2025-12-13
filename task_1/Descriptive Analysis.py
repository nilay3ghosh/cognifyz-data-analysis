# -*- coding: utf-8 -*-
"""
Created on Fri Dec 12 23:19:49 2025

@author: USER
"""

# ============================================================
# TASK: DESCRIPTIVE ANALYSIS
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------
# 1️⃣ Load Dataset
# ------------------------------------------------------------

df = pd.read_csv("D:\MACHINE_LEARNING___PYTHON\cognifyz_task\Dataset .csv")  

print("Dataset Loaded Successfully!\n")

# ------------------------------------------------------------
# 2️⃣ Basic Statistical Measures for Numerical Columns
# ------------------------------------------------------------

print("\n📌 BASIC STATISTICS FOR NUMERICAL COLUMNS:")
print(df.describe())   # mean, median, std, min, max, quartiles

# Additional stats if needed
print("\nMEDIAN OF NUMERICAL COLUMNS:")
print(df.median(numeric_only=True))

# ------------------------------------------------------------
# 3️⃣ Distribution of Categorical Variables
# ------------------------------------------------------------

categorical_cols = ["Country Code", "City", "Cuisines"]

print("\n📌 CATEGORICAL COLUMN DISTRIBUTIONS:")

for col in categorical_cols:
    if col in df.columns:
        print(f"\nTop 10 categories in '{col}':")
        print(df[col].value_counts().head(10))

        # Plot distribution
        plt.figure(figsize=(10,4))
        df[col].value_counts().head(10).plot(kind='bar')
        plt.title(f"Top Categories in {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.xticks(rotation=45)
        plt.show()
    else:
        print(f"\n❌ Column '{col}' not found in dataset!")

# ------------------------------------------------------------
# 4️⃣ Top Cuisines with Highest Number of Restaurants
# ------------------------------------------------------------

if "Cuisines" in df.columns:
    print("\n📌 TOP 10 CUISINES WITH MOST RESTAURANTS:")
    top_cuisines = df["Cuisines"].value_counts().head(10)
    print(top_cuisines)

    plt.figure(figsize=(10,4))
    top_cuisines.plot(kind='bar')
    plt.title("Top 10 Cuisines")
    plt.xlabel("Cuisine")
    plt.ylabel("Number of Restaurants")
    plt.xticks(rotation=45)
    plt.show()

else:
    print("\n❌ Column 'Cuisines' not found!")

# ------------------------------------------------------------
# 5️⃣ Top Cities with Highest Number of Restaurants
# ------------------------------------------------------------

if "City" in df.columns:
    print("\n📌 TOP 10 CITIES WITH MOST RESTAURANTS:")
    top_cities = df["City"].value_counts().head(10)
    print(top_cities)

    plt.figure(figsize=(10,4))
    top_cities.plot(kind='bar', color='green')
    plt.title("Top 10 Cities with Most Restaurants")
    plt.xlabel("City")
    plt.ylabel("Number of Restaurants")
    plt.xticks(rotation=45)
    plt.show()

else:
    print("\n❌ Column 'City' not found!")
