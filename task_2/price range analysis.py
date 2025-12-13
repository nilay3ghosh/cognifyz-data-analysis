# -*- coding: utf-8 -*-
"""
Created on Fri Dec 12 23:28:04 2025

@author: USER
"""

# ============================================================
# TASK: PRICE RANGE ANALYSIS
# ============================================================

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1️⃣ Load Dataset
# ------------------------------------------------------------

df = pd.read_csv("D:\MACHINE_LEARNING___PYTHON\cognifyz_task\Dataset .csv")   

print("Dataset Loaded Successfully!")
print(df.head())

# Check required columns
required_cols = ["Price range", "Aggregate rating", "Rating color"]

for col in required_cols:
    if col not in df.columns:
        print(f"❌ ERROR: Missing required column → {col}")
        exit()

# ------------------------------------------------------------
# 2️⃣ Most Common Price Range
# ------------------------------------------------------------

print("\n📌 MOST COMMON PRICE RANGE:")

most_common_price = df["Price range"].value_counts().idxmax()
print(f"Most Common Price Range = {most_common_price}")

plt.figure(figsize=(6,4))
df["Price range"].value_counts().plot(kind="bar", color="skyblue")
plt.title("Frequency of Price Ranges")
plt.xlabel("Price Range")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.show()

# ------------------------------------------------------------
# 3️⃣ Average Rating for Each Price Range
# ------------------------------------------------------------

print("\n📌 AVERAGE RATING BY PRICE RANGE:")

avg_rating = df.groupby("Price range")["Aggregate rating"].mean()
print(avg_rating)

plt.figure(figsize=(6,4))
avg_rating.plot(kind='bar', color='orange')
plt.title("Average Rating for Each Price Range")
plt.xlabel("Price Range")
plt.ylabel("Average Rating")
plt.xticks(rotation=0)
plt.show()

# ------------------------------------------------------------
# 4️⃣ Rating Color with Highest Average Rating
# ------------------------------------------------------------

print("\n📌 RATING COLOR WITH HIGHEST AVERAGE RATING:")

color_rating = df.groupby("Rating color")["Aggregate rating"].mean()
print(color_rating)

highest_color = color_rating.idxmax()
print(f"\n🎯 Rating Color with Highest Average Rating = {highest_color}")

plt.figure(figsize=(6,4))
color_rating.plot(kind='bar', color='green')
plt.title("Average Rating for Each Rating Color")
plt.xlabel("Rating Color")
plt.ylabel("Average Rating")
plt.xticks(rotation=45)
plt.show()
