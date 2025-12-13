# -*- coding: utf-8 -*-
"""
Created on Fri Dec 12 23:26:13 2025

@author: USER
"""

# ============================================================
# TASK: TABLE BOOKING & ONLINE DELIVERY ANALYSIS
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
required_cols = ["Has Table booking", "Has Online delivery", "Aggregate rating", "Price range"]

for col in required_cols:
    if col not in df.columns:
        print(f"❌ ERROR: Missing required column → {col}")
        exit()

# ------------------------------------------------------------
# 2️⃣ Percentage of Restaurants Offering Table Booking
# ------------------------------------------------------------

print("\n📌 Percentage of Restaurants Offering Table Booking:")

table_booking_percent = (
    df["Has Table booking"].value_counts(normalize=True) * 100
)

print(table_booking_percent)

# ------------------------------------------------------------
# 3️⃣ Percentage of Restaurants Offering Online Delivery
# ------------------------------------------------------------

print("\n📌 Percentage of Restaurants Offering Online Delivery:")

online_delivery_percent = (
    df["Has Online delivery"].value_counts(normalize=True) * 100
)

print(online_delivery_percent)

# ------------------------------------------------------------
# 4️⃣ Average Ratings: With vs Without Table Booking
# ------------------------------------------------------------

print("\n📌 Average Ratings (Table Booking vs No Booking):")

rating_comparison = df.groupby("Has Table booking")["Aggregate rating"].mean()
print(rating_comparison)

# Bar Plot
plt.figure(figsize=(6,4))
rating_comparison.plot(kind='bar', color=['green', 'red'])
plt.title("Average Ratings: Table Booking vs No Booking")
plt.ylabel("Average Rating")
plt.xlabel("Has Table Booking")
plt.xticks(rotation=0)
plt.show()

# ------------------------------------------------------------
# 5️⃣ Online Delivery Availability Across Price Ranges
# ------------------------------------------------------------

print("\n📌 Online Delivery Availability by Price Range:")

online_by_price = df.groupby("Price range")["Has Online delivery"].value_counts(normalize=True).unstack() * 100
print(online_by_price)

# Visualization
plt.figure(figsize=(8,5))
online_by_price.plot(kind='bar', stacked=True, figsize=(8,5))
plt.title("Online Delivery Availability Across Price Ranges")
plt.ylabel("Percentage (%)")
plt.xlabel("Price Range")
plt.xticks(rotation=0)
plt.legend(title="Has Online Delivery")
plt.show()
