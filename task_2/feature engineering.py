# -*- coding: utf-8 -*-
"""
Created on Fri Dec 12 23:30:13 2025

@author: USER
"""

# ============================================================
# TASK: FEATURE ENGINEERING
# ============================================================

import pandas as pd

# ------------------------------------------------------------
# 1️⃣ Load Dataset
# ------------------------------------------------------------

df = pd.read_csv("D:\MACHINE_LEARNING___PYTHON\cognifyz_task\Dataset .csv") 
print("Dataset Loaded Successfully!\n")

# Display first few rows
print(df.head())

# ------------------------------------------------------------
# 2️⃣ Extract Additional Features
# ------------------------------------------------------------

print("\n📌 Extracting New Features...")

# Length of restaurant name
if "Restaurant Name" in df.columns:
    df["Name_Length"] = df["Restaurant Name"].astype(str).apply(len)
else:
    print("❌ 'Restaurant Name' not found!")

# Length of address
if "Address" in df.columns:
    df["Address_Length"] = df["Address"].astype(str).apply(len)
else:
    print("❌ 'Address' column not found!")

# Word count in restaurant name
if "Restaurant Name" in df.columns:
    df["Name_Word_Count"] = df["Restaurant Name"].astype(str).apply(lambda x: len(x.split()))

# Word count in address
if "Address" in df.columns:
    df["Address_Word_Count"] = df["Address"].astype(str).apply(lambda x: len(x.split()))

print("Feature Extraction Completed!")

# ------------------------------------------------------------
# 3️⃣ Encode Categorical Variables
# ------------------------------------------------------------

print("\n📌 Encoding Table Booking & Online Delivery...")

# Convert Yes/No to numeric 1/0
mapping = {"Yes": 1, "No": 0}

if "Has Table booking" in df.columns:
    df["Table_Booking_Encoded"] = df["Has Table booking"].map(mapping)
else:
    print("❌ 'Has Table booking' column not found!")

if "Has Online delivery" in df.columns:
    df["Online_Delivery_Encoded"] = df["Has Online delivery"].map(mapping)
else:
    print("❌ 'Has Online delivery' column not found!")

print("Encoding Completed!")

# ------------------------------------------------------------
# 4️⃣ Show Newly Created Features
# ------------------------------------------------------------

feature_cols = [
    "Name_Length",
    "Address_Length",
    "Name_Word_Count",
    "Address_Word_Count",
    "Table_Booking_Encoded",
    "Online_Delivery_Encoded"
]

print("\n📌 Newly Created Features:")
for col in feature_cols:
    if col in df.columns:
        print(f"{col} → created successfully")
    else:
        print(f"{col} → missing!")

# Display sample of new features
print("\n📌 Preview of Engineered Features:")
print(df[feature_cols].head())

# ------------------------------------------------------------
# 5️⃣ Save Updated Dataset (Optional)
# ------------------------------------------------------------

df.to_csv("updated_feature_engineered_dataset.csv", index=False)
print("\n✅ Feature-engineered dataset saved as 'updated_feature_engineered_dataset.csv'")
