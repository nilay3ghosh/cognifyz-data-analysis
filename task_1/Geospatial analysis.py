# -*- coding: utf-8 -*-
"""
Created on Fri Dec 12 23:21:00 2025

@author: USER
"""

# ============================================================
# TASK: GEOSPATIAL ANALYSIS
# ============================================================

import pandas as pd
import numpy as np
import folium
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------
# 1️⃣ Load Dataset
# ------------------------------------------------------------

df = pd.read_csv("D:\MACHINE_LEARNING___PYTHON\cognifyz_task\Dataset .csv")     
print("Dataset Loaded Successfully!\n")

# Check required columns
required_cols = ["Latitude", "Longitude", "City", "Country Code", "Aggregate rating"]

for col in required_cols:
    if col not in df.columns:
        print(f"❌ ERROR: Column '{col}' not found in the dataset!")
        exit()

# ------------------------------------------------------------
# 2️⃣ Interactive Map (Latitude + Longitude)
# ------------------------------------------------------------

print("\n📌 Generating interactive restaurant map...")

# Center map around average location
map_center = [df["Latitude"].mean(), df["Longitude"].mean()]

restaurant_map = folium.Map(location=map_center, zoom_start=11)

# Add markers for each restaurant
for _, row in df.head(500).iterrows():  
    # limit to 500 markers to avoid heavy map
    folium.CircleMarker(
        location=[row["Latitude"], row["Longitude"]],
        radius=3,
        color="red",
        fill=True,
        fill_color="red",
        popup=f"City: {row['City']} | Rating: {row['Aggregate rating']}"
    ).add_to(restaurant_map)

restaurant_map.save("restaurants_map.html")

print("✅ Map created: restaurants_map.html (Open in browser)")

# ------------------------------------------------------------
# 3️⃣ Distribution of Restaurants by City
# ------------------------------------------------------------

print("\n📌 TOP 10 Cities with Most Restaurants:")
top_cities = df["City"].value_counts().head(10)
print(top_cities)

plt.figure(figsize=(10,4))
top_cities.plot(kind='bar', color='orange')
plt.title("Top 10 Cities by Number of Restaurants")
plt.xlabel("City")
plt.ylabel("Restaurant Count")
plt.xticks(rotation=45)
plt.show()

# ------------------------------------------------------------
# 4️⃣ Distribution of Restaurants by Country
# ------------------------------------------------------------

print("\n📌 TOP 10 Countries with Most Restaurants:")
top_countries = df["Country Code"].value_counts().head(10)
print(top_countries)

plt.figure(figsize=(10,4))
top_countries.plot(kind='bar', color='purple')
plt.title("Top 10 Countries by Number of Restaurants")
plt.xlabel("Country Code")
plt.ylabel("Restaurant Count")
plt.xticks(rotation=0)
plt.show()

# ------------------------------------------------------------
# 5️⃣ Location vs Rating Correlation
# ------------------------------------------------------------

print("\n📌 Checking correlation between location & rating:")

# Latitude vs Rating
plt.figure(figsize=(6,4))
sns.scatterplot(x=df["Latitude"], y=df["Aggregate rating"])
plt.title("Latitude vs Aggregate Rating")
plt.xlabel("Latitude")
plt.ylabel("Rating")
plt.show()

# Longitude vs Rating
plt.figure(figsize=(6,4))
sns.scatterplot(x=df["Longitude"], y=df["Aggregate rating"])
plt.title("Longitude vs Aggregate Rating")
plt.xlabel("Longitude")
plt.ylabel("Rating")
plt.show()

# Correlation values
corr_lat = df["Latitude"].corr(df["Aggregate rating"])
corr_lon = df["Longitude"].corr(df["Aggregate rating"])

print("\nCorrelation Results:")
print(f"Latitude vs Rating correlation: {corr_lat}")
print(f"Longitude vs Rating correlation: {corr_lon}")

if abs(corr_lat) < 0.2 and abs(corr_lon) < 0.2:
    print("\n➡️ Conclusion: No strong correlation between location and rating.")
else:
    print("\n➡️ Conclusion: Location has some influence on ratings.")
