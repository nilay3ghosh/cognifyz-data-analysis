# -*- coding: utf-8 -*-
"""
Created on Fri Dec 12 23:39:39 2025

@author: USER
"""

# ==================================
# DATA VISUALIZATION FOR ZOMATO DATA
# ==================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv("D:\MACHINE_LEARNING___PYTHON\cognifyz_task\Dataset .csv", encoding="latin1")

# Clean cuisines column (use first cuisine only)
df['Cuisines_clean'] = df['Cuisines'].astype(str).apply(lambda x: x.split(",")[0].strip())

# ----------------------------
# 1️⃣ DISTRIBUTION OF RATINGS
# ----------------------------

plt.figure(figsize=(10,5))
sns.histplot(df['Aggregate rating'], bins=20, kde=True)
plt.title("Distribution of Aggregate Ratings")
plt.xlabel("Rating")
plt.ylabel("Frequency")
plt.show()

# Barplot of rating counts
plt.figure(figsize=(10,5))
df['Aggregate rating'].value_counts().sort_index().plot(kind='bar', color='purple')
plt.title("Ratings Count Distribution")
plt.xlabel("Rating Value")
plt.ylabel("Count")
plt.show()

# ----------------------------
# 2️⃣ AVERAGE RATINGS BY CUISINE
# ----------------------------

cuisine_rating = df.groupby('Cuisines_clean')['Aggregate rating'].mean().sort_values(ascending=False).head(15)

plt.figure(figsize=(12,6))
sns.barplot(x=cuisine_rating.values, y=cuisine_rating.index, palette="viridis")
plt.title("Top 15 Cuisines by Average Rating")
plt.xlabel("Average Rating")
plt.ylabel("Cuisine")
plt.show()

# ----------------------------
# 3️⃣ AVERAGE RATINGS BY CITY
# ----------------------------

city_rating = df.groupby('City')['Aggregate rating'].mean().sort_values(ascending=False).head(15)

plt.figure(figsize=(12,6))
sns.barplot(x=city_rating.values, y=city_rating.index, palette="coolwarm")
plt.title("Top 15 Cities by Average Restaurant Rating")
plt.xlabel("Average Rating")
plt.ylabel("City")
plt.show()

# ----------------------------
# 4️⃣ RELATIONSHIP BETWEEN FEATURES & RATING
# ----------------------------

# Votes vs Rating
plt.figure(figsize=(10,5))
sns.scatterplot(data=df, x='Votes', y='Aggregate rating', alpha=0.5)
plt.title("Votes vs Aggregate Rating")
plt.xlabel("Number of Votes")
plt.ylabel("Rating")
plt.show()

# Price range vs Rating
plt.figure(figsize=(8,5))
sns.boxplot(x='Price range', y='Aggregate rating', data=df)
plt.title("Price Range vs Rating")
plt.xlabel("Price Range Level")
plt.ylabel("Rating")
plt.show()

# Online Delivery vs Rating
plt.figure(figsize=(8,5))
sns.boxplot(x='Has Online delivery', y='Aggregate rating', data=df)
plt.title("Online Delivery Availability vs Rating")
plt.xlabel("Has Online Delivery? (0/1)")
plt.ylabel("Rating")
plt.show()

# Table Booking vs Rating
plt.figure(figsize=(8,5))
sns.boxplot(x='Has Table booking', y='Aggregate rating', data=df)
plt.title("Table Booking Availability vs Rating")
plt.xlabel("Has Table Booking? (0/1)")
plt.ylabel("Rating")
plt.show()

# Heatmap for correlations
plt.figure(figsize=(10,7))
sns.heatmap(df[['Aggregate rating','Votes','Price range','Has Table booking','Has Online delivery']].corr(),
            annot=True, cmap="coolwarm", linewidths=1)
plt.title("Correlation Heatmap")
plt.show()
