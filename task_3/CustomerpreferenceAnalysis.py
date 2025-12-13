# -*- coding: utf-8 -*-
"""
Created on Fri Dec 12 23:35:33 2025

@author: USER
"""

# ============================
# Customer Preference Analysis
# ============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv("D:\MACHINE_LEARNING___PYTHON\cognifyz_task\Dataset .csv", encoding="latin1")

# Clean cuisines column (Some rows contain multiple cuisines separated by comma)
df['Cuisines_clean'] = df['Cuisines'].astype(str).apply(lambda x: x.split(",")[0].strip())

# ============================
# 1️⃣ Cuisine vs Rating
# ============================

cuisine_rating = df.groupby('Cuisines_clean')['Aggregate rating'].mean().sort_values(ascending=False)

print("\n=== Average Rating by Cuisine ===")
print(cuisine_rating.head(15))

# Plot
plt.figure(figsize=(12,6))
cuisine_rating.head(15).plot(kind='bar')
plt.title("Top Cuisines by Average Rating")
plt.xlabel("Cuisine")
plt.ylabel("Average Rating")
plt.tight_layout()
plt.show()

# ============================
# 2️⃣ Most Popular Cuisines (Based on Votes)
# ============================

cuisine_votes = df.groupby('Cuisines_clean')['Votes'].sum().sort_values(ascending=False)

print("\n=== Most Popular Cuisines by Total Votes ===")
print(cuisine_votes.head(15))

# Plot
plt.figure(figsize=(12,6))
cuisine_votes.head(15).plot(kind='bar', color='orange')
plt.title("Most Popular Cuisines Based on Votes")
plt.xlabel("Cuisine")
plt.ylabel("Total Votes")
plt.tight_layout()
plt.show()

# ============================
# 3️⃣ Cuisines with Highest Ratings
# ============================

top_rated_cuisines = df.groupby('Cuisines_clean')['Aggregate rating'].mean().sort_values(ascending=False)

print("\n=== Cuisines with Highest Average Ratings ===")
print(top_rated_cuisines.head(10))

# Plot
plt.figure(figsize=(12,6))
top_rated_cuisines.head(10).plot(kind="bar", color='green')
plt.title("Cuisines Receiving Highest Ratings")
plt.xlabel("Cuisine")
plt.ylabel("Average Rating")
plt.tight_layout()
plt.show()
