# ==========================================
# AI & ML Internship - Task 2
# Data Cleaning & Missing Value Handling
# ==========================================

# Step 1: Import Required Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# Step 2: Load Dataset
# ==========================================

df = pd.read_csv("titanic.csv")

# Display first 5 rows
print("First 5 Rows of Dataset:\n")
print(df.head())

# ==========================================
# Step 3: Dataset Information
# ==========================================

print("\nDataset Information:\n")
print(df.info())

# ==========================================
# Step 4: Check Missing Values
# ==========================================

print("\nMissing Values Before Cleaning:\n")
print(df.isnull().sum())

# ==========================================
# Step 5: Visualize Missing Values
# ==========================================

plt.figure(figsize=(10,6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')

plt.title("Missing Values Heatmap")
plt.show()

# ==========================================
# Step 6: Handle Missing Values
# ==========================================

# Fill Age column using Mean
df['Age'].fillna(df['Age'].mean(), inplace=True)

# Fill Embarked column using Mode
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Drop Cabin column because it has many missing values
df.drop('Cabin', axis=1, inplace=True)

# ==========================================
# Step 7: Verify Missing Values
# ==========================================

print("\nMissing Values After Cleaning:\n")
print(df.isnull().sum())

# ==========================================
# Step 8: Detect Outliers
# ==========================================

plt.figure(figsize=(8,5))
sns.boxplot(x=df['Fare'])

plt.title("Outlier Detection using Boxplot")
plt.show()

# ==========================================
# Step 9: Remove Outliers using IQR Method
# ==========================================

Q1 = df['Fare'].quantile(0.25)
Q3 = df['Fare'].quantile(0.75)

IQR = Q3 - Q1

lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

df = df[(df['Fare'] >= lower_limit) & 
        (df['Fare'] <= upper_limit)]

# ==========================================
# Step 10: Save Cleaned Dataset
# ==========================================

df.to_csv("cleaned_titanic.csv", index=False)

print("\nData Cleaning Completed Successfully")
print("\nCleaned Dataset Saved as 'cleaned_titanic.csv'")