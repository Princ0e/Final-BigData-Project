# eda_analysis.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the cleaned dataset
data = pd.read_csv('synthetic_fertilizer_dataset.csv')

# --- 1. Descriptive Statistics ---
print("Descriptive Statistics:")
print(data.describe())

print("\nDataset Info:")
print(data.info())

print("\nMissing Values:")
print(data.isnull().sum())

# --- 2. Distribution Plots ---
numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns

for col in numeric_cols:
    plt.figure(figsize=(8, 4))
    sns.histplot(data[col], kde=True, bins=30)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

# --- 3. Boxplots for Outlier Detection ---
for col in numeric_cols:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=data[col])
    plt.title(f"Boxplot of {col}")
    plt.tight_layout()
    plt.show()

# --- 4. Correlation Heatmap ---
plt.figure(figsize=(10, 6))
# --- 4. Correlation Heatmap (numeric columns only) ---
plt.figure(figsize=(10, 6))
numeric_data = data.select_dtypes(include=['int64', 'float64'])
corr = numeric_data.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()

plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()

# --- 5. Relationship Plots ---
# Example: fertilizer_amount vs crop_yield
plt.figure(figsize=(8, 5))
sns.scatterplot(x='fertilizer_amount_kg', y='crop_yield_kg_per_hectare', data=data)
plt.title("Fertilizer Amount vs Crop Yield")
plt.xlabel("Fertilizer Amount (kg)")
plt.ylabel("Crop Yield (kg/ha)")
plt.tight_layout()
plt.show()
