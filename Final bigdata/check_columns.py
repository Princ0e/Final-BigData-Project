import pandas as pd

# Load the dataset
data = pd.read_csv('synthetic_fertilizer_dataset.csv')

# Print column names
print("🧾 Column names in your dataset:")
print(data.columns.tolist())
