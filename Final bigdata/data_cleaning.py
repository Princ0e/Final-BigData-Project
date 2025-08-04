import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the dataset
data = pd.read_csv('synthetic_fertilizer_dataset.csv')

# Check for missing values
print(data.isnull().sum())

# If any missing, decide how to fill or drop
data.fillna({
    'Fertilizer_Quantity_kg': data['Fertilizer_Quantity_kg'].median(),
    'Fertilizer_Cost_RWF': data['Fertilizer_Cost_RWF'].median(),
    'Yield_kg': data['Yield_kg'].median()
}, inplace=True)

# Check inconsistent categorical formats and standardize
data['Fertilizer_Type'] = data['Fertilizer_Type'].str.capitalize()

# Detect outliers in Yield using IQR method
Q1 = data['Yield_kg'].quantile(0.25)
Q3 = data['Yield_kg'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = data[(data['Yield_kg'] < lower_bound) | (data['Yield_kg'] > upper_bound)]
print(f'Number of outliers in Yield: {len(outliers)}')

# Optionally remove or cap outliers
data['Yield_kg'] = np.where(data['Yield_kg'] > upper_bound, upper_bound,
                            np.where(data['Yield_kg'] < lower_bound, lower_bound, data['Yield_kg']))

# Encoding categorical variables
le_region = LabelEncoder()
data['Region_encoded'] = le_region.fit_transform(data['Region'])

le_crop = LabelEncoder()
data['Crop_Type_encoded'] = le_crop.fit_transform(data['Crop_Type'])

le_fert = LabelEncoder()
data['Fertilizer_Type_encoded'] = le_fert.fit_transform(data['Fertilizer_Type'])

# Scaling numeric columns (optional but useful for some ML models)
scaler = StandardScaler()
num_cols = ['Land_Size_ha', 'Fertilizer_Quantity_kg', 'Fertilizer_Cost_RWF', 'Yield_kg']
data[num_cols] = scaler.fit_transform(data[num_cols])

data.head()
