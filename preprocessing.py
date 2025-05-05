# preprocessing.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np

# Load dataset
df = pd.read_csv('healthcare-dataset-stroke-data.csv')

# Drop kolom ID
df.drop('id', axis=1, inplace=True)

# Tangani missing value
df['bmi'] = df['bmi'].fillna(df['bmi'].median())


# Encode kategorikal
categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# Pisahkan fitur dan target
X = df.drop('stroke', axis=1)
y = df['stroke']

# Normalisasi fitur
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Simpan hasil preprocessing
np.savez('processed_data.npz', X=X_scaled, y=y.to_numpy(), feature_names=X.columns.to_numpy())
print("✅ Data berhasil diproses dan disimpan ke 'processed_data.npz'")
