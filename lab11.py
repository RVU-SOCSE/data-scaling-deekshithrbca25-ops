import pandas as pd
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler

# Load dataset
df = pd.read_csv("C:/Users/Deekshith/Downloads/11prog_3Salary_Data - 11prog_3Salary_Data (1).csv")
print("Original DataFrame:\n", df)

# -------------------------------
# StandardScaler
# -------------------------------
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)
scaled_df1 = pd.DataFrame(scaled_data, columns=df.columns)
print("\nData after StandardScaler:\n", scaled_df1)

# -------------------------------
# Normalizer
# -------------------------------
normalizer = Normalizer()
normalized_data = normalizer.fit_transform(df)
normalized_df = pd.DataFrame(normalized_data, columns=df.columns)
print("\nData after Normalizer:\n", normalized_df)

# -------------------------------
# MinMaxScaler
# -------------------------------
minmax = MinMaxScaler()
minmax_data = minmax.fit_transform(df)
minmax_df = pd.DataFrame(minmax_data, columns=df.columns)
print("\nData after MinMaxScaler:\n", minmax_df)
