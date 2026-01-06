import joblib

cols = joblib.load("models/saved_models/feature_cols.pkl")
print(cols)

print(df.columns)