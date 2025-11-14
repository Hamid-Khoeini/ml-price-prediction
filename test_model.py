import joblib
import pandas as pd
from sklearn.datasets import fetch_california_housing

model = joblib.load("models/housing_model.pkl")
housing = fetch_california_housing(as_frame=True)
X_sample = housing.frame.drop("MedHouseVal", axis=1).iloc[:5]
pred = model.predict(X_sample)

print("Predict first 5 samples:", pred)
assert len(pred) == 5
print("Success test")
