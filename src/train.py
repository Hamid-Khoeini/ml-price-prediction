import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import matplotlib.pyplot as plt
import os

os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)

print("Dataset California Housing is downloading...")
housing = fetch_california_housing(as_frame=True)
df = housing.frame

df.to_csv("data/california_housing.csv", index=False)
print("Dataset saved: data/california_housing.csv")

X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"MAE: {mae:.3f}")
print(f"R²: {r2:.3f}")

joblib.dump(model, "models/housing_model.pkl")
print("Model saved: models/housing_model.pkl")

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel("Real price")
plt.ylabel("Predicted price")
plt.title(f"California home price forecast\nR² = {r2:.3f}")
plt.savefig("data/prediction_plot.png")
plt.close()
print("Diagram saved: data/prediction_plot.png")
