import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import matplotlib.pyplot as plt
import os

# ساختن پوشه‌ها
os.makedirs("../models", exist_ok=True)
os.makedirs("../data", exist_ok=True)

# دانلود دیتاست (اتوماتیک!)
print("در حال دانلود دیتاست California Housing...")
housing = fetch_california_housing(as_frame=True)
df = housing.frame

# ذخیره دیتاست خام
df.to_csv("../data/california_housing.csv", index=False)
print("دیتاست ذخیره شد: data/california_housing.csv")

# ویژگی‌ها و هدف
X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# مدل
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# پیش‌بینی و معیارها
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.3f}")
print(f"R²: {r2:.3f}")

# ذخیره مدل
joblib.dump(model, "../models/housing_model.pkl")
print("مدل ذخیره شد: models/housing_model.pkl")

# رسم نمودار پیش‌بینی در مقابل واقعی
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel("قیمت واقعی")
plt.ylabel("قیمت پیش‌بینی شده")
plt.title(f"پیش‌بینی قیمت خانه کالیفرنیا\nR² = {r2:.3f}")
plt.savefig("../data/prediction_plot.png")
plt.close()
print("نمودار ذخیره شد: data/prediction_plot.png")
