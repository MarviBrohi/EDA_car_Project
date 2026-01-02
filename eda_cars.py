import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date

# ---------- Step 1: Load dataset ----------
data = pd.read_csv("Datasets/carprice.csv")
print("Dataset Loaded Successfully")
print("Shape:", data.shape)
print(data.head())
print(data.isnull().sum())

# ---------- Step 2: Data reduction ----------
cols_to_drop = [c for c in data.columns if "id" in c.lower() or "s.no" in c.lower()]
data = data.drop(columns=cols_to_drop, errors="ignore")

# ---------- Step 3: Feature engineering ----------
# Example: car_age (if year column exists)
if "year" in data.columns:
    current_year = date.today().year
    data["car_age"] = current_year - data["year"]

# Example: brand from make/name (use what you have)
if "make" in data.columns:
    data["brand"] = data["make"]          # simple copy, or keep make only

# ---------- Step 4: Cleaning ----------
# Replace "?" with NaN in numeric-like columns, then convert
for col in ["normalized-losses", "price", "horsepower"]:
    if col in data.columns:
        data[col] = data[col].replace("?", np.nan)
        data[col] = pd.to_numeric(data[col], errors="coerce")

# Example: handle missing price (drop rows with no price)
if "price" in data.columns:
    data = data.dropna(subset=["price"])

print("After cleaning:")
print(data.isnull().sum())

# ---------- Step 5: Stats summary ----------
print("Statistical Summary:")
print(data.describe(include="all").T)

# ---------- Step 6: Univariate plots ----------
num_cols = data.select_dtypes(include=np.number).columns

for col in num_cols:
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    data[col].hist(grid=False)
    plt.title(f"Histogram of {col}")

    plt.subplot(1, 2, 2)
    sns.boxplot(x=data[col])
    plt.title(f"Boxplot of {col}")

    plt.tight_layout()
    plt.savefig(f"plots_{col}.png")
    plt.close()

# ---------- Step 7: Transformation – log of price ----------
if "price" in data.columns:
    data["price_log"] = np.log1p(data["price"])

# ---------- Step 8: Bivariate – brand vs avg price_log ----------
if "brand" in data.columns and "price_log" in data.columns:
    plt.figure(figsize=(10, 5))
    data.groupby("brand")["price_log"].mean().sort_values(ascending=False).head(15)\
        .plot(kind="bar")
    plt.title("Top 15 brands by average log price")
    plt.ylabel("Mean price_log")
    plt.tight_layout()
    plt.savefig("plots_brand_vs_price.png")
    plt.close()

# ---------- Step 9: Multivariate – correlation heatmap ----------
plt.figure(figsize=(10, 6))
sns.heatmap(
    data.select_dtypes(include=np.number).corr(),
    cmap="coolwarm", vmin=-1, vmax=1
)
plt.title("Correlation heatmap")
plt.tight_layout()
plt.savefig("plots_corr_heatmap.png")
plt.close()

print("EDA completed. Check generated plots.")
