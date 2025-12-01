import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler

from datetime import datetime


# ==============================
# LOAD & PREPARE DATA
# ==============================
path = "customer_personality_Final.csv"
df = pd.read_csv(path)

current_year = datetime.now().year
df['Age'] = current_year - df['Year_Birth']
df = df[(df['Year_Birth'] > 1900) & (df['Age'] < 120)].copy()
df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], dayfirst=True, errors='coerce')
today = pd.to_datetime(datetime.now())
df["Customer_since_years"] = (today - df["Dt_Customer"]).dt.days / 365.25
df["Income"] = df["Income"].fillna(df["Income"].median())

education_map = {"Basic":1, "2n Cycle":2, "Graduation":3, "Master":4, "PhD":5}
df["Education_Ordinal"] = df["Education"].map(education_map).fillna(0)

df["Marital_Status"] = df["Marital_Status"].replace({"Alone":"Single"}).fillna("Unknown")

# Feature engineering
df["TotalFood"] = df[["MntMeatProducts","MntFishProducts","MntFruits","MntSweetProducts"]].sum(axis=1)
df["TotalLuxury"] = df[["MntWines","MntGoldProds"]].sum(axis=1)
df["TotalSpending"] = df["TotalFood"] + df["TotalLuxury"]
df["TotalPurchases"] = df[["NumWebPurchases","NumCatalogPurchases","NumStorePurchases","NumDealsPurchases"]].sum(axis=1)

df["AvgPurchaseValue"] = (df["TotalSpending"] / df["TotalPurchases"].replace(0, np.nan)).fillna(0)
df["WebRatio"] = (df["NumWebPurchases"] / df["TotalPurchases"].replace(0, np.nan)).fillna(0)
df["WineShare"] = (df["MntWines"] / df["TotalSpending"].replace(0, np.nan)).fillna(0)
df["GoldShare"] = (df["MntGoldProds"] / df["TotalSpending"].replace(0, np.nan)).fillna(0)

df["KidsTotal"] = df["Kidhome"] + df["Teenhome"]
df["IsFamily"] = (df["KidsTotal"] > 0).astype(int)

# LOG TRANSFORM for skewed data
for col in ["TotalFood","TotalLuxury","TotalSpending","TotalPurchases","AvgPurchaseValue","Income"]:
    df[col+"_log"] = np.log1p(df[col])

# final data matrix
demo = ["Age","Income","Customer_since_years","Education_Ordinal"]
X = df[[c+"_log" for c in ["TotalFood","TotalLuxury","TotalSpending","TotalPurchases","AvgPurchaseValue","Income"]] + demo].fillna(0)

scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)



# ===============================================
#     PRE-ANALYSIS: CORRELATIONS + ANOVA
# ===============================================

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import f_oneway, kruskal

# ===== 1) Korelace mezi příznaky =====

corr_cols = [
    "Age","Income","Customer_since_years",
    "TotalSpending","TotalPurchases","AvgPurchaseValue",
    "WineShare","GoldShare","WebRatio","KidsTotal","IsFamily"
]

corr = df[corr_cols].corr(method="pearson")

plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

print("\n=== Strongest correlations (abs > 0.5) ===")
strong = corr.abs()[(corr.abs() > 0.5) & (corr.abs() < 1)]
print(strong.dropna(how="all").dropna(axis=1, how="all"))

# ===============================================
#     ANOVA: Factors influencing spending
# ===============================================

factors = ["Age","Income","KidsTotal","WebRatio","WineShare"]

for fac in factors:
    group1 = df[fac]
    group2 = df["TotalSpending"]

    # rozděl TotalSpending podle kvartilů fac (lepší než binning)
    df["bin"] = pd.qcut(df[fac], q=4, duplicates='drop')
    groups = [df[df["bin"] == b]["TotalSpending"] for b in df["bin"].unique()]

    # ANOVA
    fval, pval = f_oneway(*groups)

    print(f"\nANOVA for {fac} → TotalSpending:")
    print(f"F-value = {fval:.3f}, p-value = {pval:.5f}")

# ===============================================
#  KRUSKAL-WALLIS: If ANOVA assumptions not met
# ===============================================

for fac in factors:
    df["bin"] = pd.qcut(df[fac], q=4, duplicates='drop')
    groups = [df[df["bin"] == b]["TotalSpending"] for b in df["bin"].unique()]

    H, p = kruskal(*groups)
    print(f"\nKruskal-Wallis for {fac} → TotalSpending:")
    print(f"H = {H:.3f}, p = {p:.5f}")
