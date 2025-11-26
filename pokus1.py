# Produce cluster profiles for the successful log-transform + PCA2 + GMM result
# Using df_log, X_pca2, labels from previous run in this session; re-run minimal steps to ensure variables exist.

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
from sklearn.mixture import GaussianMixture

import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
# reload and rebuild df_log quickly (same steps)
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

# engineered features
df["TotalFood"] = df["MntMeatProducts"].fillna(0) + df["MntFishProducts"].fillna(0) + df["MntFruits"].fillna(0) + df["MntSweetProducts"].fillna(0)
df["TotalLuxury"] = df["MntWines"].fillna(0) + df["MntGoldProds"].fillna(0)
df["TotalSpending"] = df["TotalFood"] + df["TotalLuxury"]
df["TotalPurchases"] = df["NumWebPurchases"].fillna(0) + df["NumCatalogPurchases"].fillna(0) + df["NumStorePurchases"].fillna(0) + df["NumDealsPurchases"].fillna(0)
df["AvgPurchaseValue"] = (df["TotalSpending"] / df["TotalPurchases"].replace(0, np.nan)).fillna(0)
df["WebRatio"] = (df["NumWebPurchases"] / df["TotalPurchases"].replace(0, np.nan)).fillna(0)
df["CatalogRatio"] = (df["NumCatalogPurchases"] / df["TotalPurchases"].replace(0, np.nan)).fillna(0)
df["StoreRatio"] = (df["NumStorePurchases"] / df["TotalPurchases"].replace(0, np.nan)).fillna(0)
df["WineShare"] = (df["MntWines"] / df["TotalSpending"].replace(0, np.nan)).fillna(0)
df["GoldShare"] = (df["MntGoldProds"] / df["TotalSpending"].replace(0, np.nan)).fillna(0)
df["KidsTotal"] = df["Kidhome"].fillna(0) + df["Teenhome"].fillna(0)
df["IsFamily"] = (df["KidsTotal"] > 0).astype(int)

engineered = ["TotalFood","TotalLuxury","TotalSpending","TotalPurchases","AvgPurchaseValue",
              "WebRatio","CatalogRatio","StoreRatio","WineShare","GoldShare","KidsTotal","IsFamily"]
demo = ["Age","Income","Customer_since_years","Education_Ordinal"]
campaigns = [c for c in ["AcceptedCmp1","AcceptedCmp2","AcceptedCmp3","AcceptedCmp4","AcceptedCmp5","Response"] if c in df.columns]
features = engineered + demo + campaigns

# log1p transform for selected columns
for col in ["TotalFood","TotalLuxury","TotalSpending","TotalPurchases","AvgPurchaseValue","Income"]:
    df[col+"_log"] = np.log1p(df[col])

# build X for PCA using log features and keep other demos unchanged
X_log = df[[c+"_log" for c in ["TotalFood","TotalLuxury","TotalSpending","TotalPurchases","AvgPurchaseValue","Income"]] + demo].fillna(0)
X_scaled_log = RobustScaler().fit_transform(X_log)

pca2 = PCA(n_components=2, random_state=42)
X_pca2 = pca2.fit_transform(X_scaled_log)

# best GMM from previous experiment: pick k=3 # dala jsem 4
gmm = GaussianMixture(n_components=2, covariance_type="full", random_state=42)
labels = gmm.fit_predict(X_pca2)
df["Cluster_final"] = labels

# cluster summary: means of key features (original scale for interpretability)
summary = df.groupby("Cluster_final")[["TotalSpending","TotalPurchases","AvgPurchaseValue","WineShare","GoldShare","WebRatio","IsFamily","Income","Age"]].median().round(2)
counts = df["Cluster_final"].value_counts().sort_index()
print("Cluster counts:\n", counts.to_string())
print("\nMedian profiles:\n", summary.to_string())

# Show top distinguishing features per cluster (difference from overall median)
overall = df[["TotalSpending","TotalPurchases","AvgPurchaseValue","WineShare","GoldShare","WebRatio","IsFamily","Income","Age"]].median()
print("\nTop deltas vs overall (median):")
for lab in sorted(df["Cluster_final"].unique()):
    med = df[df["Cluster_final"]==lab][["TotalSpending","TotalPurchases","AvgPurchaseValue","WineShare","GoldShare","WebRatio","IsFamily","Income","Age"]].median()
    diff = (med - overall).sort_values(ascending=False)
    print(f"\nCluster {lab} (n={len(df[df['Cluster_final']==lab])}):")
    print(diff.to_string())

from sklearn.metrics import silhouette_samples, silhouette_score

# Silhouette nad PCA2 (stejný prostor jako GMM)
sil_vals = silhouette_samples(X_pca2, df["Cluster_final"])
df["silhouette"] = sil_vals

plt.figure(figsize=(8,5))
sns.boxplot(x="Cluster_final", y="silhouette", data=df)
plt.title("Silhouette score per cluster (PCA2 space)")
plt.show()

print("Global silhouette:", silhouette_score(X_pca2, df["Cluster_final"]))

