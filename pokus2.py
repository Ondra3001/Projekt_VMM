# ===============================================
#              CLEAN FINAL CLUSTERING PIPELINE
# ===============================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from umap import UMAP

# ===========================
# LOAD & FEATURE ENGINEERING
# ===========================

path = "customer_personality_Final.csv"
df = pd.read_csv(path)

current_year = datetime.now().year
df["Age"] = current_year - df["Year_Birth"]
df = df[(df["Year_Birth"] > 1900) & (df["Age"] < 120)].copy()

df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], dayfirst=True, errors="coerce")
df["Customer_since_years"] = (datetime.now() - df["Dt_Customer"]).dt.days / 365.25

df["Income"] = df["Income"].fillna(df["Income"].median())

education_map = {"Basic":1, "2n Cycle":2, "Graduation":3, "Master":4, "PhD":5}
df["Education_Ordinal"] = df["Education"].map(education_map).fillna(0)

# Feature engineering
df["TotalFood"] = df[["MntMeatProducts","MntFishProducts","MntFruits","MntSweetProducts"]].sum(axis=1)
df["TotalLuxury"] = df[["MntWines","MntGoldProds"]].sum(axis=1)
df["TotalSpending"] = df["TotalFood"] + df["TotalLuxury"]

df["TotalPurchases"] = df[["NumWebPurchases","NumCatalogPurchases",
                           "NumStorePurchases","NumDealsPurchases"]].sum(axis=1)

df["AvgPurchaseValue"] = (df["TotalSpending"] / df["TotalPurchases"].replace(0, np.nan)).fillna(0)
df["WebRatio"] = (df["NumWebPurchases"] / df["TotalPurchases"].replace(0, np.nan)).fillna(0)
df["WineShare"] = (df["MntWines"] / df["TotalSpending"].replace(0, np.nan)).fillna(0)
df["GoldShare"] = (df["MntGoldProds"] / df["TotalSpending"].replace(0, np.nan)).fillna(0)

df["KidsTotal"] = df["Kidhome"] + df["Teenhome"]
df["IsFamily"] = (df["KidsTotal"] > 0).astype(int)

# Log-transform heavy skew
log_cols = ["TotalFood","TotalLuxury","TotalSpending",
            "TotalPurchases","AvgPurchaseValue","Income"]

for c in log_cols:
    df[c+"_log"] = np.log1p(df[c])

# Final feature matrix
demo = ["Age","Income","Customer_since_years","Education_Ordinal"]
X = df[[c+"_log" for c in log_cols] + demo].fillna(0)

scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

# ===============================
# GRID SEARCH PCA × GMM
# ===============================

pca_range = range(2, 11)
cluster_range = range(2, 11)
covariance_types = ["full", "tied", "diag"]

results = []

for n_pca, k, cov in itertools.product(pca_range, cluster_range, covariance_types):

    pca = PCA(n_components=n_pca, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    gmm = GaussianMixture(n_components=k, covariance_type=cov, random_state=42)
    labels = gmm.fit_predict(X_pca)

    sil = silhouette_score(X_pca, labels)

    results.append({"PCA": n_pca, "K": k, "Cov": cov, "Silhouette": sil})

results_df = pd.DataFrame(results)
best = results_df.loc[results_df["Silhouette"].idxmax()]

best_pca = int(best["PCA"])
best_cov = best["Cov"]

# ===============================================
# FORCE NUMBER OF CLUSTERS HERE (3 or 4)
# ===============================================
forced_k = 3   # <---- ZMĚŇ TADY

pca_final = PCA(n_components=best_pca, random_state=42)
X_final = pca_final.fit_transform(X_scaled)

gmm = GaussianMixture(n_components=forced_k, covariance_type=best_cov, random_state=42)
df["Cluster"] = gmm.fit_predict(X_final)

# ===============================
# CLUSTER PROFILES
# ===============================

profile_cols = [
    "Age","Income","Customer_since_years",
    "TotalSpending","TotalPurchases","AvgPurchaseValue",
    "WineShare","GoldShare","WebRatio","KidsTotal","IsFamily"
]

cluster_profiles = df.groupby("Cluster")[profile_cols].mean().round(2)
print("\n===== CLUSTER PROFILES =====")
print(cluster_profiles)

plt.figure(figsize=(10,6))
sns.heatmap(cluster_profiles.T, annot=True, cmap="coolwarm")
plt.title("Cluster Feature Profiles")
plt.show()

# ===============================
# CAMPAIGN RESPONSE
# ===============================

campaign_cols = ["AcceptedCmp1","AcceptedCmp2","AcceptedCmp3",
                 "AcceptedCmp4","AcceptedCmp5","Response"]

campaign_summary = df.groupby("Cluster")[campaign_cols].mean().round(3)
campaign_summary["Size"] = df["Cluster"].value_counts().sort_index()

print("\n===== CAMPAIGN RESPONSE PER CLUSTER =====")
print(campaign_summary)

plt.figure(figsize=(8,5))
sns.heatmap(campaign_summary[campaign_cols], annot=True, cmap="YlGnBu")
plt.title("Campaign Response per Cluster")
plt.show()

# ===============================
# PCA VISUALIZATION
# ===============================

pca_vis = PCA(n_components=2, random_state=42)
X_vis = pca_vis.fit_transform(X_scaled)

plt.figure(figsize=(8,6))
sns.scatterplot(x=X_vis[:,0], y=X_vis[:,1], hue=df["Cluster"], palette="Set2", s=60)
plt.title("Clusters in PCA 2D")
plt.show()

# ===============================
# UMAP VISUALIZATION
# ===============================

umap_model = UMAP(n_components=2, random_state=42)
X_umap = umap_model.fit_transform(X_scaled)

plt.figure(figsize=(8,6))
sns.scatterplot(
    x=X_umap[:,0], y=X_umap[:,1],
    hue=df["Cluster"],  # <-- FIXED
    palette="Set2",
    s=60
)
plt.title("UMAP Visualization of Clusters")
plt.show()
