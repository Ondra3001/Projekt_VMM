import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
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

# ============================================
# 1) TEST DIFFERENT PCA COMPONENTS
# ============================================
pca_range = range(2, 11)
sil_scores_pca = []

for n in pca_range:
    pca = PCA(n_components=n, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    gmm = GaussianMixture(n_components=3, covariance_type="full", random_state=42)
    labels = gmm.fit_predict(X_pca)

    sil = silhouette_score(X_pca, labels)
    sil_scores_pca.append(sil)

plt.figure(figsize=(8,5))
plt.plot(pca_range, sil_scores_pca, marker='o')
plt.title("Silhouette score vs. number of PCA components")
plt.xlabel("PCA components")
plt.ylabel("Silhouette")
plt.grid(True)
plt.show()

best_pca = pca_range[np.argmax(sil_scores_pca)]
print(f"Best number of PCA components: {best_pca}")

# ============================================
# 2) TEST DIFFERENT NUMBER OF CLUSTERS (k)
# ============================================
pca = PCA(n_components=best_pca, random_state=42)
X_best_pca = pca.fit_transform(X_scaled)

cluster_range = range(2, 9)
sil_scores_k = []

for k in cluster_range:
    gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=42)
    labels = gmm.fit_predict(X_best_pca)
    sil = silhouette_score(X_best_pca, labels)
    sil_scores_k.append(sil)

plt.figure(figsize=(8,5))
plt.plot(cluster_range, sil_scores_k, marker='o')
plt.title("Silhouette score vs. number of clusters")
plt.xlabel("Clusters (k)")
plt.ylabel("Silhouette")
plt.grid(True)
plt.show()

best_k = cluster_range[np.argmax(sil_scores_k)]
print(f"Optimal number of clusters: {best_k}")

# ============================================
# 3) Fit final model
# ============================================
gmm_final = GaussianMixture(n_components=best_k, covariance_type="full", random_state=42)
labels_final = gmm_final.fit_predict(X_best_pca)
df["Cluster_optimal"] = labels_final

# ============================================
# 4) Visualization of clusters in 2D PCA
# ============================================
pca_vis = PCA(n_components=2, random_state=42)
X_vis = pca_vis.fit_transform(X_scaled)

plt.figure(figsize=(8,6))
sns.scatterplot(x=X_vis[:,0], y=X_vis[:,1], hue=df["Cluster_optimal"], palette="Set2", s=50)
plt.title("Clusters in PCA 2D space")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend(title="Cluster")
plt.show()

from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================
# GRID SEARCH: PCA components × Clusters × Covariance
# ============================================

pca_range = range(2, 11)      # test PCA(2–10)
cluster_range = range(2, 11)  # test GMM with 2–10 clusters
cov_types = ["full", "tied", "diag"]

results = []

for n_pca, k, cov in itertools.product(pca_range, cluster_range, cov_types):

    # PCA
    pca = PCA(n_components=n_pca, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    # GMM
    gmm = GaussianMixture(n_components=k, covariance_type=cov, random_state=42)
    labels = gmm.fit_predict(X_pca)

    # Silhouette
    sil = silhouette_score(X_pca, labels)

    results.append({
        "PCA_components": n_pca,
        "Clusters": k,
        "Covariance": cov,
        "Silhouette": sil
    })

# Convert to DataFrame
results_df = pd.DataFrame(results)
results_df.sort_values("Silhouette", ascending=False).head(10)

best_row = results_df.loc[results_df["Silhouette"].idxmax()]
best_pca = int(best_row["PCA_components"])
best_k = int(best_row["Clusters"])
best_cov = best_row["Covariance"]

print("===== BEST MODEL =====")
print(f"Best PCA components: {best_pca}")
print(f"Best number of clusters: {best_k}")
print(f"Best covariance type: {best_cov}")
print(f"Best silhouette: {best_row['Silhouette']:.4f}")

best_cov_df = results_df[results_df["Covariance"] == best_cov]

pivot = best_cov_df.pivot(index="PCA_components", columns="Clusters", values="Silhouette")

plt.figure(figsize=(10,6))
sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis")
plt.title(f"Silhouette heatmap (covariance={best_cov})")
plt.show()

pca_final = PCA(n_components=best_pca, random_state=42)
X_final = pca_final.fit_transform(X_scaled)

gmm_final = GaussianMixture(n_components=best_k, covariance_type=best_cov, random_state=42)
labels_final = gmm_final.fit_predict(X_final)

df["Cluster_optimal"] = labels_final

pca_vis = PCA(n_components=2, random_state=42)
X_vis = pca_vis.fit_transform(X_scaled)

plt.figure(figsize=(8,6))
sns.scatterplot(
    x=X_vis[:,0], y=X_vis[:,1],
    hue=df["Cluster_optimal"],
    palette="Set2", s=50
)
plt.title("Cluster visualisation in PCA 2D")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend(title="Cluster")
plt.show()

#KAMPANE


campaign_cols = ["AcceptedCmp1","AcceptedCmp2","AcceptedCmp3","AcceptedCmp4","AcceptedCmp5","Response"]

# Summary of conversion rates per cluster
cluster_campaign = df.groupby("Cluster_optimal")[campaign_cols].mean()

# Add cluster sizes (optional but useful)
cluster_campaign["Cluster_size"] = df["Cluster_optimal"].value_counts().sort_index()


profile_cols = [
    "Age", "Income", "Customer_since_years",
    "TotalSpending", "TotalPurchases", "AvgPurchaseValue",
    "KidsTotal", "IsFamily", "WineShare", "GoldShare", "WebRatio"
]

cluster_profiles = df.groupby("Cluster_optimal")[profile_cols].mean().round(2)

print("\n===== CLUSTER PROFILES =====")
print(cluster_profiles)


print("\n===== CAMPAIGN RESPONSE PER CLUSTER =====")
print(cluster_campaign)
plt.figure(figsize=(12,6))
sns.heatmap(cluster_profiles.T, annot=True, fmt=".1f", cmap="coolwarm")
plt.title("Cluster Feature Profiles")
plt.show()




from sklearn.preprocessing import StandardScaler

# IMPORTANT — normalize original data BEFORE aggregating
cols = ["Age","Income","Customer_since_years",
        "TotalSpending","TotalPurchases","AvgPurchaseValue",
        "KidsTotal","IsFamily","WineShare","GoldShare","WebRatio"]

df_norm = df.copy()
df_norm[cols] = StandardScaler().fit_transform(df_norm[cols])

cluster_profiles_z = df_norm.groupby("Cluster_optimal")[cols].mean()

plt.figure(figsize=(12,6))
sns.heatmap(cluster_profiles_z.T, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Normalized Cluster Feature Profiles (z-scores)")
plt.show()
