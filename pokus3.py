# UMAP + HDBSCAN pipeline s EDA, profilací a testy, lepsi clusterovani
#

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# preprocessing & scaling
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.decomposition import PCA

# embedding + clustering
from umap import UMAP
import hdbscan

# statistika
from scipy.stats import kruskal
from itertools import combinations
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# metriky
from sklearn.metrics import silhouette_score

# ----------------------------
# 1) Načtení a feature eng.
# ----------------------------
path = "customer_personality_Final.csv"
df = pd.read_csv(path)

# základní cleaning
current_year = datetime.now().year
df["Age"] = current_year - df["Year_Birth"]
df = df[(df["Year_Birth"] > 1900) & (df["Age"] < 120)].copy()
df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], dayfirst=True, errors="coerce")
df["Customer_since_years"] = (datetime.now() - df["Dt_Customer"]).dt.days / 365.25
df["Income"] = df["Income"].fillna(df["Income"].median())

education_map = {"Basic":1, "2n Cycle":2, "Graduation":3, "Master":4, "PhD":5}
df["Education_Ordinal"] = df["Education"].map(education_map).fillna(0)
df["Marital_Status"] = df["Marital_Status"].replace({"Alone":"Single"}).fillna("Unknown")

# vypocitane features
df["TotalFood"] = df[["MntMeatProducts","MntFishProducts","MntFruits","MntSweetProducts"]].sum(axis=1)
df["TotalLuxury"] = df[["MntWines","MntGoldProds"]].sum(axis=1)
df["TotalSpending"] = df["TotalFood"] + df["TotalLuxury"]
df["TotalPurchases"] = df[["NumWebPurchases","NumCatalogPurchases","NumStorePurchases","NumDealsPurchases"]].sum(axis=1)
df["AvgPurchaseValue"] = (df["TotalSpending"] / df["TotalPurchases"].replace(0, np.nan)).fillna(0)
df["WebRatio"] = (df["NumWebPurchases"] / df["TotalPurchases"].replace(0, np.nan)).fillna(0)
df["WineShare"] = (df["MntWines"] / df["TotalSpending"].replace(0, np.nan)).fillna(0)
df["GoldShare"] = (df["MntGoldProds"] / df["TotalSpending"].replace(0, np.nan)).fillna(0)
df["KidsTotal"] = df["Kidhome"].fillna(0) + df["Teenhome"].fillna(0)
df["IsFamily"] = (df["KidsTotal"] > 0).astype(int)

# krátká sanity kontrola
print("Rows:", len(df))
print("Columns (sample):", df.columns[:12].tolist())

# ----------------------------
# 2) EDA & korelace
# ----------------------------
profile_cols = [
    "Age","Income","Customer_since_years",
    "TotalSpending","TotalPurchases","AvgPurchaseValue",
    "WineShare","GoldShare","WebRatio","KidsTotal","IsFamily"
]

plt.figure(figsize=(10,8))
corr = df[profile_cols].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", center=0)
plt.title("Correlation matrix (profile features)")
plt.show()


strong = corr.unstack().abs().sort_values(ascending=False).drop_duplicates()
strong = strong[(strong < 1.0) & (strong > 0.5)]
print("\nSilné korelace (|r|>0.5):")
print(strong)

# ----------------------------
# 3) připrava vstupu pro embedding
# ----------------------------
# použijeme log transformu pro těžce zkreslené sloupce
log_cols = ["TotalFood","TotalLuxury","TotalSpending","TotalPurchases","AvgPurchaseValue","Income"]
for c in log_cols:
    df[c+"_log"] = np.log1p(df[c])

X_cols = [c+"_log" for c in log_cols] + ["Age","Customer_since_years","Education_Ordinal"]
X = df[X_cols].fillna(0)

# robust scaling kvůli outlierum , nevim jeslti to měnit
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

# PCA (už jen pro indikaci variance)
pca = PCA(n_components=6, random_state=42)
pca_comp = pca.fit_transform(X_scaled)
print("\nExplained variance (first 6 PCs):", np.round(pca.explained_variance_ratio_,3))
plt.figure(figsize=(6,4))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
plt.xlabel("n components")
plt.ylabel("cumulative explained variance")
plt.grid(True)
plt.show()

# ----------------------------
# 4) UMAP embedding
# ----------------------------
umap_model = UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.1)
X_umap = umap_model.fit_transform(X_scaled)

plt.figure(figsize=(8,6))
sns.scatterplot(x=X_umap[:,0], y=X_umap[:,1], s=30)
plt.title("UMAP (unlabeled) — raw view")
plt.show()

# ----------------------------
# 5) HDBSCAN clustering
# ----------------------------
# doporučené parametry: min_cluster_size ~ 1–3% datasetu (u malých dat snížit)?????????????????????
n = len(df)
min_cluster_size = max(15, int(0.02 * n))  # aspoň 15 nebo 2% záznamů
clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=5, metric='euclidean')
labels = clusterer.fit_predict(X_umap)  # běží na UMAP prostoru

df["Cluster_hdb"] = labels
unique, counts = np.unique(labels, return_counts=True)
print("\nHDBSCAN cluster counts (label:-1 = noise):")
print(dict(zip(unique, counts)))

# optional silhouette (only non-noise)
mask = df["Cluster_hdb"] != -1
if mask.sum() > 1:
    sil = silhouette_score(X_umap[mask.values], df.loc[mask, "Cluster_hdb"])
    print("Silhouette (non-noise):", round(sil,3))
else:
    print("Silhouette: not enough non-noise points")

# vizualizace HDBSCAN clusters
plt.figure(figsize=(9,6))
sns.scatterplot(x=X_umap[:,0], y=X_umap[:,1], hue=df["Cluster_hdb"], palette="tab10", s=35, legend='full')
plt.title("UMAP + HDBSCAN clusters")
plt.legend(title="Cluster")
plt.show()

# ----------------------------
# 6) Profilace clusterů
# ----------------------------
# 6a) počet a velikost
print("\nCluster counts (including noise label -1):")
print(df["Cluster_hdb"].value_counts().sort_index())

# 6b) mediány a průměry pro interpretaci (použij median pro odpor vůči outlierům)
profile_stats = df.groupby("Cluster_hdb")[profile_cols].median().round(2)
print("\nMedian profiles by cluster:")
print(profile_stats)

# 6c) normalized profiles for heatmap (standardize on original individuals first!)
scaler_z = StandardScaler()
df_z = df.copy()
df_z[profile_cols] = scaler_z.fit_transform(df_z[profile_cols])
cluster_profiles_z = df_z.groupby("Cluster_hdb")[profile_cols].mean()

plt.figure(figsize=(10,6))
sns.heatmap(cluster_profiles_z.T, annot=True, cmap="RdBu_r", center=0, fmt=".2f")
plt.title("Normalized (z) cluster profiles (mean z per feature)")
plt.show()

# ----------------------------
# 7) Statistické testy
# ----------------------------
# vybereme pár metrik, které chceme testovat
test_vars = ["TotalSpending","TotalPurchases","AvgPurchaseValue","Income","WebRatio","WineShare"]

print("\nKruskal-Wallis tests:")
for var in test_vars:
    groups = []
    labels_list = []
    for lab in sorted(df["Cluster_hdb"].unique()):
        grp = df.loc[df["Cluster_hdb"] == lab, var]
        # vynech noise pokud příliš málo dat
        if len(grp) >= 5:
            groups.append(grp)
            labels_list.append(str(lab))
    if len(groups) >= 2:
        H, p = kruskal(*groups)
        print(f"{var}: H={H:.3f}, p={p:.4f}")
    else:
        print(f"{var}: not enough groups for test")

# pairwise post-hoc: Tukey on z-scored numeric values (note: Tukey expects no -1 label ideally)
df_post = df_z[df_z["Cluster_hdb"] != -1].copy()
if df_post["Cluster_hdb"].nunique() > 1:
    for var in test_vars:
        try:
            res = pairwise_tukeyhsd(df_post[var], df_post["Cluster_hdb"])
            print("\nTukey HSD for", var)
            print(res.summary())
        except Exception as e:
            print("Tukey failed for", var, ":", e)

# ----------------------------
# 8) Campaign response per cluster
# ----------------------------
campaign_cols = [c for c in ["AcceptedCmp1","AcceptedCmp2","AcceptedCmp3","AcceptedCmp4","AcceptedCmp5","Response"] if c in df.columns]
campaign_summary = df.groupby("Cluster_hdb")[campaign_cols].mean().round(3)
campaign_summary["Size"] = df["Cluster_hdb"].value_counts().sort_index()
print("\nCampaign response per cluster:")
print(campaign_summary)

plt.figure(figsize=(10,5))
sns.heatmap(campaign_summary[campaign_cols].T, annot=True, cmap="YlGnBu")
plt.title("Campaign acceptance rates per cluster (rows=campaigns)")
plt.show()

# ----------------------------
# 9)

# uložit výsledky s clustery
df.to_csv("customers_with_hdb_clusters.csv", index=False)
print("\nExport saved to customers_with_hdb_clusters.csv")

# jednoduché doporučení (auto)
print("\nSimple marketing recommendations (auto):")
for lab, row in campaign_summary.iterrows():
    sz = int(row["Size"])
    avg_resp = row["Response"] if "Response" in row.index else np.nan
    label = "noise" if lab == -1 else f"cluster_{lab}"
    print(f"- {label} (n={sz}): avg Response={avg_resp}")
    if lab == -1:
        print("  > noise: pravděpodobně outliers/neopakující se chování")
    else:
        # look at spending and income medians:
        med = profile_stats.loc[lab]
        if med["Income"] > df["Income"].median():
            print("  > doporučeno: premium / personalizované kampaně (vyšší ARPU)")
        elif med["TotalSpending"] < df["TotalSpending"].median():
            print("  > doporučeno: cenové nabídky, slevy, bundly")
        else:
            print("  > doporučeno: remarketing + upsell")


#CLUSTERING UVNITR CLUSTERU 0 (ten nejvetsi), at muzeme projit mainstream customers
df0 = df[df["Cluster_hdb"] == 0].copy()

# UMAP for finer segmentation
umap_small = UMAP(n_components=2, random_state=42)
X0_umap = umap_small.fit_transform(X_scaled[df["Cluster_hdb"] == 0])

hdb_small = hdbscan.HDBSCAN(
    min_cluster_size=20,
    min_samples=10,
    cluster_selection_epsilon=0.1
)
sub_labels = hdb_small.fit_predict(X0_umap)

df0["Subcluster"] = sub_labels

sns.scatterplot(
    x=X0_umap[:,0], y=X0_umap[:,1],
    hue=sub_labels, palette="Set2"
)
plt.title("Subclusters inside cluster 0")
plt.show()


