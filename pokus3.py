
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
# 1) Načtení + predzpracovani
# ----------------------------
path = "customer_personality_Final.csv"
df = pd.read_csv(path)

# cleaning
current_year = datetime.now().year
df["Age"] = current_year - df["Year_Birth"]
df = df[(df["Year_Birth"] > 1900) & (df["Age"] < 120)].copy()
df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], dayfirst=True, errors="coerce")
df["Customer_since_years"] = (datetime.now() - df["Dt_Customer"]).dt.days / 365.25
df["Income"] = df["Income"].fillna(df["Income"].median())

education_map = {"Basic":1, "2n Cycle":2, "Graduation":3, "Master":4, "PhD":5}
df["Education_Ordinal"] = df["Education"].map(education_map)
median_edu = df["Education_Ordinal"].median()
df["Education_Ordinal"] = df["Education_Ordinal"].fillna(median_edu)


df["Marital_Status"] = df["Marital_Status"].replace({"Alone":"Single"}).fillna("Unknown")
marital_map = {
    "Single": 0,
    "Together": 1,
    "Married": 2,
    "Divorced": 3,
    "Widow": 4,
    "Unknown": -1
}

df["Marital_Status_Ordinal"] = df["Marital_Status"].map(marital_map)

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

#  kontrola
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
plt.title("Correlation matrix")
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

X_cols = [c+"_log" for c in log_cols] + ["Age","Customer_since_years","Education_Ordinal", "Marital_Status_Ordinal", "KidsTotal"]
X = df[X_cols].fillna(0)

# robust scaling kvůli outlierum
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

# PCA (pro indikaci variance, neni pouzito k analyze)
pca = PCA(n_components=6, random_state=42)
pca_comp = pca.fit_transform(X_scaled)
print("\nExplained variance (first 6 PCs):", np.round(pca.explained_variance_ratio_,3))
plt.figure(figsize=(6,4))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
plt.xlabel("n components")
plt.ylabel("cumulative explained variance")
plt.grid(True)
plt.show()
# POZNAMKY:
#struktura dat má relativně nízkou dimenzionalitu a většina informací je obsažena v několika hlavních faktorech.
# PCA tedy vhodně komprimuje data a potvrzuje, že je možné použít metody jako UMAP nebo clustering bez výrazné ztráty informace.
#--> data NEJSOU dobre  linearne separovatelna, PCA nepouzivame
# ----------------------------
# UMAP
# ----------------------------
umap_model = UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.1)
X_umap = umap_model.fit_transform(X_scaled)



# ---------------------------------------------
#  UMAP INTERPRETACE PODLE DEMOGRAFIE / CHOVÁNÍ.
#k vizualizaci vybrany proměnné co se jeví jako dulezite pro odliseni
# ---------------------------------------------

interp_features = [
    "Income",
    "TotalSpending",
    "GoldShare",
    "WineShare",
    "KidsTotal",
    "Education_Ordinal"
]

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for i, feat in enumerate(interp_features):
    ax = axes[i]
    sc = ax.scatter(
        X_umap[:, 0], X_umap[:, 1],
        c=df[feat], cmap="viridis", s=25
    )
    ax.set_title(f"{feat} (větší = světlejší)")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    plt.colorbar(sc, ax=ax)

plt.suptitle("UMAP – Interpretační mapa zákazníků dle demografie a nákupního chování")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,6))
sns.scatterplot(x=X_umap[:,0], y=X_umap[:,1], s=30)
plt.title("UMAP (unlabeled) — raw view")
plt.show()

response_corr = df[[
    "AcceptedCmp1","AcceptedCmp2","AcceptedCmp3","AcceptedCmp4","AcceptedCmp5","Response",
    "Income","TotalSpending","AvgPurchaseValue","WineShare","WebRatio","Age", "Education_Ordinal", "KidsTotal"
]].corr()

sns.heatmap(response_corr, annot=False, cmap="coolwarm", center=0)
plt.title("Korelace kampaní a zákaznických charakteristik")
plt.show()


# UMAP vizualizace kampaní v jednom obrázku


campaign_cols = ["AcceptedCmp1","AcceptedCmp2","AcceptedCmp3",
                 "AcceptedCmp4","AcceptedCmp5","Response"]

# jen sloupce, které v DF existují
campaign_cols = [c for c in campaign_cols if c in df.columns]

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for i, cmp in enumerate(campaign_cols):
    ax = axes[i]
    sns.scatterplot(
        x=X_umap[:,0],
        y=X_umap[:,1],
        hue=df[cmp],
        palette="coolwarm",
        s=35,
        ax=ax,
        legend=False
    )
    ax.set_title(f"{cmp} (1 = accepted)")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")

# případný prázdný subplot
for j in range(len(campaign_cols), len(axes)):
    axes[j].axis("off")

plt.suptitle("UMAP – Vizualizace reakcí na kampaně", fontsize=16)
plt.tight_layout()
plt.show()




# ==========================================
# STATISTICKÉ TESTY PRO KAMPANĚ
# ==========================================

from scipy.stats import mannwhitneyu

# proměnné které budeme testovat
test_features = [
    "Income","TotalSpending","TotalPurchases","AvgPurchaseValue",
    "WineShare","GoldShare","WebRatio","KidsTotal","Age","Education_Ordinal", "Marital_Status_Ordinal"
]

campaign_cols = ["AcceptedCmp1","AcceptedCmp2","AcceptedCmp3","AcceptedCmp4","AcceptedCmp5","Response"]
campaign_cols = [c for c in campaign_cols if c in df.columns]

results = []

for cmp in campaign_cols:
    for feat in test_features:

        grp1 = df[df[cmp] == 1][feat]
        grp0 = df[df[cmp] == 0][feat]

        if len(grp1) > 3:  # musí být aspoň pár respondentů
            stat, p = mannwhitneyu(grp1, grp0, alternative='two-sided')

            results.append({
                "campaign": cmp,
                "feature": feat,
                "mean_responders": grp1.mean(),
                "mean_nonresponders": grp0.mean(),
                "difference": grp1.mean() - grp0.mean(),
                "p_value": p
            })

results_df = pd.DataFrame(results)
results_df["significant"] = results_df["p_value"] < 0.05

# multiple testing correction
from statsmodels.stats.multitest import multipletests

results_df["p_adj"] = multipletests(results_df["p_value"], method='fdr_bh')[1]

# signed effect size scaled by significance
results_df["signed_log_p"] = (
    np.sign(results_df["difference"]) * (-np.log10(results_df["p_adj"]))
)

# pivot pro heatmapu
pivot_signed = results_df.pivot_table(
    index="feature",
    columns="campaign",
    values="signed_log_p"
)

plt.figure(figsize=(12, 7))
sns.heatmap(
    pivot_signed,
    cmap="coolwarm",  # modrá=nižší u respondentů, červená=vyšší u respondentů
    center=0,
    annot=False
)
plt.title("Signed effect size × significance\n( sign(diff) * -log10(p_adj) )")
plt.show()
# Filtr na statisticky významné efekty
sig = results_df[results_df["p_adj"] < 0.05].copy()

# seřadíme podle velikosti efektu × významnost
sig = sig.sort_values("signed_log_p", ascending=False)

print("\n==============================")
print("SIGNIFIKANTNÍ EFEKTY (p_adj < 0.05)")
print("==============================\n")

for _, row in sig.iterrows():
    print(
        f"Kampaň: {row['campaign']}\n"
        f"  Feature: {row['feature']}\n"
        f"  Rozdíl (respondenti – nereagující): {row['difference']:.3f}\n"
        f"  p-value: {row['p_value']:.3e}\n"
        f"  p_adj: {row['p_adj']:.3e}\n"
        f"  signed_log_p: {row['signed_log_p']:.3f}\n"
        "----------------------------------------"
    )



