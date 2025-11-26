import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
# data import 

path = "customer_personality_Final.csv"
df = pd.read_csv(path)

#print(df)

# základní přehled
#df.info()

#df.describe(include='all')

#prehled numerickych promennych

#num_cols = ['Income', 'Year_Birth', 'Age', 'MntWines', 'MntMeatProducts',
  #          'MntFruits', 'MntSweetProducts', 'MntGoldProds', 'Recency']

#plt.figure(figsize=(15, 10))
#for i, col in enumerate(num_cols, 1):
 #   plt.subplot(3, 3, i)
  #  sns.histplot(df[col], kde=True, bins=30)
   # plt.title(col)
#plt.tight_layout()
#plt.show()

#prevod year_birth na age -> odstraneni outliers
current_year = datetime.now().year
df['Age'] = current_year - df['Year_Birth']

# odstranění nereálných roků narození nebo extrémně starých zákazníků
df = df[(df['Year_Birth'] > 1900) & (df['Age'] < 120)]

# --- CUSTOMER SINCE (v letech) ---
df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], dayfirst=True)

today = pd.to_datetime(datetime.now())

df["Customer_since_years"] = (today - df["Dt_Customer"]).dt.days / 365.25


df["Income"] = df["Income"].fillna(df["Income"].median())



# prohledani
print(df['Marital_Status'].unique())
df["Marital_Status"] = df["Marital_Status"].replace({
    "Alone": "Single"
})
# educations jsou 'Graduation' 'PhD' 'Master' 'Basic' '2n Cycle'.. prevod na ordinalni
print(df['Education'].unique())

education_map = {
    "Basic": 1,
    "2n Cycle": 2,
    "Graduation": 3,
    "Master": 4,
    "PhD": 5
}

df["Education_Ordinal"] = df["Education"].map(education_map)
df["Education_Ordinal"] = df["Education_Ordinal"].fillna(0)   # UNKNOWN
# ============================================================
#   8) VÝBĚR FEATURE SETU PRO CLUSTERING
# ============================================================



# --- numerické nákupní chování ---
purchase_cols = [
    "MntWines", "MntMeatProducts", "MntFishProducts", "MntFruits",
    "MntSweetProducts", "MntGoldProds",
    "NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases",
    "NumDealsPurchases", "NumWebVisitsMonth", "Recency"
]

# demografie ---
demo_cols = [
    "Age", "Income", "Kidhome", "Teenhome", "Customer_since_years"
]

# --- kampaně ---
campaign_cols = [
    "AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3",
    "AcceptedCmp4", "AcceptedCmp5", "Response"
]

# --- kategorické proměnné (zaencodingujeme), Education zkousim dat jako ordinal ---
cat_cols = ["Marital_Status"]




#   9) ONE-HOT ENCODING pro marital


df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

df_encoded['TotalFood'] = (
    df_encoded['MntMeatProducts']
    + df_encoded['MntFishProducts']
    + df_encoded['MntFruits']
    + df_encoded['MntSweetProducts']
)
# kompletní feature set
features = (
    purchase_cols
    + demo_cols
    + campaign_cols
    + ["Education_Ordinal"]    #  ordinal education
    + [col for col in df_encoded.columns if col.startswith("Marital_")]
)


X = df_encoded[features].copy()

# ============================================================
#  10) OŠETŘENÍ OUTLIERŮ – cutoff příjmu
# ============================================================

cutoff = df_encoded["Income"].quantile(0.99)
df_encoded = df_encoded[df_encoded["Income"] <= cutoff]
X = df_encoded[features]

# ============================================================
#  11) DOPLNĚNÍ NaN + STANDARD SCALING
# ============================================================

X = X.fillna(X.median())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ============================================================
# 12) PCA pouze na NÁKUPNÍ chování
# ============================================================

from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

# --- blok dat pro PCA (výdaje + návštěvy + recency) ---
pca_cols = [
    "MntWines", "MntMeatProducts", "MntFishProducts",
    "MntFruits", "MntSweetProducts", "MntGoldProds",
    "NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases",
    "NumWebVisitsMonth", "Recency"
]

X_pca_input = X[pca_cols]

# Robust scaler místo StandardScaler
from sklearn.preprocessing import RobustScaler
scaler_pca = RobustScaler()
X_pca_scaled = scaler_pca.fit_transform(X_pca_input)

# PCA udržující 80% variance
pca = PCA(n_components=0.80)
X_pca = pca.fit_transform(X_pca_scaled)

print("\n--- PCA variances ---")
print("Explained variance:", pca.explained_variance_ratio_)
print("Cumulative:", pca.explained_variance_ratio_.cumsum())

# vytvoříme dataframe PC komponent
pca_feature_names = [f"PC{i}" for i in range(1, X_pca.shape[1] + 1)]
pca_df = pd.DataFrame(X_pca, columns=pca_feature_names)


# ============================================================
# 13) Spojení PCA + DEMOGRAFIE (nejlepší praxe)
# ============================================================

demographic_cols = [
    "Age", "Income", "Kidhome", "Teenhome", "Customer_since_years",
    "Education_Ordinal"
]

X_final = pd.concat(
    [
        pca_df.reset_index(drop=True),
        X[demographic_cols].reset_index(drop=True)
    ],
    axis=1
)

print("\nFinal shape for clustering:", X_final.shape)


# ============================================================
# 14) Gaussian Mixture Model (lepší než KMeans)
# ============================================================

gmm = GaussianMixture(
    n_components=4,
    covariance_type="full",
    random_state=42
)

clusters = gmm.fit_predict(X_final)
X_final["Cluster"] = clusters
df_encoded["Cluster"] = clusters


# ============================================================
# 15) Silhouette score
# ============================================================

sil = silhouette_score(X_final.drop("Cluster", axis=1), clusters)
print("\nSilhouette score:", sil)


# ============================================================
# 16) Vizualizace PC1–PC2
# ============================================================

plt.figure(figsize=(8, 6))
sns.scatterplot(
    x=pca_df["PC1"],
    y=pca_df["PC2"],
    hue=clusters,
    palette="tab10"
)
plt.title("GMM clustering – PCA space")
plt.show()


# ============================================================
# 17) PROFILY CLUSTERŮ
# ============================================================

cluster_summary = df_encoded.groupby("Cluster")[
    pca_cols + demographic_cols
].mean().round(2)

print("\n===== PROFILY CLUSTERŮ =====")
print(cluster_summary)
