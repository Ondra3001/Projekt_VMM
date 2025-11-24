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
#  12) PCA – snížení dimenzionality
# ============================================================

from sklearn.decomposition import PCA

# vezmeme 5 hlavních komponent, které vysvětlí cca 60–70 % variance
pca = PCA(n_components=8)
X_pca = pca.fit_transform(X_scaled)

print("Explained variance per component:", pca.explained_variance_ratio_)
print("Cumulative:", pca.explained_variance_ratio_.cumsum())

# vytvoříme DataFrame, aby se s tím lépe pracovalo
pca_df = pd.DataFrame(
    X_pca,
    columns=[f"PC{i}" for i in range(1, 9)]
)
pca_df["Cluster_ID"] = df_encoded.index  # pro pozdější spojení


# ============================================================
#  13) Clusterování na PCA datech
# ============================================================

from sklearn.cluster import KMeans

k_final = 4  # doporučuji začít s 3–5, díky PCA už to bývá stabilnější
kmeans = KMeans(n_clusters=k_final, random_state=42)
clusters = kmeans.fit_predict(pca_df.iloc[:, :8])

pca_df["Cluster"] = clusters

# přidáme clustery zpět do původního df_encoded
df_encoded["Cluster"] = clusters


# ============================================================
#  14) Silhouette score
# ============================================================

from sklearn.metrics import silhouette_score

sil = silhouette_score(X_pca, clusters)
print("Silhouette score (PCA-based):", sil)


# ============================================================
#  15) Vizualizace PC1 vs PC2
# ============================================================

plt.figure(figsize=(8, 6))
sns.scatterplot(
    x=pca_df["PC1"], y=pca_df["PC2"],
    hue=pca_df["Cluster"], palette="tab10"
)
plt.title("PCA clustering – PC1 vs PC2")
plt.show()


# ============================================================
#  16) Profil clusterů v původních features
# ============================================================

cluster_summary = df_encoded.groupby("Cluster")[features].mean().round(2)
print("\n===== PROFILY CLUSTERŮ =====")
print(cluster_summary)

#