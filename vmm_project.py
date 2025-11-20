import pandas as pd
import numpy as np
import seaborn as sns
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer

# nacteni dat
df = pd.read_csv('customer_personality_Final.csv')


# vytvireni kopie
df_clean = df.copy()

# odstraneni outliers
#vek
current_year = 2025
df_clean['Age'] = current_year - df_clean['Year_Birth']
df_clean = df_clean[(df_clean['Age'] >= 18) & (df_clean['Age'] <= 100)]

#prijem
income_median = df_clean['Income'].median()
df_clean['Income'] = df_clean['Income'].fillna(income_median)

# vzdelani
df_clean['Education'] = df_clean['Education'].replace({'2n Cycle': 'Master'})

# vztahy
df_clean['Marital_Status'] = df_clean['Marital_Status'].replace({
    'Alone': 'Single',
})
df_clean['Marital_Status'] = df_clean['Marital_Status'].fillna('Unknown')


# vsechny nakoupene veci
spending_columns = ['MntWines', 'MntFruits', 'MntMeatProducts', 
                   'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']

for col in spending_columns:
    df_clean[col] = df_clean[col].fillna(0)

df_clean['Total_Spent'] = df_clean[spending_columns].sum(axis=1)

# vsechny deti
df_clean['Total_Children'] = df_clean['Kidhome'] + df_clean['Teenhome']

# rodicove
df_clean['Is_Parent'] = (df_clean['Total_Children'] > 0).astype(int)

# vsechny kampane
campaign_columns = ['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 
                   'AcceptedCmp4', 'AcceptedCmp5', 'Response']
df_clean['Total_Accepted'] = df_clean[campaign_columns].sum(axis=1)

# jak dlouho je zakaznikem
df_clean['Dt_Customer'] = pd.to_datetime(df_clean['Dt_Customer'], dayfirst=True, errors='coerce')
latest_date = df_clean['Dt_Customer'].max()
df_clean['Customer_For_Days'] = (latest_date - df_clean['Dt_Customer']).dt.days

# odstraneni zbytecnych sloupcu 
df_clean = df_clean.drop(['Z_CostContact', 'Z_Revenue'], axis=1)

#predzpracovani pro pca
pca_columns = [
    'Income', 'Age', 'Recency',
    'MntWines', 'MntFruits', 'MntMeatProducts', 
    'MntFishProducts', 'MntSweetProducts', 'MntGoldProds',
    'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 
    'NumStorePurchases', 'NumWebVisitsMonth',
     'Total_Children', 'Total_Accepted', 'Customer_For_Days'
]
#'Total_Spent'

pca_data = df_clean[pca_columns].copy()

# prazdne hodnoty nahrazeny medianem 
imputer = SimpleImputer(strategy='median')
pca_data_imputed = pd.DataFrame(imputer.fit_transform(pca_data), 
                               columns=pca_data.columns, 
                               index=pca_data.index)


# standardizace

scaler = StandardScaler()
pca_data_scaled = scaler.fit_transform(pca_data_imputed)

print("Data standardized for PCA")
print(f"Scaled data shape: {pca_data_scaled.shape}")

#pca
pca = PCA()
pca_result = pca.fit_transform(pca_data_scaled)

pca_df = pd.DataFrame(data=pca_result, 
                     columns=[f'PC{i+1}' for i in range(pca_data_scaled.shape[1])])




# 6.1 Explained Variance
explained_variance = pca.explained_variance_ratio_
cumulative_variance = explained_variance.cumsum()

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.7)
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance by Principal Component')

plt.subplot(1, 2, 2)
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance')
plt.axhline(y=0.8, color='r', linestyle='--', label='80% Variance')
plt.axhline(y=0.9, color='g', linestyle='--', label='90% Variance')
plt.legend()

plt.tight_layout()
plt.show()

print(f"Variance explained by first 2 PCs: {cumulative_variance[1]:.3f}")
print(f"Variance explained by first 3 PCs: {cumulative_variance[2]:.3f}")

# 6.2 PCA Loadings (Feature contributions)
pca_loadings = pd.DataFrame(
    pca.components_.T,
    columns=[f'PC{i+1}' for i in range(pca_data_scaled.shape[1])],
    index=pca_columns
)

# top 3 featury pro kazdou komponentu 
print("Top features for PC1:")
print(pca_loadings['PC1'].abs().sort_values(ascending=False).head(3))
print("\nTop features for PC2:")
print(pca_loadings['PC2'].abs().sort_values(ascending=False).head(3))
print("\nTop features for PC3:")
print(pca_loadings['PC3'].abs().sort_values(ascending=False).head(3))



# vysledek pca
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(pca_df['PC1'], pca_df['PC2'], alpha=0.6, s=20)
plt.xlabel(f'PC1 ({explained_variance[0]:.2%} variance)')
plt.ylabel(f'PC2 ({explained_variance[1]:.2%} variance)')
plt.title('2D PCA Projection')

# rozdeleni podle prijmu 
plt.subplot(1, 2, 2)
scatter = plt.scatter(pca_df['PC1'], pca_df['PC2'], c=df_clean['Income'], 
                     alpha=0.6, s=20, cmap='viridis')
plt.colorbar(scatter, label='Income')
plt.xlabel(f'PC1 ({explained_variance[0]:.2%} variance)')
plt.ylabel(f'PC2 ({explained_variance[1]:.2%} variance)')
plt.title('2D PCA - Colored by Income')
plt.tight_layout()
plt.show()

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

#pocet komponent pro shlukovani 
n_components = 3
pca_for_clustering = pca_result[:, :n_components]

#pocet shluku 
optimal_k = 3  

# k means
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(pca_for_clustering)

df_clean['Cluster'] = cluster_labels
pca_df['Cluster'] = cluster_labels

print(f"Cluster distribution:\n{df_clean['Cluster'].value_counts().sort_index()}")

# vysledek shlukovani 
plt.figure(figsize=(15, 5))
scatter = plt.scatter(pca_df['PC1'], pca_df['PC2'], c=df_clean['Cluster'], 
                     cmap='viridis', alpha=0.7, s=30)
plt.xlabel(f'PC1 ({explained_variance[0]:.2%} variance)')
plt.ylabel(f'PC2 ({explained_variance[1]:.2%} variance)')
plt.title('Customer Segments in PCA Space')
plt.colorbar(scatter, label='Cluster')
plt.tight_layout()
plt.show()

# Analyze cluster characteristics
cluster_profile = df_clean.groupby('Cluster').agg({
    'Income': ['mean', 'median', 'std'],
    'Age': ['mean', 'median'],
    'Total_Spent': ['mean', 'median', 'sum'],
    'Total_Children': ['mean', 'sum'],
    'Total_Accepted': ['mean', 'sum'],
    'Recency': ['mean', 'median'],
    'Customer_For_Days': ['mean', 'median'],
    'Is_Parent': 'mean',
    'MntWines': 'mean',
    'MntMeatProducts': 'mean',
    'MntGoldProds': 'mean',
    'NumDealsPurchases': 'mean',
    'NumWebPurchases': 'mean',
    'NumStorePurchases': 'mean'
}).round(2)

print("Cluster Profiles:")
print(cluster_profile)

# Analyze spending patterns across clusters
spending_by_cluster = df_clean.groupby('Cluster')[spending_columns].mean()

plt.figure(figsize=(12, 6))
spending_by_cluster.T.plot(kind='bar', figsize=(12, 6))
plt.title('Average Spending by Product Category and Cluster')
plt.xlabel('Product Category')
plt.ylabel('Average Spending')
plt.legend(title='Cluster')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()