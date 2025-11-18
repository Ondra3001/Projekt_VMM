import pandas as pd
import numpy as np
import seaborn as sns
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer

# Load the data
df = pd.read_csv('customer_personality_Final.csv')

# Step 1: Initial Data Cleaning

# Create a copy to preserve original data
df_clean = df.copy()

# 1.1 Handle Year_Birth outliers
current_year = 2015  # Based on latest Dt_Customer being 2014
df_clean['Age'] = current_year - df_clean['Year_Birth']

# Remove impossible ages (e.g., Age > 100 or Age < 18)
df_clean = df_clean[(df_clean['Age'] >= 18) & (df_clean['Age'] <= 100)]

# 1.2 Handle Income outliers and missing values

# Impute remaining missing Income with median
income_median = df_clean['Income'].median()
df_clean['Income'] = df_clean['Income'].fillna(income_median)

# 1.3 Standardize Education values
df_clean['Education'] = df_clean['Education'].replace({'2n Cycle': 'Master'})

# 1.4 Standardize Marital_Status
df_clean['Marital_Status'] = df_clean['Marital_Status'].replace({
    'Alone': 'Single',
})

# Fill remaining missing Marital_Status with 'Unknown'
df_clean['Marital_Status'] = df_clean['Marital_Status'].fillna('Unknown')

# Step 2: Feature Engineering

# 2.1 Create Total_Spent
spending_columns = ['MntWines', 'MntFruits', 'MntMeatProducts', 
                   'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']

# Handle missing spending data - if all spending columns are NaN, assume no spending (0)
# If only some are NaN, fill with 0
for col in spending_columns:
    df_clean[col] = df_clean[col].fillna(0)

df_clean['Total_Spent'] = df_clean[spending_columns].sum(axis=1)

# 2.2 Create Total_Children
df_clean['Total_Children'] = df_clean['Kidhome'] + df_clean['Teenhome']

# 2.3 Create Is_Parent
df_clean['Is_Parent'] = (df_clean['Total_Children'] > 0).astype(int)

# 2.4 Create Total_Accepted (campaign responses)
campaign_columns = ['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 
                   'AcceptedCmp4', 'AcceptedCmp5', 'Response']
df_clean['Total_Accepted'] = df_clean[campaign_columns].sum(axis=1)

# 2.5 Create Customer_For (tenure in days)
df_clean['Dt_Customer'] = pd.to_datetime(df_clean['Dt_Customer'], dayfirst=True, errors='coerce')
latest_date = df_clean['Dt_Customer'].max()
df_clean['Customer_For_Days'] = (latest_date - df_clean['Dt_Customer']).dt.days

# 2.6 Remove constant columns
df_clean = df_clean.drop(['Z_CostContact', 'Z_Revenue'], axis=1)

# Step 3: Final Data Preparation for PCA

# Select numerical features for PCA
pca_columns = [
    'Income', 'Age', 'Recency',
    'MntWines', 'MntFruits', 'MntMeatProducts', 
    'MntFishProducts', 'MntSweetProducts', 'MntGoldProds',
    'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 
    'NumStorePurchases', 'NumWebVisitsMonth',
     'Total_Children', 'Total_Accepted', 'Customer_For_Days'
]
#'Total_Spent'
# Create PCA-ready dataset
pca_data = df_clean[pca_columns].copy()

# Check for any remaining missing values and fill with median
imputer = SimpleImputer(strategy='median')
pca_data_imputed = pd.DataFrame(imputer.fit_transform(pca_data), 
                               columns=pca_data.columns, 
                               index=pca_data.index)


# Step 4: Standardize the Data (CRITICAL for PCA)

scaler = StandardScaler()
pca_data_scaled = scaler.fit_transform(pca_data_imputed)

print("Data standardized for PCA")
print(f"Scaled data shape: {pca_data_scaled.shape}")
# Step 5: Perform PCA

pca = PCA()
pca_result = pca.fit_transform(pca_data_scaled)

# Create DataFrame for PCA results
pca_df = pd.DataFrame(data=pca_result, 
                     columns=[f'PC{i+1}' for i in range(pca_data_scaled.shape[1])])


# Step 6: Analyze PCA Results

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

# Display top contributing features for first 3 PCs
print("Top features for PC1:")
print(pca_loadings['PC1'].abs().sort_values(ascending=False).head(3))
print("\nTop features for PC2:")
print(pca_loadings['PC2'].abs().sort_values(ascending=False).head(3))
print("\nTop features for PC3:")
print(pca_loadings['PC3'].abs().sort_values(ascending=False).head(3))

# Step 7: Visualize PCA Results

# 7.1 2D PCA Plot
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(pca_df['PC1'], pca_df['PC2'], alpha=0.6, s=20)
plt.xlabel(f'PC1 ({explained_variance[0]:.2%} variance)')
plt.ylabel(f'PC2 ({explained_variance[1]:.2%} variance)')
plt.title('2D PCA Projection')

# 7.2 Color by Total_Spent
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

# Use the first few principal components for clustering (they capture most variance)
# Let's use the number that captures ~80-90% of variance
n_components = 3
pca_for_clustering = pca_result[:, :n_components]


# Choose optimal k (you can adjust based on the plots)
optimal_k = 4  # This is a common choice, adjust based on your elbow plot

# Perform K-means clustering with optimal k
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(pca_for_clustering)

# Add cluster labels to our cleaned dataframe
df_clean['Cluster'] = cluster_labels
pca_df['Cluster'] = cluster_labels

print(f"Cluster distribution:\n{df_clean['Cluster'].value_counts().sort_index()}")

# Visualize clusters in 2D PCA space
plt.figure(figsize=(15, 5))

# Plot 1: Clusters in PCA space
plt.subplot(1, 2, 1)
scatter = plt.scatter(pca_df['PC1'], pca_df['PC2'], c=df_clean['Cluster'], 
                     cmap='viridis', alpha=0.7, s=30)
plt.xlabel(f'PC1 ({explained_variance[0]:.2%} variance)')
plt.ylabel(f'PC2 ({explained_variance[1]:.2%} variance)')
plt.title('Customer Segments in PCA Space')
plt.colorbar(scatter, label='Cluster')

# Plot 2: Clusters colored by Income
plt.subplot(1, 3, 2)
scatter = plt.scatter(pca_df['PC1'], pca_df['PC2'], c=df_clean['Income'], 
                     cmap='plasma', alpha=0.7, s=30)
plt.xlabel(f'PC1 ({explained_variance[0]:.2%} variance)')
plt.ylabel(f'PC2 ({explained_variance[1]:.2%} variance)')
plt.title('PCA Space - Colored by Income')
plt.colorbar(scatter, label='Income')


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