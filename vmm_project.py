import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
# data import 

path = "customer_personality_Final.csv"
df = pd.read_csv(path)

#print(df)

# základní přehled
#df.info()

#df.describe(include='all')

#prehled numerickych promennych

num_cols = ['Income', 'Year_Birth', 'Age', 'MntWines', 'MntMeatProducts',
            'MntFruits', 'MntSweetProducts', 'MntGoldProds', 'Recency']

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

df['Age'].describe()

plt.figure(figsize=(15, 10))
for i, col in enumerate(num_cols, 1):
    plt.subplot(3, 3, i)
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(col)
plt.tight_layout()
plt.show()

df.info()
df.describe(include='all')

# === Korelační analýza ===
from scipy.stats import pearsonr

# vyber jen numerické sloupce
numeric_df = df.select_dtypes(include=['float64', 'int64'])

# korelační matice
corr = numeric_df.corr()

plt.figure(figsize=(12, 8))
sns.heatmap(corr, cmap='coolwarm', center=0, annot=False)
plt.title('Korelační matice numerických proměnných')
plt.show()

# textový přehled silných korelací
corr_unstacked = corr.unstack().sort_values(ascending=False)
strong_corr = corr_unstacked[(corr_unstacked < 1) & (corr_unstacked > 0.5)]
print("Silnější korelace (r > 0.5):")
print(strong_corr)

# === Test statistické významnosti (p-hodnoty) ===
results = []
for (var1, var2) in strong_corr.index:
    # vyber jen řádky bez NaN
    data = df[[var1, var2]].dropna()
    if len(data) > 2:
        r, p = pearsonr(data[var1], data[var2])
        results.append((var1, var2, r, p))

significance_df = pd.DataFrame(results, columns=['Variable 1', 'Variable 2', 'r', 'p-value'])
significance_df['Significant (p<0.05)'] = significance_df['p-value'] < 0.05

print("\nTest statistické významnosti pro silnější korelace:")
print(significance_df.sort_values(by='r', ascending=False))

#otulier v income, nevim jestli ho nechat?? 160000 prijem- muze byt bohac
#plt.figure(figsize=(8,5))
#sns.boxplot(x=df['Income'], color='skyblue')
#plt.title('Boxplot příjmu (Income)')
#plt.show()

#Nans zmena na unknown - nechci tam davat median?
#df['Education'] = df['Education'].fillna('Unknown')
#df['Marital_Status'] = df['Marital_Status'].fillna('Unknown')


###########################################################################



# === Výběr vhodných proměnných ===
features = [
    'Age', 'Income',
    'MntWines', 'MntMeatProducts', 'MntFishProducts',
    'MntFruits', 'MntSweetProducts', 'MntGoldProds',
    'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases',
    'NumDealsPurchases', 'NumWebVisitsMonth', 'Recency'
]

X = df[features].copy()

# Nany nahrazny medianem, nemelo by to u techto features vadit asi
X = X.fillna(X.median())

# Robustní škálování (lepší pro outliery - at se muze pouzit clustering)
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

# === Elbow metoda -- furt si myslim ze 5 je fajn
inertia = []
K_range = range(2, 10)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X_scaled)
    inertia.append(km.inertia_)

plt.plot(K_range, inertia, marker='o')
plt.xlabel("Počet clusterů")
plt.ylabel("Inertia")
plt.title("Elbow metoda")
plt.show()

# === Finální model ===
kmeans = KMeans(n_clusters=5, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# === Shrnutí clusterů ===
cluster_summary = df.groupby('Cluster')[features].mean().round(2)
print("\n=== Profil jednotlivých clusterů ===")
print(cluster_summary)

# === Vizualizace (volitelná) ===
sns.pairplot(df, hue='Cluster', vars=['Income', 'Age', 'MntWines', 'NumStorePurchases'])
plt.show()
