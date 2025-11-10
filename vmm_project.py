import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
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

#otulier v income, nevim jestli ho nechat?? 160000 prijem- muze byt bohac
plt.figure(figsize=(8,5))
sns.boxplot(x=df['Income'], color='skyblue')
plt.title('Boxplot příjmu (Income)')
plt.show()

#Nans zmena na unknown - nechci tam davat median?
df['Education'] = df['Education'].fillna('Unknown')
df['Marital_Status'] = df['Marital_Status'].fillna('Unknown')


#scaling (normalizace)
features = ['Income', 'Age', 'MntWines', 'MntMeatProducts', 'MntFishProducts',
            'MntSweetProducts', 'MntGoldProds', 'NumWebPurchases',
            'NumStorePurchases', 'NumDealsPurchases', 'Recency']

X = df[features].dropna()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


inertia = []
#for k in range(2, 10):
  #  km = KMeans(n_clusters=k, random_state=42)
   # km.fit(X_scaled)
    #inertia.append(km.inertia_)
#graf at se muzu podivat jaky k urcit, elbow method --> 5 mi prijde dobry
#plt.plot(range(2, 10), inertia, marker='o')
#plt.xlabel('Počet clusterů')
#plt.ylabel('Inertia')
#plt.title('Elbow metoda pro určení optimálního počtu clusterů')
#plt.show()

kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

#df['Cluster'] = kmeans.fit_predict(X_scaled)

sns.pairplot(df, hue='Cluster', vars=['Income', 'Age', 'MntWines', 'MntMeatProducts'])
plt.show()




