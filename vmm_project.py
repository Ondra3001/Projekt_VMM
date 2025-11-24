import pandas as pd
import numpy as np
import seaborn as sns
import prince 
from datetime import datetime
import matplotlib.pyplot as plt



from preprocessing import * 
from run_pca import * 
from new_features import *

# nacteni dat
df_clean = preprocessing('customer_personality_Final.csv') #removes nonsense values 
df_pca = new_featues(df_clean) #creates new features 

pca_features = ['Age', 'Income', 'Total_Children','Recency', 'Customer_Tenure_Days'] #numerical features 
df_pca = df_pca[pca_features].dropna() #drop na


pca_complete = run_pca(df_pca, 2) #run pca with given number of components 

pca_complete_df = pd.DataFrame(
    pca_complete,
    columns=["PC1","PC2"],
    index=df_clean.index
) #return dataframe with new coordinates 

plt.figure(figsize=(10, 8))
plt.scatter(pca_complete_df.iloc[:, 0], pca_complete_df.iloc[:, 1], alpha=0.7, s=50)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Results - 518 Customers')
plt.grid(alpha=0.3)
plt.show()