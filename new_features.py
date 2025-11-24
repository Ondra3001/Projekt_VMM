import pandas as pd
from datetime import datetime

def new_featues(dataset):
    
    pca_features = ['Year_Birth', 'Education', 'Marital_Status', 'Income', 
                  'Kidhome', 'Teenhome', 'Dt_Customer', 'Recency']
    
    pca_data = dataset[pca_features].copy()

    pca_data['Age'] = 2025 - pca_data['Year_Birth']  # Assuming current year is 2015
    pca_data['Total_Children'] = pca_data['Kidhome'] + pca_data['Teenhome']
    pca_data['Is_Parent'] = (pca_data['Total_Children'] > 0).astype(int)

# Calculate customer tenure in days
    pca_data['Dt_Customer'] = pd.to_datetime(pca_data['Dt_Customer'])
    latest_date = pca_data['Dt_Customer'].max()
    pca_data['Customer_Tenure_Days'] = (latest_date - pca_data['Dt_Customer']).dt.days

    return pca_data