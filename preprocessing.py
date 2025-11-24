
import pandas as pd

def preprocessing(dataset_path):

    df = pd.read_csv(dataset_path)

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

    return df_clean
