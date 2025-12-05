import warnings

warnings.filterwarnings("ignore")
from scipy import stats

import pandas as pd
import numpy as np
from datetime import datetime

# preprocessing & scaling
from sklearn.preprocessing import RobustScaler, StandardScaler, KBinsDiscretizer, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Model a vyhodnotenie
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# 1) Načtení a features

path = "customer_personality_Final.csv"
df = pd.read_csv(path)

# základní cleaning
current_year = datetime.now().year
df["Age"] = current_year - df["Year_Birth"]
df = df[(df["Year_Birth"] > 1900) & (df["Age"] < 120)].copy()
df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], dayfirst=True, errors="coerce")
df["Customer_since_years"] = (datetime.now() - df["Dt_Customer"]).dt.days / 365.25
df["Income"] = df["Income"].fillna(df["Income"].median())

education_map = {"Basic": 1, "2n Cycle": 2, "Graduation": 3, "Master": 4, "PhD": 5}
df["Education_Ordinal"] = df["Education"].map(education_map).fillna(0)
df["Marital_Status"] = df["Marital_Status"].replace({"Alone": "Single"}).fillna("Unknown")

# Cieľová premenná: 1 = Master/PhD, 0 = ostatné
df['Is_Higher_Education'] = df['Education'].apply(lambda x: 1 if x in ['Master', 'PhD'] else 0)

# vypocitane features
df["TotalFood"] = df[["MntMeatProducts", "MntFishProducts", "MntFruits", "MntSweetProducts"]].sum(axis=1)
df["TotalLuxury"] = df[["MntWines", "MntGoldProds"]].sum(axis=1)
df["TotalSpending"] = df["TotalFood"] + df["TotalLuxury"]
df["TotalPurchases"] = df[["NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases", "NumDealsPurchases"]].sum(
    axis=1)
df["AvgPurchaseValue"] = (df["TotalSpending"] / df["TotalPurchases"].replace(0, np.nan)).fillna(0)
df["WebRatio"] = (df["NumWebPurchases"] / df["TotalPurchases"].replace(0, np.nan)).fillna(0)
df["WineShare"] = (df["MntWines"] / df["TotalSpending"].replace(0, np.nan)).fillna(0)
df["GoldShare"] = (df["MntGoldProds"] / df["TotalSpending"].replace(0, np.nan)).fillna(0)
df["KidsTotal"] = df["Kidhome"].fillna(0) + df["Teenhome"].fillna(0)
df["IsFamily"] = (df["KidsTotal"] > 0).astype(int)

# krátká sanity kontrola
print("Rows:", len(df))
print("Columns (sample):", df.columns[:12].tolist())


# 2) VYTVORENIE CIEĽOVEJ PREMENNEJ - ÚROVEŇ CONSUMPTION (3 KATEGÓRIE)



# Celková spotreba
df['TotalConsumption'] = df[['MntWines', 'MntFruits', 'MntMeatProducts',
                             'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']].sum(axis=1)

# Rozdelenie na 3 kategórie (nízka, stredná, vysoká)
discretizer = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='quantile')
df['Consumption_Level'] = discretizer.fit_transform(df[['TotalConsumption']]).astype(int)

# Mapovanie na zrozumiteľné názvy
level_names = {0: 'Low', 1: 'Medium', 2: 'High'}
df['Consumption_Level_Name'] = df['Consumption_Level'].map(level_names)

# Štatistika rozdelenia
#print("\nRozdelenie úrovní spotreby:")
#level_counts = df['Consumption_Level_Name'].value_counts().sort_index()
#print(level_counts)
#print(f"\nPercentuálne rozdelenie:")
#print((level_counts / len(df) * 100).round(1))

# Deskriptívna štatistika spotreby podľa úrovne
print("\n 2.)Štatistika celkovej spotreby podľa úrovne:")
consumption_stats = df.groupby('Consumption_Level_Name')['TotalConsumption'].agg([
    'count', 'mean', 'median', 'std', 'min', 'max'
]).round(0)
print(consumption_stats)


# 3) PRÍPRAVA DAT PRE MODELOVANIE


print("3) PRÍPRAVA DAT PRE MODELOVANIE")


# Výber premenných
feature_columns = [
    'Education_Ordinal',  # Hlavný faktor záujmu
    'Income',
    'Age',
    'KidsTotal',
    'IsFamily',
    'Marital_Status',  # Kategorická premenná
    'Recency',
    'Customer_since_years'
]

X = df[feature_columns]
y = df['Consumption_Level']

# Rozdelenie na trénovaciu a testovaciu sadu
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"Trénovacia sada: {X_train.shape[0]} vzoriek")
print(f"Testovacia sada: {X_test.shape[0]} vzoriek")


# 4) PREPROCESSING PIPELINE


print("4) VYTVORENIE PREPROCESSING PIPELINE")


# Identifikácia typov premenných
categorical_cols = ['Marital_Status']
numerical_cols = [col for col in feature_columns if col not in categorical_cols]

print(f"Kategorické premenné: {categorical_cols}")
print(f"Numerické premenné ({len(numerical_cols)}): {numerical_cols}")

# Vytvorenie ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_cols)
    ]
)

# Získanie názvov features po transformácii (pre neskoršiu interpretáciu)
preprocessor.fit(X_train)

# Získanie názvov features
num_features = numerical_cols
cat_features = list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols))
all_feature_names = num_features + cat_features

print(f"\nCelkový počet features po transformácii: {len(all_feature_names)}")


# 5) TRÉNOVANIE LOGISTICKEJ REGRESIE


# Vytvorenie pipeline
model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(
        multi_class='multinomial',
        solver='lbfgs',
        max_iter=1000,
        random_state=42,
        class_weight='balanced'  # triedy tu boli vyrovnane
    ))
])

# Trénovanie modelu
model.fit(X_train, y_train)

# Predikcie
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)


# 5) VYHODNOTENIE MODELU


print("5.) VYHODNOTENIE MODELU")


# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nCelková Accuracy: {accuracy:.3f}")

# Classification Report
print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=['Low', 'Medium', 'High']))

# Confusion Matrix
print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Vizualizácia confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Low', 'Medium', 'High'],
            yticklabels=['Low', 'Medium', 'High'])
plt.title('Confusion Matrix - Úroveň consumption')
plt.ylabel('Skutočná hodnota')
plt.xlabel('Predikovaná hodnota')
plt.tight_layout()
plt.show()


# 7) INTERPRETÁCIA MODELU - KOEFICIENTY


print("6) INTERPRETÁCIA - VPLYV PREMENNÝCH")


# Získanie koeficientov
coefficients = model.named_steps['classifier'].coef_

# Analýza pre každú úroveň consumption
for i, level in enumerate(['Low', 'Medium', 'High']):

    print(f"CONSUMPTION : {level}")


    # Vytvorenie DF s koeficientmi
    coef_df = pd.DataFrame({
        'Feature': all_feature_names,
        'Coefficient': coefficients[i]
    })

    # Zoradenie absolutnou hodnotou
    coef_df['Abs_Coefficient'] = np.abs(coef_df['Coefficient'])
    coef_df = coef_df.sort_values('Abs_Coefficient', ascending=False)

    print(f"\nNajdoležitejšie premenné pre '{level}' spotrebu:")
    print(coef_df.head(16).to_string(index=False))

    # Interpretácia vzdelania
    edu_coef = coef_df[coef_df['Feature'] == 'Education_Ordinal']
    if not edu_coef.empty:
        coef_val = edu_coef.iloc[0]['Coefficient']
        print(f"\nVplyv vzdelania (Education_Ordinal) na '{level}' TotalConsumption: {coef_val:.3f}")
        if coef_val > 0:
            print(f"  → Vyššie vzdelanie ZVYŠUJE pravdepodobnosť '{level}' spotreby")
        else:
            print(f"  → Vyššie vzdelanie ZNIŽUJE pravdepodobnosť '{level}' spotreby")

# ===================================================================
# 8) ANALÝZA VZŤAHU VZDELANIE - SPOTREBA
# ===================================================================
print("\n" + "=" * 60)
print("8) ANALÝZA VZŤAHU VZDELANIE - SPOTREBA")
print("=" * 60)

# Deskriptívna analýza
print("\nPriemerná spotreba podľa vzdelania:")
edu_consumption = df.groupby('Education')['TotalConsumption'].agg([
    'count', 'mean', 'median', 'std'
]).round(0).sort_values('mean', ascending=False)
print(edu_consumption)

# Percentuálne rozdelenie úrovní spotreby podľa vzdelania
print("\nPercentuálne rozdelenie úrovní spotreby podľa vzdelania:")
edu_level_dist = pd.crosstab(
    df['Education'],
    df['Consumption_Level_Name'],
    normalize='index'
) * 100
print(edu_level_dist.round(1))

# Vizualizácia
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Boxplot spotreby podľa vzdelania
sns.boxplot(data=df, x='Education', y='TotalConsumption',
            order=['Basic', '2n Cycle', 'Graduation', 'Master', 'PhD'],
            ax=axes[0])
axes[0].set_title('Rozdelenie celkovej spotreby podľa vzdelania')
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45)
axes[0].set_ylabel('Celková spotreba')

# Heatmap rozdelenia úrovní spotreby
sns.heatmap(edu_level_dist, annot=True, fmt='.1f', cmap='YlOrRd',
            ax=axes[1], cbar_kws={'label': '%'})
axes[1].set_title('Rozdelenie úrovní spotreby podľa vzdelania (%)')
axes[1].set_ylabel('Vzdelanie')
axes[1].set_xlabel('Úroveň spotreby')

plt.tight_layout()
plt.show()

# ===================================================================
# 9) PREDIKCIA PRE NOVÝCH ZÁKAZNÍKOV - PRÍKLAD
# ===================================================================
print("\n" + "=" * 60)
print("9) PREDIKCIA PRE PRÍKLADOVÝCH ZÁKAZNÍKOV")
print("=" * 60)

# Príklady nových zákazníkov
new_customers = pd.DataFrame({
    'Education_Ordinal': [1, 3, 5],  # Basic, Graduation, PhD
    'Income': [30000, 60000, 100000],
    'Age': [45, 35, 50],
    'KidsTotal': [2, 0, 1],
    'IsFamily': [1, 0, 1],
    'Marital_Status': ['Married', 'Single', 'Married'],
    'Recency': [30, 10, 5],
    'Customer_since_years': [2, 5, 10],
    'TotalPurchases': [15, 25, 40],
    'WebRatio': [0.3, 0.7, 0.5],
    'WineShare': [0.1, 0.3, 0.4],
    'GoldShare': [0.05, 0.1, 0.15]
})

# Predikcia
predictions = model.predict(new_customers)
probabilities = model.predict_proba(new_customers)

print("\nPredikcie pre príkladových zákazníkov:")
for i, (pred, probs) in enumerate(zip(predictions, probabilities)):
    education_level = {1: 'Basic', 3: 'Graduation', 5: 'PhD'}[new_customers.iloc[i]['Education_Ordinal']]
    predicted_level = level_names[pred]

    print(f"\nZákazník {i + 1} (Vzdelanie: {education_level}):")
    print(f"  Predikovaná úroveň spotreby: {predicted_level}")
    print(f"  Pravdepodobnosti: Low={probs[0]:.2%}, Medium={probs[1]:.2%}, High={probs[2]:.2%}")


# 10) ZÁVEREČNÁ ANALÝZA


print("10) ZÁVEREČNÉ ZISTENIA")


# Najdôležitejšie premenné celkovo (podľa priemernej absolútnej hodnoty)
avg_importance = pd.DataFrame({
    'Feature': all_feature_names,
    'Avg_Abs_Coef': np.mean(np.abs(coefficients), axis=0)
}).sort_values('Avg_Abs_Coef', ascending=False)

print("\nNajdôležitejšie premenné celkovo (priemerný absolútny vplyv):")
print(avg_importance.head(10).to_string(index=False))

# Špeciálne pre vzdelanie
edu_importance = avg_importance[avg_importance['Feature'] == 'Education_Ordinal']
if not edu_importance.empty:
    rank = list(avg_importance['Feature']).index('Income') + 1
    print(f"\nPrijem je {rank}. najdôležitejšia premenná z {len(all_feature_names)}")



# TEST : T-TEST - BASIC/2n CYCLE VZDĚLÁNÍ V LOW vs. OSTATNÍ

alpha = 0.05

print("TEST 2: VZDĚLÁNÍ BASIC/2n CYCLE V LOW SPOTŘEBĚ")


# Vytvorenie skupín
low_edu_basic_2n = df[(df['Consumption_Level_Name'] == 'Low') &
                      (df['Education'].isin(['Basic', '2n Cycle']))]
not_low_edu_basic_2n = df[(df['Consumption_Level_Name'] != 'Low') &
                          (df['Education'].isin(['Basic', '2n Cycle']))]

# Počty
print(f"Počet Basic/2n Cycle v LOW spotrebe: {len(low_edu_basic_2n)}")
print(f"Počet Basic/2n Cycle v NON-LOW spotrebe: {len(not_low_edu_basic_2n)}")

# Percentuálne zastúpenie
total_basic_2n = len(df[df['Education'].isin(['Basic', '2n Cycle'])])
percent_in_low = len(low_edu_basic_2n) / total_basic_2n * 100
percent_in_not_low = len(not_low_edu_basic_2n) / total_basic_2n * 100

print(f"\nZákladné vzdelanie (Basic/2n Cycle):")
print(f"  {percent_in_low:.1f}% je v LOW spotrebe")
print(f"  {percent_in_not_low:.1f}% je v NIE-LOW spotrebe")

# Chi-kvadrát test pre kategorické dáta
contingency_table = pd.crosstab(
    df['Education'].isin(['Basic', '2n Cycle']),
    df['Consumption_Level_Name'] == 'Low'
)
chi2, p_chi, dof, expected = stats.chi2_contingency(contingency_table)

print(f"\nCHI-KVADRÁT TEST:")
print(f"  Chi-štatistika: {chi2:.3f}")
print(f"  p-hodnota: {p_chi:.6f}")

# Porovnanie príjmu medzi Basic/2n Cycle v Low vs. non-Low
if len(low_edu_basic_2n) > 1 and len(not_low_edu_basic_2n) > 1:
    income_low = low_edu_basic_2n['Income']
    income_not_low = not_low_edu_basic_2n['Income']

    print(f"\nPRÍJEM BASIC/2n CYCLE:")
    print(f"  V LOW spotrebe: priemer = {income_low.mean():.0f}, N = {len(income_low)}")
    print(f"  V NON-LOW spotrebe: priemer = {income_not_low.mean():.0f}, N = {len(income_not_low)}")

    # T-test pre príjem
    t_stat_edu, p_value_edu = stats.ttest_ind(income_low, income_not_low, equal_var=False)
    print(f"\nT-TEST PRÍJEMU:")
    print(f"  t-statistika: {t_stat_edu:.3f}")
    print(f"  p-hodnota: {p_value_edu:.6f}")

    if p_value_edu < alpha:
        print(f"  ŠTATISTICKY VÝZNAMNÝ rozdiel v príjme")
    else:
        print(f"  NIE je štatisticky významný rozdiel v príjme")