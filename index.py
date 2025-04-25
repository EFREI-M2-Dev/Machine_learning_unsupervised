import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import numpy as np

df = pd.read_csv('Hostel.csv')

df['Distance'] = df['Distance'].str.replace('km from city centre', '').str.strip()
df['Distance'] = df['Distance'].str.replace(',', '.').astype(float)

features = [
    'price.from', 'summary.score', 'atmosphere', 'cleanliness',
    'facilities', 'location.y', 'security', 'staff',
    'valueformoney', 'Distance'
]

X = df[features].dropna()
df_clean = df.loc[X.index]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

iso_forest = IsolationForest(contamination=0.05, random_state=42)
outliers = iso_forest.fit_predict(X_scaled)

df_clean['is_outlier'] = outliers

outliers_df = df_clean[df_clean['is_outlier'] == -1]

print("\nAuberges atypiques détectées :")
print(outliers_df[['hostel.name', 'price.from', 'summary.score', 'valueformoney', 'Distance']])

if 'hostel.name' in df.columns:
    for index, row in outliers_df.iterrows():
        print(f"\nAuberge : {row['hostel.name']}")
        print(f"  Prix: {row['price.from']}€")
        print(f"  Score résumé: {row['summary.score']}")
        print(f"  Rapport qualité/prix: {row['valueformoney']}")
        print(f"  Distance du centre-ville: {row['Distance']} km")
else:
    print("\nColonne 'hostel.name' non trouvée dans le dataset. Assurez-vous d'avoir une colonne de nom d'hôtel.")

def predict_outlier():
    print("\nEntrez les caractéristiques d'un nouvel hôtel pour prédiction (valeurs numériques) :")

    new_hotel = {}
    for feature in features:
        value = float(input(f"{feature}: "))
        new_hotel[feature] = value

    new_hotel_df = pd.DataFrame([new_hotel], columns=features)

    new_hotel_scaled = scaler.transform(new_hotel_df)

    prediction = iso_forest.predict(new_hotel_scaled)

    if prediction == -1:
        print("\nCet hôtel est atypique (anomalie).")
        
        means = np.mean(X_scaled, axis=0)
        differences = new_hotel_scaled - means
        
        print("\nRaison de l'anomalie (écart par rapport à la moyenne des hôtels) :")
        for i, feature in enumerate(features):
            print(f"{feature} : {differences[0][i]:.2f}")
    else:
        print("\nCet hôtel n'est pas atypique.")

predict_outlier()
