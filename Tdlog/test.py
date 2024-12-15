import os
import pickle
import numpy as np

# Chemins vers les fichiers
model_path = "Tdlog/model1.pkl"
scaler_path = "Tdlog/scaler1.pkl"

# Charger le modèle
with open(model_path, "rb") as file:
    model = pickle.load(file)

# Charger le scaler
with open(scaler_path, "rb") as file:
    scaler = pickle.load(file)

# Fonction pour prédire le Burn Rate
def predict_burn_rate(designation, resource_allocation, mental_fatigue_score, gender, wfh_setup):
    # Créer une liste des nouvelles données
    new_data = [[designation, resource_allocation, mental_fatigue_score,
                 1 if gender == 'Female' else 0,  # Gender_1
                 1 if gender == 'Male' else 0,    # Gender_0
                 1 if wfh_setup == 'Yes' else 0,  # WFH Setup Available_1
                 1 if wfh_setup == 'No' else 0]]  # WFH Setup Available_0

    # Convertir en array et appliquer la transformation
    new_data_scaled = scaler.transform(np.array(new_data))

    # Prédiction du Burn Rate
    return model.predict(new_data_scaled)[0]

# Exemple d'utilisation
designation = 2.0
resource_allocation = 4.0
mental_fatigue_score = 3.6
gender = 'Female'  # 'Male' ou 'Female'
wfh_setup = 'Yes'  # 'Yes' ou 'No'

# Afficher uniquement le Burn Rate prédit
burn_rate = predict_burn_rate(designation, resource_allocation, mental_fatigue_score, gender, wfh_setup)
print(burn_rate)
