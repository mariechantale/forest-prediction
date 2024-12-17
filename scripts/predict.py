
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score

# Charger le modèle sauvegardé
with open('results/best_model.pkl', 'rb') as file:
    best_model = pickle.load(file)

print("Modèle chargé avec succès.")

# Charger les données de test
test_data = pd.read_csv('data/test.csv')

# Assurez-vous que les colonnes d'entrée correspondent à celles utilisées pendant l'entraînement
# Remplacer les colonnes par celles utilisées pendant l'entraînement
features = [
    'Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology', 
    'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
    'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points'
] + [f'Wilderness_Area_{i}' for i in range(4)] + [f'Soil_Type_{i}' for i in range(40)]

# Séparer les features du test set
X_test = test_data[features]

# Effectuer les prédictions
predictions = best_model.predict(X_test)

# Ajouter les prédictions à un DataFrame
test_data['Predicted_Cover_Type'] = predictions

# Sauvegarder les prédictions dans un fichier CSV
output_path = 'results/test_predictions.csv'
test_data[['Id', 'Predicted_Cover_Type']].to_csv(output_path, index=False)

print(f"Les prédictions ont été sauvegardées dans {output_path}.")
