import pandas as pd
import numpy as np

# Charger les données d'entraînement
train_data = pd.read_csv("data/train.csv")

# Exemple : Calculer une nouvelle feature
train_data["Distance_To_Hydrology"] = np.sqrt(
    train_data["Horizontal_Distance_To_Hydrology"] ** 2 +
    train_data["Vertical_Distance_To_Hydrology"] ** 2
)

# Sauvegarder les données prétraitées
train_data.to_csv("data/train_preprocessed.csv", index=False)

print("Prétraitement terminé. Fichier sauvegardé dans data/train_preprocessed.csv")
