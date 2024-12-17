import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle
import time


# Charger les données prétraitées
data = pd.read_csv("data/train_preprocessed.csv")

# Séparer les features et la target
X = data.drop(columns=["Cover_Type"])  # Remplacez "Cover_Type" par le nom correct de la colonne cible
y = data["Cover_Type"]

# Diviser les données en train et validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Créez un scaler
scaler = StandardScaler()

# Appliquez-le aux données d'entraînement et de test
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Définir les modèles à tester
models = {
    "RandomForest": RandomForestClassifier(),
    "GradientBoosting": GradientBoostingClassifier(),
    "KNN": KNeighborsClassifier(),
    "LogisticRegression": LogisticRegression(solver='liblinear', max_iter=1000),
    "SVM": SVC()
}

# Paramètres pour la recherche de grille (Grid Search)
param_grids = {
    "RandomForest": {"n_estimators": [50, 100, 200], "max_depth": [10, 20, None]},
    "GradientBoosting": {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 0.2]},
    "KNN": {"n_neighbors": [3, 5, 7]},
    "LogisticRegression": {"C": [0.01, 0.1, 1, 10]},
    "SVM": {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]}
}

best_model = None
best_score = 0

# Effectuer une recherche de grille pour chaque modèle
for model_name, model in models.items():
    print(f"Recherche de grille pour {model_name}...")
    grid_search = GridSearchCV(model, param_grids[model_name], cv=5, scoring="accuracy", n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    print(f"Meilleur score pour {model_name}: {grid_search.best_score_}")
    if grid_search.best_score_ > best_score:
        best_model = grid_search.best_estimator_
        best_score = grid_search.best_score_

# Évaluer le meilleur modèle sur les données de validation
y_pred = model.predict(X_val_scaled)
val_accuracy = accuracy_score(y_val, y_pred)
print(f"Accuracy sur le set de validation : {val_accuracy}")

# Afficher la matrice de confusion
conf_matrix = confusion_matrix(y_val, y_pred)
print("Matrice de confusion :\n", conf_matrix)

# Supposons que vous ayez déjà fait une recherche de grille sur différents modèles
grid_search_rf = GridSearchCV(estimator=RandomForestClassifier(), param_grid=params_rf, cv=5)
grid_search_rf.fit(X_train, y_train)

grid_search_gb = GridSearchCV(estimator=GradientBoostingClassifier(), param_grid=params_gb, cv=5)
grid_search_gb.fit(X_train, y_train)

start_time = time.time()

# Fit le grid search
grid_search_svm.fit(X_train, y_train)

end_time = time.time()
print(f"GridSearch terminé en {end_time - start_time} secondes")

# Après avoir effectué la recherche pour chaque modèle, vous comparez les meilleurs scores
best_rf_score = grid_search_rf.best_score_
best_gb_score = grid_search_gb.best_score_

# Vous pouvez choisir le meilleur modèle ici, par exemple RandomForest si il a le meilleur score
best_model = grid_search_rf.best_estimator_ if best__rf_score > best_gb_score else grid_search_gb.best_estimator_

# Sauvegarder le meilleur modèle
with open("results/best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

print("Le meilleur modèle a été sauvegardé dans results/best_model.pkl")

# Charger le modèle sauvegardé
with open("results/best_model.pkl", "rb") as f:
    best_model = pickle.load(f)

# Charger les données de test (à fournir le dernier jour)
test_data = pd.read_csv("data/test.csv")

# Supposons que les features sont dans "X_test"
X_test = test_data.drop("Id", axis=1)  # Retirer des colonnes inutiles comme "Id"

# Faire des prédictions sur les données de test
test_predictions = best_model.predict(X_test)

# Sauvegarder les prédictions dans un fichier CSV
output = pd.DataFrame({"Id": test_data["Id"], "Cover_Type": test_predictions})
output.to_csv("results/test_predictions.csv", index=False)

print("Prédictions sauvegardées dans 'results/test_predictions.csv'")
