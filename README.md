



L'objectif de ce projet est d'utiliser des variables cartographiques pour classer les catégories de forêts. Vous devrez analyser les données, créer des entités et entraîner un modèle d'apprentissage automatique sur les données cartographiques pour les rendre aussi précises que possible.


allez sur https://github.com/mariechantale

Données
Les fichiers d'entrée sont train.csv, test.csvet covtype.info:

train.csv
test.csv
covtype.info
L'ensemble de données de train est utilisé pour analyser les données et calibrer les modèles . L'objectif est d'obtenir la précision la plus élevée possible sur l'ensemble de test. L'ensemble de test sera disponible à la fin de la dernière journée pour éviter le surapprentissage de l'ensemble de test.

Les données sont décrites dans covtype.info.

Structure
La structure du projet est la suivante :

project
│   README.md
│   environment.yml
│
└───data
│   │   train.csv
│   |   test.csv (not available first day)
|   |   covtype.info
│
└───notebook
│   │   EDA.ipynb
|
|───scripts
|   │   preprocessing_feature_engineering.py
|   │   model_selection.py
│   |   predict.py
│
└───results
    │   plots
    │   test_predictions.csv
    │   best_model.pkl

1. EDA et ingénierie des fonctionnalités
Créez un bloc-notes Jupyter pour analyser les ensembles de données et effectuer une analyse exploratoire des données (EDA). Ce bloc-notes ne sera pas évalué.

Astuce : Exemples de fonctionnalités intéressantes

Distance to hydrology = sqrt((Horizontal_Distance_To_Hydrology)^2 + (Vertical_Distance_To_Hydrology)^2)
Horizontal_Distance_To_Fire_Points - Horizontal_Distance_To_Roadways
2. Sélection du modèle
L'approche de sélection du modèle est une étape clé car elle doit renvoyer le meilleur modèle et garantir que les résultats sont reproductibles sur l'ensemble de test final. L'objectif de cette étape est de s'assurer que les résultats sur l'ensemble de test ne sont pas dus à un surajustement de l'ensemble de test. Cela implique de diviser l'ensemble de données comme indiqué ci-dessous :

DATA
└───TRAIN FILE (0)
│   └───── Train (1)
│   |           Fold0:
|   |                  Train
|   |                  Validation
|   |           Fold1:
|   |                   Train
|   |                   Validation
... ...         ...
|   |
|   └───── Test (1)
│
└─── TEST FILE (0) (available last day)

Règles:

Essai de train divisé
Validation croisée : au moins 5 fois
Recherche par grille sur au moins 5 modèles différents :
Gradient Boosting, KNN, Random Forest, SVM, régression logistique. N'oubliez pas que pour certains modèles, la mise à l'échelle des données est importante et pour d'autres, elle n'a pas d'importance.
Score de précision du train < 0,98 . Ensemble de train (0). Écrivez le résultat dans le champREADME.md
Précision du test (dernier jour) > 0,65 . Ensemble de test (0). Écrivez le résultat dans le champREADME.md
Affichez la matrice de confusion pour le meilleur modèle dans un DataFrame. Précisez les noms d'index et de colonnes (True label et Predicted label)
Tracez la courbe d'apprentissage pour le meilleur modèle
Enregistrer le modèle formé sous forme de fichier pickle
Conseil : Comme la recherche dans la grille prend du temps, je vous suggère de préparer et de tester le code. Une fois que vous êtes sûr qu'il fonctionne, exécutez la recherche dans la grille la nuit et analysez les résultats

Astuce : La matrice de confusion montre les erreurs de classification classe par classe. Essayez de détecter si le modèle classe mal une classe par rapport à une autre. Ensuite, faites quelques recherches sur Internet sur les deux types de couvert forestier, trouvez les différences et créez de nouvelles caractéristiques qui soulignent ces différences. Plus généralement, la méthodologie d'apprentissage d'un modèle est un cycle avec plusieurs itérations. Plus de détails ici

3. Prédire (dernier jour)
Une fois que vous avez sélectionné le meilleur modèle et que vous êtes sûr qu'il fonctionnera bien sur de nouvelles données, vous êtes prêt à prédire sur l'ensemble de test :

Charger le modèle formé
Prédire sur l'ensemble de test et calculer la précision
Enregistrer les prédictions dans un fichier csv
Ajoutez votre score sur le README.md

