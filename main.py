import numpy as np
import pandas as pd
import matplotlib
from sklearn.neighbors import KNeighborsClassifier

# matplotlib.use('Agg')  # Ou essayez 'TkAgg'
matplotlib.use('Qt5Agg')  # Ou essayez 'TkAgg'
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from pandas.plotting import scatter_matrix
import seaborn as sns

df = pd.read_csv('car_insurance.csv')

# Affichage de types
print(df.dtypes)

# Affichage des premiers enregistrements
print(df.head())

# Affichage de la taille des données
print(df.shape)

# Check for NaNs in each column and print the results
print("Columns with NaN values:")
for col in df.columns:
    if df[col].isnull().any():
        nan_count = df[col].isnull().sum()
        print(f"- {col}: {nan_count} NaN(s)")

        median_value = df[col].median()
        df[col] = df[col].fillna(median_value)

# Identify numeric columns
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

# Create histograms for each numeric column
df[numeric_cols].hist(figsize=(12, 10))
plt.suptitle('Histograms of Numeric Columns', y=1.02) # Add a general title
plt.tight_layout() # Adjust layout to prevent overlapping titles
plt.show()

## Normalisation des données numériques
# On peut voir des données abérrantes sur les Children et sur les speeding violations
median_col_children = df['children'].median()
median_col_speeding_violations = df['speeding_violations'].median()

df['children'] = df['children'].fillna(median_col_children)
df['speeding_violations'] = df['speeding_violations'].fillna(median_col_speeding_violations)

df[numeric_cols].hist(figsize=(12, 10))
plt.suptitle('Histograms of Numeric Columns', y=1.02) # Add a general title
plt.tight_layout() # Adjust layout to prevent overlapping titles
plt.show()

# Identifier les colonnes de type objet (qui contiennent généralement des chaînes)
string_cols = df.select_dtypes(include='object').columns

## Normalisation des chaînes

# Itérer sur chaque colonne de type chaîne
for col in string_cols:
    # Vérifier si la colonne contient la valeur 'none'
    if 'none' in df[col].unique():
        # Calculer la valeur la plus fréquente dans la colonne
        most_frequent = df[col].mode()[0]  # .mode() retourne une Series, on prend le premier élément

        # Remplacer toutes les occurrences de 'none' par la valeur la plus fréquente
        df[col] = df[col].replace('none', most_frequent)

# Identifier les colonnes qualitatives (de type 'object')
qualitative_cols = df.select_dtypes(include='object').columns

# Initialiser le LabelEncoder
label_encoder = LabelEncoder()

# Itérer sur chaque colonne qualitative et appliquer l'encodage
for col in qualitative_cols:
    # Vérifier si la colonne contient des valeurs non numériques (pour éviter les erreurs)
    if df[col].dtype == 'object':
        # Appliquer l'encodage LabelEncoder à la colonne
        df[col] = label_encoder.fit_transform(df[col])

        # (Optionnel) Afficher les classes encodées pour comprendre la transformation
        print(f"Classes encodées pour la colonne '{col}':")
        print(label_encoder.classes_)
        print("-" * 30)

# Afficher les premières lignes du DataFrame modifié pour vérifier
print("\nDataFrame après application de LabelEncoder sur les colonnes qualitatives:")
print(df.head())

# Afficher les types de données pour confirmer la conversion en numérique
print("\nTypes de données après encodage:")
print(df.dtypes)

def detect_outliers_percentile_indices(series, lower_percentile=0.01, upper_percentile=0.99):
    lower_threshold = series.quantile(lower_percentile)
    upper_threshold = series.quantile(upper_percentile)
    outlier_indices = series[(series < lower_threshold) | (series > upper_threshold)].index
    return outlier_indices

# Identifier les indices des outliers dans 'children'
outliers_children_indices = detect_outliers_percentile_indices(df['children'])
print("Indices des outliers (Z-score) dans 'children':")
print(outliers_children_indices)

# Identifier les indices des outliers dans 'speeding_violations'
outliers_speeding_indices = detect_outliers_percentile_indices(df['speeding_violations'])
print("\nIndices des outliers (Z-score) dans 'speeding_violations':")
print(outliers_speeding_indices)

# Calculer la médiane des colonnes 'children' et 'speeding_violations'
median_children = df['children'].median()
median_speeding = df['speeding_violations'].median()

# Remplacer les outliers par la médiane dans chaque colonne respectivement
for index in outliers_children_indices:
    df.loc[index, 'children'] = median_children

for index in outliers_speeding_indices:
    df.loc[index, 'speeding_violations'] = median_speeding

# Afficher les histogrammes du DataFrame APRÈS le remplacement des outliers
df.hist(figsize=(12, 10))
plt.suptitle('Histograms of Columns After Outlier Replacement', y=1.02)
plt.tight_layout()
plt.show()





print("\n--- Analyse de Corrélation ---")

# Calculer la matrice de corrélation pour toutes les variables numériques
correlation_matrix = df.corr(numeric_only=True)
print("\nMatrice de Corrélation :\n", correlation_matrix)

# Corrélation d'une variable particulière avec toutes les autres (par exemple, 'past_accidents')
past_accidents_correlation = correlation_matrix['past_accidents'].sort_values(ascending=False)
print("\nCorrélation de 'past_accidents' avec les autres variables :\n", past_accidents_correlation)

# Identification des variables d'entrée les plus corrélées entre elles
# On évite la diagonale et on ne regarde qu'une moitié de la matrice pour éviter les doublons
upper_triangle = np.triu(correlation_matrix, k=1)
highly_correlated_pairs = []
threshold_correlation = 0.5  # Seuil de corrélation à considérer comme élevé

for i in range(len(correlation_matrix.columns)):
    for j in range(i + 1, len(correlation_matrix.columns)):
        if abs(upper_triangle[i, j]) > threshold_correlation:
            highly_correlated_pairs.append((correlation_matrix.columns[i], correlation_matrix.columns[j], upper_triangle[i, j]))

print(f"\nPaires de variables d'entrée fortement corrélées (>{threshold_correlation}):\n", highly_correlated_pairs)

# Affichage de la matrice de corrélation sous forme de heatmap
plt.figure(figsize=(12, 10))
plt.title("Matrice de corrélation")
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.tight_layout()
plt.show()



from sklearn.model_selection import train_test_split, KFold, cross_val_score

# Identifier la variable cible (y) et les variables d'entrée (X)
# Nous supposons que 'outcome' est la variable cible.
# Si votre dataset utilise un autre nom pour la variable cible (ex: 'is_claim', 'fraud_reported'),
# veuillez ajuster la ligne suivante en conséquence.
# Si l'objectif est de prédire 'past_accidents', utilisez: y = df['past_accidents']
try:
    y = df['outcome']
    X = df.drop('outcome', axis=1)
except KeyError:
    print("La colonne 'outcome' n'a pas été trouvée. Utilisation de 'past_accidents' comme cible par défaut.")
    y = df['past_accidents']
    X = df.drop('past_accidents', axis=1)


# Convertir les DataFrames en tableaux NumPy si nécessaire (train_test_split accepte aussi les DataFrames)
X_np = X.to_numpy()
y_np = y.to_numpy()

# Diviser les données en jeux d'apprentissage et de test
# test_size=0.25 signifie 25% des données pour le test, 75% pour l'apprentissage.
# random_state=42 assure la reproductibilité des résultats.
X_train, X_test, y_train, y_test = train_test_split(X_np, y_np, test_size=0.25, random_state=42)

# Calculer les proportions
total_samples = len(df)
train_samples = len(X_train)
test_samples = len(X_test)

train_proportion = train_samples / total_samples
test_proportion = test_samples / total_samples

print(f"\n--- Extraction des jeux d'apprentissage et de test ---")
print(f"Taille totale du jeu de données : {total_samples} échantillons")
print(f"Taille du jeu d'apprentissage (X_train, y_train) : {train_samples} échantillons")
print(f"Taille du jeu de test (X_test, y_test) : {test_samples} échantillons")
print(f"Proportion du jeu d'apprentissage : {train_proportion:.2f} ({train_proportion*100:.0f}%)")
print(f"Proportion du jeu de test : {test_proportion:.2f} ({test_proportion*100:.0f}%)")

print(f"\nForme de X_train: {X_train.shape}")
print(f"Forme de y_train: {y_train.shape}")
print(f"Forme de X_test: {X_test.shape}")
print(f"Forme de y_test: {y_test.shape}")

from sklearn.linear_model import LogisticRegression, Perceptron

# Instancier le modèle de Régression Logistique
# On peut ajouter des paramètres comme solver='liblinear' ou max_iter=1000
# pour un meilleur contrôle, mais les valeurs par défaut sont souvent un bon point de départ.
model_logistic_regression = LogisticRegression(random_state=42, max_iter=10000)

# Entraîner le modèle sur le jeu d'apprentissage
print("\n--- Entraînement du modèle de Régression Logistique ---")
model_logistic_regression.fit(X_train, y_train)

print("Modèle de Régression Logistique entraîné avec succès.")

# Afficher les coefficients appris (optionnel)
# print("\nCoefficients du modèle :", model_logistic_regression.coef_)
# print("Interception (bias) du modèle :", model_logistic_regression.intercept_)

# Effectuer des prédictions sur les données d'entrée du jeu de test
y_pred = model_logistic_regression.predict(X_test)

print("\n--- Prédictions sur le jeu de test ---")
print("Quelques exemples de comparaison (Prédit vs Réel) :")
for i in range(10): # Afficher les 10 premiers exemples
    print(f"Échantillon {i+1}: Prédit = {y_pred[i]}, Réel = {y_test[i]}")

from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

print("\n--- Évaluation quantitative du modèle ---")

# 1. Accuracy Score
accuracy = accuracy_score(y_test, y_pred)
print(f"\nPrécision (Accuracy Score) : {accuracy:.4f}")

# 2. Matrice de Confusion
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nMatrice de Confusion :")
print(conf_matrix)

# 3. Précision (Precision Score)
# Pour la classification binaire, 'pos_label' est souvent 1 (la classe "positive")
precision = precision_score(y_test, y_pred, pos_label=1, average='binary') # ou average=None si on veut par classe
print(f"\nPrécision (Precision Score) : {precision:.4f}")

# 4. Rappel (Recall Score)
recall = recall_score(y_test, y_pred, pos_label=1, average='binary')
print(f"\nRappel (Recall Score) : {recall:.4f}")

# 5. F1 Score
f1 = f1_score(y_test, y_pred, pos_label=1, average='binary')
print(f"\nScore F1 : {f1:.4f}")




print("\n--- Amélioration de l'évaluation par Validation Croisée (K-Fold) ---")

# 2. Définir la stratégie de validation croisée
# KFold permet de découper les données en K plis (folds).
# n_splits = 5 est un choix courant.
# shuffle=True assure que les données sont mélangées avant de créer les plis.
# random_state=42 assure la reproductibilité du découpage.
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 3. Réaliser la validation croisée et obtenir les scores
# cross_val_score retourne un tableau des scores pour chaque pli.
# Nous allons utiliser 'accuracy' comme métrique pour une comparaison directe avec le résultat précédent.
model_logistic_regression_cv = LogisticRegression(random_state=42, max_iter=10000)

cv_scores = cross_val_score(model_logistic_regression_cv, X, y, cv=kf, scoring='accuracy')

print(f"\nScores d'exactitude pour chaque pli de la validation croisée (5 plis) :")
print(cv_scores)

# 4. Analyser les résultats de la validation croisée
mean_cv_accuracy = np.mean(cv_scores)
std_cv_accuracy = np.std(cv_scores)

print(f"\nExactitude moyenne de la validation croisée : {mean_cv_accuracy:.4f}")
print(f"Écart-type de l'exactitude de la validation croisée : {std_cv_accuracy:.4f}")

# Comparaison avec le résultat précédent (obtenu avec un seul split)
previous_accuracy = 0.8136
print(f"\n--- Comparaison ---")
print(f"Exactitude obtenue avec un seul split d'apprentissage/test : {previous_accuracy:.4f}")
print(f"Exactitude moyenne obtenue avec la validation croisée : {mean_cv_accuracy:.4f}")

# Discussion des résultats
print("\n--- Analyse et Discussion ---")
if mean_cv_accuracy < previous_accuracy:
    print(f"L'exactitude moyenne en validation croisée ({mean_cv_accuracy:.4f}) est légèrement inférieure ou similaire à celle obtenue sur le seul jeu de test ({previous_accuracy:.4f}).")
    print("Cela pourrait indiquer que l'exactitude du split unique était légèrement optimiste ou que la variation est normale entre les splits.")
else:
    print(f"L'exactitude moyenne en validation croisée ({mean_cv_accuracy:.4f}) est légèrement supérieure ou similaire à celle obtenue sur le seul jeu de test ({previous_accuracy:.4f}).")
    print("Cela suggère que l'estimation précédente était cohérente avec la performance moyenne sur différents sous-ensembles de données.")

print(f"L'écart-type de {std_cv_accuracy:.4f} indique la variabilité des performances du modèle sur les différents plis.")
print("Un écart-type faible signifie que le modèle est relativement stable et performe de manière similaire sur différents sous-ensembles de données.")
print("La validation croisée donne une estimation plus robuste et fiable de la performance généralisée du modèle, car elle teste le modèle sur plusieurs combinaisons d'apprentissage et de test.")

# Note on ConvergenceWarning: The same ConvergenceWarning may appear during cross-validation
# if the underlying LogisticRegression model still struggles to converge within max_iter=100 for some folds.
# Scaling the data (e.g., using StandardScaler) before training is generally recommended for Logistic Regression
# and can help mitigate this warning and potentially improve performance.



print("\n--- Comparaison avec d'autres algorithmes de classification ---")

# Define the cross-validation strategy (same as before)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Define the classifiers to compare with some initial hyperparameters
classifiers = {
    "Régression Logistique": LogisticRegression(random_state=42, max_iter=10000), # Increased max_iter to avoid ConvergenceWarning
    "Perceptron": Perceptron(random_state=42, max_iter=10000, tol=1e-3), # Common parameters for Perceptron
    "K-Plus Proches Voisins (K=5)": KNeighborsClassifier(n_neighbors=5) # K=5 is a common starting point
}

results = {}

for name, model in classifiers.items():
    print(f"\n--- Évaluation du modèle : {name} ---")
    try:
        # Perform cross-validation
        cv_scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')

        # Analyze results
        mean_accuracy = np.mean(cv_scores)
        std_accuracy = np.std(cv_scores)

        results[name] = {"mean_accuracy": mean_accuracy, "std_accuracy": std_accuracy, "scores": cv_scores}

        print(f"Scores d'exactitude pour chaque pli : {cv_scores}")
        print(f"Exactitude moyenne : {mean_accuracy:.4f}")
        print(f"Écart-type de l'exactitude : {std_accuracy:.4f}")

    except Exception as e:
        print(f"Une erreur est survenue lors de l'évaluation du modèle {name}: {e}")
        print("Veuillez vérifier les hyperparamètres ou la compatibilité des données.")

print("\n--- Synthèse des résultats de comparaison ---")
for name, res in results.items():
    print(f"- {name}: Exactitude moyenne = {res['mean_accuracy']:.4f} (Écart-type = {res['std_accuracy']:.4f})")

# Identify the best performing model
if results:
    best_model_name = max(results, key=lambda k: results[k]['mean_accuracy'])
    print(f"\nLe meilleur modèle basé sur l'exactitude moyenne en validation croisée est : {best_model_name}")
    print(f"Avec une exactitude moyenne de : {results[best_model_name]['mean_accuracy']:.4f}")

import pickle

print("\n--- Sauvegarde et Chargement du Modèle Entraîné ---")

model_filename = 'logistic_regression_model.pkl'
try:
    with open(model_filename, 'wb') as file:
        pickle.dump(model_logistic_regression_cv, file)
    print(f"\nModèle de Régression Logistique sauvegardé sous : {model_filename}")
except Exception as e:
    print(f"Erreur lors de la sauvegarde du modèle : {e}")

# 3. Charger le modèle sauvegardé
loaded_model = None
try:
    with open(model_filename, 'rb') as file:
        loaded_model = pickle.load(file)
    print(f"Modèle chargé avec succès depuis : {model_filename}")
except FileNotFoundError:
    print(f"Erreur : Le fichier {model_filename} n'a pas été trouvé.")
except Exception as e:
    print(f"Erreur lors du chargement du modèle : {e}")

# 4. Vérifier le modèle chargé en effectuant une prédiction (si le chargement a réussi)
if loaded_model is not None:
    print("\nTest du modèle chargé sur un échantillon du jeu de test:")
    # Prendre un échantillon du jeu de test pour la prédiction
    # Assurez-vous que X_test est bien disponible et qu'il a la bonne forme
    if X_test.shape[0] > 0:
        sample_index = 0 # Prendre le premier échantillon
        sample_data = X_test[sample_index].reshape(1, -1) # Reshape pour une seule prédiction

        predicted_outcome = loaded_model.predict(sample_data)
        actual_outcome = y_test[sample_index]

        print(f"Données de l'échantillon (première ligne de X_test) : {sample_data}")
        print(f"Prédiction du modèle chargé : {predicted_outcome[0]}")
        print(f"Valeur réelle : {actual_outcome}")

        if predicted_outcome[0] == actual_outcome:
            print("La prédiction du modèle chargé correspond à la valeur réelle pour cet échantillon.")
        else:
            print("La prédiction du modèle chargé ne correspond PAS à la valeur réelle pour cet échantillon.")
    else:
        print("Le jeu de test (X_test) est vide, impossible de tester le modèle chargé.")
else:
    print("Le modèle n'a pas pu être chargé, le test ne peut pas être effectué.")