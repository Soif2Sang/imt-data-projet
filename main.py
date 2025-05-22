import numpy as np
import pandas as pd
import matplotlib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import seaborn as sns
import pickle

# Try 'Qt5Agg' first, fall back to 'Agg' if needed (for MacOS)
try:
    matplotlib.use('Qt5Agg')
except ImportError:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("--- Chargement et Inspection Initiales des Données ---")
# --- 3. Importation des données ---
df = pd.read_csv('car_insurance.csv')

# Affichage de types
print("\nTypes de données:")
print(df.dtypes)

# Affichage des premiers enregistrements
print("\nPremiers enregistrements:")
print(df.head())

# --- 4. Examen des données

# Affichage de la taille des données
print("\nTaille du jeu de données (lignes, colonnes):")
print(df.shape)

# Affichage des valeurs manquantes (NaNs) ---
print("\n--- Affichage des Valeurs Manquantes (NaNs) ---")
print("Colonnes avec des valeurs NaN:")
for col in df.columns:
    if df[col].isnull().any():
        nan_count = df[col].isnull().sum()
        print(f"- {col}: {nan_count} NaN(s)")

## Visualisation des distributions (Histograms) ---

print("\n--- Visualisation des Distributions Numériques (Histograms) ---")
# Identifier les colonnes numériques après imputation

# Create histograms for each column
df.hist(figsize=(12, 10))
plt.suptitle('Histograms for each column', y=1.02)
plt.tight_layout()
plt.show()

# --- 5. Préparation des données ---


for col in df.columns:
    if df[col].isnull().any():
        # Remplacer les NaN par la médiane pour les colonnes numériques
        if df[col].dtype in ['int64', 'float64']:
             median_value = df[col].median()
             df[col] = df[col].fillna(median_value)
             print(f"  NaNs dans '{col}' remplacés par la médiane ({median_value:.2f}).")

# Vérifier s'il reste des NaNs après imputation
print("\nVérification des NaNs après imputation:")
if df.isnull().sum().sum() == 0:
    print("Aucune valeur NaN restante dans le jeu de données.")
else:
    print("Des valeurs NaN subsistent dans certaines colonnes:")
    print(df.isnull().sum()[df.isnull().sum() > 0])

print("\n--- Normalisation des Chaînes et Encodage Qualitatif ---")
# Identifier les colonnes de type objet (qui contiennent généralement des chaînes)
string_cols = df.select_dtypes(include='object').columns

print("\n--- Gestion des Valeurs None ---")
# Itérer sur chaque colonne de type chaîne
for col in string_cols:
    # Vérifier si la colonne contient la valeur 'none' et la remplacer par le mode
    if 'none' in df[col].unique():
        most_frequent = df[col].mode()[0]
        df[col] = df[col].replace('none', most_frequent)
        print(f"Valeurs 'none' dans '{col}' remplacées par le mode : '{most_frequent}'.")

# Identifier les colonnes qualitatives (de type 'object') après la normalisation des chaînes
qualitative_cols = df.select_dtypes(include='object').columns

# Initialiser le LabelEncoder
label_encoder = LabelEncoder()

print("\n--- Encodage des Variables Qualitatives ---")
# Itérer sur chaque colonne qualitative et appliquer l'encodage
for col in qualitative_cols:
    # Appliquer l'encodage LabelEncoder à la colonne
    df[col] = label_encoder.fit_transform(df[col])

    # Afficher les classes encodées pour comprendre la transformation
    print(f"\nClasses encodées pour la colonne '{col}':")
    print(label_encoder.classes_)
    print("-" * 30)

# Afficher les premières lignes du DataFrame modifié pour vérifier
print("\nDataFrame après application de LabelEncoder sur les colonnes qualitatives:")
print(df.head())

# Afficher les types de données pour confirmer la conversion en numérique
print("\nTypes de données après encodage:")
print(df.dtypes)

# Afficher les colonnes numériques après encodage
print("\nColonnes numériques après encodage:")
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

# Create histograms for each numeric column
df[numeric_cols].hist(figsize=(12, 10))
plt.suptitle('Histograms of Numeric Columns (After LabelEncoder)', y=1.02) # Add a general title
plt.tight_layout() # Adjust layout to prevent overlapping titles
plt.show()

print("\n--- Détection et Traitement des Outliers ---")

def detect_outliers_percentile_indices(series, lower_percentile=0.01, upper_percentile=0.99):
    """Détecte les indices des outliers en utilisant la méthode des percentiles."""
    lower_threshold = series.quantile(lower_percentile)
    upper_threshold = series.quantile(upper_percentile)
    outlier_indices = series[(series < lower_threshold) | (series > upper_threshold)].index
    return outlier_indices

# Identifier les indices des outliers dans 'children' et 'speeding_violations'
# Note: Ces colonnes ont déjà été traitées pour les NaNs, mais on peut encore avoir des outliers extrêmes.
outliers_children_indices = detect_outliers_percentile_indices(df['children'])
print("\nIndices des outliers (percentile) dans 'children':")
print(outliers_children_indices)

outliers_speeding_indices = detect_outliers_percentile_indices(df['speeding_violations'])
print("\nIndices des outliers (percentile) dans 'speeding_violations':")
print(outliers_speeding_indices)

# Calculer la médiane des colonnes 'children' et 'speeding_violations' (après imputation des NaNs)
median_children = df['children'].median()
median_speeding = df['speeding_violations'].median()

# Remplacer les outliers par la médiane dans chaque colonne respectivement
for index in outliers_children_indices:
    df.loc[index, 'children'] = median_children
    print(f"Outlier à l'index {index} dans 'children' remplacé par la médiane ({median_children}).")

for index in outliers_speeding_indices:
    df.loc[index, 'speeding_violations'] = median_speeding
    print(f"Outlier à l'index {index} dans 'speeding_violations' remplacé par la médiane ({median_speeding}).")

# Afficher les histogrammes du DataFrame APRÈS le remplacement des outliers
print("\n--- Visualisation des Distributions Numériques (Après Traitement des Outliers) ---")
df[numeric_cols].hist(figsize=(12, 10))
plt.suptitle('Histograms of Numeric Columns After Outlier Replacement', y=1.02)
plt.tight_layout()
plt.show()

# Normalisation des Données (StandardScaler) ---
print("\n--- Normalisation des Données (StandardScaler) ---")

# IMPORTANT: Define your target column and any ID columns that should NOT be scaled.
# Common names - adjust if your CSV uses different names.
target_column = 'outcome'
id_column = 'Id'

# Create a list of feature columns to be scaled
columns_to_scale = df.columns.tolist()

# Remove target column from scaling list (if it exists)
if target_column in columns_to_scale:
    columns_to_scale.remove(target_column)
    print(f"Colonne cible ('{target_column}') exclue de la normalisation.")
elif target_column in df.columns: # It exists but wasn't in columns_to_scale (e.g. if columns_to_scale was already filtered)
     print(f"Note : La colonne cible ('{target_column}') existe mais n'était pas dans la liste initiale à normaliser.")
else:
    print(f"Avertissement : La colonne cible ('{target_column}') non trouvée dans le DataFrame. "
          "Toutes les colonnes (sauf l'ID potentiel) seront normalisées. "
          "Veuillez vérifier le nom de votre colonne cible.")

# Remove ID column from scaling list (if it exists)
if id_column in columns_to_scale:
    columns_to_scale.remove(id_column)
    print(f"Colonne ID ('{id_column}') exclue de la normalisation.")
elif id_column in df.columns:
    print(f"Note : La colonne ID ('{id_column}') existe mais n'était pas dans la liste initiale à normaliser.")
else:
    print(f"Note : Colonne ID ('{id_column}') non trouvée. Aucune colonne ID à exclure spécifiquement.")


# Ensure all columns to scale are actually numeric before proceeding
final_features_to_scale = [col for col in columns_to_scale if pd.api.types.is_numeric_dtype(df[col])]
if not final_features_to_scale:
    print("Aucune colonne numérique identifiée pour la normalisation après exclusions. Vérifiez vos noms de colonnes.")
else:
    print(f"\nColonnes qui seront normalisées : {final_features_to_scale}")

    # Separate features to be scaled
    X_to_scale = df[final_features_to_scale]

    # Initialize StandardScaler
    scaler = StandardScaler()

    # Fit the scaler and transform the features (converting to NumPy array as requested)
    # .values converts the DataFrame subset to a NumPy array
    X_scaled_values = scaler.fit_transform(X_to_scale.values)

    # Convert the scaled NumPy array back to a DataFrame with original column names and index
    X_scaled_df = pd.DataFrame(X_scaled_values, columns=final_features_to_scale, index=X_to_scale.index)

    # Update the original DataFrame with the scaled feature columns
    df[final_features_to_scale] = X_scaled_df

    print("\nPremières lignes du DataFrame après normalisation des caractéristiques:")
    print(df.head())

    print("\nStatistiques descriptives des caractéristiques normalisées (vérification - moyenne proche de 0, std proche de 1):")
    print(df[final_features_to_scale].describe())

    # Visualisation après normalisation (optionnel, mais peut être utile)
    print("\n--- Visualisation des Distributions Numériques (Après Normalisation) ---")
    if final_features_to_scale: # Check if there are any scaled features to plot
        df[final_features_to_scale].hist(figsize=(12, 10))
        plt.suptitle('Histograms of Scaled Numeric Features', y=1.02)
        plt.tight_layout()
        plt.show()
    else:
        print("Aucune caractéristique n'a été normalisée pour la visualisation.")

# Afficher les types de données finaux pour s'assurer que tout est numérique comme prévu
print("\nTypes de données finaux dans df:")
print(df.dtypes)

# Le DataFrame 'df' contient maintenant les données préparées et normalisées.
# Vous pouvez maintenant procéder à la division train/test et à l'entraînement des modèles.
print("\n--- Préparation des données terminée. Prêt pour la modélisation. ---")

# Example: Separating features (X) and target (y) for model training
# This assumes 'outcome' is your target and is now numerically encoded.
if target_column in df.columns:
    X = df.drop([col for col in [target_column, id_column] if col in df.columns], axis=1)
    y = df[target_column]
    print(f"\nVariables d'entrée (X) après normalisation (premières lignes) pour la modélisation (colonne '{target_column}' et '{id_column}' exclues):")
    print(X.head())
    print(f"\nVariable cible (y) (premières lignes):")
    print(y.head())
else:
    print(f"\nLa colonne cible '{target_column}' n'a pas été trouvée. Impossible de séparer X et y automatiquement.")
    X = df.drop([col for col in [id_column] if col in df.columns], axis=1) # Drop ID if it exists
    y = None # Target is unknown
    print(f"\nVariables d'entrée (X) après normalisation (premières lignes) pour la modélisation (colonne '{id_column}' exclue, cible inconnue):")
    print(X.head())


# --- 6. Recherche de Corrélation ---
print("\n--- Analyse de Corrélation ---")

# Calculer la matrice de corrélation pour toutes les variables numériques (incluant les colonnes encodées)
correlation_matrix = df.corr(numeric_only=True)
print("\nMatrice de Corrélation :\n", correlation_matrix)

# Corrélation d'une variable particulière avec toutes les autres (par exemple, 'outcome')
outcome_correlation = correlation_matrix['outcome'].sort_values(ascending=False)
print("\nCorrélation de 'outcome' avec les autres variables :\n", outcome_correlation)

# Identification des variables d'entrée les plus corrélées entre elles
# On évite la diagonale et on ne regarde qu'une moitié de la matrice pour éviter les doublons
# upper_triangle = np.triu(correlation_matrix, k=1)
# highly_correlated_pairs = []
# threshold_correlation = 0.5  # Seuil de corrélation à considérer comme élevé
#
# # Obtenir les noms des colonnes numériques pour l'indexation
# numeric_col_names = correlation_matrix.columns.tolist()
#
# for i in range(len(numeric_col_names)):
#     for j in range(i + 1, len(numeric_col_names)):
#         if abs(upper_triangle[i, j]) > threshold_correlation:
#             highly_correlated_pairs.append((numeric_col_names[i], numeric_col_names[j], upper_triangle[i, j]))
#
# print(f"\nPaires de variables d'entrée fortement corrélées (>{threshold_correlation}):\n", highly_correlated_pairs)

# Affichage de la matrice de corrélation sous forme de heatmap
plt.figure(figsize=(14, 12)) # Increased figure size for better readability
plt.title("Matrice de corrélation")
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", annot_kws={"size": 8}) # Adjusted annot_kws for smaller font
plt.tight_layout()
plt.show()


# --- 7. Extraction des jeux d'apprentissage et de test ---
print("\n--- Extraction des jeux d'apprentissage et de test ---")

# Identifier la variable cible (y) et les variables d'entrée (X)
# Nous supposons que 'outcome' est la variable cible.
# Si votre dataset utilise un autre nom pour la variable cible (ex: 'is_claim', 'fraud_reported'),
# veuillez ajuster la ligne suivante en conséquence.
target_variable = 'outcome'
y = df[target_variable]
X = df.drop(target_variable, axis=1)

# Convertir les DataFrames en tableaux NumPy (train_test_split accepte aussi les DataFrames, mais NumPy est souvent utilisé)
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

print(f"Taille totale du jeu de données : {total_samples} échantillons")
print(f"Taille du jeu d'apprentissage (X_train, y_train) : {train_samples} échantillons")
print(f"Taille du jeu de test (X_test, y_test) : {test_samples} échantillons")
print(f"Proportion du jeu d'apprentissage : {train_proportion:.2f} ({train_proportion*100:.0f}%)")
print(f"Proportion du jeu de test : {test_proportion:.2f} ({test_proportion*100:.0f}%)")

print(f"\nForme de X_train: {X_train.shape}")
print(f"Forme de y_train: {y_train.shape}")
print(f"Forme de X_test: {X_test.shape}")
print(f"Forme de y_test: {y_test.shape}")


# --- 8. Entraînement et Évaluation du Modèle de Régression Logistique (Split unique) ---
print("\n--- Entraînement et Évaluation du Modèle de Régression Logistique (Split unique) ---")

# Instancier le modèle de Régression Logistique
# Increased max_iter to 10000 to help with convergence
model_logistic_regression = LogisticRegression(random_state=42, max_iter=100)

# Entraîner le modèle sur le jeu d'apprentissage
print("Entraînement du modèle de Régression Logistique sur le jeu d'apprentissage...")
model_logistic_regression.fit(X_train, y_train)
print("Modèle de Régression Logistique entraîné avec succès.")

# Effectuer des prédictions sur les données d'entrée du jeu de test
y_pred = model_logistic_regression.predict(X_test)

print("\n--- Prédictions sur le jeu de test (Split unique) ---")
print("Quelques exemples de comparaison (Prédit vs Réel) :")
for i in range(10): # Afficher les 10 premiers exemples
    print(f"Échantillon {i+1}: Prédit = {y_pred[i]}, Réel = {y_test[i]}")

print("\n--- Évaluation quantitative du modèle (Split unique) ---")

# 1. Accuracy Score
accuracy = accuracy_score(y_test, y_pred)
print(f"\nPrécision (Accuracy Score) : {accuracy:.4f}")

# 2. Matrice de Confusion
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nMatrice de Confusion :")
print(conf_matrix)

# 3. Précision (Precision Score)
# For binary classification, 'pos_label' is often 1 (the "positive" class)
# Use zero_division=0 to handle cases where there are no predicted positives
precision = precision_score(y_test, y_pred, pos_label=1, average='binary', zero_division=0)
print(f"\nPrécision (Precision Score) : {precision:.4f}")

# 4. Rappel (Recall Score)
# Use zero_division=0 to handle cases where there are no actual positives
recall = recall_score(y_test, y_pred, pos_label=1, average='binary', zero_division=0)
print(f"\nRappel (Recall Score) : {recall:.4f}")

# 5. F1 Score
# Use zero_division=0 to handle cases where there are no predicted or actual positives
f1 = f1_score(y_test, y_pred, pos_label=1, average='binary', zero_division=0)
print(f"\nScore F1 : {f1:.4f}")


# --- 10. Amélioration de l'évaluation par Validation Croisée (K-Fold) ---
print("\n--- Amélioration de l'évaluation par Validation Croisée (K-Fold) ---")

# Define the cross-validation strategy
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Use the same Logistic Regression model instance configuration
model_logistic_regression_cv = LogisticRegression(random_state=42, max_iter=100)

# Perform cross-validation and get the scores
# We will use 'accuracy' as the metric for comparison
cv_scores = cross_val_score(model_logistic_regression_cv, X, y, cv=kf, scoring='accuracy')

print(f"\nScores d'exactitude pour chaque pli de la validation croisée (5 plis) :")
print(cv_scores)

# Analyze cross-validation results
mean_cv_accuracy = np.mean(cv_scores)
std_cv_accuracy = np.std(cv_scores)

print(f"\nExactitude moyenne de la validation croisée : {mean_cv_accuracy:.4f}")
print(f"Écart-type de l'exactitude de la validation croisée : {std_cv_accuracy:.4f}")

# --- 11. Comparaison avec d'autres algorithmes ---
print("\n--- Comparaison avec d'autres algorithmes de classification ---")

# Define the classifiers to compare with some initial hyperparameters
classifiers = {
    "Régression Logistique": LogisticRegression(random_state=42, max_iter=100),
    "Perceptron": Perceptron(random_state=42, max_iter=100, tol=1e-3),
    "K-Plus Proches Voisins (K=15)": KNeighborsClassifier(n_neighbors=15)
}

results = {}

for name, model in classifiers.items():
    print(f"\n--- Évaluation du modèle : {name} ---")
    try:
        # Perform cross-validation
        # Use X and y (the full dataset) for cross-validation
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

# Identify the best performing model based on mean cross-validation accuracy
if results:
    best_model_name = max(results, key=lambda k: results[k]['mean_accuracy'])
    print(f"\nLe meilleur modèle basé sur l'exactitude moyenne en validation croisée est : {best_model_name}")
    print(f"Avec une exactitude moyenne de : {results[best_model_name]['mean_accuracy']:.4f}")

    # Get the actual best model instance from the classifiers dictionary
    best_model = classifiers[best_model_name]

    # Re-train the best model on the *entire* training set (X_train, y_train)
    # This is typically done before saving the model for production
    print(f"\nEntraînement du meilleur modèle ({best_model_name}) sur le jeu d'apprentissage complet avant sauvegarde...")
    best_model.fit(X_train, y_train)
    print(f"Modèle {best_model_name} entraîné avec succès sur le jeu d'apprentissage.")


# --- 11. Sauvegarde et Chargement du Modèle Entraîné ---
print("\n--- Sauvegarde et Chargement du Modèle Entraîné ---")

# Define the filename for saving the model
model_filename = 'best_classification_model.pkl' # Changed filename to be more general

# Save the trained best model
try:
    # Save the 'best_model' instance, which was fitted on X_train, y_train
    with open(model_filename, 'wb') as file:
        pickle.dump(best_model, file)
    print(f"\nModèle entraîné sauvegardé sous : {model_filename}")
except Exception as e:
    print(f"Erreur lors de la sauvegarde du modèle : {e}")

# Load the saved model
loaded_model = None
try:
    with open(model_filename, 'rb') as file:
        loaded_model = pickle.load(file)
    print(f"Modèle chargé avec succès depuis : {model_filename}")
except FileNotFoundError:
    print(f"Erreur : Le fichier {model_filename} n'a pas été trouvé.")
except Exception as e:
    print(f"Erreur lors du chargement du modèle : {e}")

# Verify the loaded model by making a prediction (if loading was successful)
if loaded_model is not None:
    print("\nTest du modèle chargé sur un échantillon du jeu de test:")
    # Take a sample from the test set for prediction
    # Ensure X_test is available and has the correct shape
    if X_test.shape[0] > 0:
        sample_index = 1
        # Reshape the sample data to be a 2D array (required by predict method for a single sample)
        sample_data = X_test[sample_index].reshape(1, -1)

        try:
            predicted_outcome = loaded_model.predict(sample_data)
            actual_outcome = y_test[sample_index]

            print(f"Données de l'échantillon (première ligne de X_test) : {sample_data}")
            print(f"Prédiction du modèle chargé : {predicted_outcome[0]}")
            print(f"Valeur réelle : {actual_outcome}")

            if predicted_outcome[0] == actual_outcome:
                print("La prédiction du modèle chargé correspond à la valeur réelle pour cet échantillon.")
            else:
                print("La prédiction du modèle chargé ne correspond PAS à la valeur réelle pour cet échantillon.")
        except Exception as e:
             print(f"Erreur lors de la prédiction avec le modèle chargé : {e}")
    else:
        print("Le jeu de test (X_test) est vide, impossible de tester le modèle chargé.")
else:
    print("Le modèle n'a pas pu être chargé, le test ne peut pas être effectué.")
