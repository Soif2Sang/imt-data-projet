import numpy as np
import pandas as pd
import matplotlib
# matplotlib.use('Agg')  # Ou essayez 'TkAgg'
matplotlib.use('Qt5Agg')  # Ou essayez 'TkAgg'
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from pandas.plotting import scatter_matrix

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

# Corrélation d'une variable particulière avec toutes les autres (par exemple, 'outcome')
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

print("\n--- Scatter Matrix ---")

# Sélection d'un sous-ensemble de variables prometteuses pour la classification
# Basé sur la corrélation avec 'outcome' et la corrélation entre elles
promising_features = ['outcome', 'speeding_violations', 'past_accidents', 'duis', 'age', 'driving_experience']
# Assurez-vous que ces colonnes existent et sont numériques
promising_df = df[promising_features]

# Affichage de la scatter matrix
scatter_matrix(promising_df, figsize=(10, 10))
plt.suptitle('Scatter Matrix des Variables Prometteuses', y=1.02)
plt.tight_layout()
plt.show()

print("\n--- Commentaires sur la Visualisation (Scatter Matrix) ---")
print("La scatter matrix affiche les nuages de points pour chaque paire de variables sélectionnées.")
print("La diagonale présente l'histogramme de chaque variable, permettant de visualiser leur distribution.")
print("En observant les nuages de points, on peut identifier des tendances ou des relations entre les variables.")
print("Par exemple, une tendance linéaire ascendante ou descendante suggère une corrélation linéaire positive ou négative.")
print("Des nuages de points dispersés sans motif clair indiquent une faible corrélation.")
print("Des regroupements de points peuvent suggérer des clusters ou des sous-groupes dans les données.")
print("Il est important d'examiner la relation de chaque variable d'entrée avec la variable de sortie ('outcome') pour identifier les prédicteurs potentiels.")
print("De même, examiner les relations entre les variables d'entrée permet de détecter une éventuelle multicolinéarité.")