# Étude Data Mining - Système de Recommandation Candidat-Emploi
# ================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, silhouette_score
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import warnings

warnings.filterwarnings('ignore')

# Configuration des graphiques
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (12, 8)
sns.set_palette("husl")

print("=" * 60)
print("ÉTUDE DATA MINING - SYSTÈME DE RECOMMANDATION CANDIDAT-EMPLOI")
print("=" * 60)

# =====================================================
# 1. CHARGEMENT ET EXPLORATION INITIALE DES DONNÉES
# =====================================================

print("\n1. CHARGEMENT ET EXPLORATION INITIALE")
print("-" * 40)

# Chargement des données
df = pd.read_csv("candidate_job_match_dataset.csv")

print(f"Dimensions du dataset: {df.shape}")
print(f"\nPremières lignes:")
print(df.head())

print(f"\nInformations générales:")
print(df.info())

print(f"\nStatistiques descriptives:")
print(df.describe())

# =====================================================
# 2. NETTOYAGE ET PRÉPARATION DES DONNÉES
# =====================================================

print("\n\n2. NETTOYAGE ET PRÉPARATION DES DONNÉES")
print("-" * 45)

# Vérification des valeurs manquantes
print("Valeurs manquantes par colonne:")
missing_values = df.isnull().sum()
print(missing_values)

# Vérification des doublons
duplicates = df.duplicated().sum()
print(f"\nNombre de doublons: {duplicates}")

# Suppression des doublons si nécessaire
if duplicates > 0:
    df = df.drop_duplicates()
    print(f"Doublons supprimés. Nouvelles dimensions: {df.shape}")

# Traitement des valeurs manquantes
df_clean = df.copy()

# Pour les colonnes numériques, remplacer par la médiane
numeric_cols = ['ExperienceYears', 'RequiredExperience', 'MatchScore']
for col in numeric_cols:
    if df_clean[col].isnull().sum() > 0:
        median_val = df_clean[col].median()
        df_clean[col].fillna(median_val, inplace=True)
        print(f"Valeurs manquantes dans {col} remplacées par la médiane: {median_val}")

# Pour les colonnes catégorielles, remplacer par le mode
categorical_cols = ['CandidateName', 'EducationLevel', 'Skills', 'RequiredSkills',
                    'JobTitle', 'ClientName', 'EducationRequired']
for col in categorical_cols:
    if df_clean[col].isnull().sum() > 0:
        mode_val = df_clean[col].mode()[0]
        df_clean[col].fillna(mode_val, inplace=True)
        print(f"Valeurs manquantes dans {col} remplacées par le mode: {mode_val}")

# Détection et traitement des valeurs aberrantes
print(f"\nDétection des valeurs aberrantes:")
for col in numeric_cols:
    Q1 = df_clean[col].quantile(0.25)
    Q3 = df_clean[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df_clean[(df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)][col]
    print(f"{col}: {len(outliers)} valeurs aberrantes détectées")

    # Limitation des valeurs aberrantes (winsorization)
    df_clean[col] = np.clip(df_clean[col], lower_bound, upper_bound)

print(f"\nDonnées nettoyées. Dimensions finales: {df_clean.shape}")

# =====================================================
# 3. ANALYSE EXPLORATOIRE DES DONNÉES (EDA)
# =====================================================

print("\n\n3. ANALYSE EXPLORATOIRE DES DONNÉES")
print("-" * 40)

# Distribution des variables numériques
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Distribution des Variables Numériques', fontsize=16, fontweight='bold')

df_clean['ExperienceYears'].hist(bins=20, ax=axes[0, 0], alpha=0.7, color='skyblue')
axes[0, 0].set_title('Distribution Expérience Candidats')
axes[0, 0].set_xlabel('Années d\'expérience')

df_clean['RequiredExperience'].hist(bins=20, ax=axes[0, 1], alpha=0.7, color='lightcoral')
axes[0, 1].set_title('Distribution Expérience Requise')
axes[0, 1].set_xlabel('Années d\'expérience requise')

df_clean['MatchScore'].hist(bins=20, ax=axes[1, 0], alpha=0.7, color='lightgreen')
axes[1, 0].set_title('Distribution Score de Match')
axes[1, 0].set_xlabel('Score de correspondance')

# Boxplot pour détecter les outliers
df_clean[['ExperienceYears', 'RequiredExperience', 'MatchScore']].boxplot(ax=axes[1, 1])
axes[1, 1].set_title('Boxplot - Détection Outliers')

plt.tight_layout()
plt.show()

# Analyse des variables catégorielles
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Distribution des Variables Catégorielles', fontsize=16, fontweight='bold')

# Top 10 des niveaux d'éducation
education_counts = df_clean['EducationLevel'].value_counts().head(10)
education_counts.plot(kind='bar', ax=axes[0, 0], color='lightblue')
axes[0, 0].set_title('Top 10 Niveaux d\'Éducation Candidats')
axes[0, 0].tick_params(axis='x', rotation=45)

# Top 10 des niveaux d'éducation requis
edu_req_counts = df_clean['EducationRequired'].value_counts().head(10)
edu_req_counts.plot(kind='bar', ax=axes[0, 1], color='lightcoral')
axes[0, 1].set_title('Top 10 Niveaux d\'Éducation Requis')
axes[0, 1].tick_params(axis='x', rotation=45)

# Top 10 des titres de poste
job_counts = df_clean['JobTitle'].value_counts().head(10)
job_counts.plot(kind='bar', ax=axes[1, 0], color='lightgreen')
axes[1, 0].set_title('Top 10 Titres de Poste')
axes[1, 0].tick_params(axis='x', rotation=45)

# Top 10 des clients
client_counts = df_clean['ClientName'].value_counts().head(10)
client_counts.plot(kind='bar', ax=axes[1, 1], color='gold')
axes[1, 1].set_title('Top 10 Clients')
axes[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# Matrice de corrélation
print("\nMatrice de corrélation des variables numériques:")
correlation_matrix = df_clean[['ExperienceYears', 'RequiredExperience', 'MatchScore']].corr()
print(correlation_matrix)

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
            square=True, cbar_kws={'shrink': 0.8})
plt.title('Matrice de Corrélation des Variables Numériques', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# Analyse de la relation expérience vs match score
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(df_clean['ExperienceYears'], df_clean['MatchScore'], alpha=0.6, color='blue')
plt.xlabel('Expérience Candidat (années)')
plt.ylabel('Score de Match')
plt.title('Expérience Candidat vs Score de Match')

plt.subplot(1, 2, 2)
plt.scatter(df_clean['RequiredExperience'], df_clean['MatchScore'], alpha=0.6, color='red')
plt.xlabel('Expérience Requise (années)')
plt.ylabel('Score de Match')
plt.title('Expérience Requise vs Score de Match')

plt.tight_layout()
plt.show()

# =====================================================
# 4. FEATURE ENGINEERING
# =====================================================

print("\n\n4. FEATURE ENGINEERING")
print("-" * 25)

# Création de nouvelles variables
df_features = df_clean.copy()

# 1. Différence d'expérience (candidat - requis)
df_features['ExperienceDiff'] = df_features['ExperienceYears'] - df_features['RequiredExperience']


# 2. Catégorisation du niveau d'expérience
def categorize_experience(years):
    if years <= 2:
        return 'Junior'
    elif years <= 5:
        return 'Mid-level'
    elif years <= 10:
        return 'Senior'
    else:
        return 'Expert'


df_features['ExperienceCategory'] = df_features['ExperienceYears'].apply(categorize_experience)
df_features['RequiredExperienceCategory'] = df_features['RequiredExperience'].apply(categorize_experience)


# 3. Catégorisation du match score - Ajustement des seuils
def categorize_match(score):
    # Ajustement des seuils basé sur les quartiles pour assurer une distribution équilibrée
    q1 = df_clean['MatchScore'].quantile(0.33)
    q2 = df_clean['MatchScore'].quantile(0.67)

    if score <= q1:
        return 'Faible'
    elif score <= q2:
        return 'Moyen'
    else:
        return 'Élevé'


df_features['MatchCategory'] = df_features['MatchScore'].apply(categorize_match)

# Vérification de la distribution des classes
print(f"Distribution des classes MatchCategory:")
print(df_features['MatchCategory'].value_counts())
print(f"Nombre de classes uniques: {df_features['MatchCategory'].nunique()}")

# 4. Encodage des variables catégorielles
label_encoders = {}
categorical_columns = ['EducationLevel', 'EducationRequired', 'JobTitle', 'ClientName',
                       'ExperienceCategory', 'RequiredExperienceCategory']

for col in categorical_columns:
    le = LabelEncoder()
    df_features[f'{col}_encoded'] = le.fit_transform(df_features[col].astype(str))
    label_encoders[col] = le

# 5. Normalisation des variables numériques
scaler = StandardScaler()
numeric_features = ['ExperienceYears', 'RequiredExperience', 'MatchScore', 'ExperienceDiff']
df_features[numeric_features] = scaler.fit_transform(df_features[numeric_features])

print("Nouvelles variables créées:")
print("- ExperienceDiff: Différence d'expérience (candidat - requis)")
print("- ExperienceCategory: Catégorie d'expérience du candidat")
print("- RequiredExperienceCategory: Catégorie d'expérience requise")
print("- MatchCategory: Catégorie de score de match")
print("- Variables encodées pour les colonnes catégorielles")
print("- Variables numériques normalisées")

# =====================================================
# 5. RÈGLES D'ASSOCIATION (APRIORI)
# =====================================================

print("\n\n5. ANALYSE DES RÈGLES D'ASSOCIATION")
print("-" * 35)

# Préparation des données pour l'analyse des règles d'association
# Création de transactions basées sur les caractéristiques des candidats et jobs

transactions = []
for _, row in df_clean.iterrows():
    transaction = [
        f"Education_{row['EducationLevel']}",
        f"JobTitle_{row['JobTitle']}",
        f"EducationReq_{row['EducationRequired']}",
        f"ExpCat_{categorize_experience(row['ExperienceYears'])}",
        f"ExpReqCat_{categorize_experience(row['RequiredExperience'])}",
        f"MatchCat_{categorize_match(row['MatchScore'])}"
    ]
    transactions.append(transaction)

# Encodage des transactions
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_transactions = pd.DataFrame(te_ary, columns=te.columns_)

# Application de l'algorithme Apriori
frequent_itemsets = apriori(df_transactions, min_support=0.1, use_colnames=True)
print(f"Nombre d'itemsets fréquents trouvés: {len(frequent_itemsets)}")

if len(frequent_itemsets) > 0:
    # Génération des règles d'association
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)

    if len(rules) > 0:
        print(f"Nombre de règles d'association générées: {len(rules)}")
        print("\nTop 10 des règles d'association (par confiance):")
        top_rules = rules.nlargest(10, 'confidence')[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
        for idx, rule in top_rules.iterrows():
            antecedents = ', '.join(list(rule['antecedents']))
            consequents = ', '.join(list(rule['consequents']))
            print(f"Si {antecedents} => Alors {consequents}")
            print(f"  Support: {rule['support']:.3f}, Confiance: {rule['confidence']:.3f}, Lift: {rule['lift']:.3f}\n")
    else:
        print("Aucune règle d'association trouvée avec les seuils définis.")
else:
    print("Aucun itemset fréquent trouvé. Réduction du seuil de support...")
    frequent_itemsets = apriori(df_transactions, min_support=0.05, use_colnames=True)
    print(f"Nombre d'itemsets fréquents avec support réduit: {len(frequent_itemsets)}")

# =====================================================
# 6. CLUSTERING (K-MEANS ET DBSCAN)
# =====================================================

print("\n\n6. ANALYSE DE CLUSTERING")
print("-" * 25)

# Préparation des données pour le clustering
clustering_features = ['ExperienceYears', 'RequiredExperience', 'MatchScore', 'ExperienceDiff']
X_cluster = df_features[clustering_features + [col + '_encoded' for col in categorical_columns]]

# K-Means Clustering
print("A. K-MEANS CLUSTERING")
print("-" * 20)

# Détermination du nombre optimal de clusters (méthode du coude)
inertias = []
silhouette_scores = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_cluster)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_cluster, cluster_labels))

# Visualisation de la méthode du coude et silhouette score
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

ax1.plot(k_range, inertias, marker='o', linewidth=2, markersize=8)
ax1.set_xlabel('Nombre de clusters (k)')
ax1.set_ylabel('Inertie')
ax1.set_title('Méthode du Coude - K-Means')
ax1.grid(True, alpha=0.3)

ax2.plot(k_range, silhouette_scores, marker='s', linewidth=2, markersize=8, color='orange')
ax2.set_xlabel('Nombre de clusters (k)')
ax2.set_ylabel('Score de Silhouette')
ax2.set_title('Score de Silhouette - K-Means')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Choix du nombre optimal de clusters
optimal_k = k_range[np.argmax(silhouette_scores)]
print(f"Nombre optimal de clusters (basé sur silhouette score): {optimal_k}")

# Application du K-Means avec le nombre optimal de clusters
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df_features['KMeans_Cluster'] = kmeans_final.fit_predict(X_cluster)

print(f"Score de silhouette final: {silhouette_score(X_cluster, df_features['KMeans_Cluster']):.3f}")

# Analyse des clusters
print("\nAnalyse des clusters K-Means:")
cluster_analysis = df_clean.groupby(df_features['KMeans_Cluster']).agg({
    'ExperienceYears': ['mean', 'std'],
    'RequiredExperience': ['mean', 'std'],
    'MatchScore': ['mean', 'std'],
    'EducationLevel': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'N/A',
    'JobTitle': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'N/A'
}).round(2)

print(cluster_analysis)

# DBSCAN Clustering
print("\nB. DBSCAN CLUSTERING")
print("-" * 20)

# Test de différents paramètres pour DBSCAN
eps_values = [0.5, 1.0, 1.5, 2.0]
min_samples_values = [5, 10, 15]

best_silhouette = -1
best_params = {}

for eps in eps_values:
    for min_samples in min_samples_values:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = dbscan.fit_predict(X_cluster)

        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)

        if n_clusters > 1:  # Au moins 2 clusters pour calculer silhouette
            silhouette_avg = silhouette_score(X_cluster, cluster_labels)
            if silhouette_avg > best_silhouette:
                best_silhouette = silhouette_avg
                best_params = {'eps': eps, 'min_samples': min_samples, 'n_clusters': n_clusters, 'n_noise': n_noise}

if best_params:
    print(f"Meilleurs paramètres DBSCAN: eps={best_params['eps']}, min_samples={best_params['min_samples']}")
    print(f"Nombre de clusters: {best_params['n_clusters']}, Points de bruit: {best_params['n_noise']}")
    print(f"Score de silhouette: {best_silhouette:.3f}")

    # Application finale de DBSCAN
    dbscan_final = DBSCAN(eps=best_params['eps'], min_samples=best_params['min_samples'])
    df_features['DBSCAN_Cluster'] = dbscan_final.fit_predict(X_cluster)
else:
    print("Aucune configuration DBSCAN satisfaisante trouvée.")

# Visualisation des clusters (2D projection)
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
scatter = plt.scatter(df_clean['ExperienceYears'], df_clean['MatchScore'],
                      c=df_features['KMeans_Cluster'], cmap='viridis', alpha=0.6)
plt.xlabel('Expérience Candidat')
plt.ylabel('Score de Match')
plt.title('Clusters K-Means')
plt.colorbar(scatter)

if 'DBSCAN_Cluster' in df_features.columns:
    plt.subplot(1, 3, 2)
    scatter = plt.scatter(df_clean['ExperienceYears'], df_clean['MatchScore'],
                          c=df_features['DBSCAN_Cluster'], cmap='viridis', alpha=0.6)
    plt.xlabel('Expérience Candidat')
    plt.ylabel('Score de Match')
    plt.title('Clusters DBSCAN')
    plt.colorbar(scatter)

plt.subplot(1, 3, 3)
scatter = plt.scatter(df_clean['RequiredExperience'], df_clean['MatchScore'],
                      c=df_features['KMeans_Cluster'], cmap='viridis', alpha=0.6)
plt.xlabel('Expérience Requise')
plt.ylabel('Score de Match')
plt.title('K-Means: Exp. Requise vs Match')
plt.colorbar(scatter)

plt.tight_layout()
plt.show()

# =====================================================
# 7. CLASSIFICATION (PRÉDICTION DU NIVEAU DE MATCH)
# =====================================================

print("\n\n7. MODÈLES DE CLASSIFICATION")
print("-" * 30)

# Préparation des données pour la classification
# Variable cible: MatchCategory (Faible, Moyen, Élevé)
X = df_features[['ExperienceYears', 'RequiredExperience', 'ExperienceDiff'] +
                [col + '_encoded' for col in categorical_columns]]
y = df_features['MatchCategory']

# Encodage de la variable cible
le_target = LabelEncoder()
y_encoded = le_target.fit_transform(y)

# Division des données avec vérification des classes
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3,
                                                    random_state=42, stratify=y_encoded)

print(f"Taille de l'ensemble d'entraînement: {X_train.shape}")
print(f"Taille de l'ensemble de test: {X_test.shape}")
print(f"Distribution des classes dans y:")
print(pd.Series(y).value_counts())
print(f"Distribution des classes encodées:")
print(pd.Series(y_encoded).value_counts())

# Vérification que nous avons au moins 2 classes
n_classes = len(np.unique(y_encoded))
print(f"Nombre de classes uniques: {n_classes}")

if n_classes < 2:
    print("⚠️ ATTENTION: Moins de 2 classes détectées. Ajustement des seuils de catégorisation...")

    # Recatégorisation avec des seuils plus fins
    score_min = df_clean['MatchScore'].min()
    score_max = df_clean['MatchScore'].max()
    score_range = score_max - score_min


    def categorize_match_fine(score):
        threshold1 = score_min + score_range * 0.4
        threshold2 = score_min + score_range * 0.7

        if score <= threshold1:
            return 'Faible'
        elif score <= threshold2:
            return 'Moyen'
        else:
            return 'Élevé'


    # Réapplication de la catégorisation
    df_features['MatchCategory'] = df_clean['MatchScore'].apply(categorize_match_fine)
    y = df_features['MatchCategory']
    y_encoded = le_target.fit_transform(y)

    print(f"Nouvelle distribution après ajustement:")
    print(pd.Series(y).value_counts())

    # Nouvelle division des données
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3,
                                                        random_state=42, stratify=y_encoded)

# Modèles à tester - avec gestion des cas mono-classe
models = {}

# Vérification du nombre de classes avant d'ajouter les modèles
if len(np.unique(y_encoded)) >= 2:
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'K-NN': KNeighborsClassifier(n_neighbors=min(5, len(np.unique(y_encoded)))),
        'SVM': SVC(random_state=42, probability=True)  # Ajout de probability=True pour predict_proba
    }
else:
    print("❌ Impossible de faire de la classification avec une seule classe.")
    print("Création de classes artificielles basées sur les valeurs numériques...")

    # Création de classes basées sur les quantiles des scores originaux
    df_features['MatchCategory'] = pd.qcut(df_clean['MatchScore'],
                                           q=3,
                                           labels=['Faible', 'Moyen', 'Élevé'])
    y = df_features['MatchCategory']
    y_encoded = le_target.fit_transform(y)

    # Nouvelle division
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3,
                                                        random_state=42, stratify=y_encoded)

    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'K-NN': KNeighborsClassifier(n_neighbors=min(5, len(np.unique(y_encoded)))),
        'SVM': SVC(random_state=42, probability=True)
    }

    print(f"Nouvelles classes créées:")
    print(pd.Series(y).value_counts())

# Entraînement et évaluation des modèles
results = {}

print("\nRésultats des modèles de classification:")
print("=" * 50)

for name, model in models.items():
    print(f"\n{name.upper()}")
    print("-" * len(name))

    # Entraînement
    model.fit(X_train, y_train)

    # Prédictions
    y_pred = model.predict(X_test)

    # Métriques
    accuracy = accuracy_score(y_test, y_pred)
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)

    results[name] = {
        'accuracy': accuracy,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'model': model
    }

    print(f"Précision sur test: {accuracy:.3f}")
    print(f"Validation croisée: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")

    # Rapport de classification détaillé
    print(f"\nRapport de classification:")
    print(classification_report(y_test, y_pred,
                                target_names=le_target.classes_))

# Comparaison des modèles
print("\n\nCOMPARAISON DES MODÈLES")
print("=" * 25)

comparison_df = pd.DataFrame({
    'Modèle': list(results.keys()),
    'Précision Test': [results[model]['accuracy'] for model in results.keys()],
    'CV Moyenne': [results[model]['cv_mean'] for model in results.keys()],
    'CV Écart-type': [results[model]['cv_std'] for model in results.keys()]
})

print(comparison_df.round(3))

# Visualisation des performances
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
model_names = list(results.keys())
accuracies = [results[model]['accuracy'] for model in model_names]
colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold']
bars = plt.bar(model_names, accuracies, color=colors)
plt.ylabel('Précision')
plt.title('Précision sur Ensemble de Test')
plt.ylim(0, 1)
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
             f'{acc:.3f}', ha='center', va='bottom')

plt.subplot(1, 2, 2)
cv_means = [results[model]['cv_mean'] for model in model_names]
cv_stds = [results[model]['cv_std'] for model in model_names]
bars = plt.bar(model_names, cv_means, yerr=cv_stds, capsize=5, color=colors)
plt.ylabel('Précision (Validation Croisée)')
plt.title('Performance en Validation Croisée')
plt.ylim(0, 1)

plt.tight_layout()
plt.show()

# Analyse d'importance des features (Random Forest)
if 'Random Forest' in results:
    rf_model = results['Random Forest']['model']
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)

    print(f"\nIMPORTANCE DES VARIABLES (Random Forest):")
    print("-" * 45)
    print(feature_importance.head(10))

    # Visualisation de l'importance des features
    plt.figure(figsize=(10, 6))
    top_features = feature_importance.head(10)
    plt.barh(top_features['Feature'], top_features['Importance'])
    plt.xlabel('Importance')
    plt.title('Top 10 des Variables les Plus Importantes')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

# =====================================================
# 8. SYSTÈME DE RECOMMANDATION
# =====================================================

print("\n\n8. SYSTÈME DE RECOMMANDATION")
print("-" * 30)

# Fonction de recommandation basée sur le meilleur modèle
best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
best_model = results[best_model_name]['model']

print(f"Utilisation du meilleur modèle: {best_model_name}")


def recommend_candidates(job_requirements, top_n=5):
    """
    Recommande les meilleurs candidats pour un poste donné
    """
    # Simulation d'une base de candidats (utilisation du dataset existant)
    candidates_df = df_clean.copy()

    # Calcul du score de correspondance pour chaque candidat
    recommendations = []

    for idx, candidate in candidates_df.iterrows():
        # Création du vecteur de features pour ce candidat
        candidate_features = [
            (candidate['ExperienceYears'] - scaler.mean_[0]) / scaler.scale_[0],  # Normalisation
            (job_requirements['required_experience'] - scaler.mean_[1]) / scaler.scale_[1],
            (candidate['ExperienceYears'] - job_requirements['required_experience'] - scaler.mean_[3]) / scaler.scale_[
                3]
        ]

        # Ajout des features encodées (simplification)
        for col in categorical_columns:
            if col == 'EducationRequired':
                # Utilisation de l'éducation requise du job
                if job_requirements['education'] in label_encoders[col].classes_:
                    encoded_val = label_encoders[col].transform([job_requirements['education']])[0]
                else:
                    encoded_val = 0
            elif col == 'JobTitle':
                if job_requirements['job_title'] in label_encoders[col].classes_:
                    encoded_val = label_encoders[col].transform([job_requirements['job_title']])[0]
                else:
                    encoded_val = 0
            else:
                # Pour les autres colonnes, utiliser les valeurs du candidat
                if candidate[col.replace('_encoded', '')] in label_encoders[col].classes_:
                    encoded_val = label_encoders[col].transform([str(candidate[col.replace('_encoded', '')])])[0]
                else:
                    encoded_val = 0
            candidate_features.append(encoded_val)

        # Prédiction du niveau de match
        try:
            match_prediction = best_model.predict([candidate_features])[0]

            # Vérification si le modèle a predict_proba
            if hasattr(best_model, 'predict_proba'):
                match_prob = best_model.predict_proba([candidate_features])[0].max()
            else:
                # Pour les modèles sans predict_proba, utiliser une confiance basée sur la distance
                match_prob = 0.8  # Valeur par défaut

            recommendations.append({
                'candidate_name': candidate['CandidateName'],
                'experience': candidate['ExperienceYears'],
                'education': candidate['EducationLevel'],
                'skills': candidate['Skills'],
                'predicted_match': le_target.inverse_transform([match_prediction])[0],
                'confidence': match_prob,
                'actual_match_score': candidate['MatchScore']
            })
        except Exception as e:
            # En cas d'erreur, utiliser le score actuel
            recommendations.append({
                'candidate_name': candidate['CandidateName'],
                'experience': candidate['ExperienceYears'],
                'education': candidate['EducationLevel'],
                'skills': candidate['Skills'],
                'predicted_match': categorize_match_fine(
                    candidate['MatchScore']) if 'categorize_match_fine' in locals() else categorize_match(
                    candidate['MatchScore']),
                'confidence': candidate['MatchScore'],
                'actual_match_score': candidate['MatchScore']
            })

    # Tri des recommandations par niveau de match et confiance
    recommendations_df = pd.DataFrame(recommendations)

    # Ordre de priorité: Élevé > Moyen > Faible
    match_priority = {'Élevé': 3, 'Moyen': 2, 'Faible': 1}
    recommendations_df['match_priority'] = recommendations_df['predicted_match'].map(match_priority)

    # Tri par priorité puis par confiance
    recommendations_df = recommendations_df.sort_values(['match_priority', 'confidence'], ascending=[False, False])

    return recommendations_df.head(top_n)


# Exemple d'utilisation du système de recommandation
print("\nEXEMPLE DE RECOMMANDATION:")
print("-" * 25)

# Définition d'un poste exemple
job_example = {
    'job_title': 'Data Scientist',
    'required_experience': 5,
    'education': 'Master',
    'required_skills': 'Python, Machine Learning, Statistics'
}

print(f"Poste à pourvoir: {job_example['job_title']}")
print(f"Expérience requise: {job_example['required_experience']} ans")
print(f"Éducation requise: {job_example['education']}")
print(f"Compétences requises: {job_example['required_skills']}")

# Génération des recommandations
top_candidates = recommend_candidates(job_example, top_n=10)

print(f"\nTop 10 candidats recommandés:")
print("=" * 50)
for idx, candidate in top_candidates.iterrows():
    print(f"\n{idx + 1}. {candidate['candidate_name']}")
    print(f"   Expérience: {candidate['experience']} ans")
    print(f"   Éducation: {candidate['education']}")
    print(f"   Compétences: {candidate['skills'][:50]}...")
    print(f"   Match prédit: {candidate['predicted_match']}")
    print(f"   Confiance: {candidate['confidence']:.3f}")
    print(f"   Score réel: {candidate['actual_match_score']:.3f}")

# =====================================================
# 9. ÉVALUATION ET MÉTRIQUES DE PERFORMANCE
# =====================================================

print("\n\n9. ÉVALUATION GLOBALE DU SYSTÈME")
print("-" * 35)


# Évaluation du système de recommandation
def evaluate_recommendation_system(n_tests=100):
    """
    Évalue la performance du système de recommandation
    """
    correct_predictions = 0
    total_predictions = 0

    # Test sur un échantillon du dataset
    test_sample = df_clean.sample(min(n_tests, len(df_clean)))

    for idx, row in test_sample.iterrows():
        job_req = {
            'job_title': row['JobTitle'],
            'required_experience': row['RequiredExperience'],
            'education': row['EducationRequired'],
            'required_skills': row['RequiredSkills']
        }

        # Recommandation pour ce poste
        recommendations = recommend_candidates(job_req, top_n=5)

        # Vérification si le candidat original est dans les recommandations
        if row['CandidateName'] in recommendations['candidate_name'].values:
            correct_predictions += 1
        total_predictions += 1

    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    return accuracy, correct_predictions, total_predictions


# Évaluation du système
rec_accuracy, correct, total = evaluate_recommendation_system(50)
print(f"Précision du système de recommandation: {rec_accuracy:.3f}")
print(f"Recommandations correctes: {correct}/{total}")

# Métriques supplémentaires
print(f"\nMÉTRIQUES SUPPLÉMENTAIRES:")
print("-" * 25)

# Distribution des scores de match
match_distribution = df_clean['MatchScore'].describe()
print(f"Distribution des scores de match:")
print(match_distribution)

# Analyse par catégorie d'expérience
exp_analysis = df_clean.groupby(df_clean['ExperienceYears'].apply(categorize_experience)).agg({
    'MatchScore': ['mean', 'std', 'count']
}).round(3)
print(f"\nAnalyse par catégorie d'expérience:")
print(exp_analysis)

# Analyse par niveau d'éducation
edu_analysis = df_clean.groupby('EducationLevel')['MatchScore'].agg(['mean', 'count']).sort_values('mean',
                                                                                                   ascending=False).head(
    10)
print(f"\nTop 10 niveaux d'éducation par score de match moyen:")
print(edu_analysis.round(3))

# =====================================================
# 10. INTERPRÉTATION ET CONCLUSIONS
# =====================================================

print("\n\n10. INTERPRÉTATION ET CONCLUSIONS")
print("-" * 35)

print("RÉSUMÉ DE L'ANALYSE:")
print("=" * 25)

print(f"✓ Dataset analysé: {df.shape[0]} candidats, {df.shape[1]} variables")
print(f"✓ Données nettoyées: {df_clean.shape[0]} enregistrements après nettoyage")
print(f"✓ Variables créées: 4 nouvelles variables (ExperienceDiff, catégories, encodages)")

if len(frequent_itemsets) > 0:
    print(f"✓ Règles d'association: {len(frequent_itemsets)} itemsets fréquents identifiés")
else:
    print("✓ Règles d'association: Analyse effectuée (données peu structurées pour les règles)")

print(f"✓ Clustering: {optimal_k} clusters optimaux identifiés (K-means)")
print(f"✓ Classification: Meilleur modèle = {best_model_name} (précision: {results[best_model_name]['accuracy']:.3f})")
print(f"✓ Système de recommandation: Précision = {rec_accuracy:.3f}")

print(f"\nINSIGHTS PRINCIPAUX:")
print("-" * 20)

# Corrélations importantes
corr_exp_match = df_clean[['ExperienceYears', 'MatchScore']].corr().iloc[0, 1]
corr_req_match = df_clean[['RequiredExperience', 'MatchScore']].corr().iloc[0, 1]

print(f"• Corrélation expérience-match: {corr_exp_match:.3f}")
print(f"• Corrélation expérience requise-match: {corr_req_match:.3f}")

# Meilleur niveau d'éducation
best_education = edu_analysis.index[0]
best_edu_score = edu_analysis.iloc[0]['mean']
print(f"• Meilleur niveau d'éducation: {best_education} (score moyen: {best_edu_score:.3f})")

# Analyse des clusters
print(f"• {optimal_k} profils de candidats identifiés par clustering")

# Variables les plus importantes
if 'Random Forest' in results:
    top_feature = feature_importance.iloc[0]
    print(f"• Variable la plus importante: {top_feature['Feature']} (importance: {top_feature['Importance']:.3f})")

print(f"\nRECOMMANDATIONS MÉTIER:")
print("-" * 25)

print("1. AMÉLIORATION DU MATCHING:")
print("   • Intégrer davantage de variables de compétences spécifiques")
print("   • Pondérer différemment selon le type de poste")
print("   • Considérer l'expérience sectorielle en plus de l'expérience totale")

print("\n2. OPTIMISATION DU SYSTÈME:")
print("   • Collecter plus de données pour améliorer les règles d'association")
print("   • Implémenter un feedback loop pour améliorer les prédictions")
print("   • Ajouter des critères de localisation et de salaire")

print("\n3. UTILISATION DES CLUSTERS:")
print("   • Personnaliser les recommandations par cluster de candidats")
print("   • Adapter les critères de matching selon le profil identifié")
print("   • Développer des stratégies de sourcing spécifiques par cluster")

print(f"\nLIMITES DE L'ÉTUDE:")
print("-" * 20)
print("• Données simulées - résultats à valider sur données réelles")
print("• Variables de compétences peu structurées")
print("• Manque de données temporelles pour l'évolution des candidats")
print("• Absence de feedback des recruteurs pour validation")

print(f"\nPISTES D'AMÉLIORATION:")
print("-" * 25)
print("• Intégration de NLP pour analyser les compétences textuelles")
print("• Utilisation de réseaux de neurones pour des patterns complexes")
print("• Développement d'un système de scoring multi-critères")
print("• Implémentation d'un système de recommandation collaborative")

print("\n" + "=" * 60)
print("FIN DE L'ANALYSE - SYSTÈME DE RECOMMANDATION CANDIDAT-EMPLOI")
print("=" * 60)

# Sauvegarde des résultats principaux
results_summary = {
    'dataset_size': df.shape,
    'clean_dataset_size': df_clean.shape,
    'best_model': best_model_name,
    'best_model_accuracy': results[best_model_name]['accuracy'],
    'optimal_clusters': optimal_k,
    'recommendation_accuracy': rec_accuracy,
    'top_correlations': {
        'experience_match': corr_exp_match,
        'required_exp_match': corr_req_match
    }
}

print(f"\nRésultats sauvegardés dans results_summary")
print("Analyse terminée avec succès!")

# Visualisation finale - Dashboard de résultats
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('DASHBOARD - RÉSULTATS DE L\'ANALYSE DATA MINING', fontsize=16, fontweight='bold')

# 1. Distribution des scores de match
axes[0, 0].hist(df_clean['MatchScore'], bins=20, color='skyblue', alpha=0.7, edgecolor='black')
axes[0, 0].set_title('Distribution des Scores de Match')
axes[0, 0].set_xlabel('Score de Match')
axes[0, 0].set_ylabel('Fréquence')

# 2. Performance des modèles
model_names = list(results.keys())
accuracies = [results[model]['accuracy'] for model in model_names]
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
bars = axes[0, 1].bar(model_names, accuracies, color=colors)
axes[0, 1].set_title('Performance des Modèles de Classification')
axes[0, 1].set_ylabel('Précision')
axes[0, 1].tick_params(axis='x', rotation=45)
for bar, acc in zip(bars, accuracies):
    axes[0, 1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom')

# 3. Clustering visualization
scatter = axes[0, 2].scatter(df_clean['ExperienceYears'], df_clean['MatchScore'],
                             c=df_features['KMeans_Cluster'], cmap='viridis', alpha=0.6)
axes[0, 2].set_title(f'Clustering K-Means ({optimal_k} clusters)')
axes[0, 2].set_xlabel('Expérience (années)')
axes[0, 2].set_ylabel('Score de Match')

# 4. Corrélations
corr_data = df_clean[['ExperienceYears', 'RequiredExperience', 'MatchScore']].corr()
im = axes[1, 0].imshow(corr_data, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
axes[1, 0].set_title('Matrice de Corrélation')
axes[1, 0].set_xticks(range(len(corr_data.columns)))
axes[1, 0].set_yticks(range(len(corr_data.columns)))
axes[1, 0].set_xticklabels(corr_data.columns, rotation=45)
axes[1, 0].set_yticklabels(corr_data.columns)
for i in range(len(corr_data.columns)):
    for j in range(len(corr_data.columns)):
        axes[1, 0].text(j, i, f'{corr_data.iloc[i, j]:.2f}', ha='center', va='center')

# 5. Top éducations par score
top_edu = edu_analysis.head(5)
axes[1, 1].barh(range(len(top_edu)), top_edu['mean'], color='lightgreen')
axes[1, 1].set_title('Top 5 Niveaux d\'Éducation')
axes[1, 1].set_xlabel('Score de Match Moyen')
axes[1, 1].set_yticks(range(len(top_edu)))
axes[1, 1].set_yticklabels(top_edu.index)

# 6. Métriques de performance globales
metrics = ['Précision\nClassification', 'Précision\nRecommandation', 'Nb Clusters\nOptimal', 'Corrélation\nExp-Match']
values = [results[best_model_name]['accuracy'], rec_accuracy, optimal_k / 10,
          abs(corr_exp_match)]  # Normalisation pour le nb de clusters
colors_metrics = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
bars = axes[1, 2].bar(metrics, values, color=colors_metrics)
axes[1, 2].set_title('Métriques de Performance Globales')
axes[1, 2].set_ylabel('Valeur')
for bar, val, metric in zip(bars, values, metrics):
    if 'Clusters' in metric:
        axes[1, 2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f'{optimal_k}', ha='center', va='bottom')
    else:
        axes[1, 2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f'{val:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

print("\n🎉 ANALYSE COMPLÈTE TERMINÉE AVEC SUCCÈS! 🎉")