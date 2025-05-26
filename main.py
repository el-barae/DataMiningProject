# Analyse Data Mining - Dataset de Recommandation d'Emploi
# Projet complet d'analyse de données avec techniques de fouille

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# Pour le clustering
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.decomposition import PCA

# Pour la classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Pour les règles d'association
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# Configuration des graphiques
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("=== PROJET DATA MINING - RECOMMANDATION D'EMPLOI ===")
print("Étapes: Nettoyage → EDA → Feature Engineering → Modélisation → Évaluation")
print("=" * 60)

# ============================================================================
# 1. CHARGEMENT ET EXPLORATION INITIALE DES DONNÉES
# ============================================================================

# Chargement du dataset (remplacez par le chemin réel)
try:
    # Si vous avez téléchargé via kagglehub, utilisez le chemin retourné
    df = pd.read_csv("job_dataset_120.csv")  # Remplacez par le bon chemin
except:
    # Création d'un dataset exemple pour la démonstration
    np.random.seed(42)
    n_samples = 1000

    # Génération de données simulées réalistes
    job_titles = ['Data Scientist', 'Software Engineer', 'Marketing Manager',
                  'Sales Representative', 'HR Specialist', 'Financial Analyst',
                  'Project Manager', 'UX Designer', 'Business Analyst', 'DevOps Engineer']

    industries = ['Technology', 'Finance', 'Healthcare', 'Education',
                  'Retail', 'Manufacturing', 'Consulting', 'Media']

    locations = ['Paris', 'Lyon', 'Marseille', 'Toulouse', 'Nice',
                 'Nantes', 'Bordeaux', 'Lille', 'Strasbourg', 'Montpellier']

    company_sizes = ['Startup (1-50)', 'Medium (51-200)', 'Large (201-1000)', 'Enterprise (1000+)']

    education_levels = ['Bachelor', 'Master', 'PhD', 'High School']

    # Création du dataset
    df = pd.DataFrame({
        'job_title': np.random.choice(job_titles, n_samples),
        'industry': np.random.choice(industries, n_samples),
        'location': np.random.choice(locations, n_samples),
        'company_size': np.random.choice(company_sizes, n_samples),
        'required_education': np.random.choice(education_levels, n_samples),
        'salary_min': np.random.normal(45000, 15000, n_samples).astype(int),
        'salary_max': np.random.normal(65000, 20000, n_samples).astype(int),
        'years_experience': np.random.choice(range(0, 15), n_samples),
        'remote_work': np.random.choice([True, False], n_samples, p=[0.3, 0.7]),
        'benefits_score': np.random.uniform(1, 5, n_samples).round(1),
        'job_satisfaction': np.random.uniform(2, 5, n_samples).round(1)
    })

    # Ajout de quelques valeurs manquantes pour simuler des données réelles
    missing_indices = np.random.choice(df.index, size=int(0.05 * len(df)), replace=False)
    df.loc[missing_indices[:20], 'salary_max'] = np.nan
    df.loc[missing_indices[20:40], 'benefits_score'] = np.nan

    print("Dataset d'exemple créé avec succès!")

print(f"Dataset chargé: {df.shape[0]} lignes, {df.shape[1]} colonnes")
print("\nPremières lignes du dataset:")
print(df.head())
print("\nInformations sur le dataset:")
print(df.info())

# ============================================================================
# 2. NETTOYAGE ET PRÉPARATION DES DONNÉES
# ============================================================================

print("\n" + "=" * 60)
print("2. NETTOYAGE ET PRÉPARATION DES DONNÉES")
print("=" * 60)

# Vérification des valeurs manquantes
print("Valeurs manquantes par colonne:")
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0])

# Traitement des valeurs manquantes
df['salary_max'].fillna(df['salary_max'].median(), inplace=True)
df['benefits_score'].fillna(df['benefits_score'].mean(), inplace=True)

# Vérification des doublons
duplicates = df.duplicated().sum()
print(f"\nNombre de doublons: {duplicates}")
if duplicates > 0:
    df.drop_duplicates(inplace=True)

# Correction des incohérences dans les salaires
df['salary_max'] = df[['salary_min', 'salary_max']].max(axis=1)
df['salary_range'] = df['salary_max'] - df['salary_min']


# Détection et traitement des outliers
def detect_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] < lower_bound) | (df[column] > upper_bound)]


outliers_salary = detect_outliers(df, 'salary_min')
print(f"\nOutliers détectés dans salary_min: {len(outliers_salary)}")

# Suppression des outliers extrêmes
df = df[(df['salary_min'] > 15000) & (df['salary_min'] < 150000)]
df = df[(df['salary_max'] > 20000) & (df['salary_max'] < 200000)]

print(f"Dataset après nettoyage: {df.shape[0]} lignes, {df.shape[1]} colonnes")

# ============================================================================
# 3. ANALYSE EXPLORATOIRE DES DONNÉES (EDA)
# ============================================================================

print("\n" + "=" * 60)
print("3. ANALYSE EXPLORATOIRE DES DONNÉES")
print("=" * 60)

# Statistiques descriptives
print("Statistiques descriptives des variables numériques:")
print(df.describe())

# Visualisations
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Analyse Exploratoire des Données - Job Recommendation Dataset', fontsize=16)

# Distribution des salaires
axes[0, 0].hist(df['salary_min'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
axes[0, 0].set_title('Distribution des Salaires Minimum')
axes[0, 0].set_xlabel('Salaire Minimum (€)')
axes[0, 0].set_ylabel('Fréquence')

# Distribution par industrie
industry_counts = df['industry'].value_counts()
axes[0, 1].pie(industry_counts.values, labels=industry_counts.index, autopct='%1.1f%%')
axes[0, 1].set_title('Répartition par Industrie')

# Relation expérience vs salaire
axes[0, 2].scatter(df['years_experience'], df['salary_min'], alpha=0.6, color='coral')
axes[0, 2].set_title('Expérience vs Salaire Minimum')
axes[0, 2].set_xlabel('Années d\'expérience')
axes[0, 2].set_ylabel('Salaire Minimum (€)')

# Box plot des salaires par taille d'entreprise
df.boxplot(column='salary_min', by='company_size', ax=axes[1, 0])
axes[1, 0].set_title('Salaires par Taille d\'Entreprise')
axes[1, 0].set_xlabel('Taille d\'Entreprise')

# Heatmap des corrélations
numeric_cols = df.select_dtypes(include=[np.number]).columns
corr_matrix = df[numeric_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1, 1])
axes[1, 1].set_title('Matrice de Corrélation')

# Distribution du télétravail
remote_counts = df['remote_work'].value_counts()
axes[1, 2].bar(['Sur site', 'Télétravail'], remote_counts.values, color=['lightcoral', 'lightgreen'])
axes[1, 2].set_title('Répartition Télétravail vs Sur site')
axes[1, 2].set_ylabel('Nombre d\'offres')

plt.tight_layout()
plt.show()

# Analyse des tendances
print("\nAnalyse des tendances principales:")
print(f"- Salaire moyen: {df['salary_min'].mean():.0f}€")
print(f"- Industrie la plus représentée: {df['industry'].mode().iloc[0]}")
print(f"- Pourcentage d'offres en télétravail: {df['remote_work'].mean() * 100:.1f}%")
print(f"- Satisfaction moyenne: {df['job_satisfaction'].mean():.2f}/5")

# ============================================================================
# 4. FEATURE ENGINEERING
# ============================================================================

print("\n" + "=" * 60)
print("4. FEATURE ENGINEERING")
print("=" * 60)

# Création de nouvelles variables
df['salary_avg'] = (df['salary_min'] + df['salary_max']) / 2
df['salary_category'] = pd.cut(df['salary_avg'],
                               bins=[0, 35000, 50000, 75000, float('inf')],
                               labels=['Low', 'Medium', 'High', 'Very High'])

df['experience_level'] = pd.cut(df['years_experience'],
                                bins=[-1, 2, 5, 10, float('inf')],
                                labels=['Junior', 'Mid', 'Senior', 'Expert'])

df['high_satisfaction'] = (df['job_satisfaction'] >= 4).astype(int)

# Encodage des variables catégorielles
le = LabelEncoder()
categorical_columns = ['job_title', 'industry', 'location', 'company_size', 'required_education']

df_encoded = df.copy()
for col in categorical_columns:
    df_encoded[col + '_encoded'] = le.fit_transform(df_encoded[col])

print("Nouvelles variables créées:")
print("- salary_avg: Salaire moyen")
print("- salary_category: Catégorie de salaire")
print("- experience_level: Niveau d'expérience")
print("- high_satisfaction: Satisfaction élevée (binaire)")
print("- Variables encodées pour les colonnes catégorielles")

# ============================================================================
# 5. CLUSTERING (K-MEANS)
# ============================================================================

print("\n" + "=" * 60)
print("5. CLUSTERING - K-MEANS")
print("=" * 60)

# Préparation des données pour le clustering
features_for_clustering = ['salary_avg', 'years_experience', 'benefits_score',
                           'job_satisfaction', 'remote_work']
X_cluster = df[features_for_clustering].copy()
X_cluster['remote_work'] = X_cluster['remote_work'].astype(int)

# Normalisation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

# Détermination du nombre optimal de clusters (méthode du coude)
inertias = []
K_range = range(2, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(K_range, inertias, 'bo-')
plt.title('Méthode du Coude pour K-Means')
plt.xlabel('Nombre de Clusters')
plt.ylabel('Inertie')
plt.show()

# Application du K-means avec k=4
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_scaled)
df['cluster'] = cluster_labels

# Analyse des clusters
print("Analyse des clusters:")
for i in range(4):
    cluster_data = df[df['cluster'] == i]
    print(f"\nCluster {i} ({len(cluster_data)} emplois):")
    print(f"  - Salaire moyen: {cluster_data['salary_avg'].mean():.0f}€")
    print(f"  - Expérience moyenne: {cluster_data['years_experience'].mean():.1f} ans")
    print(f"  - Satisfaction moyenne: {cluster_data['job_satisfaction'].mean():.2f}/5")
    print(f"  - % Télétravail: {cluster_data['remote_work'].mean() * 100:.0f}%")

# Visualisation des clusters
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
scatter = plt.scatter(df['salary_avg'], df['years_experience'], c=df['cluster'], cmap='viridis', alpha=0.6)
plt.colorbar(scatter)
plt.title('Clusters: Salaire vs Expérience')
plt.xlabel('Salaire Moyen (€)')
plt.ylabel('Années d\'Expérience')

plt.subplot(1, 2, 2)
scatter = plt.scatter(df['job_satisfaction'], df['benefits_score'], c=df['cluster'], cmap='viridis', alpha=0.6)
plt.colorbar(scatter)
plt.title('Clusters: Satisfaction vs Avantages')
plt.xlabel('Satisfaction')
plt.ylabel('Score Avantages')

plt.tight_layout()
plt.show()

# ============================================================================
# 6. CLASSIFICATION - PRÉDICTION DE LA SATISFACTION ÉLEVÉE
# ============================================================================

print("\n" + "=" * 60)
print("6. CLASSIFICATION - PRÉDICTION SATISFACTION ÉLEVÉE")
print("=" * 60)

# Préparation des données pour la classification
features_for_classification = ['salary_avg', 'years_experience', 'benefits_score',
                               'job_title_encoded', 'industry_encoded', 'company_size_encoded']
X = df_encoded[features_for_classification]
y = df_encoded['high_satisfaction']

# Division train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Normalisation des features
scaler_clf = StandardScaler()
X_train_scaled = scaler_clf.fit_transform(X_train)
X_test_scaled = scaler_clf.transform(X_test)

# Modèles à tester
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'K-NN': KNeighborsClassifier(n_neighbors=5),
    'SVM': SVC(random_state=42)
}

results = {}

print("Évaluation des modèles de classification:")
print("-" * 50)

for name, model in models.items():
    if name == 'SVM' or name == 'K-NN':
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy

    print(f"{name}: Accuracy = {accuracy:.3f}")

# Analyse détaillée du meilleur modèle
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]

if best_model_name == 'SVM' or best_model_name == 'K-NN':
    best_model.fit(X_train_scaled, y_train)
    y_pred_best = best_model.predict(X_test_scaled)
else:
    best_model.fit(X_train, y_train)
    y_pred_best = best_model.predict(X_test)

print(f"\nAnalyse détaillée du meilleur modèle ({best_model_name}):")
print("Classification Report:")
print(classification_report(y_test, y_pred_best))

# Matrice de confusion
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred_best)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f'Matrice de Confusion - {best_model_name}')
plt.ylabel('Valeurs Réelles')
plt.xlabel('Prédictions')
plt.show()

# Importance des features (si applicable)
if hasattr(best_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'feature': features_for_classification,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance, x='importance', y='feature')
    plt.title(f'Importance des Features - {best_model_name}')
    plt.xlabel('Importance')
    plt.show()

    print("\nImportance des features:")
    print(feature_importance)

# ============================================================================
# 7. RÈGLES D'ASSOCIATION
# ============================================================================

print("\n" + "=" * 60)
print("7. RÈGLES D'ASSOCIATION")
print("=" * 60)

# Préparation des données pour les règles d'association
# Conversion des variables continues en catégories
df_rules = df.copy()
df_rules['salary_range_cat'] = pd.cut(df_rules['salary_avg'],
                                      bins=3, labels=['Low_Salary', 'Medium_Salary', 'High_Salary'])
df_rules['experience_cat'] = pd.cut(df_rules['years_experience'],
                                    bins=3, labels=['Low_Exp', 'Medium_Exp', 'High_Exp'])

# Création du dataset binaire pour l'analyse des règles d'association
binary_features = []

# Ajout des features catégorielles
for col in ['job_title', 'industry', 'company_size', 'salary_range_cat', 'experience_cat']:
    dummies = pd.get_dummies(df_rules[col], prefix=col)
    binary_features.append(dummies)

# Ajout des features binaires
binary_features.append(pd.DataFrame({'remote_work': df_rules['remote_work'].astype(int)}))
binary_features.append(pd.DataFrame({'high_satisfaction': df_rules['high_satisfaction']}))

# Concaténation
df_binary = pd.concat(binary_features, axis=1)

# Application de l'algorithme Apriori
print("Application de l'algorithme Apriori...")
frequent_itemsets = apriori(df_binary, min_support=0.1, use_colnames=True)

if len(frequent_itemsets) > 0:
    # Génération des règles d'association
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)

    if len(rules) > 0:
        # Tri par lift décroissant
        rules_sorted = rules.sort_values('lift', ascending=False)

        print(f"\nTop 10 règles d'association (triées par lift):")
        print("-" * 80)

        for idx, rule in rules_sorted.head(10).iterrows():
            antecedents = ', '.join(list(rule['antecedents']))
            consequents = ', '.join(list(rule['consequents']))
            print(f"Règle {idx + 1}:")
            print(f"  Si {antecedents}")
            print(f"  Alors {consequents}")
            print(f"  Support: {rule['support']:.3f}, Confidence: {rule['confidence']:.3f}, Lift: {rule['lift']:.3f}")
            print()

        # Visualisation des règles
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(rules['support'], rules['confidence'],
                              c=rules['lift'], s=rules['lift'] * 20,
                              alpha=0.6, cmap='viridis')
        plt.colorbar(scatter, label='Lift')
        plt.xlabel('Support')
        plt.ylabel('Confidence')
        plt.title('Règles d\'Association: Support vs Confidence (taille = Lift)')
        plt.show()
    else:
        print("Aucune règle d'association trouvée avec les seuils définis.")
else:
    print("Aucun itemset fréquent trouvé avec le support minimum défini.")

# ============================================================================
# 8. SYNTHÈSE ET INTERPRÉTATION
# ============================================================================

print("\n" + "=" * 60)
print("8. SYNTHÈSE ET INTERPRÉTATION DES RÉSULTATS")
print("=" * 60)

print("RÉSULTATS PRINCIPAUX:")
print("-" * 30)

print(f"📊 DONNÉES:")
print(f"   • Dataset final: {df.shape[0]} emplois, {df.shape[1]} variables")
print(f"   • Qualité des données: Nettoyage effectué (valeurs manquantes, outliers)")

print(f"\n🔍 ANALYSE EXPLORATOIRE:")
print(f"   • Salaire moyen: {df['salary_avg'].mean():.0f}€")
print(f"   • Industrie dominante: {df['industry'].mode().iloc[0]}")
print(f"   • Télétravail: {df['remote_work'].mean() * 100:.1f}% des offres")

print(f"\n🎯 CLUSTERING:")
print(f"   • 4 clusters identifiés représentant différents profils d'emplois")
print(f"   • Segmentation basée sur salaire, expérience, satisfaction et télétravail")

print(f"\n🤖 CLASSIFICATION:")
print(f"   • Meilleur modèle: {best_model_name}")
print(f"   • Accuracy: {results[best_model_name]:.3f}")
print(f"   • Prédiction de la satisfaction élevée des employés")

print(f"\n📏 RÈGLES D'ASSOCIATION:")
if 'rules_sorted' in locals() and len(rules_sorted) > 0:
    print(f"   • {len(rules_sorted)} règles découvertes")
    print(f"   • Identification de patterns dans les caractéristiques d'emplois")
else:
    print(f"   • Analyse réalisée mais peu de règles significatives trouvées")

print(f"\n⚠️  LIMITES ET AMÉLIORATIONS:")
print(f"   • Dataset simulé pour la démonstration")
print(f"   • Collecte de données réelles recommandée")
print(f"   • Ajout de features supplémentaires (compétences, localisation précise)")
print(f"   • Tests de modèles plus avancés (ensemble methods, deep learning)")
print(f"   • Validation croisée plus robuste")

print(f"\n✅ CONCLUSION:")
print(f"   • Étude complète réalisée avec succès")
print(f"   • Techniques de data mining appliquées et évaluées")
print(f"   • Insights utiles pour la recommandation d'emplois")
print(f"   • Base solide pour un système de recommandation")

print("\n" + "=" * 60)
print("FIN DE L'ANALYSE - PROJET DATA MINING TERMINÉ")
print("=" * 60)