import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn import tree
import matplotlib.pyplot as plt

# Charger les données à partir du fichier CSV
data = pd.read_csv('jouer.csv')

# Séparer les caractéristiques (X) de la classe cible (y)
X = data.iloc[:, :-1]  # Sélectionne toutes les colonnes sauf la dernière comme caractéristiques
y = data.iloc[:, -1]   # Sélectionne la dernière colonne comme variable cible

#effectuer le codage des variables catégorielles dans un DataFrame.
X=pd.get_dummies(X)

#Division des données en ensembles d'entraînement et de test.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#Création d'un modèle d'arbre de décision avec une profondeur maximale définie
model =DecisionTreeClassifier(max_depth=4,random_state=0)

#Utilisation d'une validation croisée pour évaluer les performances du modèle
cv = StratifiedKFold(n_splits=5)
scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')

print("Scores de validation croisée :", scores)
print("Score moyen de validation croisée :", scores.mean())

#Entraînement du modèle sur les données d'entraînement.
model.fit(X_train, y_train)

#Prédiction des valeurs de la variable cible pour les données de test.
y_pred =model.predict(X_test)

#Calcul de la précision du modèle en comparant les prédictions avec les valeurs réelles
accuracy = accuracy_score(y_test, y_pred)
print("Précision du modèle :", accuracy)

# Extraire les règles de l'arbre de décision
rules = tree.export_text(model, feature_names=list(X.columns))
print("Règles de l'arbre de décision :")
print(rules)

# Visualiser l'arbre de décision
short_feature_names = [f[:3] + '...' + f[-3:] for f in X.columns]
plt.figure(figsize=(14, 10))
tree.plot_tree(model, feature_names=short_feature_names, class_names=["nepasjouer", "jouer"], filled=True, fontsize=8)
plt.show()