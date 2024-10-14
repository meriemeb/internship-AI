import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn import tree
import matplotlib.pyplot as plt

# Charger les données à partir du fichier CSV
data = pd.read_csv('cs.csv')

# Supprimer les lignes contenant des valeurs NaN
data = data.dropna()

# Séparer les caractéristiques (X) de la classe cible (y)
X = data.drop('SeriousDlqin2yrs', axis=1)
y = data['SeriousDlqin2yrs']

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Créer le classifieur de l'arbre de décision
clf = DecisionTreeClassifier(max_depth=4, random_state=0)

# Effectuer la validation croisée avec StratifiedKFold et l'exactitude comme métrique
cv = StratifiedKFold(n_splits=5)
scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')

# Afficher les scores de validation croisée
print("Scores de validation croisée :", scores)
print("Score moyen de validation croisée :", scores.mean())

# Entraîner le classifieur sur l'ensemble d'entraînement
clf.fit(X_train, y_train)

# Prédire les classes pour l'ensemble de test
y_pred = clf.predict(X_test)

# Calculer l'exactitude du classifieur
accuracy = accuracy_score(y_test, y_pred)
print("Exactitude :", accuracy)

# Extraire les règles de l'arbre de décision
rules = tree.export_text(clf, feature_names=list(X.columns))
print("Règles de l'arbre de décision :")
print(rules)

# Visualiser l'arbre de décision
short_feature_names = [f[:3] + '...' + f[-3:] for f in X.columns]
plt.figure(figsize=(14, 10))
tree.plot_tree(clf, feature_names=short_feature_names, class_names=["0", "1"], filled=True, fontsize=8)
plt.show()
àààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààààà