import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn import tree
import matplotlib.pyplot as plt

# Chargement du fichier Excel dans un DataFrame
data= pd.read_csv('golf.csv')


X = data.iloc[:, :-1]
y = data.iloc[:, -1]

X = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

tree_model = DecisionTreeClassifier(max_depth=4, random_state=0)

cv = StratifiedKFold(n_splits=5)
scores = cross_val_score(tree_model, X, y, cv=cv, scoring='accuracy')

print("Scores de validation croisée :", scores)
print("Score moyen de validation croisée :", scores.mean())

tree_model.fit(X_train, y_train)

y_pred = tree_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Précision du modèle :", accuracy)

# Extraire les règles de l'arbre de décision
rules = tree.export_text(tree_model, feature_names=list(X.columns))
print("Règles de l'arbre de décision :")
print(rules)

# Visualiser l'arbre de décision
short_feature_names = [f[:3] + '...' + f[-3:] for f in X.columns]
plt.figure(figsize=(12,8))
tree.plot_tree(tree_model, feature_names=short_feature_names, class_names=["Jouer", "NePasJouer"], filled=True, fontsize=8)
plt.show()
