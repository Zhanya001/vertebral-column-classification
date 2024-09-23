from ucimlrepo import fetch_ucirepo
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, f1_score, classification_report
from scipy.stats import randint
import matplotlib.pyplot as plt

# 1. Cargar el dataset
vertebral_column = fetch_ucirepo(id=212)
X = vertebral_column.data.features
y = vertebral_column.data.targets

# 2. Preprocesar los datos: dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Escalar los datos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Buscar hiperparámetros usando GridSearchCV
param_grid = {'max_depth': [None, 4, 6, 8, 10], 'min_samples_split': [2, 3, 4], 'min_samples_leaf': [1, 2, 3]}
grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)
print("Mejores hiperparámetros (GridSearchCV):", grid_search.best_params_)

# 5. Poda del árbol usando `max_depth` y `min_samples_leaf`
depth_scores = []
for depth in [4, 6, 8]:
    clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    depth_scores.append(accuracy)
    print(f"Accuracy (max_depth={depth}): {accuracy}")

min_samples_scores = []
for min_samples in [2, 4, 6]:
    clf = DecisionTreeClassifier(min_samples_leaf=min_samples, random_state=42)
    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    min_samples_scores.append(accuracy)
    print(f"Accuracy (min_samples_leaf={min_samples}): {accuracy}")

# 6. RandomizedSearchCV para hiperparámetros adicionales
param_dist = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 4, 6, 8],
    'min_samples_split': randint(2, 6),
    'min_samples_leaf': randint(1, 6)
}
random_search = RandomizedSearchCV(DecisionTreeClassifier(random_state=42), param_distributions=param_dist, n_iter=100, cv=5, scoring='accuracy')
random_search.fit(X_train_scaled, y_train)
print("Mejores hiperparámetros (RandomizedSearchCV):", random_search.best_params_)
print("Mejor accuracy (RandomizedSearchCV):", random_search.best_score_)

# 7. Evaluar el modelo final
clf_final = random_search.best_estimator_
y_pred_final = clf_final.predict(X_test_scaled)
print("\nEvaluación final:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_final)}")
print(classification_report(y_test, y_pred_final))

# 8. Graficar los resultados

# Gráfico de los resultados de poda por max_depth
plt.figure(figsize=(10,5))
plt.bar([4, 6, 8], depth_scores, color='lightblue')
plt.title("Accuracy vs max_depth")
plt.xlabel("max_depth")
plt.ylabel("Accuracy")
plt.show()

# Gráfico de los resultados de poda por min_samples_leaf
plt.figure(figsize=(10,5))
plt.bar([2, 4, 6], min_samples_scores, color='lightgreen')
plt.title("Accuracy vs min_samples_leaf")
plt.xlabel("min_samples_leaf")
plt.ylabel("Accuracy")
plt.show()

# Gráfico de comparación de GridSearchCV y RandomizedSearchCV
labels = ['GridSearchCV', 'RandomizedSearchCV']
accuracy_scores = [grid_search.best_score_, random_search.best_score_]

x = range(len(labels))
width = 0.35

fig, ax = plt.subplots()
ax.bar(x, accuracy_scores, width, label='Accuracy', color='lightblue')

# Configuración del gráfico
ax.set_xlabel('Métodos de Búsqueda')
ax.set_ylabel('Accuracy')
ax.set_title('Comparación de Accuracy entre GridSearchCV y RandomizedSearchCV')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

plt.show()

