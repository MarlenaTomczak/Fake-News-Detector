import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

PROCESSED_DIR = Path("data")

data = pd.read_csv('data/processed_welfake.csv')
data = data.dropna(subset=["content"])

X = data["content"]
y = data["label"]

# Podział na dane treningowe i testowe (70/30)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Podzbiór 1% danych treningowych do optymalizacji hiperparametrów
_, X_subset, _, y_subset = train_test_split(
    X_train, y_train, test_size=0.01, random_state=42, stratify=y_train
)

# Wektoryzacja dopasowana na podzbiorze (X_subset)
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_subset_vec = vectorizer.fit_transform(X_subset)

# Zakresy hiperparametrów
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear']
}

# Optymalizacja na 1% danych
grid_search = GridSearchCV(
    LogisticRegression(),
    param_grid,
    cv=5,
    scoring='accuracy',
    verbose=1
)
grid_search.fit(X_subset_vec, y_subset)

print("Najlepsze parametry:", grid_search.best_params_)

# wektoryzacja
X_train_vec = vectorizer.transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Trening  modelu na pełnym zbiorze treningowym
best_model = LogisticRegression(**grid_search.best_params_)
best_model.fit(X_train_vec, y_train)

# Ewaluacja na danych testowych
y_pred = best_model.predict(X_test_vec)
print("Dokładność (na danych testowych):", accuracy_score(y_test, y_pred))
print("Raport klasyfikacji:\n", classification_report(y_test, y_pred))

# Zapis modelu i wektoryzatora
joblib.dump(best_model, PROCESSED_DIR / "logistic_regression_model.pkl")
joblib.dump(vectorizer, PROCESSED_DIR / "tfidf_vectorizer.pkl")

# Zapis hiperparametrów
with open(PROCESSED_DIR / "logistic_regression_params.txt", "w") as f:
    for key, val in grid_search.best_params_.items():
        f.write(f"{key}: {val}\n")
