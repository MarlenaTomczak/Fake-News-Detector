import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# Wczytaj dane
data = pd.read_csv('data/processed_welfake.csv')

# Podział na dane i etykiety
X = data['text']  # zakładam, że kolumna z tekstem nazywa się 'text'
y = data['label']

# Wybierz 10% danych do optymalizacji hiperparametrów
_, X_subset, _, y_subset = train_test_split(X, y, test_size=0.1, random_state=42)

# Podział na treningowe/testowe (70/30)
X_train, X_test, y_train, y_test = train_test_split(X_subset, y_subset, test_size=0.3, random_state=42)

# Wektoryzacja tekstu
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Optymalizacja hiperparametrów
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear']
}

grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5, scoring='accuracy', verbose=1)
grid_search.fit(X_train_vec, y_train)

print("Najlepsze parametry:", grid_search.best_params_)

# Trening na najlepszych parametrach
best_model = LogisticRegression(**grid_search.best_params_)
best_model.fit(X_train_vec, y_train)

# Ewaluacja modelu
y_pred = best_model.predict(X_test_vec)
print("Dokładność:", accuracy_score(y_test, y_pred))
print("Raport klasyfikacji:\n", classification_report(y_test, y_pred))

# Zapis modelu oraz wektoryzatora
joblib.dump(best_model, 'logistic_regression_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
