from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
import joblib

PROCESSED_DIR = Path("data")

def load_processed_data() -> pd.DataFrame:
    path = PROCESSED_DIR / "processed_welfake.csv"
    return pd.read_csv(path)

def main():
    data = load_processed_data()
    data = data.dropna(subset=["content"])

    X = data["content"]
    y = data["label"]

    # Podział na dane treningowe i testowe (70/30)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Podzbiór 1% danych treningowych do optymalizacji
    _, X_subset, _, y_subset = train_test_split(
        X_train, y_train, test_size=0.01, random_state=42, stratify=y_train
    )

    # Wektoryzacja dopasowana na podzbiorze
    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
    X_subset_vec = vectorizer.fit_transform(X_subset)

    # Zakresy hiperparametrów
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Optymalizacja na podzbiorze
    grid_search = GridSearchCV(
        DecisionTreeClassifier(class_weight="balanced"),
        param_grid,
        cv=5,
        scoring="accuracy",
        verbose=1
    )
    grid_search.fit(X_subset_vec, y_subset)
    print("Najlepsze parametry:", grid_search.best_params_)

    X_train_vec = vectorizer.transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Trening na pełnym treningowym
    best_model = DecisionTreeClassifier(
        **grid_search.best_params_,
        class_weight="balanced"
    )
    best_model.fit(X_train_vec, y_train)

    # Ewaluacja na danych testowych
    y_pred = best_model.predict(X_test_vec)
    print("Dokładność (na danych testowych):", accuracy_score(y_test, y_pred))
    print("Raport klasyfikacji:\n", classification_report(y_test, y_pred))

    # Zapis modelu i wektoryzatora
    joblib.dump(best_model, PROCESSED_DIR / "decision_tree_model.pkl")
    joblib.dump(vectorizer, PROCESSED_DIR / "tfidf_vectorizer_dt.pkl")

    # Zapis hiperparametrów
    with open(PROCESSED_DIR / "decision_tree_params.txt", "w") as f:
        for key, val in grid_search.best_params_.items():
            f.write(f"{key}: {val}\n")

if __name__ == "__main__":
    main()
