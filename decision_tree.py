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

    X = data["content"]
    y = data["label"]

    _, X_subset, _, y_subset = train_test_split(X, y, test_size=0.1, random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(X_subset, y_subset, test_size=0.3, random_state=42)

    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    grid_search = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5, scoring="accuracy", verbose=1)
    grid_search.fit(X_train_vec, y_train)

    print("Najlepsze parametry:", grid_search.best_params_)

    best_model = DecisionTreeClassifier(**grid_search.best_params_)
    best_model.fit(X_train_vec, y_train)

    y_pred = best_model.predict(X_test_vec)
    print("Dokładność:", accuracy_score(y_test, y_pred))
    print("Raport klasyfikacji:\n", classification_report(y_test, y_pred))

    joblib.dump(best_model, PROCESSED_DIR / "decision_tree_model.pkl")
    joblib.dump(vectorizer, PROCESSED_DIR / "tfidf_vectorizer_dt.pkl")


if __name__ == "__main__":
    main()
