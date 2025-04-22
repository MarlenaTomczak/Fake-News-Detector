from pathlib import Path
import pandas as pd
import joblib
from data_set import perform_eda
from decision_tree import load_processed_data

PROCESSED_DIR = Path("data")


def load_model(model_name: str):
    model = joblib.load(PROCESSED_DIR / model_name)
    vectorizer_name = "tfidf_vectorizer.pkl" if "logistic" in model_name else "tfidf_vectorizer_dt.pkl"
    vectorizer = joblib.load(PROCESSED_DIR / vectorizer_name)
    return model, vectorizer


def classify_text(model, vectorizer, text: str):
    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)[0]
    return "Wiadomość prawdziwa" if prediction == 1 else "Wiadomość fałszywa"


def main():
    data = load_processed_data()

    print("\n===== Eksploracyjna Analiza Danych (EDA) =====")
    perform_eda(data)

    print("\n===== Podsumowanie ewaluacji modeli =====")
    choice = input("\nWybierz model (1 - Regresja logistyczna, 2 - Drzewo decyzyjne): ")

    if choice == "1":
        model_name = "logistic_regression_model.pkl"
    elif choice == "2":
        model_name = "decision_tree_model.pkl"
    else:
        print("Nieprawidłowy wybór. Zakończono.")
        return

    model, vectorizer = load_model(model_name)

    user_text = input("\nWklej tekst, który chcesz sklasyfikować:\n")

    result = classify_text(model, vectorizer, user_text)
    print(f"\nWynik klasyfikacji: {result}")


if __name__ == "__main__":
    main()
