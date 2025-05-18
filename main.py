from pathlib import Path
import joblib
from data_set import perform_eda
from decision_tree import load_processed_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)

PROCESSED_DIR = Path("data")

def load_model(model_name: str):
    model_path = PROCESSED_DIR / model_name

    if "logistic" in model_name:
        vectorizer_name = "tfidf_vectorizer.pkl"
    else:
        vectorizer_name = "tfidf_vectorizer_dt.pkl"

    vectorizer_path = PROCESSED_DIR / vectorizer_name

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

def classify_text(model, vectorizer, text: str):
    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)[0]
    return "Wiadomość prawdziwa" if prediction == 1 else "Wiadomość fałszywa"

def evaluate_model(model, vectorizer, X, y, model_name):
    X_vec = vectorizer.transform(X)
    y_pred = model.predict(X_vec)

    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred)
    rec = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)

    print(f"\n>>> Ewaluacja dla modelu: {model_name}")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")

    return {
        "Model": model_name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-score": f1
    }


def explain_param(name, value):
    explanations = {
        "C": f"  - C = {value}     Określa, jak bardzo model ma się dostosować do danych. Im wyższe C, tym mniej karze się za złożone rozwiązania.",
        "penalty": f"  - penalty = {value}     Rodzaj uproszczenia modelu. 'l1' i 'l2' pomagają unikać przeuczenia, ale na różne sposoby.",
        "solver": f"  - solver = {value}     Technika, którą model używa, żeby nauczyć się jak najlepiej dopasować dane.",
        "criterion": f"  - criterion = {value}     Sposób, w jaki drzewo decyduje, gdzie się rozdzielić.",
        "max_depth": f"  - max_depth = {value}     Jak głębokie może być drzewo – czyli ile kroków może wykonać, by dojść do decyzji.",
        "min_samples_split": f"  - min_samples_split = {value}     Minimalna liczba przykładów, żeby rozdzielić dane na kolejne gałęzie.",
        "min_samples_leaf": f"  - min_samples_leaf = {value}     Minimalna liczba przykładów w końcowej części drzewa (liściu)."
    }
    return explanations.get(name, f"  - {name} = {value}     (brak opisu)")



def show_params(path, model_name):
    print(f"> {model_name}")
    if path.exists():
        with open(path, "r") as f:
            for line in f:
                key, val = line.strip().split(": ")
                print(explain_param(key, val))
    else:
        print("- (Brak zapisanych parametrów)")
    print()

def main():
    data = load_processed_data()
    data = data.dropna(subset=["content"])
    perform_eda(data)

    print("\nPodsumowanie ewaluacji modeli")
    X_full = data["content"]
    y_full = data["label"]

    # Subset 20% danych do szybkiej ewaluacji
    _, X_subset, _, y_subset = train_test_split(
        X_full, y_full, test_size=0.2, random_state=42, stratify=y_full
    )

    # Ewaluacja modeli
    results = []
    logistic_model, logistic_vect = load_model("logistic_regression_model.pkl")
    results.append(
        evaluate_model(logistic_model, logistic_vect, X_subset, y_subset, "Regresja logistyczna")
    )

    tree_model, tree_vect = load_model("decision_tree_model.pkl")
    results.append(
        evaluate_model(tree_model, tree_vect, X_subset, y_subset, "Drzewo decyzyjne")
    )

    # Porównanie metryk
    best_accuracy = max(r["Accuracy"] for r in results)
    best_precision = max(r["Precision"] for r in results)
    best_recall = max(r["Recall"] for r in results)
    best_f1 = max(r["F1-score"] for r in results)

    print("\nPorównanie modeli")
    print(f"{'Model':<25} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-score':<12}")
    for r in results:
        acc = f"{r['Accuracy']:.4f}" + (" *" if r["Accuracy"] == best_accuracy else "")
        prec = f"{r['Precision']:.4f}" + (" *" if r["Precision"] == best_precision else "")
        rec = f"{r['Recall']:.4f}" + (" *" if r["Recall"] == best_recall else "")
        f1 = f"{r['F1-score']:.4f}" + (" *" if r["F1-score"] == best_f1 else "")
        print(f"{r['Model']:<25} {acc:<12} {prec:<12} {rec:<12} {f1:<12}")

    print("\nWyjaśnienie metryk:")
    print("Accuracy  : Odsetek poprawnych predykcji względem wszystkich przypadków.")
    print("Precision : Spośród wiadomości zaklasyfikowanych jako prawdziwe, ile faktycznie było prawdziwych.")
    print("Recall    : Spośród wszystkich prawdziwych wiadomości, ile zostało poprawnie wykrytych.")
    print("F1-score  : Średnia harmoniczna( precision i recall; inaczej: 2 * (precision * recall) / (precision + recall)")
    print("*         : Najlepszy wynik w danej kolumnie.")

    print("\nPrzegląd użytych hiperparametrów:")
    show_params(PROCESSED_DIR / "logistic_regression_params.txt", "Regresja logistyczna")
    show_params(PROCESSED_DIR / "decision_tree_params.txt", "Drzewo decyzyjne")

    # Klasyfikacja
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
