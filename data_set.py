from pathlib import Path
import kagglehub
import pandas as pd
import numpy as np
import re
import string
import argparse

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data")

def download_dataset() -> Path:

    dataset_path = kagglehub.dataset_download("saurabhshahane/fake-news-classification")
    csv_candidates = list(Path(dataset_path).rglob("*.csv"))
    if not csv_candidates:
        raise FileNotFoundError("No CSV file found in downloaded dataset.")
    raw_csv = csv_candidates[0]
    target_path = RAW_DIR / raw_csv.name
    if not target_path.exists():
        target_path.write_bytes(raw_csv.read_bytes())
    return target_path

def clean_text(text: str) -> str:
    """Basic text normalisation: lower‑casing, stripping HTML, numbers, punctuation
    and extra whitespace."""
    if pd.isna(text):
        return ""
    text = str(text).lower()  #małe litery
    text = re.sub(r'<.*?>', ' ', text)  #usuwanie HTML
    text = re.sub(r'http\S+|www\.\S+', ' ', text)  #usuwanie URL
    text = re.sub(r'[^a-z\s]', ' ', text)  #cyfry + interpunkcja
    text = re.sub(r'\s{2,}', ' ', text)  #nadmiar spacji
    return text.strip()  #obciecie spacji na końcach

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    df["title"] = df["title"].astype(str).apply(clean_text)
    df["text"] = df["text"].astype(str).apply(clean_text)
    df["content"] = (df["title"] + " " + df["text"]).str.strip()

    # Usuń brakujące etykiety
    df.dropna(subset=["label"], inplace=True)

    # Usuń puste teksty
    df = df[df["content"].str.strip().astype(bool)]

    # Usuń duplikaty
    df.drop_duplicates(subset=["content"], inplace=True)

    return df[["content", "label"]]


def perform_eda(df: pd.DataFrame) -> None:
    print("\nEksploracyjna Analiza Danych (EDA)")
    print(
        "Zbiór danych to zbiór wiadomości zawierający artykuły oznaczone jako prawdziwe (1) lub fałszywe (0).\n"
    )
    print(f"\nLiczba wierszy : {df.shape[0]}")
    print(f"Liczba kolumn : {df.shape[1]}")

    print("\nRozkład klas:")
    label_counts = df["label"].value_counts()
    print(f"- Fałszywe wiadomości (label = 0): {label_counts.get(0, 0)}")
    print(f"- Prawdziwe wiadomości (label = 1): {label_counts.get(1, 0)}")

    print("\nLiczba brakujących wartości  w każdej kolumnie:")
    missing = df.isna().sum()
    for col, count in missing.items():
        print(f"- {col}: {count}")

    avg_words = df["content"].dropna().str.split().apply(len).mean().round(2)
    print(f"\nŚrednia liczba słów w jednym artykule: {avg_words}")

    print("\n")


def save_processed(df: pd.DataFrame, path: Path):
    df.to_csv(path, index=False)
    print(f"Processed dataset saved to {path.resolve()}")

def main(args=None):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=Path, default=PROCESSED_DIR / "processed_welfake.csv",
                        help="Where to store cleaned dataset")
    parsed = parser.parse_args(args=args)
    raw_csv = download_dataset()
    raw_df = pd.read_csv(raw_csv)
    processed_df = preprocess(raw_df)
    perform_eda(processed_df)
    save_processed(processed_df, parsed.save_path)

if __name__ == "__main__":
    main()