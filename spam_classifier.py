import argparse
import os
import pickle
import re
import string
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


STOPWORDS = set(ENGLISH_STOP_WORDS)
DEFAULT_MODEL_PATH = "best_spam_model.pkl"
ACTIVE_MODEL: Optional[Pipeline] = None


@dataclass
class ModelArtifacts:
    name: str
    pipeline: Pipeline
    metrics: Dict[str, object]


def preprocess_text(text: str) -> str:
    """
    Preprocess a single text message:
    1) Lowercase
    2) Remove punctuation
    3) Tokenize
    4) Remove stopwords
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)

    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text)

    tokens = text.split(" ")
    filtered_tokens = [token for token in tokens if token and token not in STOPWORDS]
    return " ".join(filtered_tokens)


def preprocess_corpus(messages: pd.Series) -> pd.Series:
    """Apply preprocessing to a full pandas Series of messages."""
    return messages.fillna("").apply(preprocess_text)


def build_pipeline(model) -> Pipeline:
    """Create a text-classification pipeline with TF-IDF + classifier."""
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer()),
            ("classifier", model),
        ]
    )


def evaluate_model(model: Pipeline, x_test: pd.Series, y_test: pd.Series) -> Dict[str, object]:
    """Evaluate a trained model and return key classification metrics."""
    predictions = model.predict(x_test)
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, pos_label=1, zero_division=0),
        "recall": recall_score(y_test, predictions, pos_label=1, zero_division=0),
        "f1": f1_score(y_test, predictions, pos_label=1, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, predictions),
    }
    return metrics


def train_models(
    x_train: pd.Series, y_train: pd.Series, x_test: pd.Series, y_test: pd.Series
) -> List[ModelArtifacts]:
    """Train required models and return their artifacts with evaluation metrics."""
    candidates = [
        ("Naive Bayes", MultinomialNB()),
        ("Logistic Regression", LogisticRegression(max_iter=1000)),
    ]
    artifacts: List[ModelArtifacts] = []

    for name, model in candidates:
        pipeline = build_pipeline(model)
        pipeline.fit(x_train, y_train)
        metrics = evaluate_model(pipeline, x_test, y_test)
        artifacts.append(ModelArtifacts(name=name, pipeline=pipeline, metrics=metrics))

    return artifacts


def print_comparison(artifacts: List[ModelArtifacts]) -> None:
    """Print side-by-side model comparison."""
    print("\n=== Model Comparison ===")
    for artifact in artifacts:
        m = artifact.metrics
        print(f"\nModel: {artifact.name}")
        print(f"Accuracy : {m['accuracy']:.4f}")
        print(f"Precision: {m['precision']:.4f}")
        print(f"Recall   : {m['recall']:.4f}")
        print(f"F1 Score : {m['f1']:.4f}")
        print("Confusion Matrix:")
        print(m["confusion_matrix"])


def choose_best_model(artifacts: List[ModelArtifacts]) -> ModelArtifacts:
    """Select best model by F1 score, then accuracy as tie-breaker."""
    return max(artifacts, key=lambda a: (a.metrics["f1"], a.metrics["accuracy"]))


def save_model(pipeline: Pipeline, model_path: str) -> None:
    """Persist model pipeline to disk using pickle."""
    with open(model_path, "wb") as file:
        pickle.dump(pipeline, file)


def load_model(model_path: str) -> Pipeline:
    """Load a previously persisted model pipeline from disk."""
    with open(model_path, "rb") as file:
        model = pickle.load(file)
    return model


def predict_spam(text: str) -> str:
    """
    Predict whether a message is spam.
    Returns: "Spam" or "Not Spam"
    """
    global ACTIVE_MODEL
    if not text or not text.strip():
        raise ValueError("Input text must be a non-empty string.")

    if ACTIVE_MODEL is None:
        if not os.path.exists(DEFAULT_MODEL_PATH):
            raise FileNotFoundError(
                f"No active model found and default model path does not exist: {DEFAULT_MODEL_PATH}"
            )
        ACTIVE_MODEL = load_model(DEFAULT_MODEL_PATH)

    cleaned_text = preprocess_text(text)
    prediction = ACTIVE_MODEL.predict([cleaned_text])[0]
    return "Spam" if int(prediction) == 1 else "Not Spam"


def load_and_validate_dataset(data_path: str) -> Tuple[pd.Series, pd.Series]:
    """Load dataset and validate required columns and label format."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path)
    required_columns = {"label", "message"}
    if not required_columns.issubset(df.columns):
        raise ValueError("Dataset must contain columns: label, message")

    df = df[["label", "message"]].dropna(subset=["label", "message"])
    if df.empty:
        raise ValueError("Dataset is empty after removing invalid rows.")

    # Accept common string labels and map to numeric
    label_map = {
        "spam": 1,
        "ham": 0,
        "not spam": 0,
        "0": 0,
        "1": 1,
    }

    def normalize_label(value) -> int:
        if isinstance(value, (int, float)) and value in (0, 1):
            return int(value)
        normalized = str(value).strip().lower()
        if normalized not in label_map:
            raise ValueError(
                "Invalid label encountered. Allowed labels are: spam/ham, not spam, 1/0."
            )
        return label_map[normalized]

    df["label"] = df["label"].apply(normalize_label)
    x = preprocess_corpus(df["message"])
    y = df["label"]

    if y.nunique() < 2:
        raise ValueError("Dataset must contain at least two classes (spam and not spam).")

    return x, y


def run_training(data_path: str, model_path: str) -> Pipeline:
    """End-to-end training, evaluation, best-model selection, and persistence."""
    global ACTIVE_MODEL, DEFAULT_MODEL_PATH
    x, y = load_and_validate_dataset(data_path)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    artifacts = train_models(x_train, y_train, x_test, y_test)
    print_comparison(artifacts)

    best = choose_best_model(artifacts)
    print(f"\nBest model selected: {best.name} (F1={best.metrics['f1']:.4f})")

    save_model(best.pipeline, model_path)
    ACTIVE_MODEL = best.pipeline
    DEFAULT_MODEL_PATH = model_path
    print(f"Saved best model to: {model_path}")
    return best.pipeline


def interactive_cli() -> None:
    """Simple CLI loop for testing custom emails."""
    print("\nInteractive mode enabled. Type 'exit' to quit.")
    while True:
        user_input = input("Enter email text: ").strip()
        if user_input.lower() == "exit":
            print("Exiting interactive mode.")
            break
        try:
            print(f"Prediction: {predict_spam(user_input)}")
        except ValueError as error:
            print(f"Input error: {error}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Email Spam Classification")
    parser.add_argument(
        "--data_path", type=str, default="spam.csv", help="Path to labeled dataset CSV."
    )
    parser.add_argument(
        "--model_path", type=str, default="best_spam_model.pkl", help="Path to save best model."
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Optional single text input to classify after training.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Enable interactive CLI input after training.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        run_training(args.data_path, args.model_path)

        if args.text is not None:
            prediction = predict_spam(args.text)
            print(f"\nInput Text: {args.text}")
            print(f"Prediction: {prediction}")

        if args.interactive:
            interactive_cli()

        return 0
    except (FileNotFoundError, ValueError, pd.errors.EmptyDataError) as error:
        print(f"Error: {error}")
        return 1
    except Exception as error:  # Catch-all for unexpected runtime errors
        print(f"Unexpected error: {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
