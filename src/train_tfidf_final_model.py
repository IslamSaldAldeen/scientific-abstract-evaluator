import json
import os
import re
import math
import joblib
import numpy as np

from collections import Counter
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    cohen_kappa_score,
    confusion_matrix,
    classification_report,
)
from sklearn.base import clone


EXPERIMENT_NAME = "tfidf_strong_baseline"
OUT_DIR = "results/experiments/tfidf_strong_baseline"
MODEL_DIR = "models/experiments/tfidf_strong_baseline"

TRAIN_FILE = "data/splits/train.json"
VAL_FILE = "data/splits/val.json"
TEST_FILE = "data/splits/test.json"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def tokenize(text):
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


generic_words = {
    "important", "significant", "various", "general", "potential", "suggest",
    "suggests", "findings", "understanding", "explores", "investigated",
    "researchers", "future", "further", "overall", "beneficial", "crucial",
    "complex", "challenges", "improve", "improving"
}

fabricated_words = {
    "mystical", "ancient", "conspiracy", "kale", "cure", "universal",
    "revolutionary", "groundbreaking", "eradicate", "definitively",
    "spiritual", "herbal", "completely", "all", "immunoboost"
}


def make_text(entries):
    return [entry["submission"] for entry in entries]


def labels(entries):
    return np.array([int(entry["score"]) for entry in entries])


def numeric_features(entries):
    rows = []

    for entry in entries:
        submission = entry["submission"]
        reference = entry["reference"]

        submission_tokens = tokenize(submission)
        reference_tokens = set(tokenize(reference))

        n_tokens = max(len(submission_tokens), 1)
        unique_tokens = set(submission_tokens)

        overlap = len(unique_tokens & reference_tokens) / max(len(unique_tokens), 1)

        digits = len(re.findall(r"\d+", submission))
        percents = submission.count("%")
        parens = submission.count("(") + submission.count(")")
        semicolons = submission.count(";")
        commas = submission.count(",")

        generic_ratio = sum(1 for token in submission_tokens if token in generic_words) / n_tokens
        fabricated_count = sum(1 for token in submission_tokens if token in fabricated_words)

        avg_word_len = sum(len(token) for token in submission_tokens) / n_tokens
        long_word_ratio = sum(1 for token in submission_tokens if len(token) >= 9) / n_tokens

        rows.append([
            len(submission_tokens),
            len(unique_tokens),
            len(unique_tokens) / n_tokens,
            overlap,
            digits,
            percents,
            parens,
            semicolons,
            commas,
            generic_ratio,
            fabricated_count,
            avg_word_len,
            long_word_ratio,
        ])

    return np.array(rows, dtype=float)


def build_features(entries, word_vectorizer, char_vectorizer, scaler, fit=False):
    texts = make_text(entries)

    if fit:
        word_features = word_vectorizer.fit_transform(texts)
        char_features = char_vectorizer.fit_transform(texts)
        numeric = scaler.fit_transform(numeric_features(entries))
    else:
        word_features = word_vectorizer.transform(texts)
        char_features = char_vectorizer.transform(texts)
        numeric = scaler.transform(numeric_features(entries))

    return hstack([word_features, char_features, csr_matrix(numeric)])


def compute_metrics(y_true, y_pred, model_name):
    return {
        "model": model_name,
        "experiment": EXPERIMENT_NAME,
        "num_predictions": int(len(y_true)),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": math.sqrt(mean_squared_error(y_true, y_pred)),
        "qwk": cohen_kappa_score(y_true, y_pred, weights="quadratic"),
        "cohen_kappa_linear": cohen_kappa_score(y_true, y_pred, weights="linear"),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4]).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=[0, 1, 2, 3, 4],
            target_names=[f"Score {i}" for i in range(5)],
            output_dict=True,
            zero_division=0,
        ),
    }


def main():
    train = load_json(TRAIN_FILE)
    val = load_json(VAL_FILE)
    test = load_json(TEST_FILE)

    y_train = labels(train)
    y_val = labels(val)
    y_test = labels(test)

    word_vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        max_features=30000,
        min_df=1,
    )

    char_vectorizer = TfidfVectorizer(
        lowercase=True,
        analyzer="char_wb",
        ngram_range=(3, 5),
        max_features=30000,
        min_df=1,
    )

    scaler = StandardScaler()

    X_train = build_features(train, word_vectorizer, char_vectorizer, scaler, fit=True)
    X_val = build_features(val, word_vectorizer, char_vectorizer, scaler, fit=False)

    candidates = []

    for C in [0.05, 0.1, 0.25, 0.5, 1.0, 2.0]:
        candidates.append((
            f"LinearSVC_C{C}",
            LinearSVC(C=C, class_weight="balanced", max_iter=20000)
        ))

    candidates.append((
        "RidgeClassifier",
        RidgeClassifier(class_weight="balanced")
    ))

    best = None

    print("Selecting best model on validation set...")

    for name, model in candidates:
        model_copy = clone(model)
        model_copy.fit(X_train, y_train)

        val_pred = model_copy.predict(X_val)

        acc = accuracy_score(y_val, val_pred)
        macro_f1 = f1_score(y_val, val_pred, average="macro", zero_division=0)
        qwk = cohen_kappa_score(y_val, val_pred, weights="quadratic")

        score_tuple = (macro_f1, qwk, acc)

        print(
            name,
            "VAL accuracy=", round(acc, 4),
            "macro_f1=", round(macro_f1, 4),
            "qwk=", round(qwk, 4),
            "dist=", Counter(int(x) for x in val_pred)
        )

        if best is None or score_tuple > best[0]:
            best = (score_tuple, name, model)

    best_name = best[1]
    best_model_template = best[2]

    print("\nBest validation model:", best_name)

    trainval = train + val
    y_trainval = labels(trainval)

    final_word_vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        max_features=30000,
        min_df=1,
    )

    final_char_vectorizer = TfidfVectorizer(
        lowercase=True,
        analyzer="char_wb",
        ngram_range=(3, 5),
        max_features=30000,
        min_df=1,
    )

    final_scaler = StandardScaler()

    X_trainval = build_features(
        trainval,
        final_word_vectorizer,
        final_char_vectorizer,
        final_scaler,
        fit=True,
    )

    X_test = build_features(
        test,
        final_word_vectorizer,
        final_char_vectorizer,
        final_scaler,
        fit=False,
    )

    final_model = clone(best_model_template)
    final_model.fit(X_trainval, y_trainval)

    test_pred = final_model.predict(X_test)

    predictions = []

    for entry, pred in zip(test, test_pred):
        predictions.append({
            "reference": entry["reference"],
            "submission": entry["submission"],
            "true_score": int(entry["score"]),
            "predicted_score": int(pred),
            "gold_rationale": entry.get("rationale", "")
        })

    metrics = compute_metrics(y_test, test_pred, best_name)

    artifact = {
        "experiment": EXPERIMENT_NAME,
        "model_name": best_name,
        "model": final_model,
        "word_vectorizer": final_word_vectorizer,
        "char_vectorizer": final_char_vectorizer,
        "scaler": final_scaler,
        "generic_words": generic_words,
        "fabricated_words": fabricated_words,
    }

    model_path = os.path.join(MODEL_DIR, "tfidf_linear_svc.joblib")
    joblib.dump(artifact, model_path)

    predictions_path = os.path.join(OUT_DIR, "finetuned_predictions.json")
    metrics_path = os.path.join(OUT_DIR, "finetuned_metrics.json")

    with open(predictions_path, "w", encoding="utf-8") as f:
        json.dump({
            "experiment": EXPERIMENT_NAME,
            "model": best_name,
            "predictions": predictions,
            "metrics": metrics,
        }, f, indent=2, ensure_ascii=False)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print("\nSaved model artifact to:", model_path)
    print("Saved predictions to:", predictions_path)
    print("Saved metrics to:", metrics_path)

    print("\nTEST METRICS")
    print(json.dumps(metrics, indent=2))

    print("\nPrediction distribution:", Counter(int(x) for x in test_pred))


if __name__ == "__main__":
    main()
