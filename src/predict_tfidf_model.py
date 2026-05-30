import argparse
import json
import re
import joblib
import numpy as np

from scipy.sparse import hstack, csr_matrix


MODEL_PATH = "models/experiments/tfidf_strong_baseline/tfidf_linear_svc.joblib"


def tokenize(text):
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def numeric_features(reference, submission, generic_words, fabricated_words):
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

    return np.array([[
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
    ]], dtype=float)


def predict(reference, submission, model_path=MODEL_PATH):
    artifact = joblib.load(model_path)

    model = artifact["model"]
    word_vectorizer = artifact["word_vectorizer"]
    char_vectorizer = artifact["char_vectorizer"]
    scaler = artifact["scaler"]
    generic_words = artifact["generic_words"]
    fabricated_words = artifact["fabricated_words"]

    word_features = word_vectorizer.transform([submission])
    char_features = char_vectorizer.transform([submission])

    num = numeric_features(reference, submission, generic_words, fabricated_words)
    num_scaled = scaler.transform(num)

    X = hstack([word_features, char_features, csr_matrix(num_scaled)])

    score = int(model.predict(X)[0])

    return {
        "score": score,
        "model": artifact.get("model_name", "TF-IDF + LinearSVC"),
        "experiment": artifact.get("experiment", "tfidf_strong_baseline"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=str)
    parser.add_argument("--submission", type=str)
    parser.add_argument("--reference_file", type=str)
    parser.add_argument("--submission_file", type=str)
    parser.add_argument("--model_path", type=str, default=MODEL_PATH)

    args = parser.parse_args()

    if args.reference_file:
        reference = open(args.reference_file, encoding="utf-8").read()
    elif args.reference:
        reference = args.reference
    else:
        raise ValueError("Provide --reference or --reference_file")

    if args.submission_file:
        submission = open(args.submission_file, encoding="utf-8").read()
    elif args.submission:
        submission = args.submission
    else:
        raise ValueError("Provide --submission or --submission_file")

    result = predict(reference, submission, args.model_path)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
