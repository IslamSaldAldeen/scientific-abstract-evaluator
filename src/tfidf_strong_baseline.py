import json, os, re, math
import numpy as np
from collections import Counter
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, cohen_kappa_score, confusion_matrix, classification_report
from sklearn.base import clone

OUT_DIR = "results/experiments/tfidf_strong_baseline"
os.makedirs(OUT_DIR, exist_ok=True)

TRAIN = "data/splits/train.json"
VAL = "data/splits/val.json"
TEST = "data/splits/test.json"

def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

train = load(TRAIN)
val = load(VAL)
test = load(TEST)

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
    # submission is the main signal; reference overlap is added as numeric features
    return [e["submission"] for e in entries]

def numeric_features(entries):
    rows = []
    for e in entries:
        sub = e["submission"]
        ref = e["reference"]
        stoks = tokenize(sub)
        rtoks = set(tokenize(ref))

        n = max(len(stoks), 1)
        unique = set(stoks)
        overlap = len(unique & rtoks) / max(len(unique), 1)

        digits = len(re.findall(r"\d+", sub))
        percents = sub.count("%")
        parens = sub.count("(") + sub.count(")")
        semicolons = sub.count(";")
        commas = sub.count(",")

        generic = sum(1 for t in stoks if t in generic_words) / n
        fabricated = sum(1 for t in stoks if t in fabricated_words)

        avg_word_len = sum(len(t) for t in stoks) / n
        long_words = sum(1 for t in stoks if len(t) >= 9) / n

        rows.append([
            len(stoks),
            len(unique),
            len(unique) / n,
            overlap,
            digits,
            percents,
            parens,
            semicolons,
            commas,
            generic,
            fabricated,
            avg_word_len,
            long_words,
        ])
    return np.array(rows, dtype=float)

def labels(entries):
    return np.array([int(e["score"]) for e in entries])

Xtr_text, Xv_text, Xt_text = make_text(train), make_text(val), make_text(test)
ytr, yv, yt = labels(train), labels(val), labels(test)

word_vec = TfidfVectorizer(lowercase=True, ngram_range=(1,2), max_features=30000, min_df=1)
char_vec = TfidfVectorizer(lowercase=True, analyzer="char_wb", ngram_range=(3,5), max_features=30000, min_df=1)

Xtr_word = word_vec.fit_transform(Xtr_text)
Xv_word = word_vec.transform(Xv_text)

Xtr_char = char_vec.fit_transform(Xtr_text)
Xv_char = char_vec.transform(Xv_text)

scaler = StandardScaler()
Xtr_num = scaler.fit_transform(numeric_features(train))
Xv_num = scaler.transform(numeric_features(val))

Xtr = hstack([Xtr_word, Xtr_char, csr_matrix(Xtr_num)])
Xv = hstack([Xv_word, Xv_char, csr_matrix(Xv_num)])

candidates = []
for C in [0.05, 0.1, 0.25, 0.5, 1.0, 2.0]:
    candidates.append((f"LinearSVC_C{C}", LinearSVC(C=C, class_weight="balanced", max_iter=20000)))
# LogisticRegression removed because this sklearn/liblinear setup does not support multiclass directly.
candidates.append(("RidgeClassifier", RidgeClassifier(class_weight="balanced")))

best = None

for name, model in candidates:
    m = clone(model)
    m.fit(Xtr, ytr)
    pred = m.predict(Xv)
    acc = accuracy_score(yv, pred)
    mf1 = f1_score(yv, pred, average="macro", zero_division=0)
    qwk = cohen_kappa_score(yv, pred, weights="quadratic")
    score = (mf1, qwk, acc)
    print(name, "VAL acc=", round(acc,4), "macro_f1=", round(mf1,4), "qwk=", round(qwk,4), "dist=", Counter(pred))
    if best is None or score > best[0]:
        best = (score, name, model)

print("\nBest model on validation:", best[1])

# Final train on train + val
trainval = train + val
ytv = labels(trainval)
Xtv_text = make_text(trainval)

word_vec = TfidfVectorizer(lowercase=True, ngram_range=(1,2), max_features=30000, min_df=1)
char_vec = TfidfVectorizer(lowercase=True, analyzer="char_wb", ngram_range=(3,5), max_features=30000, min_df=1)

Xtv_word = word_vec.fit_transform(Xtv_text)
Xt_word = word_vec.transform(Xt_text)

Xtv_char = char_vec.fit_transform(Xtv_text)
Xt_char = char_vec.transform(Xt_text)

scaler = StandardScaler()
Xtv_num = scaler.fit_transform(numeric_features(trainval))
Xt_num = scaler.transform(numeric_features(test))

Xtv = hstack([Xtv_word, Xtv_char, csr_matrix(Xtv_num)])
Xt = hstack([Xt_word, Xt_char, csr_matrix(Xt_num)])

final_model = clone(best[2])
final_model.fit(Xtv, ytv)
yp = final_model.predict(Xt)

predictions = []
for e, p in zip(test, yp):
    predictions.append({
        "reference": e["reference"],
        "submission": e["submission"],
        "true_score": int(e["score"]),
        "predicted_score": int(p),
        "gold_rationale": e.get("rationale", "")
    })

metrics = {
    "model": best[1],
    "experiment": "tfidf_strong_baseline",
    "num_predictions": len(test),
    "accuracy": accuracy_score(yt, yp),
    "macro_f1": f1_score(yt, yp, average="macro", zero_division=0),
    "mae": mean_absolute_error(yt, yp),
    "rmse": math.sqrt(mean_squared_error(yt, yp)),
    "qwk": cohen_kappa_score(yt, yp, weights="quadratic"),
    "cohen_kappa_linear": cohen_kappa_score(yt, yp, weights="linear"),
    "confusion_matrix": confusion_matrix(yt, yp, labels=[0,1,2,3,4]).tolist(),
    "classification_report": classification_report(
        yt, yp,
        labels=[0,1,2,3,4],
        target_names=[f"Score {i}" for i in range(5)],
        output_dict=True,
        zero_division=0
    )
}

with open(f"{OUT_DIR}/finetuned_predictions.json", "w", encoding="utf-8") as f:
    json.dump({"experiment": "tfidf_strong_baseline", "model": best[1], "predictions": predictions, "metrics": metrics}, f, indent=2, ensure_ascii=False)

with open(f"{OUT_DIR}/finetuned_metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2, ensure_ascii=False)

print("\nTEST METRICS")
print(json.dumps(metrics, indent=2))
print("\nPrediction distribution:", Counter(yp))
