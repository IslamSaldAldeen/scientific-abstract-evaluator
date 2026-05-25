import json
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    cohen_kappa_score,
    confusion_matrix
)

def evaluate(predictions_file):
    # Load predictions
    with open(predictions_file, "r") as f:
        data = json.load(f)

    ground_truth = [entry["true_score"] for entry in data["predictions"]]
    predictions  = [entry["predicted_score"] for entry in data["predictions"]]

    # Metrics
    acc  = accuracy_score(ground_truth, predictions)
    f1   = f1_score(ground_truth, predictions, average="macro")
    mae  = mean_absolute_error(ground_truth, predictions)
    rmse = np.sqrt(np.mean((np.array(ground_truth) - np.array(predictions)) ** 2))
    qwk  = cohen_kappa_score(ground_truth, predictions, weights="quadratic")
    cks  = cohen_kappa_score(ground_truth, predictions)
    cm   = confusion_matrix(ground_truth, predictions, labels=[0, 1, 2, 3, 4])

    # Print results
    print("=" * 45)
    print(f"Model: {data['model']}")
    print("=" * 45)
    print(f"Accuracy:              {acc:.4f}")
    print(f"Macro F1:              {f1:.4f}")
    print(f"MAE:                   {mae:.4f}")
    print(f"RMSE:                  {rmse:.4f}")
    print(f"QWK:                   {qwk:.4f}")
    print(f"Cohen Kappa (linear):  {cks:.4f}")
    print("=" * 45)
    print("\nConfusion Matrix (rows=true, cols=predicted):")
    print("     0    1    2    3    4")
    for i, row in enumerate(cm):
        print(f"  {i}  {'  '.join(str(v).rjust(3) for v in row)}")

    # Save metrics back into the predictions file
    metrics = {
        "accuracy":            acc,
        "macro_f1":            f1,
        "mae":                 mae,
        "rmse":                rmse,
        "qwk":                 qwk,
        "cohen_kappa_linear":  cks,
        "confusion_matrix":    cm.tolist()
    }

    data["metrics"] = metrics

    with open(predictions_file, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nMetrics saved to {predictions_file}")

    return metrics


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python evaluate.py <predictions_file.json>")
        print("Example: python evaluate.py results/baseline_predictions.json")
    else:
        evaluate(sys.argv[1])