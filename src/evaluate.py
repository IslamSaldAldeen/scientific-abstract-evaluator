import json
import os
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    cohen_kappa_score,
    confusion_matrix,
    classification_report
)


def evaluate(predictions_file, metrics_output_file=None):
    # Load predictions
    with open(predictions_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    ground_truth = [entry["true_score"] for entry in data["predictions"]]
    predictions = [entry["predicted_score"] for entry in data["predictions"]]

    # Metrics
    acc = accuracy_score(ground_truth, predictions)
    f1 = f1_score(ground_truth, predictions, average="macro")
    mae = mean_absolute_error(ground_truth, predictions)
    rmse = np.sqrt(np.mean((np.array(ground_truth) - np.array(predictions)) ** 2))
    qwk = cohen_kappa_score(ground_truth, predictions, weights="quadratic")
    cks = cohen_kappa_score(ground_truth, predictions)

    cm = confusion_matrix(
        ground_truth,
        predictions,
        labels=[0, 1, 2, 3, 4]
    )

    report_text = classification_report(
        ground_truth,
        predictions,
        labels=[0, 1, 2, 3, 4],
        target_names=["Score 0", "Score 1", "Score 2", "Score 3", "Score 4"],
        zero_division=0
    )

    report_dict = classification_report(
        ground_truth,
        predictions,
        labels=[0, 1, 2, 3, 4],
        target_names=["Score 0", "Score 1", "Score 2", "Score 3", "Score 4"],
        zero_division=0,
        output_dict=True
    )

    metrics = {
        "model": data.get("model", "unknown"),
        "experiment": data.get("experiment", "unknown"),
        "predictions_file": predictions_file,
        "num_predictions": len(predictions),
        "accuracy": float(acc),
        "macro_f1": float(f1),
        "mae": float(mae),
        "rmse": float(rmse),
        "qwk": float(qwk),
        "cohen_kappa_linear": float(cks),
        "confusion_matrix": cm.tolist(),
        "classification_report": report_dict
    }

    # Print results
    print("=" * 45)
    print(f"Model:      {metrics['model']}")
    print(f"Experiment: {metrics['experiment']}")
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

    print("\nClassification Report:")
    print(report_text)

    # Keep metrics inside predictions file too
    data["metrics"] = metrics

    with open(predictions_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\nMetrics also saved inside {predictions_file}")

    # Save separate metrics file
    if metrics_output_file is not None:
        os.makedirs(os.path.dirname(metrics_output_file), exist_ok=True)

        with open(metrics_output_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        print(f"Separate metrics file saved to {metrics_output_file}")

    return metrics


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python src/evaluate.py <predictions_file.json> [metrics_output_file.json]")
        print("Example: python src/evaluate.py results/experiments/exp01_v2_lora/baseline_predictions.json results/experiments/exp01_v2_lora/baseline_metrics.json")
    else:
        predictions_file = sys.argv[1]
        metrics_output_file = sys.argv[2] if len(sys.argv) >= 3 else None
        evaluate(predictions_file, metrics_output_file)