import argparse
import json
import os
import re
import yaml
import torch
import unsloth

from unsloth import FastLanguageModel


# ============================================================
# Load Config
# ============================================================
def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    type=str,
    default="configs/experiments/exp01_v2_lora.yaml",
    help="Path to experiment config YAML file"
)
args = parser.parse_args()

config = load_config(args.config)

experiment_name = config["experiment"]["name"]

MODEL_DIR = config["outputs"]["model_dir"]
TEST_FILE = config["dataset"]["test_path"]
OUTPUT_FILE = config["outputs"]["finetuned_predictions"]

METRICS_FILE = config["outputs"]["finetuned_metrics"]

MAX_SEQ_LENGTH = config["model"]["max_seq_length"]
LOAD_IN_4BIT = config["model"]["load_in_4bit"]

MAX_NEW_TOKENS = config["inference"]["max_new_tokens"]
DO_SAMPLE = config["inference"]["do_sample"]


# ============================================================
# Load Fine-tuned Model
# ============================================================
print("=" * 60)
print(f"Experiment: {experiment_name}")
print(f"Config:     {args.config}")
print(f"Model dir:  {MODEL_DIR}")
print(f"Test file:  {TEST_FILE}")
print(f"Output:     {OUTPUT_FILE}")
print(f"Metrics:    {METRICS_FILE}")
print("=" * 60)

print("Loading fine-tuned model...")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_DIR,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=LOAD_IN_4BIT,
)

FastLanguageModel.for_inference(model)

print("Model loaded!")


# ============================================================
# Load Test Set
# ============================================================
with open(TEST_FILE, "r", encoding="utf-8") as f:
    test_data = json.load(f)

print(f"Test entries: {len(test_data)}")


# ============================================================
# Prompt Function
# ============================================================
def build_inference_prompt(entry):
    rubric_text = "\n".join([f"- {k}: {v}" for k, v in entry["rubric"].items()])

    prompt = f"""[INST] You are a scientific abstract quality evaluator.

Task: {entry["task"]}

Paper Content:
{entry["reference"]}

Abstract to Evaluate:
{entry["submission"]}

Rubric:
{rubric_text}


Evaluate the abstract and respond with ONLY this format:
Score: <number from 0 to 4>
Rationale: <one sentence explanation>ve Score 0 when it is unrelated, contradictory, fabricated, or contains serious unsupported claims.

[/INST]"""

    return prompt


# ============================================================
# Inference Helpers
# ============================================================
def extract_score(response):
    match = re.search(r"Score:\s*([0-4])", response)
    if match:
        return int(match.group(1))
    return None


def get_prediction(entry):
    prompt = build_inference_prompt(entry)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_SEQ_LENGTH
    ).to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=DO_SAMPLE,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )

    score = extract_score(response.strip())

    return score, response.strip()


# ============================================================
# Run on Test Set
# ============================================================
results = []

for i, entry in enumerate(test_data):
    print(f"Processing {i + 1}/{len(test_data)}...")

    score, model_rationale = get_prediction(entry)

    if score is None:
        print("  Warning: Could not parse score, defaulting to 2")
        score = 2

    results.append({
        "reference": entry["reference"],
        "submission": entry["submission"],
        "true_score": entry["score"],
        "predicted_score": score,
        "gold_rationale": entry.get("rationale", "No gold rationale found in test entry"),
        "model_rationale": model_rationale
    })

    print(f"  True: {entry['score']} | Predicted: {score}")


# ============================================================
# Save Results
# ============================================================
finetuned_results = {
    "experiment": experiment_name,
    "model": "Mistral-7B-Instruct-v0.2 (fine-tuned)",
    "config": args.config,
    "predictions": results
}

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(finetuned_results, f, indent=2, ensure_ascii=False)

print(f"Saved to {OUTPUT_FILE}")


# ============================================================
# Run Evaluation
# ============================================================
print("\nRunning evaluation...")

from evaluate import evaluate

metrics = evaluate(OUTPUT_FILE, METRICS_FILE)