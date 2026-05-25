import json
import torch
import sys
import os
from transformers import AutoTokenizer
from unsloth import FastLanguageModel

# ============================================================
# Configuration
# ============================================================
MODEL_DIR  = "models/finetuned-mistral"
TEST_FILE  = "data/test.json"
OUTPUT_FILE = "results/finetuned_predictions.json"
MAX_SEQ_LENGTH = 2048

# ============================================================
# Load Fine-tuned Model
# ============================================================
print("Loading fine-tuned model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = MODEL_DIR,
    max_seq_length = MAX_SEQ_LENGTH,
    load_in_4bit   = True,
)
FastLanguageModel.for_inference(model)
print("Model loaded!")

# ============================================================
# Load Test Set
# ============================================================
with open(TEST_FILE, "r") as f:
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
Rationale: <one sentence explanation>
[/INST]"""
    return prompt

# ============================================================
# Inference Function
# ============================================================
def get_prediction(entry):
    prompt = build_inference_prompt(entry)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    score = None
    for line in response.strip().split("\n"):
        if line.startswith("Score:"):
            try:
                score = int(line.split(":")[1].strip()[0])
                if score < 0 or score > 4:
                    score = None
            except:
                score = None

    return score, response.strip()

# ============================================================
# Run on Test Set
# ============================================================
results = []

for i, entry in enumerate(test_data):
    print(f"Processing {i+1}/{len(test_data)}...")

    score, rationale = get_prediction(entry)

    if score is None:
        print(f"  Warning: Could not parse score, defaulting to 2")
        score = 2

    results.append({
        "reference": entry["reference"][:200],
        "submission": entry["submission"][:200],
        "true_score": entry["score"],
        "predicted_score": score,
        "rationale": rationale
    })

    print(f"  True: {entry['score']} | Predicted: {score}")

# ============================================================
# Save Results
# ============================================================
finetuned_results = {
    "model": "Mistral-7B-Instruct-v0.2 (fine-tuned)",
    "predictions": results
}

with open(OUTPUT_FILE, "w") as f:
    json.dump(finetuned_results, f, indent=2)

print(f"Saved to {OUTPUT_FILE}")

# ============================================================
# Run Evaluation
# ============================================================
print("\nRunning evaluation...")
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from evaluate import evaluate
metrics = evaluate(OUTPUT_FILE)