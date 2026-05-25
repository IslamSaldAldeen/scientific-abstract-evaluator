import json
import torch
import sys
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Load test set
with open("data/test.json", "r") as f:
    test_data = json.load(f)

print(f"Test entries: {len(test_data)}")

# Load Mistral model
model_name = "mistralai/Mistral-7B-Instruct-v0.2"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)
print("Model loaded!")

# Prompt function
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

# Inference function
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

# Run on test set
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

# Save results
baseline_results = {
    "model": "Mistral-7B-Instruct-v0.2 (base)",
    "predictions": results
}

with open("results/baseline_predictions.json", "w") as f:
    json.dump(baseline_results, f, indent=2)

print("Saved to results/baseline_predictions.json")

# Run evaluation
print("\nRunning evaluation...")
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from evaluate import evaluate
metrics = evaluate("results/baseline_predictions.json")