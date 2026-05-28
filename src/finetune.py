import argparse
import json
import os
import shutil
import yaml
import torch

from datasets import Dataset
from transformers import TrainingArguments, EarlyStoppingCallback
from unsloth import FastLanguageModel
from trl import SFTTrainer


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

MODEL_NAME = config["model"]["base_model"]
MAX_SEQ_LENGTH = config["model"]["max_seq_length"]
LOAD_IN_4BIT = config["model"]["load_in_4bit"]

TRAIN_FILE = config["dataset"]["train_path"]
VAL_FILE = config["dataset"]["val_path"]

OUTPUT_DIR = config["outputs"]["model_dir"]
EXPERIMENT_DIR = config["outputs"]["experiment_dir"]
CONFIG_USED_PATH = config["outputs"]["config_used"]

LORA_R = config["lora"]["r"]
LORA_ALPHA = config["lora"]["alpha"]
LORA_DROPOUT = config["lora"]["dropout"]
TARGET_MODULES = config["lora"]["target_modules"]

EPOCHS = config["training"]["epochs"]
TRAIN_BATCH_SIZE = config["training"]["train_batch_size"]
EVAL_BATCH_SIZE = config["training"]["eval_batch_size"]
LEARNING_RATE = config["training"]["learning_rate"]
WARMUP_STEPS = config["training"]["warmup_steps"]
LR_SCHEDULER_TYPE = config["training"]["lr_scheduler_type"]
PATIENCE = config["training"]["early_stopping_patience"]
FP16 = config["training"]["fp16"]
BF16 = config["training"]["bf16"]
REPORT_TO = config["training"]["report_to"]


# ============================================================
# Prepare Experiment Folders
# ============================================================
os.makedirs(EXPERIMENT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

shutil.copyfile(args.config, CONFIG_USED_PATH)

print("=" * 60)
print(f"Experiment: {experiment_name}")
print(f"Config:     {args.config}")
print(f"Train file: {TRAIN_FILE}")
print(f"Val file:   {VAL_FILE}")
print(f"Model out:  {OUTPUT_DIR}")
print("=" * 60)


# ============================================================
# Load Model
# ============================================================
print("Loading model...")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=LOAD_IN_4BIT,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=TARGET_MODULES,
    bias="none",
    use_gradient_checkpointing=True,
)

print("Model loaded!")


# ============================================================
# Format Data
# ============================================================
def format_entry(entry):
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
Rationale: <one sentence explanation>
Score: <number from 0 to 4>

Important scoring guidance:
- Give Score 4 if the abstract covers the main problem, objective, methodology, key results, and conclusion, and is faithful, clear, concise, and free of references.
- Do NOT reduce a Score 4 to Score 3 only because of a very minor missing detail that does not change the main meaning.
- Give Score 3 when the abstract is mostly correct but misses one important element or has a minor faithfulness issue.
- Give Score 2 when it covers only some important elements and misses major parts such as results or conclusion.
- Give Score 1 when it is mostly vague, generic, or weakly connected to the paper.
- Give Score 0 when it is unrelated, contradictory, fabricated, or contains serious unsupported claims.

[/INST]
Rationale: {entry["rationale"]}
Score: {entry["score"]}"""

    return {"text": prompt}


print("Loading and formatting data...")

with open(TRAIN_FILE, "r", encoding="utf-8") as f:
    train_raw = json.load(f)

with open(VAL_FILE, "r", encoding="utf-8") as f:
    val_raw = json.load(f)

train_dataset = Dataset.from_list([format_entry(e) for e in train_raw])
val_dataset = Dataset.from_list([format_entry(e) for e in val_raw])

print(f"Train: {len(train_dataset)} entries")
print(f"Val:   {len(val_dataset)} entries")


# ============================================================
# Training Arguments
# ============================================================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    logging_steps=10,
    warmup_steps=WARMUP_STEPS,
    lr_scheduler_type=LR_SCHEDULER_TYPE,
    fp16=FP16,
    bf16=BF16,
    report_to=REPORT_TO,
)


# ============================================================
# Trainer
# ============================================================
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    args=training_args,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=PATIENCE)],
)


# ============================================================
# Train
# ============================================================
print("Starting training...")
trainer.train()


# ============================================================
# Save Model
# ============================================================
print("Saving model...")

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"Model saved to {OUTPUT_DIR}")
print(f"Config copy saved to {CONFIG_USED_PATH}")