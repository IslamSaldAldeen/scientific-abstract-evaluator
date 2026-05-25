import json
import torch
from datasets import Dataset
from transformers import TrainingArguments, EarlyStoppingCallback
from unsloth import FastLanguageModel
from trl import SFTTrainer

# ============================================================
# Configuration
# ============================================================
MODEL_NAME      = "unsloth/mistral-7b-instruct-v0.2-bnb-4bit"
MAX_SEQ_LENGTH  = 2048
TRAIN_FILE      = "data/train.json"
VAL_FILE        = "data/val.json"
OUTPUT_DIR      = "models/finetuned-mistral"

LORA_R          = 16
LORA_ALPHA      = 32
LORA_DROPOUT    = 0.05

EPOCHS          = 10
BATCH_SIZE      = 4
LEARNING_RATE   = 2e-4
PATIENCE        = 3

# ============================================================
# Load Model
# ============================================================
print("Loading model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name      = MODEL_NAME,
    max_seq_length  = MAX_SEQ_LENGTH,
    load_in_4bit    = True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r                   = LORA_R,
    lora_alpha          = LORA_ALPHA,
    lora_dropout        = LORA_DROPOUT,
    target_modules      = ["q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj"],
    bias                = "none",
    use_gradient_checkpointing = True,
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
Score: <number from 0 to 4>
Rationale: <one sentence explanation>
[/INST]
Score: {entry["score"]}
Rationale: {entry["rationale"]}"""
    
    return {"text": prompt}

print("Loading and formatting data...")

with open(TRAIN_FILE, "r") as f:
    train_raw = json.load(f)

with open(VAL_FILE, "r") as f:
    val_raw = json.load(f)

train_dataset = Dataset.from_list([format_entry(e) for e in train_raw])
val_dataset   = Dataset.from_list([format_entry(e) for e in val_raw])

print(f"Train: {len(train_dataset)} entries")
print(f"Val:   {len(val_dataset)} entries")

# ============================================================
# Training Arguments
# ============================================================
training_args = TrainingArguments(
    output_dir                  = OUTPUT_DIR,
    num_train_epochs            = EPOCHS,
    per_device_train_batch_size = BATCH_SIZE,
    per_device_eval_batch_size  = BATCH_SIZE,
    learning_rate               = LEARNING_RATE,
    eval_strategy               = "epoch",
    save_strategy               = "epoch",
    load_best_model_at_end      = True,
    metric_for_best_model       = "eval_loss",
    greater_is_better           = False,
    logging_steps               = 10,
    warmup_steps                = 50,
    lr_scheduler_type           = "cosine",
    fp16                        = False,
    bf16                        = True,
    report_to                   = "none",
)

# ============================================================
# Trainer
# ============================================================
trainer = SFTTrainer(
    model           = model,
    tokenizer       = tokenizer,
    train_dataset   = train_dataset,
    eval_dataset    = val_dataset,
    dataset_text_field = "text",
    max_seq_length  = MAX_SEQ_LENGTH,
    args            = training_args,
    callbacks       = [EarlyStoppingCallback(early_stopping_patience=PATIENCE)],
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