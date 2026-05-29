import json
import re
import torch

from fastapi import FastAPI
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel


# =========================
# App Config
# =========================

app = FastAPI(title="Scientific Abstract Evaluator API")

BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
ADAPTER_MODEL = "1aRR0w1/scientific-abstract-evaluator-lora"

tokenizer = None
model = None


# =========================
# Request Schema
# =========================

class EvalRequest(BaseModel):
    reference: str = Field(..., description="Original paper/reference content")
    submission: str = Field(..., description="Abstract submission to evaluate")


# =========================
# Helper Functions
# =========================

def build_prompt(reference: str, submission: str) -> str:
    return f"""
You are a strict scientific abstract evaluator.

Your task is to compare the Submission against the Reference ONLY.

Reference:
{reference}

Submission:
{submission}

Rubric:
Score 4 = Excellent:
The submission covers the main problem, objective, method, results, and conclusion; it is faithful to the reference, clear, concise, and includes no references.

Score 3 = Good:
The submission mostly covers the reference but misses one minor important element or has minor clarity/conciseness issues.

Score 2 = Partial:
The submission covers only some important elements and misses major parts such as the method, results, or conclusion.

Score 1 = Weak:
The submission is related to the topic but very vague and misses most important information.

Score 0 = Invalid:
The submission is unrelated, contradictory, fabricated, or contains serious unsupported claims.

Important rules:
- Evaluate ONLY based on the given Reference and Submission.
- Do NOT require information that is not present in the Reference.
- Do NOT invent missing requirements.
- Before saying that something is missing, verify that it is not already mentioned in the Submission.
- If the Submission mentions an item directly or by paraphrase, consider it covered.
- Do NOT claim that metrics, methods, results, or limitations are missing unless they are clearly absent from the Submission.
- The rationale must be short and general.
- Do not list specific missing details unless you are completely sure they are absent.
- Focus on whether the submission is excellent, good, partial, weak, or invalid.

Return ONLY valid JSON in this exact format:
{{
  "score": 0,
  "rationale": "brief explanation"
}}

The score must be an integer from 0 to 4.
Do not write anything before or after the JSON.
"""

def extract_first_json(text: str) -> dict:
    """
    Extract the first valid JSON object from the model output.
    This protects us when the model writes extra text after the JSON.
    """
    text = text.strip()

    # Try direct JSON first
    try:
        parsed = json.loads(text)
        return normalize_result(parsed, text)
    except Exception:
        pass

    # Try to extract JSON object using regex
    matches = re.findall(r"\{.*?\}", text, re.DOTALL)

    for match in matches:
        try:
            parsed = json.loads(match)
            return normalize_result(parsed, text)
        except Exception:
            continue

    # Fallback if no valid JSON found
    return {
        "score": None,
        "rationale": text,
        "raw_output": text
    }


def normalize_result(parsed: dict, raw_output: str) -> dict:
    """
    Normalize model output into clean API response.
    """
    score = parsed.get("score", None)
    rationale = parsed.get("rationale", "")

    try:
        score = int(score)
        if score < 0:
            score = 0
        if score > 4:
            score = 4
    except Exception:
        score = None

    if not isinstance(rationale, str):
        rationale = str(rationale)

    return {
        "score": score,
        "rationale": rationale.strip(),
        "raw_output": raw_output.strip()
    }


# =========================
# Load Model Once
# =========================

@app.on_event("startup")
def load_model():
    global tokenizer, model

    print("====================================")
    print("Starting Scientific Abstract Evaluator API")
    print("====================================")
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        print("WARNING: CUDA is not available. This model may be very slow on CPU.")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading 4-bit quantization config...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.float16,
    )

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_MODEL)
    model.eval()

    print("Model loaded successfully!")
    print("====================================")


# =========================
# Routes
# =========================

@app.get("/")
def home():
    return {
        "status": "API is running",
        "base_model": BASE_MODEL,
        "adapter_model": ADAPTER_MODEL,
        "cuda_available": torch.cuda.is_available()
    }


@app.post("/predict")
def predict(req: EvalRequest):
    reference = req.reference.strip()
    submission = req.submission.strip()

    if not reference or not submission:
        return {
            "score": None,
            "rationale": "Both reference and submission are required.",
            "raw_output": ""
        }

    user_prompt = build_prompt(reference, submission)

    messages = [
        {
            "role": "user",
            "content": user_prompt
        }
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=4096
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=220,
            do_sample=False,
            repetition_penalty=1.05,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
    answer = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    ).strip()

    result = extract_first_json(answer)

    return result
