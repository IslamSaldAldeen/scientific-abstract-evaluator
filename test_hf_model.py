import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
ADAPTER_MODEL = "1aRR0w1/scientific-abstract-evaluator-lora"

print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading 4-bit config...")
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

reference = """
This study investigates the effectiveness of a machine learning model for evaluating scientific abstracts.
The model is trained to assess content coverage, faithfulness, language quality, conciseness, and absence of references.
The method uses supervised fine-tuning on rubric-based examples. The expected output is a score from 0 to 4 and a short rationale.
"""

submission = """
This paper presents a machine learning model that evaluates scientific abstracts based on quality criteria including coverage and language.
"""

user_prompt = f"""
Evaluate the quality of the scientific abstract based on the rubric.

Reference:
{reference}

Submission:
{submission}

Rubric:
Content Coverage, Faithfulness, Language Quality, Conciseness, No References.

Return ONLY a valid JSON object in this exact format:
{{
  "score": 0,
  "rationale": "brief explanation"
}}

Do not repeat the rubric.
Do not explain outside the JSON.
"""

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

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

print("Generating...")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=180,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
answer = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

print("\nMODEL OUTPUT:")
print(answer)
