# Scientific Abstract Evaluator

A web-based scientific abstract evaluation system powered by a fine-tuned instruction-based large language model.
The system compares a submitted abstract against reference paper content and returns a quality score from **0 to 4** with a short rationale.

## Project Overview

This project aims to evaluate the quality of scientific abstracts using a rubric-based approach.
The model checks whether a submitted abstract properly reflects the original paper content based on:

* Content Coverage
* Faithfulness
* Language Quality
* Conciseness
* Absence of References

The system uses a fine-tuned **Mistral-7B-Instruct-v0.2** model with a **LoRA adapter**.
The adapter is uploaded to Hugging Face and loaded by the backend during inference.

## Model

Base model:

```text
mistralai/Mistral-7B-Instruct-v0.2
```

Fine-tuned LoRA adapter:

```text
1aRR0w1/scientific-abstract-evaluator-lora
```

The model outputs a JSON response containing:

```json
{
  "score": 3,
  "rationale": "The abstract mostly covers the reference but misses some minor details."
}
```

## Scoring Rubric

| Score | Meaning                                                                                                                                        |
| ----- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| 4     | Excellent abstract: covers the main problem, objective, method, results, and conclusion; faithful, clear, concise, and includes no references. |
| 3     | Good abstract: mostly covers the reference but misses one minor important element or has minor clarity/conciseness issues.                     |
| 2     | Partial abstract: covers only some important elements and misses major parts such as the method, results, or conclusion.                       |
| 1     | Weak abstract: related to the topic but vague and misses most important information.                                                           |
| 0     | Invalid abstract: unrelated, contradictory, fabricated, or contains serious unsupported claims.                                                |

## System Architecture

```text
Streamlit Frontend
        ↓
FastAPI Backend
        ↓
Mistral-7B-Instruct + LoRA Adapter
        ↓
Score + Rationale
```

## Project Structure

```text
scientific-abstract-evaluator/
│
├── app/
│   ├── backend/
│   │   └── main.py
│   │
│   └── frontend/
│       └── streamlit_app.py
│
├── configs/
├── data/
├── docs/
├── models/
├── notebooks/
├── results/
├── src/
│
├── test_hf_model.py
├── upload_to_hf.py
├── requirements.txt
└── README.md
```

## Backend

The backend is implemented using **FastAPI**.

Main endpoint:

```text
POST /predict
```

Input:

```json
{
  "reference": "Original paper content or reference text",
  "submission": "Submitted abstract to evaluate"
}
```

Output:

```json
{
  "score": 3,
  "rationale": "Short explanation of the evaluation",
  "raw_output": "Raw model output"
}
```

## Frontend

The frontend is implemented using **Streamlit**.

The user can paste:

1. Reference paper content
2. Submitted abstract

The interface then displays:

* Final score
* Rationale
* Raw model output

## Installation

Create and activate a virtual environment:

```bash
python -m venv myenv
source myenv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

If needed, install the web demo dependencies:

```bash
pip install fastapi uvicorn streamlit requests transformers peft accelerate bitsandbytes sentencepiece safetensors
```

## Running the Project

### Terminal 1: Run FastAPI Backend

```bash
source ~/venvs/myenv/bin/activate
cd ~/scientific-abstract-evaluator

uvicorn app.backend.main:app --host 0.0.0.0 --port 8000
```

### Terminal 2: Run Streamlit Frontend

```bash
source ~/venvs/myenv/bin/activate
cd ~/scientific-abstract-evaluator

streamlit run app/frontend/streamlit_app.py --server.port 8502 --server.address 0.0.0.0
```

### Terminal 3: SSH Tunnel

From the local Windows PowerShell:

```powershell
ssh -i $env:USERPROFILE\.ssh\my-gcp-key -L 8503:localhost:8502 islam@34.158.93.211
```

Then open the website locally:

```text
http://localhost:8503
```

## Example Test

Reference:

```text
This study introduces a fine-tuned instruction-based language model for evaluating the quality of scientific abstracts. The model compares a submitted abstract with the original paper content and assigns a score from 0 to 4 based on five criteria: content coverage, faithfulness, language quality, conciseness, and absence of references. The system uses supervised fine-tuning on rubric-labeled examples, where each training entry includes a reference text, a submitted abstract, a score, and a rationale. The model is evaluated using accuracy, macro F1, mean absolute error, root mean squared error, quadratic weighted kappa, and Cohen’s kappa. The results show that the fine-tuned model can generate useful rubric-based scores and explanations, but it still has difficulty distinguishing between middle-quality abstracts, especially scores 2 and 3. The study concludes that instruction-tuned language models can support abstract quality assessment, but further dataset improvement and human validation are needed before high-stakes academic use.
```

Submission:

```text
This study presents a fine-tuned instruction-based language model for evaluating scientific abstracts by comparing a submitted abstract with the original paper content. The model assigns scores from 0 to 4 using five criteria: content coverage, faithfulness, language quality, conciseness, and absence of references. It is trained with supervised rubric-labeled examples that include reference texts, submissions, scores, and rationales. The system is evaluated using accuracy, macro F1, MAE, RMSE, quadratic weighted kappa, and Cohen’s kappa. Results show that the model can produce useful rubric-based scores and explanations, although it still struggles to distinguish some middle-quality abstracts, especially scores 2 and 3. The study concludes that instruction-tuned models can support abstract evaluation, but more dataset improvement and human validation are needed before high-stakes use.
```

Expected score:

```text
4
```

## Current Limitations

* The model may sometimes produce an imperfect rationale even when the score is reasonable.
* The model can occasionally over-penalize abstracts or claim that a detail is missing when it was paraphrased.
* The system is intended as a research/demo tool, not as a final academic grading system.
* Further dataset improvement and human validation are needed for more reliable evaluation.

## Technologies Used

* Python
* FastAPI
* Streamlit
* Hugging Face Transformers
* PEFT / LoRA
* BitsAndBytes 4-bit quantization
* PyTorch
* Hugging Face Hub
* Git / GitHub

## Author

Islam SaldAldeen
Nemeh Abu Issa
Abd Al-rahman Remawi

## Repository

```text
https://github.com/IslamSaldAldeen/scientific-abstract-evaluator
```
