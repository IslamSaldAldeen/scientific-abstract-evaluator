# Scientific Abstract Quality Evaluator

An automatic evaluator for scientific abstract quality using an instruction-tuned transformer model fine-tuned with LoRA.

The system evaluates a submitted scientific abstract against a reference paper and produces:

* A numeric score from **0 to 4**
* A natural-language rationale explaining the score

The project follows a rubric-based evaluation methodology using structured input fields: task, reference, submission, rubric, score, and rationale.

---

## Project Overview

Scientific abstracts are expected to summarize the main content of a research paper, including the problem, objective, methodology, main results, and conclusion. Manually evaluating abstract quality can be time-consuming and subjective.

This project builds an automatic abstract quality evaluator using a fine-tuned instruction-tuned large language model. The evaluator takes a reference paper and a submitted abstract, then assigns a quality score based on an explicit rubric.

---

## Final Model

The final selected model is:

**Gemma-3-12B-Instruct fine-tuned with LoRA**

Experiment name:

```text
gemma3_12b_rationale_v1
```

The model was selected because it achieved the best performance among the instruction-tuned LLM experiments tested in this project.

---

## Task Definition

The model receives a structured input containing:

| Field        | Description                                         |
| ------------ | --------------------------------------------------- |
| `task`       | The instruction describing what should be evaluated |
| `reference`  | The reference paper or verified source content      |
| `submission` | The submitted abstract to evaluate                  |
| `rubric`     | The scoring criteria                                |
| `score`      | Ground-truth score from 0 to 4                      |
| `rationale`  | Explanation justifying the ground-truth score       |

The model outputs:

```text
Score: <number from 0 to 4>
Rationale: <one sentence explanation>
```

---

## Scoring Rubric

| Score | Label     | Meaning                                                                                                                      |
| ----- | --------- | ---------------------------------------------------------------------------------------------------------------------------- |
| 0     | Invalid   | The abstract is unrelated, fabricated, contradictory, or contains serious unsupported claims.                                |
| 1     | Weak      | The abstract is related to the broad topic but very vague and misses most study-specific details.                            |
| 2     | Partial   | The abstract contains some correct information but misses major components such as methodology, main results, or conclusion. |
| 3     | Good      | The abstract is mostly correct and faithful but misses minor details or has minor specificity/language issues.               |
| 4     | Excellent | The abstract covers the problem, objective, methodology, main results, and conclusion clearly and faithfully.                |

---

## Dataset

The dataset was created for scientific abstract quality evaluation.

Each entry includes the six required roles:

```json
{
  "task": "Evaluate the quality of the submitted abstract against the reference paper.",
  "reference": "Reference scientific paper content.",
  "submission": "Candidate abstract to evaluate.",
  "rubric": {
    "0": "Invalid: unrelated, contradictory, fabricated, or contains serious unsupported claims.",
    "1": "Weak: related to the topic but very vague and misses most important details.",
    "2": "Partial: covers some important elements but misses major parts such as methodology, results, or conclusion.",
    "3": "Good: mostly covers the paper but misses one minor element or has minor specificity/language issues.",
    "4": "Excellent: covers problem, objective, methodology, main results, and conclusion; faithful, clear, concise."
  },
  "score": 3,
  "rationale": "The abstract is mostly accurate but misses a minor methodological detail."
}
```

The dataset was split into:

```text
data/splits/train.json
data/splits/val.json
data/splits/test.json
```

The test set contains **50 examples**.

---

## Methodology

The project follows this pipeline:

1. Prepare a structured instruction-tuning dataset.
2. Split the dataset into train, validation, and test sets.
3. Format each example as an instruction prompt.
4. Evaluate the base model on the held-out test set.
5. Fine-tune the model using LoRA.
6. Evaluate the fine-tuned model on the same test set.
7. Compare base and fine-tuned results using quantitative metrics.
8. Inspect generated rationales qualitatively.

---

## Model Training

The final model was trained using:

| Component          | Setting              |
| ------------------ | -------------------- |
| Base model         | Gemma-3-12B-Instruct |
| Fine-tuning method | LoRA                 |
| Quantization       | 4-bit                |
| Output format      | Score + rationale    |
| Test set size      | 50 examples          |
| Score range        | 0–4                  |

Training configuration:

```yaml
experiment:
  name: gemma3_12b_rationale_v1

model:
  base_model: unsloth/gemma-3-12b-it-bnb-4bit
  max_seq_length: 4096
  load_in_4bit: true

training:
  epochs: 3
  learning_rate: 0.00003
  train_batch_size: 2
  eval_batch_size: 2
  bf16: true
```

---

## Evaluation Metrics

The following metrics were used:

| Metric             | Purpose                                             |
| ------------------ | --------------------------------------------------- |
| Accuracy           | Exact score match                                   |
| Macro F1           | Balanced class-level performance                    |
| MAE                | Average absolute score error                        |
| RMSE               | Penalizes larger score errors                       |
| QWK                | Ordinal agreement between predicted and true scores |
| Cohen Kappa Linear | Agreement adjusted for chance                       |

Quadratic Weighted Kappa (QWK) is especially important because the scores are ordinal. A prediction of 3 instead of 4 is less severe than a prediction of 0 instead of 4.

---

## Results

### Base Model Results

The base Gemma-3-12B-Instruct model was evaluated before fine-tuning.

| Metric             | Base Gemma |
| ------------------ | ---------: |
| Accuracy           |       0.34 |
| Macro F1           |      0.324 |
| MAE                |       0.88 |
| RMSE               |      1.249 |
| QWK                |      0.614 |
| Cohen Kappa Linear |      0.175 |

### Fine-Tuned Model Results

After LoRA fine-tuning, the model achieved:

| Metric             | Fine-tuned Gemma |
| ------------------ | ---------------: |
| Accuracy           |             0.58 |
| Macro F1           |            0.539 |
| MAE                |             0.54 |
| RMSE               |            0.927 |
| QWK                |            0.770 |
| Cohen Kappa Linear |            0.475 |

### Before vs. After Fine-Tuning

| Metric             | Base Model | Fine-Tuned Model | Change |
| ------------------ | ---------: | ---------------: | -----: |
| Accuracy           |       0.34 |             0.58 |  +0.24 |
| Macro F1           |      0.324 |            0.539 | +0.215 |
| MAE                |       0.88 |             0.54 |  -0.34 |
| RMSE               |      1.249 |            0.927 | -0.322 |
| QWK                |      0.614 |            0.770 | +0.156 |
| Cohen Kappa Linear |      0.175 |            0.475 | +0.300 |

Fine-tuning improved performance across all major metrics.

---

## Fine-Tuned Confusion Matrix

```text
Rows = true score, columns = predicted score

        0   1   2   3   4
0       8   1   0   1   0
1       0   10  0   0   0
2       0   4   1   4   1
3       0   1   1   7   1
4       0   1   0   6   3
```

The model performs well on scores 0 and 1. The most challenging class is score 2, which is often confused with scores 1 and 3. This is expected because score 2 represents partially complete abstracts, which are harder to distinguish from weak or mostly good abstracts.

---

## Files and Directories

```text
scientific-abstract-evaluator/
│
├── configs/
│   └── experiments/
│       └── gemma3_12b_rationale_v1.yaml
│
├── data/
│   └── splits/
│       ├── train.json
│       ├── val.json
│       └── test.json
│
├── results/
│   └── experiments/
│       └── gemma3_12b_rationale_v1/
│           ├── baseline_predictions.json
│           ├── baseline_metrics.json
│           ├── finetuned_predictions.json
│           ├── finetuned_metrics.json
│           └── config_used.yaml
│
├── src/
│   ├── baseline.py
│   ├── finetune.py
│   ├── inference.py
│   └── evaluate.py
│
└── app/
    ├── backend/
    │   └── main.py
    └── frontend/
        └── streamlit_app.py
```

---

## How to Run

### 1. Create and activate environment

```bash
python3 -m venv ~/venvs/myenv
source ~/venvs/myenv/bin/activate
```

Or if using `virtualenv`:

```bash
source ~/venvs/myenv/bin/activate
```

### 2. Install requirements

```bash
pip install -r requirements.txt
```

If needed:

```bash
pip install fastapi uvicorn streamlit requests
```

### 3. Run base model evaluation

```bash
python src/baseline.py --config configs/experiments/gemma3_12b_rationale_v1.yaml
```

### 4. Fine-tune the model

```bash
python src/finetune.py --config configs/experiments/gemma3_12b_rationale_v1.yaml
```

### 5. Run inference with the fine-tuned model

```bash
python src/inference.py --config configs/experiments/gemma3_12b_rationale_v1.yaml
```

### 6. View metrics

```bash
cat results/experiments/gemma3_12b_rationale_v1/finetuned_metrics.json
```

---

## Web Demo

The project includes a simple demo using:

* **FastAPI** backend
* **Streamlit** frontend

### Run backend

```bash
uvicorn app.backend.main:app --host 0.0.0.0 --port 8000
```

### Run frontend

Open a second terminal:

```bash
streamlit run app/frontend/streamlit_app.py --server.address 0.0.0.0 --server.port 8501
```

Then open:

```text
http://<server-ip>:8501
```

The user can paste:

1. Reference paper content
2. Submitted abstract

The system returns:

1. Predicted score
2. Score label
3. Rationale
4. Model name
5. Experiment name

---

## Example Output

```text
Score: 3
Rationale: The abstract is mostly faithful and covers the main objective and results, but it misses a minor methodological detail.
```

---

## Key Findings

Fine-tuning had a clear positive impact.

The base model achieved:

```text
Accuracy = 0.34
QWK = 0.614
MAE = 0.88
```

The fine-tuned model achieved:

```text
Accuracy = 0.58
QWK = 0.770
MAE = 0.54
```

This shows that LoRA fine-tuning helped the model better align with the abstract evaluation rubric.

---

## Challenges

The main challenges were:

1. **Middle-score ambiguity:** Score 2 was difficult because it lies between weak and mostly complete abstracts.
2. **Score 4 under-prediction:** Some excellent abstracts were predicted as score 3.
3. **Small dataset size:** The dataset size limited the model's ability to fully generalize.
4. **Output control:** Some generated rationales were longer than expected.
5. **Prompt sensitivity:** Small changes in the rubric wording sometimes changed the prediction distribution.

---

## Future Work

Possible improvements include:

* Expanding the dataset with more scientific papers.
* Adding more balanced examples for score 2.
* Improving JSON output formatting.
* Adding confidence scores.
* Evaluating rationale quality manually.
* Testing additional instruction-tuned models.
* Improving the web interface with component-level feedback.

---

## Final Conclusion

This project demonstrates that an instruction-tuned transformer can be fine-tuned to perform rubric-based scientific abstract quality evaluation.

The final Gemma-3-12B-Instruct model fine-tuned with LoRA improved significantly over the base model:

```text
Accuracy: 0.34 → 0.58
QWK: 0.614 → 0.770
MAE: 0.88 → 0.54
```

The model produces both a score and a rationale, making the evaluation more interpretable and useful for academic assessment.
