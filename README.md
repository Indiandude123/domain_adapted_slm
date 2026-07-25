# Domain-Adapted SLM for Legal Case Classification

QLoRA fine-tuning of an open-weights small language model (Phi-3-mini-4k-instruct, 3.8B) to
classify U.S. Supreme Court opinions by issue area — 14 classes, single-label, meaningfully
imbalanced. Ships as a 4-bit quantized model behind a FastAPI inference endpoint.

Pivoted from an earlier "BERT fine-tuning" framing to a modern PEFT/quantization workflow;
see [project_idea.txt](project_idea.txt) for the original scope notes.

## Problem

- **Dataset**: [LexGLUE](https://huggingface.co/datasets/coastalcph/lex_glue) — SCOTUS subtask.
  Long-form Supreme Court opinions labeled with one of 14 Supreme Court Database "issue area"
  categories (Criminal Procedure, Civil Rights, First Amendment, ...).
- **Challenge**: real class imbalance across issue areas, and documents that routinely exceed
  the base model's context window.
- **Task framing**: sequence classification (`AutoModelForSequenceClassification`, not
  generative/prompted classification) so imbalance can be handled with standard class-weighted
  loss and evaluated with standard metrics.

## Approach

1. **QLoRA**: 4-bit (nf4) quantized base model + LoRA adapters on attention/MLP projections;
   the classification head (`score`) is kept trainable in full precision via
   `modules_to_save`. See [src/model/qlora.py](src/model/qlora.py).
2. **Imbalance handling**: inverse-frequency class weights fed into a class-weighted
   cross-entropy loss (`WeightedLossTrainer` in [src/train/trainer.py](src/train/trainer.py)).
   Trained both with and without the weighting to measure its actual effect.
3. **Evaluation**: macro-F1, weighted-F1, per-class precision/recall, and a confusion matrix —
   accuracy alone is misleading on an imbalanced 14-class problem. See
   [src/eval/metrics.py](src/eval/metrics.py).
4. **Inference optimization**: memory footprint and latency/throughput benchmarked for the
   4-bit model vs an fp16 baseline load. See [src/eval/benchmark.py](src/eval/benchmark.py).
5. **Serving**: a FastAPI `/predict` endpoint loads the quantized base model + LoRA adapter
   once at startup and returns a predicted label with per-class probabilities. See
   [src/api/main.py](src/api/main.py).

## Repo layout

```
notebooks/         EDA and the Kaggle training notebook
src/data/           dataset loading, tokenization, class-weight/sampler utilities
src/model/          QLoRA model construction (4-bit base + LoRA)
src/train/          weighted-loss Trainer subclass + training CLI
src/eval/           imbalance-aware metrics + inference benchmarking
src/api/            FastAPI inference service
configs/            training hyperparameters (YAML)
tests/              API smoke tests (stubbed model, no GPU/download needed)
```

## Running it

Training runs on Kaggle Notebooks (see
[notebooks/02_train_qlora.ipynb](notebooks/02_train_qlora.ipynb) for accelerator/internet
settings and setup). Locally:

```bash
pip install -r requirements.txt

# EDA
jupyter notebook notebooks/01_eda.ipynb

# Training (needs a CUDA GPU)
python -m src.train.run_train --config configs/scotus_phi3.yaml --weighted

# Serve the trained adapter
MODEL_PATH=outputs/weighted/final uvicorn src.api.main:app --reload
curl -X POST localhost:8000/predict -H 'content-type: application/json' \
  -d '{"text": "The petitioner challenges the search under the Fourth Amendment..."}'

# Tests
pytest tests/
```

## Results

_TBD — filled in after training runs complete on Kaggle._

| Model                          | Accuracy | Macro-F1 | Weighted-F1 |
|---------------------------------|----------|----------|-------------|
| Baseline (TF-IDF + LogReg)       |          |          |             |
| QLoRA, unweighted loss           |          |          |             |
| QLoRA, class-weighted loss        |          |          |             |

| Metric                         | fp16 baseline | 4-bit quantized |
|----------------------------------|---------------|------------------|
| Peak memory (MB)                 |               |                  |
| Throughput (examples/sec)         |               |                  |

## Known limitations

- SCOTUS opinions are truncated to the model's context window (first N tokens) rather than
  chunked or summarized — long documents lose information past the truncation point.
- Single base model evaluated (Phi-3-mini-4k-instruct); Llama-3.1-8B-Instruct is a documented
  stretch goal if more GPU headroom is available.
