![CI](https://github.com/MithunKaveriappa/banking-intent-classifier/actions/workflows/ci.yml/badge.svg)

# Banking Intent Classifier

Classifies customer messages into 77 banking intents (e.g. `card_arrival`,
`top_up_failed`, `pending_transfer`). I compared three approaches, fine-tuned
DistilBERT, served it through a FastAPI service, and packaged it with Docker
and CI.

![demo](docs/demo.gif)

**Model on Hugging Face:** https://huggingface.co/MithunKaveriappa/banking77-distilbert

## Results

Evaluated on the Banking77 test set (3,076 examples, `mteb/banking77`).
All models were trained on the same 90% training split.

| Model | Accuracy | Macro-F1 |
|---|---|---|
| TF-IDF + Logistic Regression | 0.8898 | 0.8899 |
| BGE-small embeddings + Logistic Regression | **0.9311** | **0.9310** |
| DistilBERT (fine-tuned, deployed) | 0.9217 | 0.9217 |

**Takeaways**

- Fine-tuned DistilBERT improved on the TF-IDF baseline by about 3.2 points.
- In this experiment, frozen BGE-small embeddings with a Logistic Regression
  head scored about 0.9 points higher than fine-tuned DistilBERT. This is a
  single run (one seed, one set of hyperparameters), so the gap is suggestive,
  not conclusive. DistilBERT was trained for only 5 epochs with default
  settings; more tuning was not tried.
- I deployed the DistilBERT model because it is the end-to-end fine-tuning
  pipeline this project set out to build. A natural next step is switching the
  service to the BGE model or tuning DistilBERT further.

## Error analysis

About 241 of 3,076 test messages were misclassified. The 15 most frequent
confusion pairs account for 60 of them (about 25%), and nearly all are intents
that overlap in meaning:

| True intent | Predicted as | Count |
|---|---|---|
| why_verify_identity | verify_my_identity | 7 |
| declined_transfer | declined_card_payment | 6 |
| get_disposable_virtual_card | getting_virtual_card | 5 |
| wrong_exchange_rate_for_cash_withdrawal | card_payment_wrong_exchange_rate | 4 |
| verify_my_identity | why_verify_identity | 4 |
| compromised_card | lost_or_stolen_card | 4 |
| card_arrival | card_delivery_estimate | 4 |

Full list: `results/top_confused_pairs.csv`. Many of these "errors" are messages
that could reasonably belong to either category, so label overlap in the
dataset likely caps achievable accuracy.

## Limitations

- The model always predicts one of the 77 intents, even for unrelated text. In
  a small manual check, out-of-scope messages ("what is the weather today",
  random characters) scored 10-17% top confidence versus 97% for a clear
  in-scope message. The API therefore returns a `confident` flag using a 0.30
  threshold (configurable with `CONFIDENCE_THRESHOLD`). The threshold came from
  a few manual examples and should be validated on a larger out-of-scope set
  before any production use.
- English only, short banking messages only.
- Results come from a single training run, so small differences between models
  may not be significant.

## Project structure

```
banking-intent-classifier/
├── app/                  # FastAPI service
├── src/                  # model loading and prediction logic
├── demo/                 # local Gradio demo
├── tests/                # pytest API tests (8)
├── notebooks/            # data exploration, baseline, fine-tuning, analysis
├── results/              # metrics and confusion-pair CSVs
├── scripts/              # push model to the Hugging Face Hub
├── docs/                 # demo GIF
├── Dockerfile
├── requirements-api.txt
├── requirements-dev.txt
├── requirements-demo.txt
└── .github/workflows/ci.yml
```

## How to run

**API (local)**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-dev.txt
$env:MODEL_PATH = "MithunKaveriappa/banking77-distilbert"
uvicorn app.main:app
```

Open http://127.0.0.1:8000/docs

**Docker**

```bash
docker build --build-arg MODEL_ID=MithunKaveriappa/banking77-distilbert -t banking-intent-classifier .
docker run --rm -p 8000:8000 banking-intent-classifier
```

**Gradio demo**

```bash
pip install -r requirements-demo.txt
python demo/app.py
```

**Tests**

```bash
pytest -v
```

## API example

`POST /predict`

```json
{"text": "My card has not arrived yet", "top_k": 3}
```

Response: the top-k intents with scores, plus a `confident` flag.
`GET /health` reports whether the model is loaded.

## Tech stack

Python, PyTorch, Hugging Face Transformers and Hub, sentence-transformers,
scikit-learn, FastAPI, Pydantic, Gradio, Docker, pytest, GitHub Actions

## Acknowledgements

Built on `distilbert-base-uncased`, `BAAI/bge-small-en-v1.5` and the Banking77
dataset (Casanueva et al., 2020), loaded from `mteb/banking77`.