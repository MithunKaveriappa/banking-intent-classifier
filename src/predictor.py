import os
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parent.parent / "models" / "banking77_distilbert"
)


class IntentClassifier:
    """Loads the fine-tuned model and predicts banking intents."""

    def __init__(self, model_path=None):
        # MODEL_PATH can be a local folder or, later, a Hugging Face Hub ID
        model_path = str(model_path or os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH))
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()
        self.id2label = self.model.config.id2label

    @torch.inference_mode()
    def predict(self, texts, top_k=3):
        inputs = self.tokenizer(
            texts, truncation=True, max_length=64,
            padding=True, return_tensors="pt",
        )
        probs = torch.softmax(self.model(**inputs).logits, dim=-1)
        k = min(top_k, probs.shape[-1])
        top_p, top_i = probs.topk(k, dim=-1)

        results = []
        for p_row, i_row in zip(top_p, top_i):
            results.append([
                {"label": self.id2label[int(i)], "score": round(float(p), 4)}
                for p, i in zip(p_row, i_row)
            ])
        return results