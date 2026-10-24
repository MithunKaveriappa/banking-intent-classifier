import os

import gradio as gr
from transformers import pipeline

MODEL_ID = os.getenv("MODEL_PATH", "MithunKaveriappa/banking77-distilbert")
clf = pipeline("text-classification", model=MODEL_ID)


def classify(text):
    if not text or not text.strip():
        return {}
    out = clf(text, top_k=5, truncation=True, max_length=64)
    if out and isinstance(out[0], list):   # some versions nest the result
        out = out[0]
    return {r["label"]: float(r["score"]) for r in out}


demo = gr.Interface(
    fn=classify,
    inputs=gr.Textbox(lines=2, label="Customer message",
                      placeholder="e.g. My card has not arrived yet"),
    outputs=gr.Label(num_top_classes=5, label="Predicted intent"),
    title="Banking Intent Classifier",
    description=(
        "DistilBERT fine-tuned on Banking77 to classify customer messages into "
        "77 banking intents. Note: it always picks one of the 77 intents, so a "
        "low confidence score means the message may be out of scope."
    ),
    examples=[
        ["My card has not arrived yet"],
        ["Why was I charged twice for the same purchase?"],
        ["I forgot my PIN, how do I reset it?"],
        ["My top up failed but money left my account"],
        ["How do I change my address?"],
    ],
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()