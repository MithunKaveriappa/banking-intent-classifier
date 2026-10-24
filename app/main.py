from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.predictor import IntentClassifier

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["clf"] = IntentClassifier()   # load the model once at startup
    yield
    state.clear()


app = FastAPI(
    title="Banking Intent Classifier",
    description="Classifies customer messages into 77 banking intents "
                "using a fine-tuned DistilBERT model.",
    version="1.0.0",
    lifespan=lifespan,
)


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=500,
                      examples=["My card has not arrived yet"])
    top_k: int = Field(3, ge=1, le=10)


class Prediction(BaseModel):
    label: str
    score: float


class PredictResponse(BaseModel):
    text: str
    predictions: list[Prediction]


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "clf" in state}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if not req.text.strip():
        raise HTTPException(status_code=422, detail="Text must not be blank.")
    preds = state["clf"].predict([req.text], top_k=req.top_k)[0]
    return PredictResponse(text=req.text, predictions=preds)