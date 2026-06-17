from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from clearml import Model
import joblib
import time

app = FastAPI(title="Sentiment Inference API")

MODEL_ID = "258f45e2527a4eb38eb4b329af147819"

model = None
vectorizer = None


def load_model():
    global model, vectorizer

    if model is not None and vectorizer is not None:
        return

    try:
        print("Loading model from ClearML Registry...")
        model_obj = Model(model_id=MODEL_ID)
        path = model_obj.get_local_copy()

        bundle = joblib.load(path)
        model = bundle["model"]
        vectorizer = bundle["vectorizer"]

        print("Model loaded successfully.")
    except Exception as e:
        raise RuntimeError(f"Failed to load model from ClearML: {e}")


class Request(BaseModel):
    text: str


@app.on_event("startup")
def startup_event():
    load_model()


@app.post("/predict")
def predict(req: Request):
    try:
        start = time.perf_counter()

        vec = vectorizer.transform([req.text])
        label = model.predict(vec)[0]

        confidence = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(vec)[0]
            confidence = float(max(proba))

        latency_ms = round((time.perf_counter() - start) * 1000, 2)

        return {
            "label": str(label),
            "confidence": round(confidence, 3) if confidence is not None else None,
            "latency_ms": latency_ms,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok"}
