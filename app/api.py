from fastapi import FastAPI
from pydantic import BaseModel
from .model import load_model, load_vectorizer
from .preprocess import preprocess


app = FastAPI(title="Sentiment Analysis API")

model = load_model()
vectorizer = load_vectorizer()

class TextRequest(BaseModel):
    text: str

@app.post("/predict")
def predict_sentiment(request: TextRequest):
    input_texts = [request.text]
    X_new = preprocess(input_texts, vectorizer)
    prediction = model.predict(X_new)[0]
    label = "Positive" if prediction == 1 else "Negative"
    return {
        "input_text": request.text,
        "predicted_class": int(prediction),
        "sentiment_label": label
    }
