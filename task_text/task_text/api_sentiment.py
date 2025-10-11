# task_text/api_sentiment.py
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()
sentiment = pipeline("sentiment-analysis")

class Item(BaseModel):
    text: str

@app.post("/predict")
def predict(item: Item):
    res = sentiment(item.text)
    return {"result": res}

# 运行： uvicorn task_text.api_sentiment:app --reload --port 8000