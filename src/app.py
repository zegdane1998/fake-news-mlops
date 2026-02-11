from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import os
import pandas as pd
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from datetime import datetime
from src.preprocessing import clean_text

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Load model and tokenizer once on startup
MODEL = tf.keras.models.load_model('models/cnn_v1.h5')
with open('models/tokenizer.pkl', 'rb') as f:
    TOKENIZER = pickle.load(f)

def get_pipeline_status():
    """Reads the metadata from the automated GitHub Actions runs"""
    scrape_dir = "data/new_scraped"
    try:
        files = [os.path.join(scrape_dir, f) for f in os.listdir(scrape_dir) if f.endswith('.csv')]
        if not files:
            return "No data yet", "Unknown"
        latest_file = max(files, key=os.path.getmtime)
        last_scrape = datetime.fromtimestamp(os.path.getmtime(latest_file)).strftime('%Y-%m-%d %H:%M')
        status = "Healthy" if os.path.exists("models/cnn_v1.h5") else "Model Missing"
        return last_scrape, status
    except Exception:
        return "Not available", "Error"
    return last_scrape, status, "United States"

@app.get("/")
async def home(request: Request):
    last_scrape, model_status = get_pipeline_status()
    return templates.TemplateResponse("index.html", {
        "request": request,
        "last_scrape": last_scrape,
        "model_status": model_status
    })

@app.post("/predict")
async def predict(request: Request):
    data = await request.json()
    text = data.get("text", "")
    
    # 1. Preprocess
    cleaned = clean_text(text)
    # 2. Tokenize & Pad
    seq = TOKENIZER.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=50)
    # 3. Predict
    pred = MODEL.predict(padded)[0][0]
    
    verdict = "Real News" if pred > 0.5 else "Fake News"
    confidence = float(pred if pred > 0.5 else 1 - pred)

    return {
        "verdict": verdict,
        "confidence": confidence,
        "cleaned_text": cleaned
    }