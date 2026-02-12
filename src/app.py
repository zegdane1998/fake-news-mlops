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
    scrape_dir = "data/new_scraped"
    try:
        # Existing logic to get the file date
        files = [os.path.join(scrape_dir, f) for f in os.listdir(scrape_dir) if f.endswith('.csv')]
        latest_file = max(files, key=os.path.getmtime)
        last_scrape = datetime.fromtimestamp(os.path.getmtime(latest_file)).strftime('%Y-%m-%d %H:%M')
        
        status = "Healthy" if os.path.exists("models/cnn_v1.h5") else "Model Missing"
        
        # YOU MUST ADD THIS THIRD RETURN VALUE
        region = "United States" 
        
        return last_scrape, status, region 
    except Exception:
        return "No data yet", "Unknown", "United States"

@app.get("/")
async def home(request: Request):
    last_scrape, model_status, region = get_pipeline_status()
    
    # 1. Load the latest scraped data
    scraped_results = []
    scrape_dir = "data/new_scraped"
    files = [os.path.join(scrape_dir, f) for f in os.listdir(scrape_dir) if f.endswith('.csv')]
    
    if files:
        latest_file = max(files, key=os.path.getmtime)
        df = pd.read_csv(latest_file)
        
        # 2. Analyze each tweet in the CSV
        for text in df['text'].head(5): # Show top 5 for the dashboard
            cleaned = clean_text(text)
            seq = TOKENIZER.texts_to_sequences([cleaned])
            padded = pad_sequences(seq, maxlen=50)
            pred = MODEL.predict(padded)[0][0]
            
            scraped_results.append({
                "original": text,
                "verdict": "Real" if pred > 0.5 else "Fake",
                "confidence": f"{float(pred if pred > 0.5 else 1-pred)*100:.1f}%"
            })

    return templates.TemplateResponse("index.html", {
        "request": request,
        "last_scrape": last_scrape,
        "model_status": model_status,
        "scraped_results": scraped_results # Pass this to the HTML
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