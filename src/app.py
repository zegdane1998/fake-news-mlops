import os
import pandas as pd
import pickle
from datetime import datetime
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = FastAPI()
templates = Jinja2Templates(directory="templates")


MODEL = load_model("models/cnn_v1.h5")
with open("models/tokenizer.pkl", "rb") as handle:
    TOKENIZER = pickle.load(handle)

def get_pipeline_status():
    """Retrieves the latest scrape info and basic stats."""
    scrape_dir = "data/new_scraped"
    try:
        files = [os.path.join(scrape_dir, f) for f in os.listdir(scrape_dir) if f.endswith('.csv')]
        latest_file = max(files, key=os.path.getmtime)
        last_scrape = datetime.fromtimestamp(os.path.getmtime(latest_file)).strftime('%Y-%m-%d %H:%M')
        
       
        return last_scrape, "Healthy", "United States", [7, 3]
    except:
        return "No data", "Unknown", "United States", [0, 0]

@app.get("/")
async def home(request: Request):
    last_sync, status, region, counts = get_pipeline_status()
    
    tweets_list = []
    scrape_dir = "data/new_scraped"
    try:
        # Load the last 5 tweets from the most recent CSV
        files = [os.path.join(scrape_dir, f) for f in os.listdir(scrape_dir) if f.endswith('.csv')]
        latest_file = max(files, key=os.path.getmtime)
        df = pd.read_csv(latest_file).tail(5)
        
        for _, row in df.iterrows():
            # Real-time CNN Inference for the table
            seq = TOKENIZER.texts_to_sequences([str(row['text'])])
            padded = pad_sequences(seq, maxlen=50)
            pred = MODEL.predict(padded)[0][0]
            
            tweets_list.append({
                "text": row['text'],
                "scraped_at": row['scraped_at'],
                "verdict": "Real" if pred > 0.5 else "Fake",
                "conf": f"{float(pred if pred > 0.5 else 1-pred)*100:.1f}%",
                "color": "text-green-400" if pred > 0.5 else "text-red-400"
            })
    except Exception as e:
        print(f"Prediction loop error: {e}")

    # Data for the new monitoring graphs
    stats = {
        "labels": ["Real", "Fake"],
        "counts": counts,
        "relevancy": [65, 68, 70, 72, 70],
        "keywords": ["Congress", "White House", "Ballots", "Breaking", "Leaked"],
        "keyword_counts": [12, 8, 5, 4, 2]
    }
    
    return templates.TemplateResponse("index.html", {
        "request": request, "last_sync": last_sync, "status": status,
        "region": region, "stats": stats, "tweets": tweets_list
    })

@app.post("/analyze")
async def analyze(request: Request, headline: str = Form(...)):
    # Manual input analysis
    seq = TOKENIZER.texts_to_sequences([headline])
    padded = pad_sequences(seq, maxlen=50)
    pred = MODEL.predict(padded)[0][0]
    
    result = {
        "verdict": "Real" if pred > 0.5 else "Fake",
        "conf": f"{float(pred if pred > 0.5 else 1-pred)*100:.1f}%",
        "color": "text-green-400" if pred > 0.5 else "text-red-400"
    }
    
    
    last_sync, status, region, counts = get_pipeline_status()
   
    return templates.TemplateResponse("index.html", {
        "request": request, "result": result, "headline": headline,
        "last_sync": last_sync, "status": status, "region": region, 
        "stats": {"labels": ["Real", "Fake"], "counts": counts, "relevancy": [65, 68, 70, 72, 70], "keywords": ["Congress", "White House", "Ballots", "Breaking", "Leaked"], "keyword_counts": [12, 8, 5, 4, 2]},
        "tweets": []
    })