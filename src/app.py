import os
import pickle
import tensorflow as tf
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Import your modular cleaning function from preprocessing.py
from .preprocessing import clean_text

app = FastAPI(title="Political Fake News Detector - MLOps Thesis")

# 1. Setup Directories for Frontend
# Ensure these folders exist: 'templates' and 'static'
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# 2. Load ML Artifacts
MODEL_PATH = 'models/cnn_v1.h5'
TOKENIZER_PATH = 'models/tokenizer.pkl'

if not os.path.exists(MODEL_PATH) or not os.path.exists(TOKENIZER_PATH):
    raise RuntimeError("Model or Tokenizer not found. Please run src/train.py first.")

model = tf.keras.models.load_model(MODEL_PATH)
with open(TOKENIZER_PATH, 'rb') as handle:
    tokenizer = pickle.load(handle)

# 3. Data Models
class TweetRequest(BaseModel):
    text: str

# 4. Routes
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serves the interactive dashboard."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(request: TweetRequest):
    """Handles real-time inference requests."""
    try:
        # Step A: Clean the incoming text using modular logic
        cleaned = clean_text(request.text)
        
        # Step B: Tokenize and Pad (Matches training parameters)
        seq = tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(seq, maxlen=50)
        
        # Step C: CNN Inference
        prediction = model.predict(padded)[0][0]
        
        # Step D: Determine Verdict (Threshold: 0.5)
        verdict = "Fake" if prediction < 0.5 else "Real"
        
        return {
            "verdict": verdict,
            "confidence": float(prediction),
            "cleaned_text": cleaned
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Run the server
    uvicorn.run(app, host="127.0.0.1", port=8000)