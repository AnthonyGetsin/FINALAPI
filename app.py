import os
import requests
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

MODEL_URL = "https://huggingface.co/your-username/your-repo/resolve/main/car_DealScore_423.pkl"
MODEL_PATH = "car_DealScore_423.pkl"

# Download the model if it doesn't exist
def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model...")
        response = requests.get(MODEL_URL)
        with open(MODEL_PATH, "wb") as f:
            f.write(response.content)
        print("Download complete.")

download_model()

# Load model
model = joblib.load(MODEL_PATH)

# Input schema
class CarFeatures(BaseModel):
    year: int
    odometer: float
    manufacturer: str
    model: str
    condition: str
    cylinders: str
    fuel: str
    title_status: str
    transmission: str
    drive: str
    type: str
    paint_color: str
    state: str
    lat: float
    long: float
    region: str

@app.post("/predict")
def predict(features: CarFeatures):
    try:
        df = pd.DataFrame([features.dict()])
        prediction = model.predict(df)[0]
        return {"predicted_price": round(prediction, 2)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
