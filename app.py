import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import uvicorn
from catboost import CatBoostRegressor

# Загрузка модели и файлов
model = CatBoostRegressor()
model.load_model('catboost_log.cbm')

feature_names = joblib.load('feature_names.pkl')
cat_features = joblib.load('cat_features.pkl')

try:
    ensemble = joblib.load('ensemble_models.pkl')
    print("Ensemble loaded")
except:
    ensemble = None
    print("No ensemble, using ±5% interval")

app = FastAPI()

class PredictionRequest(BaseModel):
    data: Dict[str, Any]

@app.get("/")
def root():
    return {"message": "API is ready. Use POST /predict"}

@app.post("/predict")
def predict(request: PredictionRequest):
    missing = set(feature_names) - set(request.data.keys())
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing features: {missing}")
    
    df_input = pd.DataFrame([request.data])[feature_names]
    for col in cat_features:
        if col in df_input.columns:
            df_input[col] = df_input[col].astype(str)
    
    pred_log = model.predict(df_input)[0]
    pred_demand = np.expm1(pred_log)
    
    if ensemble:
        preds = []
        for m in ensemble:
            preds.append(np.expm1(m.predict(df_input)[0]))
        lower = np.percentile(preds, 10)
        upper = np.percentile(preds, 90)
    else:
        lower = pred_demand * 0.95
        upper = pred_demand * 1.05
    
    return {
        "predicted_demand_mil_rub": round(pred_demand, 2),
        "confidence_interval": [round(lower, 2), round(upper, 2)]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)