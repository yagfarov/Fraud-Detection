from fastapi import FastAPI
from pydantic import BaseModel
from catboost import CatBoostClassifier
import numpy as np
import yaml
from pathlib import Path

# -------------------------
# 1. Загрузка конфигурации
# -------------------------

CONFIG_PATH = Path("config.yaml")

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)
    print(config)
    
MODEL_PATH = config["model_path"]
THRESHOLD = float(config["threshold"])
MODEL_VERSION = config.get("version", "unknown")


# -------------------------
# 2. Инициализация FastAPI
# -------------------------

app = FastAPI(title="Fraud Detection API")


# -------------------------
# 3. Загрузка модели при старте
# -------------------------

model = CatBoostClassifier()
model.load_model(MODEL_PATH)


# -------------------------
# 4. Pydantic схема
# -------------------------


class Transaction(BaseModel):
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float



@app.post("/predict")
def predict(tx: Transaction):
    # Преобразуем вход в numpy-массив в правильном порядке
    features = np.array([[
        tx.V1, tx.V2, tx.V3, tx.V4, tx.V5, tx.V6,
        tx.V7, tx.V8, tx.V9, tx.V10, tx.V11, tx.V12,
        tx.V13, tx.V14, tx.V15, tx.V16, tx.V17, tx.V18,
        tx.V19, tx.V20, tx.V21, tx.V22, tx.V23, tx.V24,
        tx.V25, tx.V26, tx.V27, tx.V28, tx.Amount
    ]])

    proba = model.predict_proba(features)[0, 1]
    is_fraud = proba >= THRESHOLD

    return {
        "fraud_probability": float(proba),
        "is_fraud": bool(is_fraud),
        "threshold_used": THRESHOLD,
        "model_version": MODEL_VERSION
    }