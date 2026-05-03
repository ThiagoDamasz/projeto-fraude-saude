from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.pipeline import Pipeline
import joblib
import pandas as pd
import os
from typing import Dict, Any

app = FastAPI(title="API de detecção de fraudes de saúde")

# 1. Pega o caminho da pasta onde este arquivo (main.py) está: /app/src/app
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Constrói o caminho subindo um nível (..) para sair de 'app' e entrar em 'modelos'
model_path = os.path.join(BASE_DIR, "..", "modelos", "pipeline_fraude_saude_v1.pkl")

# 3. Carrega o modelo
model = joblib.load(model_path)

# 2. Definir o esquema de entrada (Exemplo com algumas colunas)
class ClaimData(BaseModel):
    Patient_Age: float
    Claim_Amount: float
    Approved_Amount: float
    Days_Between_Service_and_Claim: float
    Number_of_Claims_Per_Provider_Monthly: float
    Prior_Visits_12m: float
    Submission_Month: int
    Submission_DayOfWeek: int
    
    # Campo para capturar todas as colunas dummy (0 ou 1)
    # Usando o Dict o pandas pega todas as colunas dummies e as classifica como int
    features_adicionais: Dict[str, int] 

    class Config:
        # Permite que campos extras sejam enviados no JSON
        extra = "allow"

@app.get("/")
def read_root():
    return {"message": "API de Detecção de Fraude está online!"}

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True}

@app.get("/model-info")
def model_info():
    # Se você estiver usando o pipeline do Scikit-Learn
    # podemos extrair informações dinamicamente do objeto carregado
    scikit_model = model.named_steps['model']
    
    return {
        "model_type": "Logistic Regression",
        "algorithm": "Generalized Linear Model",
        "parameters": {
            "penalty": scikit_model.get_params().get("penalty"),
            "C_regularization": scikit_model.get_params().get("C"),
            "solver": scikit_model.get_params().get("solver")
        },
        "input_features_count": len(model.feature_names_in_) if hasattr(model, 'feature_names_in_') else 47,
        "feature_names": list(model.feature_names_in_) if hasattr(model, 'feature_names_in_') else "Lista de 47 features",
        "output_type": "Probability (0 to 1)",
        "target_class": "Is_Fraud"
    }

@app.post("/predict")
def predict_fraud(data: ClaimData):
    # Converter os dados recebidos para DataFrame (formato que o pipeline aceita)
    input_df = pd.DataFrame([data.dict()])
    
    # Fazer a predição
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    
    return {
        "is_fraud": int(prediction),
        "fraud_probability": round(float(probability), 4),
        "status": "High Risk" if prediction == 1 else "Normal"
    }