from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os
from typing import Dict, Any

app = FastAPI(title="API de detecção de fraudes de saúde")

# 1. Carregar o modelo ao iniciar a API
# Ajuste o caminho conforme sua estrutura de pastas
model_path = os.path.join(os.path.dirname(__file__), r'C:\Users\User\Desktop\projetos\projeto-fraude-saude\src\modelos\pipeline_fraude_saude_v1.pkl')
model = joblib.load(model_path)

# 2. Definir o esquema de entrada (Exemplo com algumas colunas)
# Você deve incluir aqui todas as colunas que o modelo espera (as 47 colunas)
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