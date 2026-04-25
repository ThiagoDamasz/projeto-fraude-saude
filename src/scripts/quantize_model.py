import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import roc_auc_score, recall_score
import os

def quantizacao_int8_modelo(model_path, data_path):

    # 1. Carregar o Pipeline e os Dados
    pipeline = joblib.load(model_path)
    df = pd.read_csv(data_path)

    X_test = df.drop('Is_Fraud', axis=1)
    y_test = df['Is_Fraud']

    # 2. Extrair pesos do modelo original (Logistic Regression)
    # Acessamos o 'model' dentro da pipeline
    original_weights = pipeline.named_steps['model'].coef_
    original_intercept = pipeline.named_steps['model'].intercept_

    # 3. Processo de Quantização (Escalonamento para -128 a 127)
    # Encontramos o fator de escala baseado no valor absoluto máximo
    max_val = np.max(np.abs(original_weights))
    scale = 127 / max_val

    # Convertendo para INT8
    quantized_weights = np.round(original_weights * scale).astype(np.int8)
    quantized_intercept = np.round(original_intercept * scale).astype(np.int8)

    print(f"Fator de Escala aplicado: {scale:.4f}")
    print(f"Peso original (exemplo): {original_weights[0][0]:.6f}")
    print(f"Peso quantizado INT8: {quantized_weights[0][0]}")

    # 4. Simular Predição no Hardware (usando os pesos inteiros)
    # Primeiro, passamos os dados pelo Scaler da pipeline (isso ainda é float)
    X_test_scaled = pipeline.named_steps['scaler'].transform(X_test)

    # Cálculo da decisão: (X * Pesos_Int8) + Intercept_Int8
    # Na prática, o hardware faria isso via instruções de inteiros
    raw_scores = np.dot(X_test_scaled, quantized_weights.T) + quantized_intercept

    # Aplicar a função Sigmóide para obter probabilidades
    y_proba_quantized = 1 / (1 + np.exp(-raw_scores / scale)) # Re-escalamos para a métrica
    y_pred_quantized = (y_proba_quantized > 0.5).astype(int)

    # 5. Comparar Resultados
    auc_q = roc_auc_score(y_test, y_proba_quantized)
    recall_q = recall_score(y_test, y_pred_quantized)

    print("\n=== RESULTADOS PÓS-QUANTIZAÇÃO (INT8) ===")
    print(f"Novo AUC: {auc_q:.4f}")
    print(f"Novo Recall: {recall_q:.4f}")

    # Salvar pesos quantizados para simular o "Firmware"
    np.save('C:\\Users\\Thiag\\OneDrive\\Área de Trabalho\\projetos-IA\\projeto-fraude-saude\\src\\modelos\\weights_int8.npy', quantized_weights, quantized_weights)
    print("\nArquivo weights_int8.npy salvo para uso em hardware.")

if __name__ == "__main__":
    MODEL_FILE = 'C:\\Users\\Thiag\\OneDrive\\Área de Trabalho\\projetos-IA\\projeto-fraude-saude\\src\\modelos\\pipeline_fraude_saude_v1.pkl'
    DATA_FILE = 'C:\\Users\\Thiag\\OneDrive\\Área de Trabalho\\projetos-IA\\projeto-fraude-saude\\data\\processed\\healthcare_fraud_final_model.csv'
    quantizacao_int8_modelo(MODEL_FILE, DATA_FILE)
