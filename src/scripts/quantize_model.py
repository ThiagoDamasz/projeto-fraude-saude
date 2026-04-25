import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import mlflow
import os

def quantizar_com_tflite(model_path, data_path):
    mlflow.set_tracking_uri("file:../../mlruns")
    mlflow.set_experiment("Quantizacao_TFLite_Fraude")

    with mlflow.start_run(run_name="TFLite_INT8_PTQ"):
        # 1. Carregar Pipeline e Dados
        pipeline = joblib.load(model_path)
        df = pd.read_csv(data_path)
        X_test = df.drop('Is_Fraud', axis=1).values.astype(np.float32)

        # 2. Criar um modelo Keras equivalente à sua Regressão Logística
        # Uma Regressão Logística é apenas uma camada Dense com ativação Sigmoid
        model_keras = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(X_test.shape[1],)),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        # Importante: Transferir os pesos do Scikit-Learn para o Keras
        scikit_model = pipeline.named_steps['model']
        weights = [scikit_model.coef_.T, scikit_model.intercept_]
        model_keras.set_weights(weights)

        # 3. Configurar o Conversor TFLite para Quantização INT8
        converter = tf.lite.TFLiteConverter.from_keras_model(model_keras)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Para INT8 puro, precisamos de um "Representative Dataset" 
        # (Isso ajuda o TFLite a entender a escala dos seus dados reais)
        def representative_data_gen():
            for i in range(100):
                yield [X_test[i:i+1]]

        converter.representative_dataset = representative_data_gen
        # Forçar a saída para INT8
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8

        # 4. Converter
        tflite_model_quant = converter.convert()

        # 5. Salvar e Registrar
        tflite_path = r"src\modelos\model_quantized.tflite"
        with open(tflite_path, "wb") as f:
            f.write(tflite_model_quant)
        
        mlflow.log_artifact(tflite_path)
        print(f"Conversão TFLite INT8 concluída! Tamanho: {len(tflite_model_quant) / 1024:.2f} KB")

if __name__ == "__main__":
    MODEL_FILE = r'src/modelos/pipeline_fraude_saude_v1.pkl'
    DATA_FILE = r'data/processed/healthcare_fraud_final_model.csv'
    quantizar_com_tflite(MODEL_FILE, DATA_FILE)