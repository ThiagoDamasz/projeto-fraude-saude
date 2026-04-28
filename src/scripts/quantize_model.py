import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import time
import os

def evaluate_tflite_metrics(tflite_model, X_test, y_test):
    """Avalia acurácia e tempo médio de inferência para modelos TFLite."""
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    correct_predictions = 0
    start_time = time.time()

    for i in range(len(X_test)):
        input_data = X_test[i:i+1].astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        prediction = (output_data[0][0] > 0.5).astype(int)
        if prediction == y_test.iloc[i]:
            correct_predictions += 1
            
    end_time = time.time()
    avg_time = (end_time - start_time) / len(X_test)
    accuracy = correct_predictions / len(X_test)
    return accuracy, avg_time

def quantizar_completo(pipeline_path, data_path):
    # 1. Carregar Pipeline e Dados
    pipeline = joblib.load(pipeline_path)
    df = pd.read_csv(data_path)
    X = df.drop('Is_Fraud', axis=1)
    y = df['Is_Fraud']

    # O Pipeline já contém o Scaler, então aplicamos ele
    X_scaled = pipeline.named_steps['scaler'].transform(X).astype(np.float32)
    
    # Extrair o modelo Keras de dentro do Pipeline (SciKeras)
    model_keras = pipeline.named_steps['model'].model_

    # 2. Conversões TFLite
    formats = []
    
    # --- FLOAT 32 ---
    converter = tf.lite.TFLiteConverter.from_keras_model(model_keras)
    tflite_f32 = converter.convert()
    acc_32, time_32 = evaluate_tflite_metrics(tflite_f32, X_scaled[:500], y[:500])
    formats.append(('Float32', tflite_f32, acc_32, time_32))

    # --- FLOAT 16 ---
    converter = tf.lite.TFLiteConverter.from_keras_model(model_keras)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_f16 = converter.convert()
    acc_16, time_16 = evaluate_tflite_metrics(tflite_f16, X_scaled[:500], y[:500])
    formats.append(('Float16', tflite_f16, acc_16, time_16))

    # --- INT8 (PTQ) ---
    converter = tf.lite.TFLiteConverter.from_keras_model(model_keras)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    def representative_data_gen():
        for i in range(100):
            yield [X_scaled[i:i+1]]
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    tflite_int8 = converter.convert()
    acc_i8, time_i8 = evaluate_tflite_metrics(tflite_int8, X_scaled[:500], y[:500])
    formats.append(('INT8', tflite_int8, acc_i8, time_i8))

    # 3. TABELA COMPARATIVA
    print('='*85)
    print(' TABELA COMPARATIVA — FORMATOS DE QUANTIZAÇÃO (DNN FRAUDE)')
    print('='*85)
    print(f'{"Formato":>10} | {"Tamanho":>12} | {"Tempo/amostra":>14} | {"Acurácia":>10} | {"Saving"}')
    print('-'*85)
    
    f32_size = len(tflite_f32)
    for name, model_bytes, acc, t in formats:
        size_kb = len(model_bytes) / 1024
        reduction = f32_size / len(model_bytes)
        print(f'{name:>10} | {size_kb:>9.2f} KB | {t*1000:>11.4f} ms | {acc:>10.4f} | {reduction:>5.1f}x')
    print('='*85)

    # 4. ESTIMATIVA DE CONSUMO (Cortex-M4)
    print('\n' + '='*75)
    print(' ESTIMATIVA DE CONSUMO — 1000 INFERÊNCIAS (Ref: ARM Cortex-M4)')
    print('='*75)
    print(f'{"Formato":>10} | {"Energia":>12} | {"Saving (%)":>12} | {"Inf. por bateria"}')
    print('-'*75)
    
    # Referências proporcionais ao seu exemplo
    energy_ref = {"Float32": 42.0, "Float16": 16.0, "INT8": 5.7}
    inf_ref = {"Float32": "~9.4M", "Float16": "~24.7M", "INT8": "~69.3M"}

    for name, _, _, _ in formats:
        saving = 100 * (1 - (energy_ref[name]/42.0))
        print(f'{name:>10} | {energy_ref[name]:>9.1f} mJ | {saving:>11.1f}% | {inf_ref[name]:>16}')
    print('='*75)

if __name__ == "__main__":
    PIPELINE_PATH = r'src/modelos/pipeline_dnn_fraude.pkl'
    DATA_PATH = r'data/processed/healthcare_fraud_final_model.csv'
    quantizar_completo(PIPELINE_PATH, DATA_PATH)