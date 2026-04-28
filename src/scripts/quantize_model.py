import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import time
import os
import psutil
import mlflow
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models

# REQUISITO PARA O JOBLIB
def create_dnn_model(meta=None):
    n_features = meta["n_features_in_"] if meta else 47 
    model = models.Sequential([
        layers.Dense(64, activation='relu', input_shape=(n_features,)),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def get_ram_usage():
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)

def evaluate_tflite(tflite_model, X_test, y_test):
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    input_idx = interpreter.get_input_details()[0]['index']
    output_idx = interpreter.get_output_details()[0]['index']

    correct = 0
    start_time = time.time()
    ram_initial = get_ram_usage()

    for i in range(len(X_test)):
        input_data = X_test[i:i+1].astype(np.float32)
        interpreter.set_tensor(input_idx, input_data)
        interpreter.invoke()
        output = interpreter.get_tensor(output_idx)
        if (output[0][0] > 0.5) == y_test.iloc[i]:
            correct += 1

    total_time = (time.time() - start_time) / len(X_test)
    ram_final = get_ram_usage()
    return (correct / len(X_test)), total_time, (ram_final - ram_initial)

def gerar_dashboard(results):
    names = [r['name'] for r in results]
    sizes = [float(r['size'].split()[0]) for r in results]
    times = [float(r['time'].split()[0]) for r in results]
    accs = [float(r['acc']) for r in results]

    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    
    ax[0].bar(names, sizes, color=['blue', 'orange', 'green'])
    ax[0].set_title('Tamanho do Modelo (KB)')
    
    ax[1].bar(names, times, color=['blue', 'orange', 'green'])
    ax[1].set_title('Latência (ms/amostra)')

    ax[2].plot(names, accs, marker='o', linestyle='-', color='red')
    ax[2].set_title('Acurácia')
    ax[2].set_ylim([min(accs)-0.01, 1.0])

    plt.tight_layout()
    plt.savefig('dashboard_quantizacao.png')
    print("\n[INFO] Dashboard salvo como 'dashboard_quantizacao.png'")
    plt.show()

def comparar_e_registrar_mlflow(pipeline_path, data_path):
    caminho_db = os.path.abspath(r"src\scripts\mlflow.db")
    mlflow.set_tracking_uri(f"sqlite:///{caminho_db}")
    mlflow.set_experiment("Quantizacao_Deep_Learning")

    pipeline = joblib.load(pipeline_path)
    df = pd.read_csv(data_path)
    X_scaled = pipeline.named_steps['scaler'].transform(df.drop('Is_Fraud', axis=1)).astype(np.float32)
    y = df['Is_Fraud']
    model_keras = pipeline.named_steps['model'].model_

    X_eval, y_eval = X_scaled[:500], y[:500]
    results_to_table = []
    
    configs = [
        ('Float32', None, False),
        ('Float16', [tf.lite.Optimize.DEFAULT], False),
        ('INT8', [tf.lite.Optimize.DEFAULT], True)
    ]

    for name, opt, is_int8 in configs:
        with mlflow.start_run(run_name=f"TFLite_{name}"):
            converter = tf.lite.TFLiteConverter.from_keras_model(model_keras)
            if opt: converter.optimizations = opt
            if name == 'Float16': converter.target_spec.supported_types = [tf.float16]
            if is_int8:
                converter.representative_dataset = lambda: ([X_eval[i:i+1]] for i in range(100))
                converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

            tflite_model = converter.convert()
            acc, t_sample, ram_delta = evaluate_tflite(tflite_model, X_eval, y_eval)
            
            # Salvamento local temporário para o MLflow
            tflite_filename = f"model_{name}.tflite"
            with open(tflite_filename, "wb") as f:
                f.write(tflite_model)

            # Log MLflow
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("latency_ms", t_sample * 1000)
            mlflow.log_metric("size_kb", len(tflite_model)/1024)
            mlflow.log_artifact(tflite_filename)
            
            results_to_table.append({
                'name': name, 'size': f"{len(tflite_model)/1024:.1f} KB",
                'time': f"{t_sample*1000:.3f} ms", 'ram': f"{max(ram_delta, 0.01):.3f} MB", 'acc': f"{acc:.4f}"
            })
            os.remove(tflite_filename)

    # Exibir Tabela
    print('\n' + '='*76)
    print(f'{"Formato":>10} | {"Tamanho":>12} | {"Tempo/amostra":>14} | {"RAM":>8} | {"Acurácia":>9}')
    print('-'*76)
    for r in results_to_table:
        print(f"{r['name']:>10} | {r['size']:>12} | {r['time']:>14} | {r['ram']:>8} | {r['acc']:>9}")
    
    gerar_dashboard(results_to_table)

if __name__ == "__main__":
    comparar_e_registrar_mlflow(r'src/modelos/pipeline_dnn_fraude.pkl', 
                                 r'data/processed/healthcare_fraud_final_model.csv')