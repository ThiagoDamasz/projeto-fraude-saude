import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from scikeras.wrappers import KerasClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline as ImbPipeline # Importante para usar SMOTE
from imblearn.over_sampling import SMOTE
import joblib
import mlflow

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

def treinar_dnn_com_pipeline(data_path):
    mlflow.set_tracking_uri("file:../../mlruns")
    mlflow.set_experiment("DNN_Pipeline_Fraude")

    df = pd.read_csv(data_path)
    X = df.drop('Is_Fraud', axis=1)
    y = df['Is_Fraud']

    # Criando o Wrapper do Keras para o Scikit-Learn
    dnn_clf = KerasClassifier(
        model=create_dnn_model,
        epochs=20,
        batch_size=32,
        verbose=0
    )

    # Construindo o Pipeline (Usamos ImbPipeline por causa do SMOTE)
    pipeline = ImbPipeline([
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=42)),
        ('model', dnn_clf)
    ])

    with mlflow.start_run(run_name="DNN_Full_Pipeline"):
        print("Iniciando treinamento do Pipeline com DNN...")
        pipeline.fit(X, y)
        
        # Salvando o Pipeline completo
        # Nota: O joblib salvará o scaler + o modelo keras dentro dele
        model_path = "src\modelos\pipeline_dnn_fraude.pkl"
        joblib.dump(pipeline, model_path)
        
        mlflow.log_param("model_type", "DNN")
        mlflow.log_artifact(model_path)
        print(f"Pipeline DNN exportado com sucesso: {model_path}")

if __name__ == "__main__":
    DATA_FILE = r'data\processed\healthcare_fraud_final_model.csv'
    treinar_dnn_com_pipeline(DATA_FILE)