import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, recall_score, precision_score

def run_train():
    # 1. Configuração do MLflow
    mlflow.set_experiment("Monitoramento_Fraude_Saude")
    
    with mlflow.start_run(run_name="Regressao_Logistica_Final"):
        # 2. Carga de Dados
        df = pd.read_csv('../../data/processed/healthcare_fraud_final_model.csv')
        X = df.drop('Is_Fraud', axis=1)
        y = df['Is_Fraud']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # 3. Definição da Pipeline
        # Nota: O Scaler deve ser aplicado apenas nas numéricas
        # Para simplificar no script, vamos escalar tudo o que não for binário
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('resample', SMOTE(random_state=42)),
            ('model', LogisticRegression(max_iter=1000, random_state=42))
        ])

        # 4. Treinamento
        pipeline.fit(X_train, y_train)

        # 5. Avaliação
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        
        metrics = {
            "auc": roc_auc_score(y_test, y_proba),
            "recall": recall_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred)
        }

        # 6. Registro no MLflow
        # Log de parâmetros do modelo
        mlflow.log_params(pipeline.named_steps['model'].get_params())
        mlflow.log_param("sampling_strategy", "SMOTE")
        
        # Log de métricas
        mlflow.log_metrics(metrics)
        
        # Log do Modelo (O arquivo .pkl fica rastreado aqui)
        mlflow.sklearn.log_model(pipeline, "fraud_model_pipeline")
        
        print(f"Treinamento concluído! AUC: {metrics['auc']:.4f}")

if __name__ == "__main__":
    run_train()