🩺 Healthcare Fraud Detection System

Este projeto utiliza Machine Learning para identificar fraudes em cobranças de planos de saúde. O sistema abrange desde a análise exploratória e pré-processamento de dados até o deploy de uma API escalável e monitoramento de experimentos.

🚀 Tecnologias Utilizadas
Linguagem: Python 3.10+

Análise & ML: Pandas, Scikit-Learn, Imbalanced-learn (SMOTE)

Tracking: MLflow (Gestão de experimentos e métricas)

API: FastAPI & Uvicorn

Ambiente: Jupyter Notebooks (Desenvolvimento Inicial)

📊 Resultados do Modelo
O modelo de Regressão Logística foi treinado utilizando uma abordagem de Pipeline com normalização e balanceamento sintético.

Métrica,Resultado
AUC-ROC,0.9982
Recall (Fraude),94%
Precisão (Fraude),87%
F1-Score,0.90

Nota: O alto valor de Recall é fundamental neste domínio para garantir que a grande maioria das fraudes seja detectada, minimizando prejuízos financeiros.

🏗️ Estrutura do Projeto

├── data/
│   ├── raw/            # Dados brutos originais
│   └── processed/      # Dados limpos e preparados para o modelo
├── models/             # Exportação da pipeline (.pkl)
├── notebooks/          # EDA e Treinamento Experimental
├── src/
│   ├── app/            # Código da API (FastAPI)
│   └── scripts/        # Scripts de automação (Treinamento MLflow)
└── requirements.txt    # Dependências do projeto

🛠️ Como Executar

1. Instalação

pip install -r requirements.txt

2. Treinamento e Monitoramento (MLflow)
Para treinar o modelo e registrar as métricas na interface do MLflow:

python src/scripts/train_model.py
mlflow ui

Acesse http://localhost:5000 para comparar as execuções.

3. Rodando a API (Local)
Para subir o serviço de predição:

cd src/app
uvicorn main:app --reload

Acesse a documentação interativa em http://localhost:8000/docs.

🧠 Metodologia Aplicada
EDA: Análise de correlação e identificação de desbalanceamento severo (apenas 8% de fraude).

Pré-processamento: Tratamento de nulos, One-Hot Encoding e remoção de redundâncias.

Pipeline: Implementação de StandardScaler e SMOTE integrados para evitar data leakage.

Validação: Validação Cruzada (K-Fold) e avaliação via Curva ROC.

Desenvolvido por: Thiago Damas Ferreira Silva

Software Engineering Student @ Inatel