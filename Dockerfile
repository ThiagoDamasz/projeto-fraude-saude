# 1. Imagem base oficial do Python (versão slim para ser leve e rápida)
FROM python:3.11-slim

# 2. Define o diretório de trabalho dentro do container
# Tudo o que fizermos agora será dentro da pasta /app no "Linux do Docker"
WORKDIR /app

# 3. Evita que o Python gere arquivos .pyc desnecessários e permite logs em tempo real
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 4. Instala dependências do sistema necessárias para bibliotecas de ML
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 5. Copia primeiro apenas o requirements.txt para aproveitar o cache do Docker
COPY requirements.txt .

# 6. Instala as bibliotecas de Python (Scikit-learn, FastAPI, Uvicorn, etc.)
RUN pip install --no-cache-dir -r requirements.txt

# 7. Copia apenas o necessário para a execução
# Copia o código da aplicação para dentro de /app/src/app
COPY src/app/ ./src/app/

# Copia o modelo específico para a pasta de modelos
COPY src/modelos/pipeline_fraude_saude_v1.pkl ./src/modelos/pipeline_fraude_saude_v1.pkl

# Copia a pasta de experimentos do MLflow e o banco de dados
COPY src/scripts/mlruns/ ./src/scripts/mlruns/
COPY src/scripts/mlflow.db ./src/scripts/mlflow.db

# 8. Expõe a porta 8000 (a mesma que o FastAPI usa)
EXPOSE 8000

# 9. Comando para iniciar a API
CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]