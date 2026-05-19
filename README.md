# 🛡️ Sistema Inteligente de Detecção de Fraudes em Saúde

Este projeto consiste em uma solução de ponta a ponta (End-to-End) para a detecção de fraudes em sinistros e transações de saúde corporativa. A arquitetura foi desenhada seguindo as melhores práticas de **MLOps**, utilizando rastreamento de experimentos, otimização hiperparamétrica automatizada, deploy conteinerizado e monitoramento estatístico de dados em produção.

---

## 🏗️ Arquitetura do Projeto

O ecossistema é dividido em três pilares principais:
1. **Treinamento e Otimização:** Busca automatizada da melhor arquitetura de Rede Neural (MLP) utilizando Optuna, gerenciada por arquivos de configuração dinâmicos (Hydra) e registrada via MLflow.
2. **Serviço de Predição (API):** Uma API de alta performance desenvolvida em FastAPI que carrega o modelo otimizado e serve predições em tempo real.
3. **Monitoramento (Data Drift):** Script estatístico para identificar desvios no perfil dos dados de entrada comparado ao histórico de treino, prevenindo a degradação do modelo.

---

## 🛠️ Tecnologias Utilizadas

* **Linguagem Principal:** Python 3.13
* **Framework de IA:** TensorFlow / Keras (Redes Neurais Multicamadas)
* **Gerenciamento de Experimentos:** MLflow
* **Otimização de Hiperparâmetros:** Optuna
* **Configuração Dinâmica:** Hydra & OmegaConf
* **Desenvolvimento da API:** FastAPI & Uvicorn
* **Análise de Drift:** Evidently AI
* **Conteinerização:** Docker & Docker Desktop

---

## 📁 Estrutura de Diretórios

```text
projeto-fraude-saude/
├── conf/                         # Configurações globais do Hydra (.yaml)
│   └── config.yaml
├── data/                         # Bases de dados (Ignoradas no Git)
│   ├── train.csv
│   └── logs/                     # Dados coletados em produção pela API
├── src/
│   ├── app/                      # Código fonte da API FastAPI
│   │   ├── __init__.py
│   │   └── main.py
│   ├── otimizacao/               # Scripts de treinamento e Optuna
│   │   ├── __init__.py
│   │   └── otm.py
│   └── monitoring/               # Validação e integridade do modelo
│       ├── __init__.py
│       └── drift_analysis.py
├── Dockerfile                    # Instruções de build do container
├── requirements.txt              # Dependências do projeto
└── README.md
```
## 🚀 Como Executar o Projeto Localmente

### Pré-requisitos
* Python 3.13 instalado.
* Docker Desktop rodando (caso vá utilizar o ambiente conteinerizado).

### 1. Configurando o Ambiente Virtual
No terminal, na raiz do projeto, crie e ative seu ambiente virtual:

```bash
python -m venv venv

# No Windows (PowerShell):
.\venv\Scripts\activate

# No Linux/Mac:
source venv/bin/activate
```
Instale todas as dependências necessárias:
```bash
pip install -r requirements.txt
```

### 2. Executando o Ciclo de Otimização (Optuna + MLflow)
Antes de rodar o treinamento, inicie o servidor do MLflow em um terminal separado para acompanhar os gráficos de evolução:

```bash
mlflow ui --port 5000
```
Agora, execute o script de otimização para treinar os modelos e buscar os melhores hiperparâmetros:

```bash
python src/otimizacao/otm.py
```

💡 Dica: Você pode sobrescrever qualquer parâmetro do arquivo config.yaml direto pelo terminal graças ao Hydra, por exemplo:
python src/otimizacao/otm.py optuna.n_trials=50

### 3. Rodando a API Localmente
Para testar a API localmente via Uvicorn antes de buildar o container:

```bash
uvicorn src.app.main:app --reload
```
Acesse a documentação interativa da API (Swagger UI) em: http://localhost:8000/docs

## 🐳 Executando com Docker
Para garantir que a aplicação rode perfeitamente em qualquer ambiente corporativo, a API foi totalmente conteinerizada.

1. Construir a Imagem Docker
```bash
docker build -t projeto-fraude-api .
```
2. Inicializar o Container Ativo
Execute o container mapeando as portas de comunicação com o seu sistema operacional:
```bash
docker run -d -p 8000:8000 --name api-fraude-ativa projeto-fraude-api
```
3. Comandos Úteis de Gerenciamento
- Verificar se o container está de pé: docker ps

- Checar logs internos de inicialização/erro: docker logs api-fraude-ativa

- Acompanhar logs em tempo real: docker logs -f api-fraude-ativa

- Parar o serviço: docker stop api-fraude-ativa

- Iniciar um container existente: docker start api-fraude-ativa

## 📊 Monitoramento de Data Drift
Os fraudadores mudam de tática constantemente, o que gera o fenômeno de Concept/Data Drift (mudança no padrão dos dados). Para garantir a saúde do modelo, o script de monitoramento compara estatisticamente a base histórica de treino com os dados reais salvos pela API.

Para gerar um relatório visual completo de integridade
```bash
python src/monitoring/drift_analysis.pydocker run -d -p 8000:8000 --name api-fraude-ativa projeto-fraude-api
```
O script gerará um dashboard interativo em reports/drift_report.html, indicando se há necessidade ou não de disparar um novo gatilho de retreinamento automatizado.

Desenvolvido como projeto de Engenharia de Software / Inteligência Artificial.

