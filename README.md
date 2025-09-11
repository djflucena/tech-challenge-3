# 🧠 Diagnóstico de Câncer de Mama com IA e Python

Este projeto tem como objetivo aplicar técnicas de Inteligência Artificial e Engenharia de Dados para diagnosticar tumores de mama como **malignos** ou **benignos**, utilizando a base de dados [Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic). O projeto é dividido em duas partes: análise e modelagem em Jupyter Notebook, e disponibilização do modelo via API com Python.

---

## 📁 Estrutura de Diretórios

tech-challenge-3/  
├── data/                  # Dados brutos e tratados  
├── notebooks/             # Jupyter Notebook com análise e modelagem  
├── src/                   # Scripts de pré-processamento e treinamento  
├── models/                # Modelos treinados (.pkl ou .joblib)  
├── api/                   # Código da API FastAPI  
├── tests/                 # Testes unitários e de integração  
├── Dockerfile             # Containerização da API (opcional)  
├── requirements.txt       # Dependências do projeto  
└── README.md              # Documentação principal  
  
## 📘 Etapas do Projeto

### 🔬 Jupyter Notebook

#### 1. Apresentação da Base de Dados
- Análise estatística dos atributos
- Identificação da variável alvo (`Diagnosis`)
- Limpeza de dados (remoção de colunas irrelevantes, tratamento de valores nulos)
- Normalização ou padronização dos dados (com justificativa técnica)
- Definição da métrica de avaliação (ex: F1-score, AUC, acurácia)

#### 2. Redução de Dimensionalidade
- Comparação entre:
  - Métodos de seleção de atributos (ex: `SelectKBest`, `RFE`)
  - Métodos de transformação (ex: PCA) *(opcional)*

#### 3. Avaliação de Modelos
- Treinamento e ajuste de hiperparâmetros para:
  - Modelo baseado em função (ex: SVM ou Regressão Logística)
  - Modelo baseado em árvore (ex: Random Forest ou XGBoost)
  - Modelo baseado em rede neural *(opcional)*

#### 4. Escolha do Melhor Modelo
- Comparação de desempenho entre os modelos
- Justificativa da escolha com base nas métricas

---

### 🌐 Projeto Web com Python

#### 1. Criação de API REST
- Implementação com **FastAPI**
- Endpoint para envio de dados e retorno do diagnóstico
- Formato de comunicação: JSON

#### 2. Funcionalidades Adicionais *(opcionais)*
- Autenticação via JWT
- Integração com Gateway/API Manager

---

## 🛠 Tecnologias Utilizadas

| Categoria         | Ferramentas                     |
|------------------|----------------------------------|
| Linguagem         | Python 3.10+                    |
| Análise de Dados  | Pandas, NumPy, Seaborn          |
| Machine Learning  | Scikit-learn, XGBoost, Keras    |
| Redução Dimensional| PCA, SelectKBest, RFE          |
| API Web           | FastAPI, Uvicorn                |
| Deploy            | Docker *(opcional)*             |

---

## 📦 Como Executar

### 1. Clonar o repositório
```bash
git clone https://github.com/djflucena/tech-challenge-3.git
cd tech-challenge-3
