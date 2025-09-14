# 🧠 Diagnóstico de Câncer de Mama com IA e Python

Este projeto tem como objetivo aplicar técnicas de Inteligência Artificial e Engenharia de Dados para diagnosticar tumores de mama como **malignos** ou **benignos**, utilizando a base de dados [Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic). O projeto é dividido em duas partes: análise e modelagem em Jupyter Notebook, e disponibilização do modelo via API com Python.

---

## 📁 Estrutura de Diretórios

```
tech-challenge-3/
├── data/              # Dados brutos e tratados
├── notebooks/         # Jupyter Notebooks com análise e modelagem
├── src/               # Scripts de pré-processamento e treinamento
│   ├── data/          # Script para importar e salvar o dataset
│   └── eda/           # Funções de Análise Exploratória de Dados (EDA)
├── models/            # Modelos treinados (.pkl ou .joblib)
├── api/               # Código da API FastAPI
├── tests/             # Testes unitários e de integração
├── Dockerfile         # Containerização da API (opcional)
├── requirements.txt   # Dependências do projeto
└── README.md          # Documentação principal
```

## 📘 Etapas do Projeto

### 🔬 Jupyter Notebook

#### 1. Apresentação da Base de Dados
- Análise estatística dos atributos
- Identificação da variável alvo (`Diagnosis`)
- Limpeza de dados (remoção de colunas irrelevantes, tratamento de valores nulos)
- Normalização ou padronização dos dados (com justificativa técnica)
- Definição da métrica de avaliação (ex: F1-score, AUC, acurácia)

### 2. Seleção de Variáveis e Interpretação

**Objetivo.** Reduzir o número de variáveis **sem perda relevante de desempenho**, priorizando **Recall da classe Maligno** e mantendo interpretabilidade.

**Caminho seguido.**
1. **EDA e qualidade dos dados:** ausência de valores ausentes; assimetrias/caudas tratáveis. Para modelos lineares, usamos **imputação mediana + Yeo–Johnson**; para árvores/boosting, apenas **imputação**.
2. **PCA para interpretação (não para redução):** *scree plot*, dispersão PC1×PC2, **círculo de correlações** e **contribuições/cos²** para compreender famílias de variáveis (tamanho/escala, irregularidade/forma, textura/suavidade/simetria, fractalidade).
3. **Ranqueamento supervisionado de variáveis:** |correlação| com o alvo (ponto-bisserial), **Mutual Information**, **contribuições PCA** (PC1/PC2) e **cos²**, compondo um **score**.
4. **Poda por correlação (Spearman):** remoção gulosa de redundâncias com **|ρ| ≥ 0,90**.
5. **Seletores embutidos (SelectFromModel) com CV:** **LR L1**, **LR ElasticNet**, **Random Forest** e **XGBoost**. Calculamos a **frequência de seleção** por variável (estabilidade) ao longo dos folds/métodos.
6. **Conciliação final:** priorizamos variáveis mais **estáveis** (maior *mean_freq*) e reaplicamos a **poda por correlação**.

**Painéis avaliados em CV (5 folds; modelos: LR_L2, LR_EN, RF, XGB; métrica primária = Recall Maligno):**
- **Full** (todas as variáveis);
- **FinalSelected16** (16 variáveis);
- **MoreStable12** → **12 variáveis** (poda estrita):

```text
['worst_concavity', 'worst_area', 'worst_texture', 'worst_smoothness',
'mean_concave_points', 'area_error', 'mean_compactness', 'compactness_error',
'symmetry_error', 'worst_symmetry', 'texture_error', 'concave_points_error']

```

**Resumo dos resultados (métrica primária Recall):**
- **LR_L2**: MoreStable12 **0.9717** = FinalSelected16 **0.9717** (empate).  
  AUC: MoreStable12 **0.9974** > 0.9970.  
  AP: MoreStable12 **0.9966** > 0.9962. 
- **LR_EN**: MoreStable12 **0.9717** > 0.9670.  
  AUC: MoreStable12 **0.9972** > 0.9968.  
  AP: MoreStable12 **0.9963** > 0.9959. 
- **RF**: MoreStable12 **0,9434** > 0,9340.  
  AUC: MoreStable12 **0.9952** > 0.9942.  
  AP: MoreStable12 **0.9931** > 0.9924.  
- **XGB**: MoreStable12 **0.9763** > 0.9623.  
  AUC: MoreStable12 **0.9962** > 0.9958.  
  AP: MoreStable12 **0.9946** > 0.9942. 


**Escolhemos o painel _MoreStable12_ com 12 variáveis.**  

Motivos:
1. **Recall** (métrica primária) **≥** FinalSelected16 em **todos** os modelos.
2. **AUC/AP** invariavelmente **melhores** também em todos os modelos.
3. **Menos variáveis** (12 vs 16) ⇒ **menos custo** e **mais usabilidade** (formulário enxuto).
4. Mantém **cobertura** das famílias de sinal (tamanho/escala; irregularidade/forma; textura/suavidade/simetria; variabilidade/error).


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

| Categoria                      | Ferramentas                                                                                    |
|--------------------------------|-------------------------------------------------------------------------------------------------|
| Linguagem                      | Python 3.10+                                                                                   |
| Análise de Dados               | Pandas, NumPy, SciPy, Matplotlib (Seaborn opcional)                                            |
| Machine Learning               | Scikit-learn, XGBoost                                                                          |
| Seleção de Variáveis & Interpretação | SelectFromModel (LR L1/ElasticNet, RF, XGB), Mutual Information, Correlação (Spearman), **PCA para interpretação** |
| API Web                        | FastAPI, Uvicorn                                                                               |
| Deploy                         | Docker *(opcional)*                                                                            |


---

## 📦 Como Executar

### 1. Clonar o repositório
```bash
git clone https://github.com/djflucena/tech-challenge-3.git
cd tech-challenge-3
```
