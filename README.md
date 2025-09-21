# 🧠 Diagnóstico de Câncer de Mama com IA e Python

Este projeto tem como objetivo aplicar técnicas de Inteligência Artificial e Engenharia de Dados para diagnosticar tumores de mama como **malignos** ou **benignos**, utilizando a base de dados [Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic). O projeto é dividido em duas partes: análise e modelagem em Jupyter Notebook, e disponibilização do modelo via API com Python.

---

## 📁 Estrutura de Diretórios

```
tech-challenge-3/
├── dash_app             # Dashboard para consumir o modelo
│   ├── assets           # Recursos estáticos
│   ├── models           # Modelos treinados (.pkl ou .joblib) e arquivos json auxiliares
│   ├── paginas          # Paǵinas HTML
│   ├── requirements.txt # Dependências do Dashboard
│   └── Dockerfile       # Containerização do Dashboard
├── data                 # Dados brutos e tratados
├── src/                 # Scripts de pré-processamento e treinamento
│   ├── data/            # Script para importar e salvar o dataset
│   ├── eda/             # Funções de Análise Exploratória de Dados (EDA)
│   └── plots            # Gráficos de avaliação de modelos
├── notebooks            # Jupyter Notebooks com análise e modelagem
├── tests/               # Testes unitários e de integração
├── requirements.txt     # Dependências do projeto
└── README.md            # Documentação principal
```

## 📘 Etapas do Projeto

### 🔬 Jupyter Notebook

Notebook: `notebooks/01_exploracao_modelagem.ipynb`

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
| Modelo | Recall (MoreStable12 × FinalSelected16) | AUC (MoreStable12 × FinalSelected16) | AP (MoreStable12 × FinalSelected16) |
|--------|------------------------------------------|---------------------------------------|-------------------------------------|
| LR_L2  | **0.9717** = 0.9717 (empate)             | **0.9974** > 0.9970                   | **0.9966** > 0.9962                 |
| LR_EN  | **0.9717** > 0.9670                      | **0.9972** > 0.9968                   | **0.9963** > 0.9959                 |
| RF     | **0.9434** > 0.9340                      | **0.9952** > 0.9942                   | **0.9931** > 0.9924                 |
| XGB    | **0.9763** > 0.9623                      | **0.9962** > 0.9958                   | **0.9946** > 0.9942                 |


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

> **Confiabilidade do sinal (Explicabilidade):** A análise com **DALEx** confirmou os sinais aprendidos pelo **LR L2 + MoreStable12** (Permutation Importance, PDP/ICE e Break Down coerentes com coeficientes e PCA). As divergências pontuais entre **importância por permutação** e **coeficientes** são esperadas devido a colinearidade/redundância e refletem impacto **no desempenho global** vs **peso linear**.


### 5. Explicabilidade do Modelo — *DALEx*

Notebook: `notebooks/02_explicabilidade_dalex.ipynb`

**Objetivo.** Explicar **global** e **localmente** o modelo final (**LR L2 + MoreStable12**), validando se os sinais aprendidos são clinicamente coerentes e consistentes com as análises anteriores.

**O que entregamos:**

* **Permutation Importance** (baseada em queda de **AP/PR-AUC**): mede impacto prático de cada variável no desempenho global ao embaralhar seus valores.
* **PDP/ICE** (parciais e individuais): mostram como a probabilidade prevista varia ao longo do domínio de cada variável (efeitos médios e heterogeneidade entre observações).
* **Break Down** (local): decompõe a previsão de **6 amostras** (p_min, p_max e ~0.20/0.40/0.60/0.80) em contribuições pró/contra **Maligno**, partindo do intercepto.

**Principais insights:**

* **Permutation Importance vs. Coeficientes LR**: podem divergir — coeficientes medem **peso linear** condicionado aos demais preditores; permutação mede **efeito no desempenho** (captura interação/colinearidade e redundâncias).
* Variáveis de "pior caso" (**worst_area**, **worst_texture**, **worst_concavity**) e erros (**area_error**, **concave_points_error**) aparecem **entre as mais influentes**, coerente com os coeficientes e com o PCA interpretativo.
* Nos **Break Down**, casos de baixa probabilidade têm **medidas "pior" baixas** (empurram para **Benigno**); casos altos combinam valores "pior" e "erro" elevados (empurram para **Maligno**).

---

### 🌐 Projeto Web com Python

#### 1. Criação de Dashboard Interativo
- Implementação com **Dash** + **Bootstrap**
- Visualização de métricas e hiperparâmetros do modelo LR_L2 (MoreStable12)
- Exploração interativa de variáveis e faixas
- Formulário para inferência em tempo real
- Preparação para execução local e em contêiner Docker

##### Funcionalidades:
- **Página Início**: Métricas do modelo, hiperparâmetros e informações sobre o painel MoreStable12
- **Página Formulário**: Inputs para as 12 variáveis com validação e botão de predição
- **Página Gráficos**: Importância das variáveis e histogramas exploratórios

#### 2. Funcionalidades Adicionais *(opcionais)*
- Autenticação via JWT
- Integração com Gateway/API Manager

---

## 🛠 Tecnologias Utilizadas

| Categoria                      | Ferramentas                                                                                    |
|--------------------------------|-------------------------------------------------------------------------------------------------|
| Linguagem                      | Python 3.10+                                                                                   |
| Análise de Dados               | Pandas, NumPy, SciPy, Matplotlib e Seaborn                                                     |
| Machine Learning               | Scikit-learn, XGBoost                                                                          |
| Seleção de Variáveis & Interpretação | SelectFromModel (LR L1/ElasticNet, RF, XGB), Mutual Information, Correlação (Spearman), **PCA para interpretação** |
| Explicabilidade                | DALEx (Permutation Importance, PDP/ICE, Break Down)                                            |
| Dashboard                      | Dash, Dash-bootstrap-components, Plotly, Flask e Gunicorn                                      |
| Deploy                         | Docker *(opcional)*                                                                            |


---

## 📦 Como Executar

### 1. Clonar o repositório
```bash
git clone https://github.com/djflucena/tech-challenge-3.git
cd tech-challenge-3
```
