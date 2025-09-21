# paginas/inicio.py
from dash import html, dcc
import dash_bootstrap_components as dbc
import pandas as pd

from app import app
from paginas.compartilhado import (
    load_pipeline,
    load_metrics,
    load_hyperparams,
    metrics_tables
)

# Carrega artefatos
pipe = load_pipeline()
rep = load_metrics()
hyper = load_hyperparams()
CLASS, AGG, ACC, TOT = metrics_tables(rep)


data_card = dbc.Card(dbc.CardBody([
    html.H5("De onde vêm os dados", className="card-title"),
    dcc.Markdown(
        "- **Fonte:** *Breast Cancer Wisconsin (Diagnostic)* — carregado de `data/wdbc.csv` "
        "ou via `sklearn.datasets.load_breast_cancer` quando o CSV não está presente. "
        "- **Amostras & atributos:** **569** observações, **30** atributos numéricos "
        "(estatísticas de textura/forma do tecido: *mean*, *error*, *worst*).\n"
        "- **Alvo (`diagnosis`):** binário — **1 = Maligno**, **0 = Benigno**.\n"
        "- **Distribuição de classes (aprox.):** ~**37%** Maligno • ~**63%** Benigno "
        "(desbalanceamento moderado).\n"
        "- **Artefatos carregados no app:** pipeline treinado (`.joblib`), métricas de teste "
        "(`metrics_*.json`), hiperparâmetros não-padrão e faixas **min–max** por variável — "
        "todos em `dash_app/models/`.\n\n"
        "**Reposítório oficial:** [UCI Repository - Breast Cancer](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)\n"

    ),
]), className="mb-4")


overview_card = dbc.Card(dbc.CardBody([
    html.H5("Modelo e pipeline (resumo)", className="card-title"),
    dcc.Markdown(
        "- **Pré-processamento (numéricas):** `SimpleImputer(median)` → `PowerTransformer (Yeo–Johnson)`\n"
        "- **Algoritmo:** `LogisticRegression` (**L2**, classe positiva = Maligno).\n"
        "- **Painel de variáveis:** **MoreStable12** (seleção por estabilidade + poda por correlação ≤ 0,90).\n"
        "- **Ajuste de hiperparâmetros:** duas etapas — 1) **RandomizedSearchCV** (recall); 2) **GridSearchCV** fino.\n"
        "- **Validação:** **CV estratificada (5 folds)** e ajuste de **limiar por OOF** (usado para análise; decisão final com `thr=0.50`).\n"
        "- **Métricas foco:** **Recall** (primária - alvo ≥ 0.95), ROC-AUC, AP (PR-AUC) e F1.\n"
        "- **Guard-rails:** ΔRecall ≤ 0,02 e ΔAUC ≤ 0,01 vs. modelo completo (segurança clínica).\n\n"
        "**Projeto:** [github.com/djflucena/tech-challenge-3](https://github.com/djflucena/tech-challenge-3)"
    ),
]), className="mb-4")


metrics_card = dbc.Card(dbc.CardBody([
    html.H5("Desempenho (conjunto de teste)", className="card-title"),
    dbc.Row([
        dbc.Col([
            html.H6("Por classe"),
            dbc.Table.from_dataframe(CLASS.round(4), striped=True, bordered=True, hover=True, size="sm")
        ], md=7),
        dbc.Col([
            html.H6("Médias"),
            dbc.Table.from_dataframe(AGG.round(4), striped=True, bordered=True, hover=True, size="sm"),
            html.Div([
                dbc.Badge(f"Acurácia: {ACC:.2%}", color="info", className="me-2"),
                html.Small(f"Relatório em {TOT} amostras de teste.", className="text-muted")
            ], className="mt-2")
        ], md=5),
    ])
]), className="mb-4")

params_df = pd.DataFrame(
    [{"Hiperparâmetro": k, "Valor": v} for k, v in hyper.items()]
)

params_card = dbc.Card(dbc.CardBody([
    html.H5("Hiperparâmetros ajustados", className="card-title"),
    dbc.Table.from_dataframe(
        params_df, striped=True, bordered=True, hover=True, size="sm"
    ),
]), className="mb-4")

layout = dbc.Container([
    html.H2("Previsão de Câncer de Mama — visão geral", className="text-center mb-3 mt-2"),
    dbc.Row([
        dbc.Col(data_card, md=6),
        dbc.Col(overview_card, md=6),
    ], className="g-3"),
    dbc.Row([
        dbc.Col(metrics_card, md=8),
        dbc.Col(params_card, md=4),
    ], className="g-3"),
], fluid=True)

