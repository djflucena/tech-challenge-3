# paginas/compartilhado.py
from __future__ import annotations

from functools import lru_cache
from typing import Tuple, Dict, List
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.datasets import load_breast_cancer

# Explicabilidade: Explainer e gráfico de breakdown
import dalex as dx
import plotly.graph_objects as go


# Caminhos padrão (relativos à raiz da app)
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = MODELS_DIR / "lr_l2_morestable12.joblib"
METRICS_PATH = MODELS_DIR / "metrics_lr_l2_classification_report.json"
HYPER_PATH = MODELS_DIR / "hyperparams_lr_l2.json"
FEATURES_PATH = MODELS_DIR / "feature_order_more_stable_12.json"
RANGES_PATH = MODELS_DIR / "variables_range_more_stable_12.json"


# =========================
# Carregadores com cache
# =========================

@lru_cache(maxsize=1)
def load_pipeline():
    """
    Carrega o pipeline treinado (pré-processamento + classificador) via joblib.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Pipeline completo usado na predição.
    """
    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def load_metrics():
    """
    Carrega o relatório de métricas de teste (JSON exportado do treino).

    Returns
    -------
    dict
        Dicionário com métricas (ex.: classification_report serializado).
    """
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def load_hyperparams():
    """
    Carrega hiperparâmetros ajustados do modelo (JSON).

    Returns
    -------
    dict
        Hiperparâmetros não padrão utilizados no ajuste/treino.
    """
    with open(HYPER_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def load_feature_order() -> List[str]:
    """
    Carrega a ordem das features utilizada no treino/inferência.

    Returns
    -------
    list[str]
        Lista com os nomes das colunas na ordem esperada pelo pipeline.
    """
    with open(FEATURES_PATH, "r", encoding="utf-8") as f:
        return list(json.load(f))


@lru_cache(maxsize=1)
def load_ranges() -> Dict[str, Dict[str, float]]:
    """
    Carrega os intervalos (min–max) históricos por variável.

    Returns
    -------
    dict[str, dict[str, float]]
        Mapeamento coluna -> {'min': float, 'max': float}.
    """
    with open(RANGES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# =========================
# Dataset BRCA (sklearn)
# =========================

@lru_cache(maxsize=1)
def load_brca_df() -> pd.DataFrame:
    """
    Carrega o dataset Breast Cancer (sklearn) já normalizado para o projeto.

    Ajustes aplicados:
      - Substitui espaços em nomes de colunas por '_';
      - Cria coluna 'diagnosis' (1=Maligno, 0=Benigno);
      - Cria coluna 'diagnosis_label' ('Maligno'/'Benigno').

    Returns
    -------
    pandas.DataFrame
        DataFrame completo com preditoras + colunas auxiliares ('diagnosis', 'diagnosis_label').
    """
    brca = load_breast_cancer(as_frame=True)
    df = brca.frame.copy()
    df.columns = [c.replace(" ", "_") for c in df.columns]
    idx_malignant = int(np.where(brca.target_names == "malignant")[0][0])
    df["diagnosis"] = (brca.target == idx_malignant).astype(int)
    df["diagnosis_label"] = df["diagnosis"].map({0: "Benigno", 1: "Maligno"})
    return df


@lru_cache(maxsize=1)
def get_brca_for_plots():
    """
    Prepara artefatos para a página de gráficos.

    Returns
    -------
    tuple
        (df_brca, order_by_importance, coef_df), onde:
        - df_brca : pandas.DataFrame
            Dataset BRCA preparado (ver `load_brca_df`).
        - order_by_importance : list[str]
            Lista de features ordenadas por |coef| (impacto da LR em módulo),
            filtrando apenas as colunas presentes no df_brca.
        - coef_df : pandas.DataFrame
            Tabela com coeficientes da LR (colunas: feature, coef, coef_abs, sinal).
    """
    df_brca = load_brca_df()
    pipe = load_pipeline()
    features = load_feature_order()
    coef_df = lr_coef_table(pipe, features)
    order_by_importance = [
        c for c in coef_df.sort_values("coef_abs", ascending=False)["feature"].tolist()
        if c in df_brca.columns
    ]
    return df_brca, order_by_importance, coef_df


# =========================
# Explainer (dalex) + Plot
# =========================

def predict_proba_pos(model, X: pd.DataFrame):
    """
    Função picklável que retorna P(y=1) do modelo.

    Parameters
    ----------
    model : sklearn.base.BaseEstimator
        Estimador compatível com `predict_proba`.
    X : pandas.DataFrame
        Matriz de atributos.

    Returns
    -------
    numpy.ndarray
        Probabilidades da classe positiva, shape (n_samples,).
    """
    return model.predict_proba(X)[:, 1]


@lru_cache(maxsize=1)
def _load_background_from_sklearn(cols_key: tuple, n: int = 800, seed: int = 42) -> pd.DataFrame:
    """
    Constrói um background leve a partir do BRCA (sklearn) contendo apenas as colunas
    informadas e amostrando até `n` linhas.

    Parameters
    ----------
    cols_key : tuple
        Tupla de nomes de colunas (hashable para o cache).
    n : int, default=800
        Tamanho máximo do background.
    seed : int, default=42
        Semente para amostragem reprodutível.

    Returns
    -------
    pandas.DataFrame
        Background com colunas na ordem do treino e até `n` linhas.
    """
    cols = list(cols_key)
    brca = load_breast_cancer(as_frame=True)
    df = brca.frame.copy()
    df.columns = [c.replace(" ", "_") for c in df.columns]
    if "target" in df.columns:
        df = df.drop(columns=["target"])
    present = [c for c in cols if c in df.columns]
    X_bg = df[present].copy()
    X_bg = ensure_feature_order(X_bg, cols)
    if len(X_bg) > n:
        X_bg = X_bg.sample(n=n, random_state=seed).reset_index(drop=True)
    return X_bg


@lru_cache(maxsize=1)
def get_explainer() -> dx.Explainer:
    """
    Cria (e cacheia) um `dx.Explainer` leve para breakdowns individuais.

    O background é gerado a partir do BRCA (sklearn), mantendo independência de arquivos locais.
    O `predict_function` é `predict_proba_pos`, garantindo P(y=1).

    Returns
    -------
    dalex.Explainer
        Objeto explainer pronto para `predict_parts`/`model_profile`.
    """
    pipe = load_pipeline()
    feats = tuple(load_feature_order())  # tupla para uso no cache
    bg = _load_background_from_sklearn(feats, n=800, seed=42)
    return dx.Explainer(
        model=pipe,
        data=bg,
        y=None,
        predict_function=predict_proba_pos,
        label="LR_L2",
        verbose=False,
    )


def _fmt_max4(v):
    """
    Formata números com até 4 casas decimais (sem zeros à direita).

    Parameters
    ----------
    v : Any
        Valor a ser formatado.

    Returns
    -------
    str
        Valor formatado (ou `str(v)` caso não seja numérico).
    """
    try:
        x = float(v)
        s = f"{x:.4f}".rstrip("0").rstrip(".")
        return s if s != "-0" else "0"
    except (TypeError, ValueError):
        return str(v)


def plot_breakdown_from_bd(bd, title: str | None = None):
    """
    Constrói um gráfico Waterfall (horizontal) a partir de `bd.result` (dalex),
    exibindo contribution (Δ) e cumulative (antes/depois) no hover.

    A ordem das barras segue a do dalex: intercept (topo) → ... → prediction (base).
    A linha tracejada vertical marca o valor do intercept (acumulado após o 1º passo).

    Parameters
    ----------
    bd : dalex.PredictionParts
        Resultado de `explainer.predict_parts(..., type="break_down")`.
    title : str | None, optional
        Título do gráfico.

    Returns
    -------
    plotly.graph_objects.Figure
        Figura pronta para renderização em Dash/Plotly.
    """
    res = bd.result.copy()
    labels = res["variable"].astype(str).tolist()
    contrib = res["contribution"].astype(float).to_numpy()   # Δ
    cum_after = res["cumulative"].astype(float).to_numpy()   # após passo
    cum_before = cum_after - contrib                          # antes do passo

    intercept_x = cum_after[0] if len(cum_after) else 0.0

    measure = ["relative"] * len(res)
    if measure:
        measure[-1] = "total"  # prediction

    fig = go.Figure()
    fig.add_shape(
        type="line",
        x0=intercept_x, x1=intercept_x, y0=0, y1=1, yref="paper",
        line=dict(color="rgb(90,90,90)", width=1.5, dash="dot"),
        layer="below",
    )

    cd = np.stack([contrib, cum_before, cum_after], axis=1)  # Δ, antes, depois

    fig.add_trace(go.Waterfall(
        orientation="h",
        measure=measure,
        y=labels,
        x=contrib,  # passamos Δ; hover usa customdata
        text=[f"{c:+.3f}" for c in contrib],
        textposition="outside",
        customdata=cd,
        hovertemplate=(
            "%{y}"
            "<br>contribution (Δ): %{customdata[0]:.3f}"
            "<br>cum. before: %{customdata[1]:.3f}"
            "<br>cum. after: %{customdata[2]:.3f}"
            "<extra></extra>"
        ),
        connector=dict(line=dict(color="rgb(90,90,90)")),
        decreasing=dict(marker=dict(color="seagreen")),
        increasing=dict(marker=dict(color="crimson")),
        totals=dict(marker=dict(color="royalblue")),
    ))

    fig.update_yaxes(categoryorder="array", categoryarray=labels)
    fig.update_layout(
        title=title or "Breakdown",
        showlegend=False,
        xaxis_title="contribution",
        hovermode="closest",
        margin=dict(t=70, b=40, l=10, r=10),
        template="plotly_white",
    )
    return fig


# =========================
# Imputação p/ exibição
# =========================

def get_imputer(pipe) -> SimpleImputer | None:
    """
    Recupera o `SimpleImputer` treinado dentro do pipeline.

    Suposição mais comum:
        pipe.named_steps['pre'] -> Pipeline(...)
        ... que contém step 'imp' -> SimpleImputer(strategy='median')

    Há um fallback que procura por qualquer `Pipeline` no topo com step 'imp'.

    Parameters
    ----------
    pipe : sklearn.pipeline.Pipeline
        Pipeline completo de predição.

    Returns
    -------
    sklearn.impute.SimpleImputer | None
        O imputador encontrado (ou None se não existir).
    """
    try:
        imp = pipe.named_steps["pre"].named_steps["imp"]
        if isinstance(imp, SimpleImputer):
            return imp
    except Exception:
        pass
    for _, step in getattr(pipe, "named_steps", {}).items():
        if isinstance(step, SkPipeline) and "imp" in step.named_steps:
            if isinstance(step.named_steps["imp"], SimpleImputer):
                return step.named_steps["imp"]
    return None


def impute_for_display(pipe, X_row: pd.DataFrame, feature_order: list[str]) -> pd.DataFrame:
    """
    Preenche NaNs de uma linha/DF usando as medianas aprendidas pelo `SimpleImputer`
    do pipeline. Útil para exibir valores concretos no gráfico de breakdown.

    Observação: não altera a predição (o pipeline imputará internamente novamente).

    Parameters
    ----------
    pipe : sklearn.pipeline.Pipeline
        Pipeline completo de predição (de onde as medianas foram aprendidas).
    X_row : pandas.DataFrame
        Linha (ou poucas linhas) com as features do formulário.
    feature_order : list[str]
        Ordem das colunas esperada pelo modelo.

    Returns
    -------
    pandas.DataFrame
        Cópia de `X_row` com NaNs imputados pelas medianas do treino.
    """
    X = ensure_feature_order(X_row, feature_order).copy()
    imp = get_imputer(pipe)
    if imp is None:
        return X

    cols_imp = list(getattr(imp, "feature_names_in_", feature_order))
    stats = imp.statistics_
    med_map = {c: s for c, s in zip(cols_imp, stats)}

    for c in feature_order:
        if c in med_map:
            X[c] = X[c].fillna(med_map[c])
    return X


# =========================
# Helpers de exibição
# =========================

def metrics_tables(rep: dict) -> Tuple[pd.DataFrame, pd.DataFrame, float, int]:
    """
    Converte um `classification_report` serializado (dict) em duas tabelas pandas.

    Parameters
    ----------
    rep : dict
        Dicionário no formato do `classification_report` exportado no treino.

    Returns
    -------
    tuple
        (CLASS, AGG, ACC, TOT), onde:
        - CLASS: DataFrame com métricas por classe (Benigno/Maligno);
        - AGG: DataFrame com macro e weighted average;
        - ACC: float com acurácia;
        - TOT: int com total de amostras do relatório.
    """
    cls_rows = [
        {"Classe": "Benigno", "Precisão": rep["Benigno"]["precision"],
         "Recall": rep["Benigno"]["recall"], "F1-Score": rep["Benigno"]["f1-score"],
         "Suporte": rep["Benigno"]["support"]},
        {"Classe": "Maligno", "Precisão": rep["Maligno"]["precision"],
         "Recall": rep["Maligno"]["recall"], "F1-Score": rep["Maligno"]["f1-score"],
         "Suporte": rep["Maligno"]["support"]},
    ]
    CLASS = pd.DataFrame(cls_rows)

    agg_rows = [
        {"Métrica": "macro avg", "Precisão": rep["macro avg"]["precision"],
         "Recall": rep["macro avg"]["recall"], "F1-Score": rep["macro avg"]["f1-score"],
         "Suporte": rep["macro avg"]["support"]},
        {"Métrica": "weighted avg", "Precisão": rep["weighted avg"]["precision"],
         "Recall": rep["weighted avg"]["recall"], "F1-Score": rep["weighted avg"]["f1-score"],
         "Suporte": rep["weighted avg"]["support"]},
    ]
    AGG = pd.DataFrame(agg_rows)

    ACC = float(rep["accuracy"])
    TOT = int(CLASS["Suporte"].sum())
    return CLASS, AGG, ACC, TOT


def lr_coef_table(pipe, feature_order: List[str]) -> pd.DataFrame:
    """
    Gera uma tabela com coeficientes da Regressão Logística.

    Parameters
    ----------
    pipe : sklearn.pipeline.Pipeline
        Pipeline contendo o estimador `clf` (LogisticRegression).
    feature_order : list[str]
        Ordem das colunas usada no treino.

    Returns
    -------
    pandas.DataFrame
        DataFrame com colunas:
        - feature: nome da variável,
        - coef: coeficiente (sinal),
        - coef_abs: |coef|,
        - sinal: '+' ou '−' (conveniência para visualização).
    """
    clf = pipe.named_steps["clf"]
    coefs = clf.coef_.ravel()
    df = pd.DataFrame({
        "feature": feature_order,
        "coef": coefs,
        "coef_abs": np.abs(coefs),
        "sinal": np.where(coefs >= 0, "+", "−")
    }).sort_values("coef_abs", ascending=False)
    return df


def ensure_feature_order(df_in: pd.DataFrame, order: List[str]) -> pd.DataFrame:
    """
    Garante que um DataFrame tenha todas as colunas na ordem esperada.

    Colunas ausentes são criadas e preenchidas com NaN (o pipeline tratará via imputação).

    Parameters
    ----------
    df_in : pandas.DataFrame
        DataFrame de entrada.
    order : list[str]
        Ordem de colunas desejada.

    Returns
    -------
    pandas.DataFrame
        DataFrame reordenado e com colunas faltantes adicionadas.
    """
    df = df_in.copy()
    for c in order:
        if c not in df.columns:
            df[c] = np.nan
    return df[order]
