# paginas/formulario.py
from dash import html, dcc, no_update
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np

from app import app

from paginas.compartilhado import (
    load_pipeline, load_feature_order, load_ranges, ensure_feature_order,
    get_explainer, plot_breakdown_from_bd, impute_for_display
)

pipe = load_pipeline()
FEATURES = load_feature_order()
RANGES = load_ranges()

def num_input(id_, label, rng):
    """
    Cria um grupo de entrada numérica (Bootstrap) para uma variável do formulário.

    Parâmetros
    ----------
    id_ : str
        ID do componente `dbc.Input` (usado no callback).
    label : str
        Texto do rótulo exibido ao lado da caixa de entrada.
    rng : dict
        Dicionário com limites históricos {'min': float, 'max': float}; usado no placeholder.

    Retorna
    -------
    dash_bootstrap_components.CardGroup
        Componente contendo `dbc.Label` + `dbc.Input(type='number')`.
    """
    ph = f"{rng['min']:.2f} … {rng['max']:.2f}"
    # Removemos o min/max para não bloquear a entrada
    return dbc.CardGroup([
        dbc.Label(label),
        dbc.Input(
            id=id_, 
            type="number",
            placeholder=ph, 
            step="any"
        )
    ], className="mb-3")


# Mapeia id do input -> nome da coluna
FIELDS = [(f"f_{i}", col) for i, col in enumerate(FEATURES)]
ID_TO_COL = dict(FIELDS)

def make_column(ids_cols):
    """
    Monta uma coluna (Bootstrap) com um conjunto de inputs numéricos.

    Parâmetros
    ----------
    ids_cols : list[tuple[str, str]]
        Lista de pares (id_do_input, nome_da_coluna) a renderizar.

    Retorna
    -------
    dash_bootstrap_components.Col
        Coluna contendo um `CardGroup` por variável.
    """
    return dbc.Col([
        num_input(id_, col, RANGES.get(col, {"min": 0.0, "max": 1.0}))
        for id_, col in ids_cols
    ])


layout = dbc.Container([
    html.H2("Formulário de previsão", className="text-center mt-3 mb-2"),
    html.Div("Preencha as 12 variáveis (numéricas). Faltantes serão imputados pelo próprio pipeline.", className="text-muted mb-3"),
    dbc.Row([
        make_column(FIELDS[:6]),
        make_column(FIELDS[6:])
    ]),
    dbc.Button("Prever", id="btn-predict", color="success", n_clicks=0, className="mb-3"),
    html.Div(id="pred-out"),
    dcc.Location(id="scroll-to", refresh=False),
], fluid=True)


@app.callback(
    Output("pred-out", "children"),
    Output("scroll-to", "hash"),
    Input("btn-predict", "n_clicks"),
    [State(fid, "value") for fid, _ in FIELDS],
    prevent_initial_call=True
)
def fazer_previsao(n_clicks, *values):
    """
    Processa uma submissão do formulário, calcula a predição e renderiza o breakdown.

    Regras de negócio:
      - Constrói um DataFrame com os valores informados e garante a ordem das features;
      - Bloqueia a predição se houver valores negativos (retorna alerta de erro);
      - Emite alerta de atenção para valores fora do intervalo histórico (min–max);
      - Prediz probabilidade de classe positiva (Maligno) via pipeline carregado;
      - Gera gráfico de **breakdown** usando um `Explainer` cacheado (dalex),
        com os **valores imputados pela mediana** apenas para exibição (rótulos);
      - Faz a página rolar automaticamente até o gráfico via atualização do `hash`
        para `#breakdown-anchor`.

    Parâmetros
    ----------
    n_clicks : int
        Número de cliques no botão "Prever" (controlado pelo Dash).
    *values : tuple
        Tupla com os valores dos 12 inputs numéricos, na mesma ordem de `FIELDS`.

    Retorna
    -------
    tuple
        (children, hash), onde:
        - children : dash.development.base_component.Component
            Árvore de componentes com alertas e gráfico.
        - hash : str | dash.no_update
            `"#breakdown-anchor"` para rolar até o gráfico; `no_update` em caso de erro.
    """
    # Monta o dicionário com colunas e valores
    row = {col: (np.nan if v is None else float(v)) for (fid, col), v in zip(FIELDS, values)}
    
    # Verifica valores negativos (impede previsão)
    negative_fields = []
    for col, val in row.items():
        if pd.notna(val) and val < 0:
            negative_fields.append(f"{col} ({val})")
    
    if negative_fields:
        error_msg = f"❌ Erro: Os seguintes campos não podem ter valores negativos: {', '.join(negative_fields)}"
        # Dois outputs: children + hash (não alterar o hash em caso de erro)
        return dbc.Alert(error_msg, color="danger", className="mt-2"), no_update
    
    # Verifica campos fora do range histórico (apenas aviso)
    out_of_range = []
    for col, val in row.items():
        if pd.isna(val):
            continue
        rng = RANGES.get(col, {})
        vmin = rng.get("min", -np.inf)
        vmax = rng.get("max", np.inf)
        if not (vmin <= val <= vmax):
            direcao = "abaixo" if val < vmin else "acima"
            out_of_range.append(
                f"- **{col}**: {val:.4g} ({direcao} de {vmin:.4g}–{vmax:.4g})"
            )

    # Cria DataFrame e ordena conforme modelo
    X = pd.DataFrame([row])
    X = ensure_feature_order(X, FEATURES)

    # Predição
    proba = float(pipe.predict_proba(X)[0, 1])
    yhat = int(proba >= 0.50)
    pred_msg = (f"Probabilidade de **Maligno** = {proba:.3f} → "
                f"**{'Maligno' if yhat == 1 else 'Benigno'}** (threshold 0.50)")
    alert_color = "danger" if yhat == 1 else "secondary"

    # Para o breakdown, mostramos os valores JÁ IMPUTADOS pela mediana (rótulos legíveis)
    X_disp = impute_for_display(pipe, X, FEATURES)

    # Breakdown via explainer cacheado
    try:
        expl = get_explainer()
        bd = expl.predict_parts(new_observation=X_disp[FEATURES], type="break_down", keep_distributions=True)
        fig = plot_breakdown_from_bd(bd, title=f"Breakdown — proba={proba:.3f}")
        graph_block = dbc.Card(dbc.CardBody([
            html.H6("Perfil de decomposição", className="mb-2"),
            dcc.Graph(figure=fig, config={"displayModeBar": False})
        ]), className="mt-2")
    except Exception as e:
        graph_block = dbc.Alert(f"Falha ao gerar breakdown: {e}", color="warning", className="mt-2")

    blocks = []
    if out_of_range:
        warning_md = (
            "⚠️ **Atenção**: valores fora do intervalo esperado:\n\n"
            + "\n".join(out_of_range)
            + "\n\n> Revise os valores. A previsão foi calculada mesmo assim."
        )
        blocks.append(dbc.Alert(dcc.Markdown(warning_md), color="warning", className="mb-2"))

    blocks.append(dbc.Alert(dcc.Markdown(pred_msg), color=alert_color, className="mt-2"))
    blocks.append(graph_block)
    blocks.append(html.Span(id="breakdown-anchor"))
    return html.Div(blocks), "#breakdown-anchor"
