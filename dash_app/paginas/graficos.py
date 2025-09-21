# paginas/graficos.py
from dash import html, dcc
import dash_bootstrap_components as dbc
import plotly.express as px

from paginas.compartilhado import get_brca_for_plots

# Carrega tudo pronto do módulo compartilhado (uma vez, cacheado)
df_brca, order_by_importance, coef_df = get_brca_for_plots()


def fig_coef():
    """
    Cria um gráfico de barras horizontal com os coeficientes absolutos (|coef|)
    da Regressão Logística, exibindo também o coeficiente com sinal no texto.

    Returns
    -------
    plotly.graph_objs._figure.Figure
        Figura Plotly pronta para renderização em Dash.
    """
    dfp = coef_df.copy().sort_values("coef_abs", ascending=True)
    dfp["label"] = dfp["feature"] + " (" + dfp["sinal"] + ")"
    fig = px.bar(
        dfp,
        x="coef_abs",
        y="label",
        orientation="h",
        title="Contribuição (|coef|) — LR_L2",
        text=dfp["coef"].map(lambda x: f"{x:+.3f}"),
    )
    fig.update_traces(
        textposition="outside",
        hovertemplate="%{y}<br>|coef|=%{x:.3f}<br>coef=%{text}<extra></extra>",
    )
    fig.update_layout(
        xaxis_title="|coef|",
        yaxis_title="Variável",
        margin=dict(l=10, r=10, t=60, b=10),
    )
    return fig


def fig_box_for(feature: str):
    """
    Cria um boxplot comparando as distribuições de uma variável numérica
    entre as classes Benigno e Maligno no dataset BRCA do sklearn.

    Parameters
    ----------
    feature : str
        Nome da coluna numérica a ser plotada (deve existir em `df_brca`).

    Returns
    -------
    plotly.graph_objs._figure.Figure
        Figura Plotly com boxplots agrupados por classe.
    """
    fig = px.box(
        df_brca,
        x="diagnosis_label",
        y=feature,
        color="diagnosis_label",
        category_orders={"diagnosis_label": ["Benigno", "Maligno"]},
        title="Distribuição por classe",
        points=False,
    )
    fig.update_layout(
        margin=dict(l=10, r=10, t=50, b=10),
        xaxis_title="Classe",
        yaxis_title=feature,
        boxmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
    )
    fig.update_traces(marker_line_width=0, selector=dict(type="box"))
    return fig


def boxplot_grid(features_list: list[str]):
    """
    Monta um grid (2 colunas) de boxplots para uma lista de variáveis,
    preservando a ordem fornecida.

    Parameters
    ----------
    features_list : list[str]
        Lista de nomes de colunas numéricas a serem exibidas, na ordem desejada.

    Returns
    -------
    list[dash_bootstrap_components.Row]
        Lista de linhas (Dash Bootstrap) contendo os cards com gráficos.
    """
    cards = []
    for f in features_list:
        cards.append(
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.H6(f, className="mb-2"),
                            dcc.Graph(figure=fig_box_for(f), config={"displayModeBar": False}),
                        ]
                    ),
                    className="mb-3 h-100",
                ),
                md=6,
            )
        )
    rows = []
    for i in range(0, len(cards), 2):
        rows.append(dbc.Row(cards[i : i + 2], className="g-3 align-items-stretch"))
    return rows


# Seção com os 12 boxplots ordenados por |coef|
boxplots_section = dbc.Card(
    dbc.CardBody(
        [
            html.H5("Boxplots das 12 variáveis preditoras (ordenadas por |coef|)", className="card-title"),
            dcc.Markdown(
                "Cada gráfico compara a distribuição **Benigno vs Maligno** para a variável indicada. "
                "As variáveis estão ordenadas por **|coef|** do modelo LR_L2 (maior impacto primeiro)."
            ),
            *boxplot_grid(order_by_importance),
        ]
    ),
    className="mb-4",
)


# Layout da página
layout = dbc.Container(
    [
        html.H2("Gráficos", className="text-center mb-4 mt-3"),
        dbc.Card(
            dbc.CardBody(
                [
                    html.H5("Coeficientes (ordenados por |coef|)", className="card-title"),
                    dcc.Markdown("Valores positivos → aumentam probabilidade de **Maligno**; negativos → reduzem."),
                    dcc.Graph(figure=fig_coef(), config={"displayModeBar": False}),
                ]
            ),
            className="mb-4",
        ),
        boxplots_section,
    ],
    fluid=True,
)
