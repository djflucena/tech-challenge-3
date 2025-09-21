# main.py
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import os
from app import app, server
import paginas

navegacao = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Início", href="/")),
        dbc.NavItem(dbc.NavLink("Gráficos", href="/graficos")),
        dbc.NavItem(dbc.NavLink("Formulário", href="/formulario")),
    ],
    brand="Breast Cancer – Dashboard",
    brand_href="/",
    color="primary",
    dark=True,
    className="mb-3"
)

app.layout = html.Div([
    dcc.Location(id="url", refresh=False),
    navegacao,
    html.Div(id="conteudo", className="m-3")
])

@app.callback(
    Output("conteudo", "children"),
    Input("url", "pathname")
)
def mostrar_pagina(pathname):
    if pathname == "/formulario":
        return paginas.formulario.layout
    elif pathname == "/graficos":
        return paginas.graficos.layout
    elif pathname in ["/", None]:
        return paginas.inicio.layout
    return html.P("Página não encontrada.", className="m-3")

if __name__ == "__main__":
    # Executa com o servidor embutido (desenvolvimento).
    # Em produção via Docker/Gunicorn, será usado: gunicorn -b 0.0.0.0:9080 main:server
    port = int(os.environ.get("PORT", 9080))  # 9080 no dev; $PORT no Render
    app.run_server(debug=False, host="0.0.0.0", port=port)
