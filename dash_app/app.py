# app.py
from dash import Dash
import dash_bootstrap_components as dbc
import warnings
warnings.filterwarnings(
    "ignore",
    message="When grouping with a length-1 list-like, you will need to pass a length-1 tuple to get_group",
    category=FutureWarning,
    module="plotly"
)


app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY, "/assets/main.css"],
    suppress_callback_exceptions=True,
    title="Breast Cancer – Dashboard"
)

# Necessário para gunicorn/render
server = app.server

