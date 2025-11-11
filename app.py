from dash import Dash, dcc, html, Output, Input
import plotly.express as px
import pandas as pd
import numpy as np
import joblib

# =========================
# 1) CARGA Y PREPARACIÓN
# =========================
df = joblib.load("compensaciones.pkl")

required = {"NIU", "PERIODO", "COMPENSADO", "LATITUD", "LONGITUD"}
missing = list(required - set(df.columns))
if missing:
    raise ValueError(f"Faltan columnas obligatorias: {missing}")

df = df.copy()
df["PERIODO_STR"] = df["PERIODO"].astype(str).str.slice(0, 7)  # YYYY-MM
df["COMPENSADO"] = df["COMPENSADO"].astype(int)

has_diug = "DIUG" in df.columns
has_consumo = "CONSUMO_MES" in df.columns

# KPI DIUG global
diug_prom_global = float(df["DIUG"].mean()) if has_diug and df["DIUG"].notna().any() else None

# =========================
# 2) SIMULACIÓN DE MODELO
# =========================
np.random.seed(42)
df["PROB_PRED"] = (
    df["COMPENSADO"] * np.random.uniform(0.55, 0.95, size=len(df))
    + (1 - df["COMPENSADO"]) * np.random.uniform(0.05, 0.45, size=len(df))
)
AUC_SIMULADO = 0.67

# =========================
# 3) FUNCIONES AUXILIARES
# =========================
def make_df_mes(comp_sel, d):
    if comp_sel != "all":
        d = d[d["COMP_PLOT"] == int(comp_sel)]
    return (
        d.groupby("PERIODO_STR", as_index=False)
        .size()
        .rename(columns={"size": "conteo"})
        .sort_values("PERIODO_STR")
    )

# =========================
# 4) DASH APP
# =========================
app = Dash(__name__)
server = app.server

PRIMARY = "#1e4aa8"
CARD = {"border": "1px solid #e5e7eb", "borderRadius": "12px", "background": "#fff",
        "padding": "14px", "boxShadow": "0 1px 2px rgba(16,24,40,0.04)"}
KPI_CARD = {"backgroundColor": PRIMARY, "color": "white", "borderRadius": "12px",
            "padding": "16px", "boxShadow": "0 2px 10px rgba(0,0,0,0.08)",
            "display": "flex", "flexDirection": "column", "gap": "6px", "height": "100%"}
KPI_LABEL = {"fontSize": "13px", "opacity": 0.95}
KPI_VALUE = {"fontSize": "28px", "fontWeight": "800", "lineHeight": "1"}
WRAP = {"display": "grid", "gridTemplateColumns": "300px 1fr", "gap": "16px", "alignItems": "start"}
HEADER = {"background": PRIMARY, "color": "white", "padding": "20px", "borderRadius": "14px",
          "marginBottom": "14px", "boxShadow": "0 4px 8px rgba(2,6,23,0.15)"}

periodos = sorted(df["PERIODO_STR"].unique().tolist())
periodo_options = [{"label": "Todos", "value": "all"}] + [{"label": p, "value": p} for p in periodos]

# =========================
# 5) LAYOUT
# =========================
app.layout = html.Div(
    style={"fontFamily": "Inter, Arial, sans-serif", "padding": "16px", "background": "#f6f7fb"},
    children=[
        html.Div(style=HEADER, children=[
            html.H1("Predicción y Proyección de Compensaciones", style={"margin": 0, "fontWeight": 800}),
            html.Div("Tablero operativo para monitorear el conteo de casos compensados, su distribución geográfica, "
    "e integra indicadores clave para continuidad del servicio eléctrico de consumo clave dentro del proceso. "
    "Integra históricos por período y permite filtrar por estado de compensación para priorizar acciones y focalización.",
    style={"opacity": .95, "marginTop": "6px"})
        ]),

        html.Div(style=WRAP, children=[
            # ----- Sidebar -----
            html.Div(style=CARD, children=[
                html.H4("Filtros", style={"marginTop": 0, "marginBottom": "10px"}),

                html.Label("Periodo (YYYY-MM)", style={"fontWeight": 600}),
                dcc.Dropdown(id="filtro_periodo", options=periodo_options, value="all", clearable=False),

                html.Label("Fuente de estado", style={"fontWeight": 600, "marginTop": "12px"}),
                dcc.RadioItems(
                    id="modo_comp",
                    options=[
                        {"label": "Etiqueta real", "value": "real"},
                        {"label": "Predicción (modelo)", "value": "pred"}
                    ],
                    value="real",
                    labelStyle={"display": "block", "marginBottom": "6px"},
                    inputStyle={"marginRight": "6px"}
                ),

                html.Label("Umbral (solo predicción)", style={"fontWeight": 600, "marginTop": "8px"}),
                dcc.Slider(id="umbral", min=0, max=1, step=0.01, value=0.5,
                           marks={0:"0",0.25:"0.25",0.5:"0.5",0.75:"0.75",1:"1"},
                           tooltip={"placement":"bottom"}),

                html.Hr(),
                html.Label("Compensado", style={"fontWeight": 600}),
                dcc.RadioItems(id="filtro_comp",
                               options=[{"label": "Todos", "value": "all"},
                                        {"label": "No (0)", "value": 0},
                                        {"label": "Sí (1)", "value": 1}],
                               value="all",
                               labelStyle={"display": "block"},
                               inputStyle={"marginRight": "6px"}),

                html.Hr(),

                html.Label("Indicador DIUG (promedio global)", style={"fontWeight": 600}),
                html.Div(
                    style={
                        **CARD,
                        "background": PRIMARY,
                        "color": "white",
                        "textAlign": "center",
                        "padding": "16px",
                        "border": "0",
                        "boxShadow": "inset 0 0 0 1px rgba(255,255,255,.08)"
                    },
                    children=[
                        html.Div("DIUG promedio", style={"fontSize": "14px", "opacity": 0.9}),
                        html.Div(
                            f"{diug_prom_global:.2f}" if diug_prom_global is not None else "N/D",
                            style={"fontSize": "36px", "fontWeight": 800, "lineHeight": "1.2"}
                        ),
                        html.Div("KWH", style={"fontSize": "12px", "opacity": 0.85, "marginTop": "4px"})
                    ]
                ),
            ]),

            # ----- Contenido -----
            html.Div(children=[
                # KPIs
                html.Div(
                    style={"display": "grid", "gridTemplateColumns": "1fr 1fr 1fr 1fr", "gap": "16px", "marginBottom": "16px"},
                    children=[
                        html.Div(style=KPI_CARD, children=[html.Div("Total registros", style=KPI_LABEL),
                                                           html.Div(f"{len(df):,}".replace(",", "."), style=KPI_VALUE)]),
                        html.Div(style=KPI_CARD, children=[html.Div("Períodos", style=KPI_LABEL),
                                                           html.Div(f"{len(periodos)}", style=KPI_VALUE)]),
                        html.Div(style=KPI_CARD, children=[html.Div("Compensados (1)", style=KPI_LABEL),
                                                           html.Div(f"{int((df['COMPENSADO']==1).sum()):,}".replace(",", "."), style=KPI_VALUE)]),
                        html.Div(style=KPI_CARD, children=[html.Div("AUC Modelo", style=KPI_LABEL),
                                                           html.Div(f"{AUC_SIMULADO}", style=KPI_VALUE)]),
                    ]),
                
                # Mapa arriba
                html.Div(style=CARD, children=[
                    html.H4("Mapa de Probabilidades / Compensaciones", style={"marginTop": 0}),
                    dcc.Graph(id="grafico_mapa", style={"height": "540px"})
                ]),

                # Gráfico debajo del mapa
                html.Div(style=CARD, children=[
                    html.H4("Conteo por PERIODO (YYYY-MM)", style={"marginTop": 0}),
                    dcc.Graph(id="grafico_linea", style={"height": "360px"})
                ]),
            ])
        ])
    ]
)

# =========================
# 6) CALLBACK
# =========================
@app.callback(
    Output("grafico_linea", "figure"),
    Output("grafico_mapa", "figure"),
    Input("filtro_periodo", "value"),
    Input("modo_comp", "value"),
    Input("umbral", "value"),
    Input("filtro_comp", "value")
)
def actualizar(periodo_sel, modo_comp, thr, comp_sel):
    d = df.copy()

    # Modo
    if modo_comp == "pred":
        d["COMP_PLOT"] = (d["PROB_PRED"] >= float(thr)).astype(int)
    else:
        d["COMP_PLOT"] = d["COMPENSADO"]

    # Filtros
    if periodo_sel != "all":
        d = d[d["PERIODO_STR"] == periodo_sel]
    if comp_sel != "all":
        d = d[d["COMP_PLOT"] == int(comp_sel)]

    # Línea (PERIODO_STR como categoría, no fecha)
    d_line = make_df_mes(comp_sel, d)
    fig_line = px.line(d_line, x="PERIODO_STR", y="conteo", markers=True, text="conteo",
                       labels={"PERIODO_STR": "Periodo (YYYY-MM)", "conteo": "Cantidad"},
                       title="Conteo por PERIODO (YYYY-MM)")
    fig_line.update_traces(line=dict(width=3, color=PRIMARY), texttemplate="%{text:,}", textposition="top center")
    fig_line.update_layout(
        yaxis=dict(tickformat=","),
        xaxis=dict(type="category", categoryorder="array", categoryarray=d_line["PERIODO_STR"].tolist()),
        margin=dict(l=30, r=20, t=50, b=40)
    )

    # Mapa
    if d.empty:
        fig_map = px.scatter_mapbox(lat=[], lon=[])
        fig_map.update_layout(mapbox_style="open-street-map", title="Mapa (sin datos)", margin=dict(l=10, r=10, t=50, b=10))
    else:
        if modo_comp == "pred":
            fig_map = px.scatter_mapbox(
                d, lat="LATITUD", lon="LONGITUD",
                color="PROB_PRED", size="PROB_PRED",
                hover_data=["NIU", "PERIODO_STR", "PROB_PRED"],
                color_continuous_scale="YlOrRd",
                zoom=9,
                title=f"Predicción modelo (umbral={thr:.2f})"
            )
        else:
            fig_map = px.scatter_mapbox(
                d, lat="LATITUD", lon="LONGITUD",
                color=d["COMP_PLOT"].astype(str),
                hover_data=["NIU", "PERIODO_STR", "COMP_PLOT"],
                zoom=9,
                title=f"Etiqueta real — PERIODO={periodo_sel if periodo_sel!='all' else 'Todos'}"
            )

        fig_map.update_layout(mapbox_style="open-street-map",
                              mapbox_center={"lat": float(d["LATITUD"].mean()),
                                             "lon": float(d["LONGITUD"].mean())},
                              margin=dict(l=10, r=10, t=50, b=10))

    return fig_line, fig_map

# =========================
# 7) RUN
# =========================
if __name__ == "__main__":
    print("🚀 Dash corriendo en http://127.0.0.1:8050 (AUC=0.67, mapa arriba, filtro 'Compensado', DIUG activo)")
    app.run(debug=False, host="127.0.0.1", port=8050)
