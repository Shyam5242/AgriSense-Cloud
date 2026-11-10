"""
architecture.py - data behind the 3D system-architecture page (paper Fig. 1 / PPT slide 11).
No Streamlit or Plotly imports at module level, so it can be tested on its own.
"""
from __future__ import annotations

# id: (label, x, y, z, layer, description)
NODES = {
    "farmer":   ("Farmer input (form / photo)", -3.0, -1.0, 3.0, "Input",
                 "Soil and weather values or a leaf photo, entered in the web app."),
    "iot":      ("IoT sensor simulator", 0.0, 0.0, 3.0, "Input",
                 "Synthetic soil-moisture, temperature and humidity readings."),
    "satellite": ("Satellite ingestion (Sentinel-2)", 3.0, 1.0, 3.0, "Input",
                  "NDVI from Sentinel-2 through Google Earth Engine."),
    "gateway":  ("API gateway + preprocessing", 0.0, 0.0, 2.0, "Processing",
                 "FastAPI service that validates and routes requests to the models."),
    "yield":    ("Yield regressor (ground + NDVI)", -4.5, 0.8, 1.0, "Models",
                 "XGBoost or Random Forest on soil, weather and NDVI features."),
    "leaf":     ("Leaf-disease CNN (PlantVillage)", -1.5, -0.8, 1.0, "Models",
                 "MobileNetV2 transfer learning, 38 classes."),
    "crop":     ("Crop recommender", 1.5, 0.8, 1.0, "Models",
                 "Random Forest over 22 crops from soil and climate."),
    "land":     ("Land-cover CNN (EuroSAT)", 4.5, -0.8, 1.0, "Models",
                 "MobileNetV2 transfer learning, 10 land-cover classes."),
    "app":      ("Farmer app", -3.0, 0.0, 0.0, "Output",
                 "Predictions and recommendations in the Streamlit web app."),
    "storage":  ("Cloud storage", 0.0, 1.2, 0.0, "Output",
                 "Results and datasets. Local files today; Google Cloud Storage is optional."),
    "dash":     ("Analytics dashboard", 3.0, 0.0, 0.0, "Output",
                 "Regional yield, NDVI, soil-moisture and disease trends."),
}

EDGES = [
    ("farmer", "gateway"), ("iot", "gateway"), ("satellite", "gateway"),
    ("gateway", "yield"), ("gateway", "leaf"), ("gateway", "crop"), ("gateway", "land"),
    ("yield", "app"), ("leaf", "app"), ("crop", "app"), ("land", "app"),
    ("yield", "storage"), ("leaf", "storage"), ("crop", "storage"), ("land", "storage"),
    ("storage", "dash"),
]

LAYER_COLORS = {"Input": "#00f5c4", "Processing": "#38bdf8", "Models": "#fbbf24", "Output": "#a78bfa"}

# component: (how this app implements it, status)
STATUS = [
    ("Farmer input", "Streamlit forms and image uploads", "Implemented"),
    ("IoT sensor simulator", "sensors.py, synthetic data", "Simulated"),
    ("Satellite ingestion", "Earth Engine NDVI on demand (not scheduled)", "Partial"),
    ("API gateway", "FastAPI api.py; the Streamlit app calls the models directly", "Partial"),
    ("Yield regressor", "Random Forest; comparison page tests XGBoost and ANN", "Implemented"),
    ("Leaf-disease CNN", "MobileNetV2 trained on PlantVillage in Colab", "Implemented"),
    ("Crop recommender", "Random Forest", "Implemented"),
    ("Land-cover CNN", "MobileNetV2 trained on EuroSAT in Colab", "Implemented"),
    ("Cloud storage", "Local files; Google Cloud Storage optional", "Partial"),
    ("Analytics dashboard", "Regional analytics page", "Implemented"),
]


def edge_coordinates():
    """x, y, z lists for one Plotly line trace; None separates the edges."""
    xs, ys, zs = [], [], []
    for a, b in EDGES:
        for key in (a, b):
            _, x, y, z, *_ = NODES[key]
            xs.append(x); ys.append(y); zs.append(z)
        xs.append(None); ys.append(None); zs.append(None)
    return xs, ys, zs


def build_figure(go):
    """Build the 3D architecture figure. `go` is plotly.graph_objects."""
    xs, ys, zs = edge_coordinates()
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", hoverinfo="skip", name="data flow",
                               line=dict(color="#64748b", width=3)))
    for layer, color in LAYER_COLORS.items():
        ids = [k for k, v in NODES.items() if v[4] == layer]
        fig.add_trace(go.Scatter3d(
            x=[NODES[k][1] for k in ids], y=[NODES[k][2] for k in ids], z=[NODES[k][3] for k in ids],
            mode="markers+text", name=layer, text=[NODES[k][0] for k in ids], textposition="top center",
            hovertext=[NODES[k][5] for k in ids], hoverinfo="text",
            marker=dict(size=11, color=color, line=dict(width=1, color="#ffffff"))))
    fig.update_layout(scene=dict(
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        zaxis=dict(visible=False), camera=dict(eye=dict(x=1.5, y=-1.9, z=0.9))))
    return fig


def status_table(gateway_connected: bool, ingestion: dict | None = None, yield_model: str = "Random Forest"):
    """STATUS rows, with the API-gateway and satellite-ingestion rows based on what is actually running/stored."""
    ingestion = ingestion or {}
    rows = []
    for comp, impl, status in STATUS:
        if comp == "API gateway":
            if gateway_connected:
                impl, status = "FastAPI api.py; this web app calls it over HTTP", "Implemented"
            else:
                impl, status = "FastAPI api.py exists, but this web app is calling the models directly (start with start_all.bat)", "Partial"
        elif comp == "Satellite ingestion":
            if ingestion.get("rows", 0) > 0:
                impl = (f"ingest_satellite.py stores NDVI per registered field "
                        f"({ingestion['rows']} observations, last {ingestion['last_date']}); schedule it with run_ingestion.bat")
                status = "Implemented"
            else:
                impl, status = "ingest_satellite.py is ready, but no observations are stored yet (register fields and run it)", "Partial"
        elif comp == "Yield regressor":
            impl = f"{yield_model}; the comparison page also tests Random Forest, XGBoost and an ANN"
        rows.append((comp, impl, status))
    return rows
