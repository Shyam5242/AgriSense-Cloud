"""
app.py - AgriSense Cloud web app (Streamlit), redesigned UI.   Run:  streamlit run app.py
"""
import hmac
import io
import json
import os
import random
import time

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
from PIL import Image

from common import CROP_FEATURES, MODELS, YIELD_FEATURES, Predictor, models_ready
from regional import (SEASONS, correlations, district_table, kpis, load_regional, underperformance, yield_trend)
import architecture
import auth_token
import examples
import ingest_satellite as ingest
from api_client import ApiError, ApiPredictor
from model_comparison import ndvi_gain, load_comparison, run_comparison, save_comparison, xgboost_available
from sensors import simulate_sensors

st.set_page_config(page_title="AgriSense Cloud", page_icon="🌱", layout="wide")

# ----------------------------------------------------------------------------- look & feel
_THEME = st.session_state.get("theme", "Neon")
_BASE_CSS = """
<style>
html, body, [class*="css"], .stApp { font-family: "Segoe UI", system-ui, sans-serif; }
.stApp { background: radial-gradient(1200px 600px at 85% -10%, #12306b55 0%, transparent 60%),
                     radial-gradient(900px 500px at -10% 110%, #00f5c422 0%, transparent 55%), #060b1a; color: #e6f7f1; }
h1, h2, h3 { font-family: "Segoe UI", system-ui, sans-serif !important; font-weight: 700 !important; letter-spacing: -0.01em;
             background: linear-gradient(90deg, #00f5c4, #38bdf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
section[data-testid="stSidebar"] { background: linear-gradient(180deg, #08122b 0%, #050a1a 100%); border-right: 1px solid #00f5c433; }
.hero { background: linear-gradient(120deg, #0b1d4a 0%, #0e5a63 55%, #00c9a0 100%); border: 1px solid #00f5c455;
        border-radius: 20px; padding: 36px 40px; color: #fff; margin-bottom: 24px;
        box-shadow: 0 0 40px #00f5c422, inset 0 0 60px #ffffff08; transform: perspective(1200px) rotateX(1.5deg); }
.hero h1 { margin: 0 0 8px 0; font-size: 2.4rem; background: none; -webkit-text-fill-color: #fff; color: #fff; }
.hero p { margin: 0; font-size: 1.05rem; opacity: .9; max-width: 780px; }
.card { background: linear-gradient(145deg, #ffffff12, #ffffff05); backdrop-filter: blur(8px); border: 1px solid #00f5c42e;
        border-radius: 16px; padding: 18px 20px; height: 100%; transition: transform .25s ease, box-shadow .25s ease, border-color .25s;
        transform: perspective(900px) rotateX(0deg); }
.card:hover { transform: perspective(900px) rotateX(4deg) translateY(-6px); border-color: #00f5c4aa;
              box-shadow: 0 14px 34px #00f5c433, 0 0 0 1px #00f5c455; }
.card .big { font-size: 2.1rem; font-weight: 700; color: #00f5c4; text-shadow: 0 0 18px #00f5c477; line-height: 1.1; }
.card .lbl { font-weight: 600; margin-top: 6px; color: #e6f7f1; }
.card .sub { font-size: .82rem; color: #8fa6b8; margin-top: 3px; }
.badge { display: inline-block; font-size: .7rem; font-weight: 700; padding: 2px 9px; border-radius: 99px; }
.real { background: #00f5c422; color: #00f5c4; border: 1px solid #00f5c466; }
.sim { background: #ffb02e22; color: #ffb02e; border: 1px solid #ffb02e66; }
.result { background: linear-gradient(135deg, #0b1d4a, #0a3a45); border: 1px solid #00f5c466; border-radius: 18px;
          padding: 22px 28px; color: #fff; margin: 10px 0 18px 0; box-shadow: 0 0 30px #00f5c433; }
.result .big { font-size: 2.5rem; font-weight: 700; color: #00f5c4; text-shadow: 0 0 22px #00f5c488; }
.result .sub { opacity: .8; }
.footer { color: #5f7488; font-size: .8rem; margin-top: 36px; }
</style>
"""
_FOREST = {"#00f5c4": "#8bd450", "#38bdf8": "#d9b44a", "#a78bfa": "#9acd7a", "#060b1a": "#08150d",
           "#12306b55": "#1f5a3355", "#0b1d4a": "#14301f", "#0e5a63": "#2f6b3a", "#00c9a0": "#8fb860",
           "#08122b": "#0c2216", "#050a1a": "#07150e", "#0a3a45": "#1f4d2c"}
if _THEME == "Forest":
    for _old, _new in _FOREST.items():
        _BASE_CSS = _BASE_CSS.replace(_old, _new)
st.markdown(_BASE_CSS, unsafe_allow_html=True)

# ----------------------------------------------------------------------------- optional password gate
# Set AGRISENSE_PASSWORD before starting to require a login (recommended when sharing through a tunnel).
PASSWORD = os.getenv("AGRISENSE_PASSWORD", "")
if PASSWORD and not st.session_state.get("auth_ok"):
    # a browser refresh starts a new session, so look for a valid signed token in the address
    if auth_token.valid_token(PASSWORD, st.query_params.get("auth", "")):
        st.session_state["auth_ok"] = True
if PASSWORD and not st.session_state.get("auth_ok"):
    tiles = "".join(f'<i style="animation-delay:{(i % 40) * 0.08 + (i // 40) * 0.3:.2f}s"></i>' for i in range(240))
    st.markdown("""
<style>
.pulse-floor { position: fixed; left: -10%; right: -10%; bottom: -30px; z-index: 1; pointer-events: none;
  display: grid; grid-template-columns: repeat(40, 1fr); gap: 8px; transform: perspective(320px) rotateX(62deg); transform-origin: 50% 100%;
  -webkit-mask-image: linear-gradient(to top, #000, transparent); mask-image: linear-gradient(to top, #000, transparent); }
.pulse-floor i { aspect-ratio: 1; border-radius: 3px; background: #5eead4; opacity: .12; animation: tilepulse 2.8s ease-in-out infinite; }
@keyframes tilepulse { 50% { opacity: .75; } }
.login-wrap { position: relative; z-index: 2; text-align: center; padding-top: 14px; }
.earth { width: 190px; height: 190px; margin: 0 auto 16px; border-radius: 50%; background-color: #1d4ed8;
  background-image: url(https://cdn.jsdelivr.net/npm/three-globe/example/img/earth-blue-marble.jpg); background-size: auto 100%; background-repeat: repeat-x;
  animation: earthspin 40s linear infinite;
  box-shadow: inset -44px -24px 50px rgba(0,0,0,.72), inset 10px 10px 20px rgba(255,255,255,.12), 0 0 34px rgba(96,165,250,.45); }
@keyframes earthspin { from { background-position-x: 0; } to { background-position-x: 380px; } }
.login-title { font-size: 2.4rem; font-weight: 800; letter-spacing: -0.01em; margin: 4px 0 2px 0;
  background: linear-gradient(90deg, #00f5c4, #38bdf8, #a78bfa); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.login-sub { color: #8fa6b8; margin: 0 0 22px 0; letter-spacing: .06em; font-size: .9rem; }
[data-testid="stForm"] { position: relative; z-index: 2; max-width: 430px; margin: 0 auto; padding: 26px 28px;
  background: linear-gradient(145deg, #ffffff16, #ffffff06); backdrop-filter: blur(10px); border: 1px solid #00f5c466;
  border-radius: 20px; box-shadow: 0 24px 60px #00f5c433, 0 0 0 1px #00f5c422;
  transform: perspective(1000px) rotateX(3deg); transition: transform .3s ease, box-shadow .3s ease; }
[data-testid="stForm"]:hover { transform: perspective(1000px) rotateX(0deg) translateY(-3px); box-shadow: 0 30px 70px #00f5c455; }
[data-testid="stForm"] button { width: 100%; font-weight: 700; color: #04101c; border: none;
  background: linear-gradient(90deg, #00f5c4, #38bdf8); box-shadow: 0 0 24px #00f5c477; }
</style>
<div class="pulse-floor">__TILES__</div>
<div class="login-wrap">
  <div class="earth"></div>
  <div class="login-title">AgriSense Cloud</div>
  <p class="login-sub">SECURE ACCESS &nbsp;·&nbsp; SATELLITE + AI FOR SMARTER FARMING</p>
</div>
""".replace("__TILES__", tiles), unsafe_allow_html=True)
    with st.form("login"):
        pw = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in")
    if submitted:
        if hmac.compare_digest(pw.encode(), PASSWORD.encode()):
            st.session_state["auth_ok"] = True
            st.query_params["auth"] = auth_token.make_token(PASSWORD)     # keeps you signed in after a refresh
            st.rerun()
        else:
            time.sleep(1)          # slows down guessing
            st.error("Wrong password.")
    st.stop()

if not models_ready():
    st.error("Models not found. Run `python train_and_save.py` first, then restart this app.")
    st.stop()


API_URL = os.getenv("AGRISENSE_API_URL", "").strip()


@st.cache_resource(show_spinner="Loading models...")
def get_predictor(api_url=""):
    if api_url:
        return ApiPredictor(api_url)       # models run inside the FastAPI gateway
    return Predictor()                     # models run inside this app


try:
    P = get_predictor(API_URL)
except ApiError as e:
    st.error(f"Could not use the API gateway: {e}")
    st.info("Start it first (start_all.bat does this), or remove AGRISENSE_API_URL to run the models inside the app.")
    st.stop()
USING_API = bool(API_URL)
meta = P.meta
REAL_IMAGES = (MODELS / "image_meta.json").exists()


def clean(name):
    return str(name).replace("___", " - ").replace("_", " ")


def card(big, label, sub="", badge=None):
    b = f'<span class="badge {badge[0]}">{badge[1]}</span>' if badge else ""
    return f'<div class="card"><div class="big">{big}</div><div class="lbl">{label} {b}</div><div class="sub">{sub}</div></div>'


def result_box(title, value, sub=""):
    st.markdown(f'<div class="result"><div class="sub">{title}</div><div class="big">{value}</div>'
                f'<div class="sub">{sub}</div></div>', unsafe_allow_html=True)


def number_grid(features, defaults, ranges, key, cols=4):
    values, grid = {}, st.columns(cols)
    for i, f in enumerate(features):
        lo, hi = ranges[f]
        step = max((hi - lo) / 100, 0.001)
        k = f"{key}_{f}"
        kwargs = {} if k in st.session_state else {"value": float(round(defaults[f], 3))}   # state wins once set
        values[f] = grid[i % cols].number_input(
            f.replace("_", " "), step=float(round(step, 3)), format="%.3f", key=k, **kwargs)
    return values


def prob_bars(pairs, top=5):
    for name, p in pairs[:top]:
        c1, c2 = st.columns([3, 1])
        c1.progress(min(max(float(p), 0.0), 1.0), text=clean(name))
        c2.write(f"{p*100:.1f}%")


def sample_images(npz_name):
    """{name: (PIL image, known class)} from synthetic samples (if the models are synthetic) and examples/<kind>/."""
    out = {}
    path = MODELS.parent / "data" / npz_name
    if not REAL_IMAGES and path.exists():
        d = np.load(path)
        for c in sorted(set(d["y"])):
            arr = d["X"][np.where(d["y"] == c)[0][3]]
            out[f"Synthetic: {c}"] = (Image.fromarray((arr * 255).astype("uint8")).resize((192, 192), Image.NEAREST), str(c))
    kind = "leaf" if npz_name.startswith("leaf") else "landcover"
    for ex in examples.list_example_images(MODELS.parent, kind):
        try:
            out[ex["name"]] = (Image.open(ex["path"]).convert("RGB"), ex["label"])
        except Exception:
            continue
    return out


def image_page(title, npz_name, predict_fn, what, tip):
    if REAL_IMAGES:
        st.info(f"This model was trained on real images ({what}). {tip}")
    else:
        st.warning("These image models were trained on **synthetic** images. Real photos give meaningless answers "
                   "until you train on real data (see INSTRUCTIONS.md).")
    left, right = st.columns([1, 1.3])
    with left:
        up = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "webp"], key=title)
        store = "last_img_" + title            # survives switching pages (widget state does not)
        if up is not None:
            st.session_state[store] = (up.name, up.getvalue())
        samples = sample_images(npz_name)
        choice = st.selectbox("...or try a sample", ["(none)"] + list(samples), key=title + "_s") if samples else "(none)"
        img, caption, known = None, "Your input", None
        if choice != "(none)":
            img, known = samples[choice]
            caption = "Example image"
        elif store in st.session_state:
            name, data = st.session_state[store]
            img, caption = Image.open(io.BytesIO(data)), f"Your input: {name}"
        if img is not None:
            st.image(img, caption=caption, width=260)
            if store in st.session_state and st.button("Clear image", key=title + "_clear"):
                st.session_state.pop(store, None)
                st.rerun()
    with right:
        if img is None:
            st.info("Upload an image to see a prediction.")
            return
        res = predict_fn(img)
        top, conf = clean(res[0][0]), res[0][1]
        result_box("Prediction", top, f"{conf*100:.1f}% confidence")
        if conf < 0.6:
            st.warning("Low confidence. The model is unsure; this image may not be something it was trained on "
                       "(for leaves: a crop outside the 38 PlantVillage classes).")
        prob_bars(res, top=5)
        if known:
            st.caption(f"Known class of this example: {clean(known)}")


def satellite_map(lat, lon, radius=500, height=420):
    """Satellite basemap (Esri World Imagery) with a circle marking the area NDVI is averaged over."""
    html = f"""
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<div id="m" style="height:{height}px;border-radius:12px;"></div>
<script>
var map = L.map('m').setView([{lat}, {lon}], 15);
L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}',
            {{maxZoom: 19, attribution: 'Imagery &copy; Esri, Maxar, Earthstar Geographics'}}).addTo(map);
L.circle([{lat}, {lon}], {{radius: {radius}, color: '#ff3b30', weight: 3, fillColor: '#ff3b30', fillOpacity: 0.15}}).addTo(map);
L.circleMarker([{lat}, {lon}], {{radius: 6, color: '#ffffff', weight: 2, fillColor: '#ff3b30', fillOpacity: 1}}).addTo(map);
</script>"""
    components.html(html, height=height + 10)


def plotly_missing():
    st.info("Install the 3D chart library to see this view: `pip install plotly`, then restart the app.")


def dark3d(fig, height=520):
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", height=height,
                      margin=dict(l=0, r=0, t=30, b=0), scene=dict(bgcolor="rgba(0,0,0,0)"))
    st.plotly_chart(fig, use_container_width=True)


def dark2d(fig, height=420):
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      height=height, margin=dict(l=10, r=10, t=30, b=10), legend_title_text="")
    st.plotly_chart(fig, use_container_width=True)


def satellite_map_multi(points, height=420):
    """Satellite map with one marker per registered field. points: list of (name, lat, lon, ndvi or None)."""
    data = json.dumps([{"name": n, "lat": la, "lon": lo, "ndvi": nd} for n, la, lo, nd in points])
    html = """
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<div id="m" style="height:__H__px;border-radius:12px;"></div>
<script>
var pts = __DATA__;
var map = L.map('m').setView([pts[0].lat, pts[0].lon], 12);
L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            {maxZoom: 19, attribution: 'Imagery &copy; Esri, Maxar, Earthstar Geographics'}).addTo(map);
var group = [];
pts.forEach(function (p) {
  var color = p.ndvi === null ? '#94a3b8' : (p.ndvi < 0.3 ? '#f59e0b' : '#22c55e');
  var m = L.circleMarker([p.lat, p.lon], {radius: 9, color: '#ffffff', weight: 2, fillColor: color, fillOpacity: 1}).addTo(map);
  var d = document.createElement('div');
  d.textContent = p.name + (p.ndvi === null ? ': no NDVI yet' : ': NDVI ' + p.ndvi.toFixed(3));
  m.bindPopup(d); group.push(m);
});
if (group.length > 1) { map.fitBounds(L.featureGroup(group).getBounds().pad(0.3), {maxZoom: 14}); }
</script>""".replace("__H__", str(height)).replace("__DATA__", data)
    components.html(html, height=height + 10)


def ndvi_label(v):
    if v < 0.2: return "Bare soil, water or built-up area"
    if v < 0.4: return "Sparse or stressed vegetation"
    if v < 0.6: return "Moderate vegetation"
    return "Dense, healthy vegetation"


# ----------------------------------------------------------------------------- sidebar
st.sidebar.markdown("## 🌱 AgriSense Cloud")
page = st.sidebar.radio("Explore", ["Home", "Example data", "Yield prediction", "Crop recommendation", "Leaf disease (image)",
                                    "Land cover (image)", "Satellite NDVI", "Sensors (simulated)", "Regional analytics", "Model comparison",
                                    "System architecture", "Data & results"])
st.sidebar.caption("Soil, weather, leaf photos and satellite imagery in one place.")
st.sidebar.caption("Models: through the API gateway" if USING_API else "Models: running inside this app")
st.sidebar.radio("Theme", ["Neon", "Forest"], key="theme", horizontal=True)
if PASSWORD and st.sidebar.button("Sign out"):
    st.session_state.pop("auth_ok", None)
    if "auth" in st.query_params:
        del st.query_params["auth"]
    st.rerun()

# ----------------------------------------------------------------------------- home
if page == "Home":
    st.markdown('<div class="hero"><h1>From soil and satellite to decisions</h1>'
                '<p>AgriSense Cloud combines machine learning, deep learning and Sentinel-2 satellite data to predict '
                'crop yield, recommend crops, spot leaf disease and classify land cover.</p></div>', unsafe_allow_html=True)
    m = meta["metrics"]
    c = st.columns(4)
    c[0].markdown(card(f"{m['landcover']['accuracy']*100:.1f}%", "Land-cover CNN", "10 classes, EuroSAT",
                       ("real", "real data") if REAL_IMAGES else ("sim", "synthetic")), unsafe_allow_html=True)
    c[1].markdown(card(f"{m['leaf']['accuracy']*100:.1f}%", "Leaf-disease CNN", "38 classes, PlantVillage",
                       ("real", "real data") if REAL_IMAGES else ("sim", "synthetic")), unsafe_allow_html=True)
    c[2].markdown(card(f"{m['crop']['accuracy']*100:.1f}%", "Crop recommender", "22 crops, Random Forest"), unsafe_allow_html=True)
    c[3].markdown(card(f"{m['yield']['R2']:.2f}", "Yield model R²", "soil + weather + NDVI", ("sim", "simulated")), unsafe_allow_html=True)
    st.write("")
    st.subheader("What you can do here")
    a, b, c2 = st.columns(3)
    a.markdown(card("🌾", "Yield and crop advice", "Enter soil, weather and NDVI values, or upload a CSV for many fields."), unsafe_allow_html=True)
    b.markdown(card("🍃", "See the field", "Upload a leaf photo or a satellite tile and get a classification with confidence."), unsafe_allow_html=True)
    c2.markdown(card("🛰️", "Live satellite NDVI", "Pull vegetation greenness for any coordinates from Sentinel-2."), unsafe_allow_html=True)

    crop_csv = MODELS.parent / "data" / "crop_recommendation.csv"
    if crop_csv.exists():
        st.subheader("Crop data in 3D (nitrogen, phosphorus, potassium)")
        if HAS_PLOTLY:
            cdf = pd.read_csv(crop_csv)
            if {"N", "P", "K", "label"} <= set(cdf.columns):
                cdf = cdf.sample(min(len(cdf), 700), random_state=1)
                dark3d(px.scatter_3d(cdf, x="N", y="P", z="K", color="label", opacity=0.85), 560)
                st.caption("Drag to rotate, scroll to zoom. Each dot is one soil sample, coloured by the crop that suits it.")
        else:
            plotly_missing()

# ----------------------------------------------------------------------------- yield
elif page == "Yield prediction":
    st.title("Crop yield prediction")
    st.caption(f"{meta['metrics']['yield'].get('model', 'Random Forest')} on soil, weather and satellite NDVI features.")
    try:
        yield_ex = examples.yield_examples(MODELS.parent / "data")
    except Exception:
        yield_ex = {}

    def _apply_yield_example():
        picked = st.session_state.get("yield_example", "(none)")
        for f_, v_ in yield_ex.get(picked, {}).get("values", {}).items():
            st.session_state[f"y_{f_}"] = float(v_)

    if yield_ex:
        st.selectbox("Load an example field", ["(none)"] + list(yield_ex), key="yield_example", on_change=_apply_yield_example)
    vals = number_grid(YIELD_FEATURES, meta["yield_defaults"], meta["yield_ranges"], "y")
    pred = P.predict_yield([vals])[0]
    result_box("Predicted yield", f"{pred:,.0f} kg/ha", "For the field values above")
    chosen_ex = yield_ex.get(st.session_state.get("yield_example", "(none)"))
    if chosen_ex and all(abs(vals[f_] - chosen_ex["values"][f_]) < 1e-3 for f_ in YIELD_FEATURES):
        st.caption(f"The example dataset recorded {chosen_ex['actual']:,.0f} kg/ha for this field "
                   "(simulated data unless you replaced it).")

    if "ndvi_mean" in YIELD_FEATURES:
        st.subheader("What if the crop were greener?")
        grid = np.linspace(0.1, 0.9, 17)
        rows = [{**vals, "ndvi_mean": float(g)} for g in grid]
        ys = P.predict_yield(rows)
        st.line_chart(pd.DataFrame({"NDVI (mean)": grid, "Predicted yield (kg/ha)": ys}).set_index("NDVI (mean)"))
        st.caption("Everything else is held fixed; only mean NDVI changes.")

        if HAS_PLOTLY and "rainfall" in YIELD_FEATURES:
            st.subheader("Yield landscape in 3D")
            lo, hi = meta["yield_ranges"]["rainfall"]
            ng, rg = np.linspace(0.1, 0.9, 14), np.linspace(lo, hi, 14)
            pts = [{**vals, "ndvi_mean": float(n), "rainfall": float(r)} for r in rg for n in ng]
            Z = np.array(P.predict_yield(pts)).reshape(len(rg), len(ng))
            fig = go.Figure(go.Surface(x=ng, y=rg, z=Z, colorscale="Viridis"))
            fig.update_layout(scene=dict(xaxis_title="NDVI", yaxis_title="Rainfall", zaxis_title="Yield kg/ha"))
            dark3d(fig, 560)
            st.caption("Drag to rotate. The surface shows how predicted yield changes with NDVI and rainfall, other inputs fixed.")
        elif not HAS_PLOTLY:
            plotly_missing()

    st.subheader("Batch from CSV")
    st.write("Required columns: `" + ", ".join(YIELD_FEATURES) + "`")
    template = pd.DataFrame([meta["yield_defaults"]] * 3)[YIELD_FEATURES]
    st.download_button("Download CSV template", template.to_csv(index=False), "yield_template.csv", "text/csv")
    up = st.file_uploader("Upload CSV", type="csv")
    use_example = st.checkbox("Use the example batch instead (40 fields with known yields)", disabled=not yield_ex)
    if up or (use_example and yield_ex):
        df = pd.read_csv(up) if up else examples.yield_batch(MODELS.parent / "data")
        df.columns = [c.strip().lower() for c in df.columns]
        missing = [c for c in YIELD_FEATURES if c not in df.columns]
        if missing:
            st.error(f"Missing column(s): {', '.join(missing)}")
        else:
            df["predicted_yield_kg_ha"] = np.round(P.predict_yield(df[YIELD_FEATURES].values), 1)
            st.success(f"{len(df)} rows - mean predicted yield {df['predicted_yield_kg_ha'].mean():,.0f} kg/ha")
            if "actual_yield_kg_ha" in df.columns:
                err = df["predicted_yield_kg_ha"] - df["actual_yield_kg_ha"]
                ss_tot = float(((df["actual_yield_kg_ha"] - df["actual_yield_kg_ha"].mean()) ** 2).sum())
                m1, m2 = st.columns(2)
                m1.metric("MAE vs actual (kg/ha)", f"{err.abs().mean():,.0f}")
                m2.metric("R² vs actual", f"{1 - float((err ** 2).sum()) / ss_tot:.3f}" if ss_tot else "n/a")
                st.caption("Actual values come from the example dataset. If you have not replaced it with real data, they are simulated.")
            st.dataframe(df, use_container_width=True)
            st.download_button("Download predictions", df.to_csv(index=False), "yield_predictions.csv", "text/csv")

# ----------------------------------------------------------------------------- crop
elif page == "Crop recommendation":
    st.title("Crop recommendation")
    st.caption("Random Forest over 22 crops using soil nutrients and climate.")
    crop_ex = examples.crop_scenarios(MODELS.parent / "data")

    def _apply_crop_example():
        picked = st.session_state.get("crop_example", "(none)")
        for f_, v_ in crop_ex.get(picked, {}).items():
            st.session_state[f"c_{f_}"] = float(v_)

    st.selectbox("Load an example scenario", ["(none)"] + list(crop_ex), key="crop_example", on_change=_apply_crop_example)
    st.caption("Typical conditions for each crop, taken from your crop dataset. Edit any value afterwards.")
    vals = number_grid(CROP_FEATURES, meta["crop_defaults"], meta["crop_ranges"], "c")
    res = P.recommend_crop(vals)
    result_box("Best match", clean(res[0][0]).title(), f"{res[0][1]*100:.1f}% confidence")
    st.subheader("Other good options")
    prob_bars(res[1:], top=4)

# ----------------------------------------------------------------------------- images
elif page == "Leaf disease (image)":
    st.title("Leaf disease detection")
    image_page("Leaf disease CNN", "leaf_images.npz", P.detect_disease, "PlantVillage, 38 crop-and-disease classes",
               "Works best on a single leaf on a plain background, from crops like tomato, potato, apple, corn or grape.")

elif page == "Land cover (image)":
    st.title("Land-cover classification")
    image_page("Land cover CNN", "landcover_images.npz", P.classify_landcover, "EuroSAT Sentinel-2 tiles, 10 classes",
               "Works best on small top-down satellite tiles covering about 600 m.")

# ----------------------------------------------------------------------------- satellite / sensors
elif page == "Satellite NDVI":
    st.title("Satellite NDVI")
    st.caption("Vegetation greenness from Sentinel-2 via Google Earth Engine. Higher means greener.")
    if not os.getenv("GEE_PROJECT"):
        st.warning("Earth Engine is not configured. Set GEE_PROJECT and run `earthengine authenticate` (see INSTRUCTIONS.md).")
    else:
        c1, c2, c3 = st.columns(3)
        lat = c1.number_input("Latitude", value=13.0827, format="%.5f")
        lon = c2.number_input("Longitude", value=80.2707, format="%.5f")
        days = c3.slider("Look-back window (days)", 7, 90, 30)
        satellite_map(lat, lon, radius=500)
        st.caption("The red circle (500 m radius) is the exact area whose NDVI is averaged.")
        if st.button("Fetch NDVI", type="primary"):
            try:
                from gee_ndvi import mean_ndvi
                with st.spinner("Asking Earth Engine..."):
                    v = mean_ndvi(lat, lon, days)
                if v is None:
                    st.warning("No cloud-free scene in that window. Try a longer look-back.")
                else:
                    result_box("Mean NDVI around this point", f"{v:.3f}", ndvi_label(v))
                    st.progress(min(max(v, 0.0), 1.0))
            except Exception as e:
                st.error(f"Earth Engine error: {e}")

    st.divider()
    st.subheader("Registered fields (scheduled ingestion)")
    st.caption("ingest_satellite.py fetches NDVI for every field below and stores it in data/ndvi_observations.csv. "
               "Run it automatically with run_ingestion.bat (see INSTRUCTIONS.md).")
    FIELDS_PATH, OBS_PATH = ingest.FIELDS_PATH, ingest.OBS_PATH
    if "flash" in st.session_state:
        st.success(st.session_state.pop("flash"))
    _bad = st.session_state.pop("ingest_bad", None)
    if _bad is not None:
        st.warning("Some fields were not updated:")
        st.dataframe(_bad, use_container_width=True, hide_index=True)
    with st.form("add_field", clear_on_submit=True):
        r1, r2, r3 = st.columns(3)
        fid = r1.text_input("Field name", key="reg_name")
        flat = r2.number_input("Latitude ", value=13.0, format="%.5f", key="reg_lat")
        flon = r3.number_input("Longitude ", value=80.0, format="%.5f", key="reg_lon")
        if st.form_submit_button("Register field"):
            try:
                ingest.add_field(FIELDS_PATH, fid, flat, flon)
                st.success(f"Registered {fid.strip()}.")
            except ValueError as e:
                st.error(str(e))
    if st.button("Add the 4 example fields (urban, lake, delta, hills)"):
        added = 0
        for ex_ in examples.EXAMPLE_FIELDS:
            try:
                ingest.add_field(FIELDS_PATH, ex_["field_id"], ex_["lat"], ex_["lon"])
                added += 1
            except ValueError:
                pass
        st.session_state["flash"] = (f"Added {added} example field(s). The points are approximate: check each marker on the "
                                     "map and move it onto a real field if needed.")
        st.rerun()
    fields_df = ingest.load_fields(FIELDS_PATH)
    obs = ingest.load_observations(OBS_PATH)
    if fields_df.empty:
        st.info("No fields registered yet. Add one above.")
    else:
        latest = ingest.latest_by_field(obs)
        latest["field_id"] = latest["field_id"].astype(str)
        table = fields_df.assign(field_id=fields_df["field_id"].astype(str)).merge(latest, on="field_id", how="left")
        st.dataframe(table, use_container_width=True, hide_index=True)
        with st.expander("Remove a field"):
            del_id = st.selectbox("Field to remove", list(fields_df["field_id"]), key="del_field")
            del_hist = st.checkbox("Also delete its stored NDVI history", key="del_hist")
            if st.button("Remove field", key="del_btn"):
                try:
                    ingest.remove_field(FIELDS_PATH, del_id, OBS_PATH, del_hist)
                    st.session_state["flash"] = f"Removed {del_id}" + (" and its history." if del_hist else ". Its stored history was kept.")
                    st.rerun()
                except ValueError as e:
                    st.error(str(e))
        if st.button("Run ingestion now"):
            if not os.getenv("GEE_PROJECT"):
                st.error("Set GEE_PROJECT and run `earthengine authenticate` first.")
            else:
                from gee_ndvi import mean_ndvi
                with st.spinner("Fetching NDVI for every field..."):
                    new = ingest.ingest(fields_df, mean_ndvi, days=30, out=OBS_PATH)
                bad = new[new["status"] != "ok"]
                st.session_state["flash"] = f"{int((new['status'] == 'ok').sum())} of {len(new)} fields updated."
                st.session_state["ingest_bad"] = bad if not bad.empty else None
                st.rerun()                      # redraw the table, map and chart with the new values
        points = [(r.field_id, float(r.lat), float(r.lon), None if pd.isna(r.latest_ndvi) else float(r.latest_ndvi))
                  for r in table.itertuples()]
        satellite_map_multi(points)
        st.caption("Green markers have NDVI of 0.3 or more, amber below 0.3, grey have no NDVI stored yet.")
        history = obs[obs["ndvi"].notna() & obs["field_id"].isin(fields_df["field_id"])]
        if not history.empty:
            st.subheader("NDVI history per field")
            if HAS_PLOTLY:
                dark2d(px.line(history, x="date", y="ndvi", color="field_id", markers=True, labels={"ndvi": "NDVI", "date": ""}))
            else:
                st.line_chart(history.pivot(index="date", columns="field_id", values="ndvi"))

elif page == "Sensors (simulated)":
    st.title("Field sensors")
    st.markdown('<span class="badge sim">SIMULATED DATA</span>', unsafe_allow_html=True)
    st.caption("Synthetic IoT readings that follow a daily temperature cycle, a seasonal baseline, and rain events that "
               "are more likely in the monsoon months. Replace with a real sensor feed in production.")
    s1, s2 = st.columns(2)
    hours = s1.slider("Hours of history", 24, 168, 72, step=24)
    seed = s2.number_input("Random seed", value=7, step=1)
    sdf = simulate_sensors(hours=hours, seed=int(seed))
    last = sdf.iloc[-1]
    c = st.columns(3)
    c[0].markdown(card(f"{last['soil_moisture_pct']:.1f}%", "Soil moisture", "latest reading"), unsafe_allow_html=True)
    c[1].markdown(card(f"{last['temperature_c']:.1f} C", "Temperature", "latest reading"), unsafe_allow_html=True)
    c[2].markdown(card(f"{last['humidity_pct']:.1f}%", "Humidity", "latest reading"), unsafe_allow_html=True)
    st.write("")
    st.line_chart(sdf.set_index("timestamp")[["soil_moisture_pct", "temperature_c", "humidity_pct"]])
    st.download_button("Download readings (CSV)", sdf.to_csv(index=False), "simulated_sensors.csv", "text/csv")
    if HAS_PLOTLY:
        st.subheader("Readings in 3D")
        fig = go.Figure(go.Scatter3d(
            x=list(range(len(sdf))), y=sdf["soil_moisture_pct"], z=sdf["temperature_c"],
            mode="lines+markers", line=dict(width=5, color="#00f5c4"),
            marker=dict(size=4, color=sdf["humidity_pct"], colorscale="Turbo", showscale=True,
                        colorbar=dict(title="Humidity"))))
        fig.update_layout(scene=dict(xaxis_title="Hours", yaxis_title="Soil moisture", zaxis_title="Temperature"))
        dark3d(fig, 500)

# ----------------------------------------------------------------------------- regional analytics
elif page == "Regional analytics":
    st.title("Regional analytics")
    try:
        reg = load_regional(MODELS.parent / "data")
    except ValueError as e:
        st.error(str(e))
        st.stop()
    df, monthly, disease, simulated = reg["summary"], reg["monthly"], reg["disease"], reg["simulated"]
    if simulated:
        st.markdown('<span class="badge sim">SIMULATED REGIONAL DATA</span>', unsafe_allow_html=True)
        st.caption("Demo data, not measurements. To use your own, add data/regional_summary.csv (see INSTRUCTIONS.md).")
    else:
        st.markdown('<span class="badge real">YOUR REGIONAL DATA</span>', unsafe_allow_html=True)
    if not HAS_PLOTLY:
        plotly_missing()
        st.stop()

    f1, f2 = st.columns([3, 1])
    all_districts = sorted(df["district"].unique())
    chosen = f1.multiselect("Districts", all_districts, default=all_districts)
    seasons_here = [x for x in SEASONS if x in set(df["season"])]
    season = f2.selectbox("Season", ["All"] + seasons_here)
    sub = df[df["district"].isin(chosen)]
    if season != "All":
        sub = sub[sub["season"] == season]
    if sub.empty:
        st.warning("No data for this selection.")
        st.stop()

    k = kpis(sub)
    c = st.columns(4)
    c[0].markdown(card(f"{k['mean_yield']:,.0f}", "Mean yield (kg/ha)", f"{k['yield_change_pct']:+.1f}% from {k['first_year']} to {k['last_year']}"), unsafe_allow_html=True)
    c[1].markdown(card(f"{k['mean_ndvi']:.2f}", "Mean NDVI", f"{k['districts']} districts"), unsafe_allow_html=True)
    c[2].markdown(card(f"{k['mean_soil_moisture']:.1f}%", "Soil moisture", "simulated IoT layer" if simulated else "from your data"), unsafe_allow_html=True)
    c[3].markdown(card(f"{k['mean_disease']:.1f}%", "Disease incidence", "share of plants affected"), unsafe_allow_html=True)
    st.write("")

    t1, t2, t3, t4, t5, t6 = st.tabs(["Yield trends", "NDVI trajectory", "Correlations", "Regional comparison",
                                      "Disease and sensors", "3D view"])
    with t1:
        d, order = yield_trend(sub)
        dark2d(px.line(d, x="period", y="yield_kg_ha", color="district", markers=True,
                       category_orders={"period": order}, labels={"yield_kg_ha": "Yield (kg/ha)", "period": ""}))
        st.caption("Season-over-season yield for each selected district.")
    with t2:
        if monthly is None:
            st.info("Add data/regional_monthly.csv (district, year, season, month, ndvi, soil_moisture_pct) to see NDVI through a season.")
        else:
            m = monthly[monthly["district"].isin(chosen)]
            years = sorted(m["year"].unique())
            m1, m2 = st.columns(2)
            yr = m1.selectbox("Year", years, index=len(years) - 1)
            ss = m2.selectbox("Season ", seasons_here if season == "All" else [season])
            m = m[(m["year"] == yr) & (m["season"] == ss)]
            if m.empty:
                st.warning("No monthly data for this choice.")
            else:
                dark2d(px.line(m, x="month", y="ndvi", color="district", markers=True,
                               labels={"ndvi": "NDVI", "month": "Month of season"}))
                st.caption("NDVI rises to a peak mid-season and falls toward harvest. This is the mean, peak and slope signal the yield model uses.")
    with t3:
        corr = correlations(sub)
        dark2d(px.imshow(corr, text_auto=".2f", zmin=-1, zmax=1, color_continuous_scale="RdBu", aspect="auto"), 480)
        st.caption("Pearson correlation across all selected district-seasons. Correlation is not causation.")
    with t4:
        gap = underperformance(sub)
        if gap.empty:
            st.info("Not enough rows for a comparison. Select more districts or seasons.")
        else:
            dark2d(px.bar(gap, x="district", y="gap_pct", color="status",
                          color_discrete_map={"Below peers": "#ff6b6b", "On par or above": "#00f5c4"},
                          labels={"gap_pct": "Yield vs expected (%)", "district": ""}))
            st.caption("Expected yield comes from rainfall, temperature and NDVI across all selected districts. "
                       "Below -5% means the district yields less than similar conditions would suggest.")
        st.dataframe(district_table(sub), use_container_width=True, hide_index=True)
    with t5:
        a1, a2 = st.columns(2)
        with a1:
            sm = sub.groupby("district", as_index=False)["soil_moisture_pct"].mean()
            dark2d(px.bar(sm, x="district", y="soil_moisture_pct", labels={"soil_moisture_pct": "Soil moisture (%)", "district": ""}), 360)
            st.caption("Average soil moisture" + (" (simulated IoT)." if simulated else "."))
        with a2:
            di = sub.groupby("district", as_index=False)["disease_incidence_pct"].mean()
            dark2d(px.bar(di, x="district", y="disease_incidence_pct", labels={"disease_incidence_pct": "Disease incidence (%)", "district": ""}), 360)
            st.caption("Average share of plants affected.")
        if disease is not None:
            dd = disease[disease["district"].isin(chosen)].groupby(["district", "disease"], as_index=False)["cases"].sum()
            dark2d(px.bar(dd, x="disease", y="cases", color="district", labels={"cases": "Cases", "disease": ""}), 380)
            st.caption("Disease-incidence frequency by type and district.")
    with t6:
        dark3d(px.scatter_3d(sub, x="rainfall_mm", y="ndvi_mean", z="yield_kg_ha", color="district", opacity=0.85,
                             labels={"rainfall_mm": "Rainfall", "ndvi_mean": "NDVI", "yield_kg_ha": "Yield"}), 580)
        st.caption("Each dot is one district-season. Drag to rotate: greener, wetter seasons sit higher.")

# ----------------------------------------------------------------------------- model comparison
elif page == "Model comparison":
    st.title("Yield model comparison")
    st.caption("Random Forest, XGBoost and a shallow ANN, each trained on ground features only and on ground + NDVI "
               "features, scored with 5-fold cross-validation.")
    data_dir = MODELS.parent / "data"
    if not xgboost_available():
        st.warning("XGBoost is not installed, so it is left out. Install it with `pip install xgboost` and run again.")
    if st.button("Run comparison now (about a minute)", type="primary"):
        if not (data_dir / "soil_weather_yield.csv").exists() or not (data_dir / "ndvi_timeseries.csv").exists():
            st.error("data/soil_weather_yield.csv and data/ndvi_timeseries.csv are needed. Run train_and_save.py first.")
        else:
            with st.spinner("Training models with cross-validation..."):
                save_comparison(run_comparison(data_dir))
    res = load_comparison()
    if res is None:
        st.info("No results yet. Click the button above.")
    elif not HAS_PLOTLY:
        plotly_missing()
    else:
        st.markdown('<span class="badge sim">DATA: data/soil_weather_yield.csv</span>', unsafe_allow_html=True)
        st.caption(f"{res['n_rows']} fields, {res['folds']}-fold cross-validation, run {res['created']}. "
                   "Results are only as real as that data. If you have not replaced it with real data, they are simulated.")
        rdf = pd.DataFrame(res["results"])
        gain = ndvi_gain(res)
        if not gain.empty:
            cols = st.columns(len(gain))
            for col, (_, g) in zip(cols, gain.iterrows()):
                col.markdown(card(f"{g['R2 gain']:+.3f}", f"{g['model']}: R² gain from NDVI",
                                  f"{g['R2 ground only']:.3f} to {g['R2 ground + NDVI']:.3f}"), unsafe_allow_html=True)
            st.write("")
        metric = st.radio("Metric", ["R2", "RMSE", "MAE"], horizontal=True)
        dark2d(px.bar(rdf, x="model", y=metric, color="features", barmode="group", error_y=f"{metric}_std",
                      labels={"model": ""}))
        st.caption("Higher R² is better; lower RMSE and MAE are better. Error bars show the spread across folds.")
        show = rdf.round(3)
        st.dataframe(show, use_container_width=True, hide_index=True)
        st.download_button("Download results (CSV)", show.to_csv(index=False), "model_comparison.csv", "text/csv")
        st.subheader("Reported in the paper (not measured here)")
        st.dataframe(pd.DataFrame([
            {"Model": "XGBoost", "Features": "Ground only", "R2": 0.87},
            {"Model": "XGBoost", "Features": "Ground + NDVI", "R2": 0.93},
            {"Model": "ANN", "Features": "Ground + NDVI", "R2": 0.89}]), use_container_width=True, hide_index=True)
        st.caption("These come from Table III of the paper. Use your own measured numbers above in your report.")

# ----------------------------------------------------------------------------- system architecture
elif page == "System architecture":
    st.title("System architecture")
    st.caption("The data flow of AgriSense Cloud (paper Fig. 1). Drag to rotate, scroll to zoom, hover a node for details.")
    if HAS_PLOTLY:
        dark3d(architecture.build_figure(go), 620)
    else:
        plotly_missing()
    st.subheader("How each part is built in this project")
    st.dataframe(pd.DataFrame(architecture.status_table(USING_API, ingest.observation_summary(ingest.OBS_PATH), meta["metrics"]["yield"].get("model", "Random Forest")),
                              columns=["Component", "Implementation", "Status"]),
                 use_container_width=True, hide_index=True)
    st.caption("The paper names AWS for hosting. This project uses Google Earth Engine for satellite data, "
               "and the Cloud Run deployment is optional and not deployed.")

# ----------------------------------------------------------------------------- example data
elif page == "Example data":
    st.title("Example data")
    st.caption("Ready-made data to showcase every feature. Pick a dataset to preview it and save it. Most examples are also "
               "built into the pages themselves, as the page column below shows.")
    data_dir = MODELS.parent / "data"
    root = MODELS.parent
    names = ["Crop scenarios (one per crop)", "Yield example fields (low, typical, high)", "Yield batch (40 fields, with actual yields)",
             "Regional summary (simulated)", "Regional monthly NDVI (simulated)", "Regional disease cases (simulated)",
             "Sensor stream (72 hours, simulated)", "Example fields for satellite NDVI", "Leaf images", "Satellite tiles"]
    pick = st.selectbox("Dataset", names)

    def show_table(df, file_name, used_in):
        st.caption(f"Used in: {used_in}")
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button("Save as CSV", examples.csv_bytes(df), file_name, "text/csv")

    try:
        if pick == names[0]:
            sc = examples.crop_scenarios(data_dir)
            show_table(pd.DataFrame([{"scenario": k, **v} for k, v in sc.items()]), "crop_scenarios.csv",
                       "Crop recommendation, 'Load an example scenario'")
        elif pick == names[1]:
            ye = examples.yield_examples(data_dir)
            show_table(pd.DataFrame([{"field": k, **v["values"], "actual_yield_kg_ha": v["actual"]} for k, v in ye.items()]),
                       "yield_example_fields.csv", "Yield prediction, 'Load an example field'")
        elif pick == names[2]:
            show_table(examples.yield_batch(data_dir), "yield_batch_example.csv",
                       "Yield prediction, batch section ('Use the example batch'), or upload the saved file")
        elif pick in names[3:6]:
            files = examples.regional_examples()
            fname = ["regional_summary.csv", "regional_monthly.csv", "disease_cases.csv"][names.index(pick) - 3]
            show_table(files[fname], fname, "Regional analytics. To use your own version, save it in the data folder under this name.")
        elif pick == names[6]:
            show_table(examples.sensor_example(72), "sensor_stream_example.csv", "Sensors page")
        elif pick == names[7]:
            show_table(examples.fields_example(), "example_fields.csv", "Satellite NDVI, 'Add the 4 example fields'")
            st.info("These points are approximate. Check each marker on the satellite map before relying on it.")
        else:
            kind = "leaf" if pick == "Leaf images" else "landcover"
            items = examples.list_example_images(root, kind)
            st.caption("Used in: " + ("Leaf disease (image)" if kind == "leaf" else "Land cover (image)") + ", 'or try a sample'")
            if not items:
                st.info("No example images yet. Run colab_make_example_images.py in Google Colab, unzip the result into the "
                        f"project's examples folder ({root / 'examples'}), then refresh this page.")
            else:
                st.success(f"{len(items)} example images found.")
                cols = st.columns(4)
                for n_, it in enumerate(items):
                    cols[n_ % 4].image(it["path"], caption=it["name"], use_container_width=True)
                st.download_button("Save all as ZIP", examples.zip_examples(root, kind), f"{kind}_examples.zip", "application/zip")
    except FileNotFoundError:
        st.error("The training data files are missing. Run `python train_and_save.py` first.")
    except Exception as e:
        st.error(f"Could not build this example: {e}")

    st.subheader("Showcase checklist")
    st.dataframe(pd.DataFrame(examples.SHOWCASE, columns=["Feature", "Page", "Example to use", "What to look for"]),
                 use_container_width=True, hide_index=True)

# ----------------------------------------------------------------------------- results
else:
    st.title("Data and results")
    m = meta["metrics"]
    obs_sum = ingest.observation_summary(ingest.OBS_PATH)
    ndvi_result = (f"{obs_sum['rows']} readings, {obs_sum['fields']} fields, last {obs_sum['last_date']}"
                   if obs_sum["rows"] else "No readings stored yet")
    try:
        reg_df = load_regional(MODELS.parent / "data")["summary"]
        regional_result = f"{reg_df['district'].nunique()} districts, {len(reg_df)} district-seasons"
    except Exception:
        regional_result = "Data file has a problem"
    cmp_res = load_comparison()
    if cmp_res and cmp_res.get("results"):
        best = max(cmp_res["results"], key=lambda r: r["R2"])
        cmp_result = f"Best: {best['model']}, R² {best['R2']:.3f} ({best['features']})"
    else:
        cmp_result = "Not run yet (open Model comparison)"
    rows = [
        ("Leaf-disease CNN", f"{m['leaf']['accuracy']*100:.1f}% accuracy", "PlantVillage" if REAL_IMAGES else "synthetic images",
         "Real" if REAL_IMAGES else "Simulated"),
        ("Land-cover CNN", f"{m['landcover']['accuracy']*100:.1f}% accuracy", "EuroSAT" if REAL_IMAGES else "synthetic images",
         "Real" if REAL_IMAGES else "Simulated"),
        ("Crop recommender", f"{m['crop']['accuracy']*100:.1f}% accuracy", "Kaggle CSV if you added one, else synthetic", "Check your data folder"),
        ("Yield model", f"{m['yield'].get('model', 'Random Forest')}: R² {m['yield']['R2']}, RMSE {m['yield']['RMSE']} kg/ha", "synthetic yield + NDVI series", "Simulated"),
        ("Live NDVI", ndvi_result, "Sentinel-2 via Earth Engine", "Real" if obs_sum["rows"] else "Ready, no readings yet"),
        ("Sensors", "No metric (simulated stream)", "sensors.py daily-cycle generator", "Simulated"),
        ("Regional analytics", regional_result, "data/regional_summary.csv if present, else generated", "Simulated unless you add your CSV"),
        ("Model comparison", cmp_result, "data/soil_weather_yield.csv", "Real only if that data is real"),
    ]
    st.dataframe(pd.DataFrame(rows, columns=["Component", "Result", "Data", "Status"]), use_container_width=True, hide_index=True)
    st.info("Quote only results marked Real as real-world performance. Accuracy is measured on a held-out split of each dataset.")

st.markdown('<div class="footer">AgriSense Cloud - machine learning and satellite data for smarter farming.</div>', unsafe_allow_html=True)
