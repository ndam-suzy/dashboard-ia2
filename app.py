import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image as PILImage
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="IA2 — Deep Learning",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown("""
<style>
/* ══ NAVIGATION PRINCIPALE ══════════════════════════════════════ */
div[data-testid="stRadio"] > div {
  display: flex !important;
  flex-direction: row !important;
  gap: 8px !important;
  background: var(--white) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--rad) !important;
  padding: 5px !important;
  margin-bottom: 1.75rem !important;
  width: fit-content !important;
}
div[data-testid="stRadio"] > div > label {
  background: transparent !important;
  border: 1px solid transparent !important;
  border-radius: var(--rad-sm) !important;
  padding: 8px 20px !important;
  font-size: 0.75rem !important;
  font-weight: 600 !important;
  letter-spacing: 0.08em !important;
  text-transform: uppercase !important;
  color: var(--muted) !important;
  cursor: pointer !important;
  transition: all 0.18s !important;
  white-space: nowrap !important;
}
div[data-testid="stRadio"] > div > label:hover {
  background: var(--br-pl) !important;
  color: var(--br-dk) !important;
}
div[data-testid="stRadio"] > div > label[data-checked="true"] {
  background: var(--br-dk) !important;
  color: var(--white) !important;
  border-color: var(--br-dk) !important;
}
/* Cacher le cercle radio natif */
div[data-testid="stRadio"] > div > label > div:first-child {
  display: none !important;
}
[data-testid="collapsedControl"] {
    display: block !important;
    visibility: visible !important;
    opacity: 1 !important;
}
[data-testid="stSidebar"] {
  background: var(--br-dk) !important;
  border-right: 1px solid rgba(255,255,255,0.06) !important;
}
[data-testid="stSidebar"] > div {
  background: var(--br-dk) !important;
}
</style>
""", unsafe_allow_html=True)
IMG = "graphiques"
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MOD = os.path.join(BASE_DIR, "modeles")
CLASS_NAMES = ["T-shirt","Pantalon","Pull","Robe","Manteau",
               "Sandale","Chemise","Sneaker","Sac","Bottine"]

# ══════════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;1,400&display=swap');

:root {
  --br-dk:  #3D1F0E;
  --br-mn:  #8B5E3C;
  --br-md:  #B8845A;
  --br-wm:  #D4A47A;
  --br-cr:  #E8C49A;
  --br-lt:  #F0DEC8;
  --br-pl:  #F8EFE4;
  --bg:     #FDFAF6;
  --white:  #FFFFFF;
  --txt:    #2C1A0E;
  --muted:  #9E8070;
  --border: rgba(212,164,122,0.22);
  --rad:    14px;
  --rad-sm: 9px;
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"] {
  font-family: 'DM Sans', -apple-system, sans-serif;
  background: var(--bg);
  color: var(--txt);
  font-size: 14px;
}

/* ── scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--br-lt); border-radius: 99px; }

/* ══ SIDEBAR ══════════════════════════════════════════════════════ */
[data-testid="stSidebar"] {
  background: var(--br-dk);
  border-right: 1px solid rgba(255,255,255,0.06);
  min-width: 220px !important;
  max-width: 220px !important;
}
[data-testid="stSidebar"] > div:first-child {
  padding: 0 !important;
}
[data-testid="stSidebar"] * { color: var(--br-pl) !important; }

/* sidebar collapse button */
[data-testid="collapsedControl"] {
  background: var(--br-dk) !important;
  border-right: 1px solid rgba(255,255,255,0.08) !important;
  color: var(--br-lt) !important;
}


/* radio labels hide */
[data-testid="stSidebar"] .stRadio > label { display: none !important; }
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] {
  display: flex; flex-direction: column; gap: 2px; padding: 0 10px;
}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {
  display: flex !important; align-items: center;
  padding: 9px 12px; border-radius: var(--rad-sm);
  font-size: 0.8rem; font-weight: 500; cursor: pointer;
  transition: background 0.18s; border: 1px solid transparent;
  color: rgba(248,239,228,0.65) !important;
}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover {
  background: rgba(255,255,255,0.07);
  color: var(--br-pl) !important;
}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label[data-checked="true"],
[data-testid="stSidebar"] input[type="radio"]:checked + div {
  background: rgba(255,255,255,0.12) !important;
  border-color: rgba(212,164,122,0.35) !important;
  color: var(--white) !important;
}

/* ══ MAIN ════════════════════════════════════════════════════════ */
.main .block-container {
  padding: 1.75rem 2rem 3rem 2rem;
  max-width: 1280px;
}

/* ══ TYPOGRAPHY ══════════════════════════════════════════════════ */
.pg-title {
  font-family: 'Syne', sans-serif;
  font-size: clamp(1.5rem, 2.5vw, 2rem);
  font-weight: 800;
  color: var(--br-dk);
  line-height: 1.15;
  margin-bottom: 0.2rem;
}
.pg-sub {
  font-size: 0.82rem; color: var(--muted); font-weight: 400;
  letter-spacing: 0.02em; margin-bottom: 1.5rem;
}
.sec-title {
  font-family: 'Syne', sans-serif;
  font-size: 0.95rem; font-weight: 700;
  color: var(--br-dk); margin: 1.4rem 0 0.75rem;
}
.sec-label {
  font-size: 0.63rem; font-weight: 700;
  letter-spacing: 0.15em; text-transform: uppercase;
  color: var(--br-md); margin: 1.2rem 0 0.55rem;
}

/* ══ DIVIDER ════════════════════════════════════════════════════ */
.div { border: none; border-top: 1px solid var(--border); margin: 1.4rem 0; }

/* ══ METRICS ROW ════════════════════════════════════════════════ */
.mrow {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(110px,1fr));
  gap: 0.75rem; margin-bottom: 1.5rem;
}
.mpill {
  background: var(--white); border: 1px solid var(--border);
  border-radius: var(--rad); padding: 0.9rem 1rem; text-align: center;
  transition: box-shadow 0.2s;
}
.mpill:hover { box-shadow: 0 4px 18px rgba(139,94,60,0.1); }
.mpill.hi   { border-color: rgba(212,164,122,0.5); background: var(--br-pl); }
.mpill.best {
  background: linear-gradient(135deg, rgba(212,164,122,0.2), rgba(139,94,60,0.08));
  border-color: rgba(212,164,122,0.55);
}
.m-lbl { font-size: 0.6rem; font-weight: 700; letter-spacing: 0.13em;
         text-transform: uppercase; color: var(--br-md); margin-bottom: 0.3rem; }
.m-val { font-family: 'Syne', sans-serif; font-size: 1.45rem;
         font-weight: 800; color: var(--br-dk); line-height: 1; }
.m-val.sm { font-size: 0.95rem; }

/* ══ CARD ═══════════════════════════════════════════════════════ */
.card {
  background: var(--white); border: 1px solid var(--border);
  border-radius: var(--rad); padding: 1.25rem 1.5rem;
  margin-bottom: 1rem;
}
.card-title {
  font-family: 'Syne', sans-serif; font-size: 0.85rem; font-weight: 700;
  color: var(--br-dk); margin-bottom: 0.7rem;
  padding-bottom: 0.55rem; border-bottom: 1px solid var(--border);
}
.card p { font-size: 0.855rem; line-height: 1.8; color: var(--txt); margin-bottom: 0.45rem; }
.card ul { padding-left: 1.1rem; font-size: 0.855rem; line-height: 1.85; color: var(--txt); }
.card li { margin-bottom: 0.25rem; }
.card strong { color: var(--br-dk); font-weight: 600; }
.card code {
  background: rgba(139,94,60,0.09); color: var(--br-mn);
  padding: 0.1em 0.45em; border-radius: 5px;
  font-size: 0.78em; font-family: 'Fira Code', monospace;
}

/* ══ TABLE ══════════════════════════════════════════════════════ */
.tbl-wrap {
  background: var(--white); border: 1px solid var(--border);
  border-radius: var(--rad); overflow: hidden; margin: 0.6rem 0 1rem;
}
.tbl { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
.tbl thead tr { background: var(--br-pl); }
.tbl th {
  padding: 0.6rem 0.9rem; text-align: left;
  font-size: 0.63rem; font-weight: 700;
  letter-spacing: 0.1em; text-transform: uppercase;
  color: var(--br-mn); border-bottom: 1px solid var(--border);
}
.tbl td { padding: 0.55rem 0.9rem; border-bottom: 1px solid rgba(212,164,122,0.1); color: var(--txt); }
.tbl tr:last-child td { border-bottom: none; }
.tbl tr:hover td { background: var(--br-pl); }
.tbl td.best { font-weight: 700; color: var(--br-dk); }

/* ══ CODE BLOCK ════════════════════════════════════════════════ */
.codeblock {
  background: #2C1A0E; border-radius: var(--rad-sm);
  padding: 1rem 1.25rem;
  font-family: 'Fira Code', 'Courier New', monospace;
  font-size: 0.75rem; line-height: 1.75; color: var(--br-cr);
  overflow-x: auto; margin: 0.6rem 0;
  white-space: pre; border: 1px solid rgba(255,255,255,0.05);
}

/* ══ PREDICTION ════════════════════════════════════════════════ */
.pred-box {
  border-radius: var(--rad); padding: 1.75rem 1.5rem;
  text-align: center; margin-top: 0.75rem;
  border: 1px solid var(--border);
}
.pred-yes { background: rgba(212,164,122,0.15); border-color: rgba(212,164,122,0.5); }
.pred-no  { background: var(--white); }
.pred-lbl {
  font-family: 'Syne', sans-serif; font-size: 1.35rem;
  font-weight: 800; color: var(--br-dk); margin-bottom: 0.7rem;
}
.pred-bar-bg {
  height: 5px; background: var(--br-lt); border-radius: 99px;
  overflow: hidden; margin: 0.6rem auto; width: 70%;
}
.pred-bar-fill {
  height: 100%; border-radius: 99px;
  background: linear-gradient(90deg, var(--br-cr), var(--br-mn));
}
.pred-prob { font-size: 0.8rem; color: var(--muted); font-weight: 500; }

/* ══ EMPTY STATE ═══════════════════════════════════════════════ */
.empty {
  display: flex; flex-direction: column; align-items: center;
  justify-content: center; padding: 2.5rem;
  text-align: center; color: var(--muted); font-size: 0.8rem;
  font-weight: 500; letter-spacing: 0.07em; text-transform: uppercase;
  background: var(--white); border: 1px dashed var(--br-lt);
  border-radius: var(--rad);
}

/* ══ TABS ═══════════════════════════════════════════════════════ */
.stTabs [data-baseweb="tab-list"] {
  gap: 2px; background: var(--white); border-radius: var(--rad);
  padding: 4px; border: 1px solid var(--border);
  margin-bottom: 1.5rem; flex-wrap: wrap;
  overflow-x: auto; scrollbar-width: none;
}
.stTabs [data-baseweb="tab-list"]::-webkit-scrollbar { display: none; }
.stTabs [data-baseweb="tab"] {
  background: transparent; border: none; border-radius: var(--rad-sm);
  font-size: 0.72rem; font-weight: 600; letter-spacing: 0.07em;
  text-transform: uppercase; color: var(--muted);
  padding: 6px 13px; transition: all 0.18s; white-space: nowrap; flex-shrink: 0;
}
.stTabs [data-baseweb="tab"]:hover { color: var(--br-dk); background: var(--br-pl); }
.stTabs [aria-selected="true"] {
  background: var(--br-dk) !important; color: var(--white) !important;
  font-weight: 700 !important;
}
.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] { display: none !important; }
.stTabs [data-baseweb="tab-panel"] { padding: 0 !important; }

/* ══ BUTTONS ════════════════════════════════════════════════════ */
.stButton > button {
  background: var(--br-dk) !important; color: var(--white) !important;
  border: none !important; border-radius: var(--rad-sm) !important;
  font-family: 'DM Sans', sans-serif !important;
  font-size: 0.78rem !important; font-weight: 600 !important;
  letter-spacing: 0.07em !important; text-transform: uppercase !important;
  padding: 0.6rem 1.5rem !important;
  transition: opacity 0.18s, transform 0.18s !important;
}
.stButton > button:hover { opacity: 0.85 !important; transform: translateY(-1px) !important; }

/* ══ FORM ELEMENTS ══════════════════════════════════════════════ */
div[data-testid="stNumberInput"] label,
div[data-testid="stSlider"] label,
div[data-testid="stSelectbox"] label,
div[data-testid="stFileUploader"] label {
  font-size: 0.72rem !important; font-weight: 600 !important;
  color: var(--muted) !important; letter-spacing: 0.07em !important;
  text-transform: uppercase !important;
}
div[data-testid="stNumberInput"] input {
  border: 1px solid var(--border) !important; border-radius: var(--rad-sm) !important;
  background: var(--white) !important; font-size: 0.875rem !important;
}
div[data-testid="stSelectbox"] > div > div {
  border: 1px solid var(--border) !important; border-radius: var(--rad-sm) !important;
}

/* ══ IMG CAPTION ════════════════════════════════════════════════ */
.img-cap { font-size: 0.71rem; color: var(--muted); text-align: center;
           font-style: italic; margin-top: 0.3rem; margin-bottom: 1rem; }

/* ══ FILE UPLOADER ══════════════════════════════════════════════ */
div[data-testid="stFileUploader"] {
  background: var(--white); border: 2px dashed var(--br-lt);
  border-radius: var(--rad); padding: 0.75rem; transition: border-color 0.2s;
}
div[data-testid="stFileUploader"]:hover { border-color: var(--br-md); }

/* ══ SIDEBAR BRAND ══════════════════════════════════════════════ */
.sb-wrap { padding: 1.25rem 1rem; }
.sb-title { font-family: 'Syne', sans-serif; font-size: 0.95rem;
            font-weight: 800; color: rgba(248,239,228,0.95); line-height: 1.3; }
.sb-sub { font-size: 0.63rem; letter-spacing: 0.12em; text-transform: uppercase;
          color: rgba(212,164,122,0.65); margin-top: 0.25rem; }
.sb-divider { border: none; border-top: 1px solid rgba(255,255,255,0.07); margin: 0.9rem 0; }
.sb-nav-lbl { font-size: 0.58rem; font-weight: 700; letter-spacing: 0.18em;
              text-transform: uppercase; color: rgba(212,164,122,0.45);
              padding: 0 12px; margin-bottom: 0.4rem; }
.sb-footer { font-size: 0.63rem; color: rgba(212,164,122,0.4);
             line-height: 1.7; padding: 0.9rem 1rem 1.25rem; }

/* ══ MISC ═══════════════════════════════════════════════════════ */
#MainMenu, footer, header { visibility: hidden; }
.stAlert { border-radius: var(--rad-sm) !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════

def show_img(filename, caption=""):
    path = os.path.join(IMG, filename)
    if os.path.exists(path):
        st.image(PILImage.open(path), use_container_width=True)
        if caption:
            st.markdown(f'<div class="img-cap">{caption}</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            f'<div class="card"><p>Image non trouvee : <code>{filename}</code></p></div>',
            unsafe_allow_html=True
        )

def card(title, *body):
    content = "".join(body)
    st.markdown(f'<div class="card"><div class="card-title">{title}</div>{content}</div>',
                unsafe_allow_html=True)

def mrow(items):
    pills = ""
    for lbl, val, cls in items:
        sm = "sm" if len(str(val)) > 7 else ""
        pills += (f'<div class="mpill {cls}"><div class="m-lbl">{lbl}</div>'
                  f'<div class="m-val {sm}">{val}</div></div>')
    st.markdown(f'<div class="mrow">{pills}</div>', unsafe_allow_html=True)

def htable(cols, rows, best=None):
    ths = "".join(f"<th>{c}</th>" for c in cols)
    trs = ""
    for row in rows:
        cells = "".join(
            f'<td class="{"best" if c==best else ""}">{v}</td>'
            for c, v in zip(cols, row)
        )
        trs += f"<tr>{cells}</tr>"
    st.markdown(
        f'<div class="tbl-wrap"><table class="tbl">'
        f'<thead><tr>{ths}</tr></thead><tbody>{trs}</tbody>'
        f'</table></div>', unsafe_allow_html=True
    )

def codeblock(code):
    st.markdown(f'<div class="codeblock">{code}</div>', unsafe_allow_html=True)

def div():
    st.markdown('<hr class="div">', unsafe_allow_html=True)

def sec(text):
    st.markdown(f'<div class="sec-label">{text}</div>', unsafe_allow_html=True)

def title(main, sub=""):
    st.markdown(f'<div class="pg-title">{main}</div>', unsafe_allow_html=True)
    if sub:
        st.markdown(f'<div class="pg-sub">{sub}</div>', unsafe_allow_html=True)

def pred_box(label, prob, positive=True):
    pct = int(prob * 100)
    cls = "pred-yes" if positive else "pred-no"
    st.markdown(
        f'<div class="pred-box {cls}">'
        f'<div class="pred-lbl">{label}</div>'
        f'<div class="pred-bar-bg"><div class="pred-bar-fill" style="width:{pct}%"></div></div>'
        f'<div class="pred-prob">Probabilite estimee : {prob:.1%}</div>'
        f'</div>', unsafe_allow_html=True
    )

def empty(text="Lancez une prediction pour voir le resultat"):
    st.markdown(
        f'<div class="empty"><span style="font-size:1.5rem;opacity:.3;margin-bottom:.5rem">◇</span>'
        f'{text}</div>', unsafe_allow_html=True
    )


# ══════════════════════════════════════════════════════════════════════
# MODEL LOADERS — robustes, sans crash
# ══════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_classic():
    try:
        m = joblib.load(os.path.join(MOD, "all_models.pkl"))
        f = joblib.load(os.path.join(MOD, "feature_names.pkl"))
        return m, f
    except Exception:
        return None, None

@st.cache_resource
def load_nn():
    try:
        import pickle
        import tensorflow as tf
        scaler_path = os.path.join(MOD, "telemarketing_scaler.pkl")
        feats_path  = os.path.join(MOD, "telemarketing_features.pkl")
        model_path  = os.path.join(MOD, "telemarketing.keras")
        if not all(os.path.exists(p) for p in [scaler_path, feats_path, model_path]):
            return None, None, None
        with open(scaler_path, "rb") as fp:
            scaler = pickle.load(fp)
        with open(feats_path, "rb") as fp:
            feats = pickle.load(fp)
        model = tf.keras.models.load_model(model_path)
        return model, scaler, feats
    except Exception:
        return None, None, None

@st.cache_resource
def load_cnn():
    try:
        import tensorflow as tf
        for fname in ["resnet_fashion.keras", "bank-tel.keras"]:
            p = os.path.join(MOD, fname)
            if os.path.exists(p):
                return tf.keras.models.load_model(p)
        return None
    except Exception:
        return None

def tf_available():
    try:
        import tensorflow as tf
        return True
    except Exception:
        return False


# ══════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════

partie = st.radio(
    "",
    [
        "Partie 1 — Modeles Classiques",
        "Partie 2 — Reseau de Neurones",
        "Partie 3 — Deep Learning CNN",
    ],
    horizontal=True,
    label_visibility="collapsed",
)

# ══════════════════════════════════════════════════════════════════════
# PARTIE 1
# ══════════════════════════════════════════════════════════════════════
if partie == "Partie 1 — Modeles Classiques":

    title("Modeles Classiques de Machine Learning",
          "Bank Telemarketing Dataset — UCI Repository — Classification binaire supervisee")

    tabs = st.tabs([
        "Donnees", "Pretraitement", "Comparaison",
        "Arbre de decision", "Bagging & RF", "Boosting",
        "Variables & ROC", "Prediction"
    ])

    # ── TAB 0 : Données ──────────────────────────────────────────────
    with tabs[0]:
        mrow([
            ("Instances","41 188","best"),
            ("Variables","20 → 47",""),
            ("Numeriques","10",""),
            ("Categorielles","10",""),
            ("Classes","2","hi"),
            ("Desequilibre","88 / 12 %","hi"),
        ])
        c1, c2 = st.columns([1.1, 1])
        with c1:
            show_img("distribution_cible.png", "Distribution de la variable cible (y)")
        with c2:
            card("Contexte et problematique",
                "<p>Le <strong>Bank Marketing Dataset</strong> (UCI) recense les resultats "
                "d'une campagne de telemarketing d'une banque portugaise (2008-2010). "
                "Objectif : predire si un client souscrit a un <strong>depot a terme</strong>.</p>"
                "<ul>"
                "<li><strong>no :</strong> 36 548 clients (88.73 %)</li>"
                "<li><strong>yes :</strong> 4 640 clients (11.27 %)</li>"
                "</ul>"
                "<p>Fort desequilibre — accuracy trompeuse. Metriques privilegiees : "
                "<strong>F1-score</strong> et <strong>AUC-ROC</strong>.</p>"
            )
        div()
        sec("Variables numeriques")
        show_img("distribution_numeriques.png", "Histogrammes des 10 variables numeriques")
        card("Observations cles",
            "<ul>"
            "<li><strong>age</strong> — distribution quasi-normale, 25-60 ans.</li>"
            "<li><strong>duration</strong> — asymetrique. Non disponible avant l'appel — exclure du deploiement.</li>"
            "<li><strong>campaign</strong> — majorite contactee 1 a 3 fois.</li>"
            "<li><strong>pdays</strong> — 86 % a la valeur 999 (jamais contactes avant).</li>"
            "<li><strong>Variables macro</strong> (emp.var.rate, euribor3m, nr.employed) "
            "— distributions discretes par periodes economiques.</li>"
            "</ul>"
        )
        div()
        sec("Matrice de correlation")
        c3, c4 = st.columns(2)
        with c3:
            show_img("matrice_correlation.png", "Matrice de correlation")
        with c4:
            card("Correlations remarquables",
                "<p><strong>Correlations fortes (r2 &gt; 0.5) :</strong></p>"
                "<ul>"
                "<li>emp.var.rate vs euribor3m : r2 = 0.945</li>"
                "<li>euribor3m vs nr.employed : r2 = 0.893</li>"
                "<li>emp.var.rate vs nr.employed : r2 = 0.823</li>"
                "</ul>"
                "<p>Ces trois variables mesurent le meme phenomene. "
                "Multicolinearite severe si conservees ensemble.</p>"
                "<p><strong>Moderees :</strong> pdays vs previous (r2=0.345), "
                "emp.var.rate vs cons.price.idx (r2=0.601).</p>"
            )
        div()
        sec("Nuages de points — croisements deux a deux")
        groupe = st.selectbox("Groupe de variables", [
            "Groupe 1 — age vs autres",
            "Groupe 2 — duration vs autres",
            "Groupe 3 — campaign & pdays",
            "Groupe 4 — previous & macroeconomiques",
            "Groupe 5 — macroeconomiques entre elles",
        ])
        mapping_sc = {
            "Groupe 1 — age vs autres":          ("scatter_groupe_1.png","age vs toutes les autres"),
            "Groupe 2 — duration vs autres":     ("scatter_groupe_2.png","duration vs toutes les autres"),
            "Groupe 3 — campaign & pdays":       ("scatter_groupe_3.png","campaign et pdays vs autres"),
            "Groupe 4 — previous & macroeconomiques":("scatter_groupe_4.png","previous vs macroeconomiques"),
            "Groupe 5 — macroeconomiques entre elles":("scatter_groupe_5.png","Correlations macroeconomiques"),
        }
        show_img(*mapping_sc[groupe])

    # ── TAB 1 : Pretraitement ─────────────────────────────────────────
    with tabs[1]:
        card("Etape 1 — Gestion des valeurs manquantes (unknown)",
            "<p>Six variables contiennent des valeurs <code>unknown</code> remplacees par le "
            "<strong>mode</strong> de chaque colonne, preservant les 41 188 instances.</p>"
        )
        htable(
            ["Variable","Valeur de remplacement","Occurrences"],
            [["job","admin.","330"],["marital","married","80"],
             ["education","university.degree","1 731"],["default","no","8 597"],
             ["housing","yes","990"],["loan","no","990"]]
        )
        div()
        c1, c2 = st.columns(2)
        with c1:
            card("Etape 2 — Encodage",
                "<ul>"
                "<li>Cible : <code>no</code> → 0, <code>yes</code> → 1</li>"
                "<li><code>pd.get_dummies(drop_first=True)</code> sur 10 variables categ.</li>"
                "<li>37 nouvelles colonnes binaires creees</li>"
                "<li>Dimensionnalite : 21 → <strong>48 colonnes</strong></li>"
                "</ul>"
            )
        with c2:
            card("Etape 3 — Separation train / test",
                "<ul>"
                "<li>Entrainement : 32 950 clients (80 %)</li>"
                "<li>Test : 8 238 clients (20 %)</li>"
                "<li><code>stratify=y</code> — proportions 88.7/11.3 % preservees</li>"
                "<li><code>random_state=42</code> — reproductibilite</li>"
                "</ul>"
            )

    # ── TAB 2 : Comparaison ───────────────────────────────────────────
    with tabs[2]:
        card("Protocole",
            "<p>9 modeles evalues par <strong>validation croisee stratifiee a 5 plis</strong>. "
            "Metriques principales : <strong>F1-score et AUC-ROC</strong>.</p>"
        )
        show_img("comparaison_modeles.png", "Comparaison des 9 classifieurs")
        div()
        htable(
            ["Modele","Accuracy","F1","Precision","Recall","AUC-ROC"],
            [
                ["Gradient Boosting","0.9160","0.5843","0.6605","0.5242","0.9448"],
                ["Bagging","0.9103","0.5719","0.6192","0.5315","0.9389"],
                ["Random Forest","0.9100","0.5275","0.6456","0.4461","0.9390"],
                ["Decision Tree","0.8880","0.5104","0.5029","0.5186","0.7267"],
                ["Logistic Regression","0.9093","0.5078","0.6531","0.4154","0.9329"],
                ["AdaBoost","0.9039","0.4700","0.6215","0.3790","0.9316"],
                ["Naive Bayes","0.8726","0.4588","0.4401","0.4793","0.8399"],
                ["SVM","0.9054","0.4587","0.6464","0.3559","0.9258"],
                ["KNN","0.8952","0.4093","0.5615","0.3222","0.8112"],
            ],
            best="AUC-ROC"
        )
        card("Conclusions",
            "<ul>"
            "<li><strong>Gradient Boosting</strong> domine sur F1 (0.584) et AUC-ROC (0.945).</li>"
            "<li><strong>Bagging</strong> : meilleur recall (0.531) — detecte le plus de souscripteurs.</li>"
            "<li>Les <strong>methodes d'ensemble</strong> dominent toutes les methodes simples.</li>"
            "<li><strong>KNN</strong> : fleau de la dimensionnalite sur 47 variables.</li>"
            "</ul>"
        )

    # ── TAB 3 : Arbre de decision ─────────────────────────────────────
    with tabs[3]:
        mrow([
            ("Profondeur opt. F1","4","hi"),
            ("Profondeur opt. AUC","6","hi"),
            ("F1 apres GridSearch","0.611","best"),
            ("AUC apres GridSearch","0.909","best"),
        ])
        show_img("arbre_decision_base.png","Arbre de decision optimise (max_depth=3, criterion=entropy)")
        div()
        show_img("variation_profondeur.png","Impact de max_depth sur F1 et AUC-ROC")
        c1, c2 = st.columns(2)
        with c1:
            card("Analyse",
                "<ul>"
                "<li><strong>depth=1</strong> : sous-apprentissage total — F1 Test=0.000.</li>"
                "<li><strong>depth 2-4</strong> : zone optimale, F1 Test monte a 0.607.</li>"
                "<li><strong>depth &gt; 4</strong> : sur-apprentissage — F1 Train → 1.000, "
                "F1 Test decroit a 0.533 (depth=None).</li>"
                "</ul>"
            )
        with c2:
            card("GridSearchCV — 108 combinaisons x 5 plis = 540 entrainements","")
            htable(
                ["Parametre","Valeurs testees","Optimal"],
                [["max_depth","3,4,5,6,7,8","3"],
                 ["criterion","gini, entropy","entropy"],
                 ["min_samples_split","2,5,10","2"],
                 ["min_samples_leaf","1,2,4","1"]]
            )

    # ── TAB 4 : Bagging & RF ──────────────────────────────────────────
    with tabs[4]:
        sec("Bagging — impact du nombre d'arbres B")
        show_img("bagging_variation_B.png","F1 et AUC en fonction de B")
        htable(
            ["B","F1 Train","F1 Test","AUC Test"],
            [[1,0.8146,0.5328,0.7426],[5,0.9463,0.5797,0.9006],
             [10,0.9700,0.5657,0.9239],[20,0.9921,0.5882,0.9369],
             [50,0.9991,0.5961,0.9428],[100,0.9999,0.6024,0.9451],
             [200,1.0000,0.6032,0.9479]]
        )
        card("Conclusion Bagging",
            "<ul>"
            "<li><strong>B=100</strong> — plateau atteint, gain negligeable au-dela.</li>"
            "<li>Complexite lineaire avec B.</li>"
            "<li>Bagging = <code>RandomForestClassifier(max_features=1.0)</code>.</li>"
            "</ul>"
        )
        div()
        sec("Random Forest — erreur OOB et optimisation de p")
        mrow([
            ("Score OOB","0.9104",""),("Erreur OOB","0.0896",""),
            ("Accuracy Test","0.9192",""),("Ecart OOB/Test","0.0088","hi"),
            ("p optimal","15","best"),
        ])
        show_img("optimisation_p_rf.png","F1 et AUC en fonction du parametre p")
        show_img("comparaison_modeles_rf.png","Comparaison : Arbre / RF de base / RF optimise")
        card("Interpretation",
            "<p>Erreur OOB (0.0896) tres proche de l'erreur test (0.0808) — ecart de "
            "<strong>0.0088</strong>. L'OOB est un proxy fiable de la generalisation.</p>"
            "<p><strong>p=15</strong> superieur a la recommandation par defaut (sqrt(47)≈7) — "
            "les fortes correlations macro necessitent plus de diversite a chaque coupure.</p>"
        )

    # ── TAB 5 : Boosting ──────────────────────────────────────────────
    with tabs[5]:
        htable(
            ["Parametre","Role","Valeur retenue"],
            [
                ["n_estimators (B)","Nombre d'arbres — trop grand = surapprentissage","263"],
                ["learning_rate","Taux de correction — λ petit + B grand = robustesse","0.05"],
                ["max_depth","Profondeur des arbres de base","3"],
                ["validation_fraction","Part du train pour early stopping","0.10"],
                ["n_iter_no_change","Patience avant arret automatique","20"],
            ]
        )
        card("Equivalence AdaBoost",
            "<p><code>loss='exponential'</code>, <code>max_depth=1</code>, "
            "<code>learning_rate=1.0</code> → comportement equivalent a AdaBoost.</p>"
        )
        div()
        show_img("boosting_early_stopping.png","Evolution de l'erreur — Early Stopping a B=263")
        div()
        sec("Comparaison finale des trois classifieurs optimises")
        mrow([
            ("Arbre DT — AUC","0.909",""),
            ("Random Forest — AUC","0.949","hi"),
            ("Gradient Boosting — AUC","0.955","best"),
        ])
        htable(
            ["Classifieur","Accuracy","F1-score","AUC-ROC"],
            [["Arbre de decision","0.9130","0.6105","0.9086"],
             ["Random Forest","0.9190","0.6074","0.9485"],
             ["Gradient Boosting","0.9247","0.6158","0.9550"]],
            best="AUC-ROC"
        )

    # ── TAB 6 : Variables & ROC ───────────────────────────────────────
    with tabs[6]:
        sec("Feature Importances — Top 10")
        show_img("feature_importances.png","Top 10 variables — 3 classifieurs")
        c1, c2, c3 = st.columns(3)
        with c1:
            card("Arbre de decision",
                "<table class='tbl'><thead><tr><th>Variable</th><th>Score</th></tr></thead><tbody>"
                "<tr><td>duration</td><td>0.381</td></tr>"
                "<tr><td>nr.employed</td><td>0.187</td></tr>"
                "<tr><td>euribor3m</td><td>0.080</td></tr>"
                "<tr><td>age</td><td>0.077</td></tr>"
                "<tr><td>pdays</td><td>0.030</td></tr>"
                "</tbody></table>"
            )
        with c2:
            card("Random Forest",
                "<table class='tbl'><thead><tr><th>Variable</th><th>Score</th></tr></thead><tbody>"
                "<tr><td>duration</td><td>0.324</td></tr>"
                "<tr><td>euribor3m</td><td>0.106</td></tr>"
                "<tr><td>age</td><td>0.089</td></tr>"
                "<tr><td>nr.employed</td><td>0.079</td></tr>"
                "<tr><td>campaign</td><td>0.039</td></tr>"
                "</tbody></table>"
            )
        with c3:
            card("Gradient Boosting",
                "<table class='tbl'><thead><tr><th>Variable</th><th>Score</th></tr></thead><tbody>"
                "<tr><td>duration</td><td>0.459</td></tr>"
                "<tr><td>nr.employed</td><td>0.261</td></tr>"
                "<tr><td>euribor3m</td><td>0.083</td></tr>"
                "<tr><td>pdays</td><td>0.036</td></tr>"
                "<tr><td>age</td><td>0.029</td></tr>"
                "</tbody></table>"
            )
        card("Conclusion",
            "<ul>"
            "<li><strong>duration</strong> domine avec 33-46 % de l'importance totale.</li>"
            "<li><strong>nr.employed et euribor3m</strong> confirment le role du contexte macro.</li>"
            "<li>Noyau robuste commun : duration, nr.employed, euribor3m, age, pdays.</li>"
            "</ul>"
        )
        div()
        sec("Courbes ROC")
        mrow([
            ("Arbre de decision","AUC 0.784",""),
            ("Random Forest","AUC 0.949","hi"),
            ("Gradient Boosting","AUC 0.955","best"),
        ])
        show_img("roc_curves.png","Courbes ROC comparatives — 3 classifieurs")
        card("Interpretation",
            "<ul>"
            "<li><strong>Arbre (0.784)</strong> : courbe en escalier, peu granulaire.</li>"
            "<li><strong>RF (0.949) et GB (0.955)</strong> : montee rapide vers le coin superieur "
            "gauche — excellent pouvoir discriminant.</li>"
            "<li>La courbe ROC permet d'ajuster le seuil selon le contexte metier.</li>"
            "</ul>"
        )

    # ── TAB 7 : Prediction ────────────────────────────────────────────
    with tabs[7]:
        models_data, feature_names = load_classic()
        demo = models_data is None

        if demo:
            st.info("Fichiers all_models.pkl / feature_names.pkl non trouves — mode demonstration actif (resultats aleatoires).")

        c_form, c_res = st.columns([3, 2])

        with c_form:
            c1, c2, c3 = st.columns(3)
            with c1:
                age_v      = st.number_input("Age", 18, 95, 40, key="p1_age")
                duration_v = st.number_input("Duree appel (s)", 0, 5000, 300, key="p1_dur")
                campaign_v = st.number_input("Nb contacts", 1, 50, 2, key="p1_camp")
            with c2:
                pdays_v     = st.number_input("pdays", 0, 999, 999, key="p1_pdays")
                euribor_v   = st.number_input("Euribor 3m", -2.0, 6.0, 1.3, key="p1_eur")
                nr_emp_v    = st.number_input("Nr employed", 4900.0, 5300.0, 5099.1, key="p1_nremp")
            with c3:
                emp_var_v   = st.number_input("Emp var rate", -4.0, 2.0, -1.8, key="p1_emp")
                cons_p_v    = st.number_input("Cons price idx", 92.0, 95.0, 93.2, key="p1_consp")
                cons_c_v    = st.number_input("Cons conf idx", -55.0, -25.0, -41.8, key="p1_consc")

            choix_modele = st.selectbox(
                "Modele", ["Gradient Boosting","Random Forest","Arbre de decision"],
                key="p1_model"
            )
            go = st.button("Predire", use_container_width=True, key="p1_go")

        with c_res:
            if go:
                if demo:
                    import random
                    proba = round(random.uniform(0.05, 0.95), 3)
                    pred  = 1 if proba > 0.5 else 0
                else:
                    try:
                        # Creer la ligne avec toutes les features a zero
                        row = pd.DataFrame(0.0, index=[0], columns=feature_names)
                        mapping = {
                            "age": age_v, "duration": duration_v, "campaign": campaign_v,
                            "pdays": pdays_v, "euribor3m": euribor_v,
                            "nr.employed": nr_emp_v, "emp.var.rate": emp_var_v,
                            "cons.price.idx": cons_p_v, "cons.conf.idx": cons_c_v,
                        }
                        for k, v in mapping.items():
                            if k in row.columns:
                                row[k] = v

                        # Mapping nom -> cle dans all_models.pkl
                        key_map = {
                            "Gradient Boosting": "gb_best",
                            "Random Forest":     "rf_opt",
                            "Arbre de decision": "best_dt",
                        }
                        clf   = models_data[key_map[choix_modele]]
                        pred  = int(clf.predict(row)[0])
                        proba = float(clf.predict_proba(row)[0][1])
                    except Exception as e:
                        st.error(f"Erreur de prediction : {e}")
                        pred, proba = 0, 0.0

                lbl = "Souscription probable" if pred == 1 else "Souscription improbable"
                pred_box(lbl, proba, positive=(pred == 1))
            else:
                empty("Remplissez le formulaire et cliquez sur Predire")


# ══════════════════════════════════════════════════════════════════════
# PARTIE 2 — RESEAU DE NEURONES
# ══════════════════════════════════════════════════════════════════════
elif partie == "Partie 2 — Reseau de Neurones":

    title("Reseau de Neurones Artificiel",
          "TensorFlow / Keras — Bank Telemarketing — Classification binaire")

    tabs = st.tabs(["Architecture","Entrainement","Performances","Prediction"])

    with tabs[0]:
        card("Contexte et motivation",
            "<p>Un ANN apprend des representations non lineaires automatiquement. "
            "Chaque neurone calcule : <code>sortie = activation(w1.x1 + ... + wn.xn + biais)</code>. "
            "Sans activation, empiler des couches = simple regression lineaire.</p>"
            "<p><strong>Normalisation StandardScaler obligatoire</strong> : sans elle, "
            "loss initiale a 3.34 avec sauts erratiques. Apres : 0.27, descente reguliere.</p>"
        )
        div()
        c1, c2 = st.columns(2)
        with c1:
            card("Architecture simple (2 couches cachees)","")
            codeblock(
                "model = Sequential([\n"
                "    Dense(64, activation='relu', input_shape=(47,)),\n"
                "    Dense(32, activation='relu'),\n"
                "    Dense(1,  activation='sigmoid')\n"
                "])"
            )
            htable(["Couche","Neurones","Parametres"],
                   [["Dense relu","64","3 072"],["Dense relu","32","2 080"],
                    ["Dense sigmoid","1","33"],["Total","—","5 185"]])
        with c2:
            card("Architecture finale (3 couches + Dropout)","")
            codeblock(
                "model = Sequential([\n"
                "    Dense(128, activation='relu', input_shape=(47,)),\n"
                "    Dropout(0.3),\n"
                "    Dense(64,  activation='relu'),\n"
                "    Dropout(0.2),\n"
                "    Dense(32,  activation='relu'),\n"
                "    Dense(1,   activation='sigmoid')\n"
                "])\n"
                "model.compile(optimizer='adam',\n"
                "              loss='binary_crossentropy',\n"
                "              metrics=['accuracy'])"
            )
            htable(["Couche","Neurones","Parametres"],
                   [["Dense relu","128","6 144"],["Dropout 0.3","—","0"],
                    ["Dense relu","64","8 256"],["Dropout 0.2","—","0"],
                    ["Dense relu","32","2 080"],["Dense sigmoid","1","33"],
                    ["Total","—","16 513"]])
        div()
        card("Justification des choix",
            "<ul>"
            "<li><strong>ReLU</strong> — f(x)=max(0,x), evite le vanishing gradient.</li>"
            "<li><strong>Sigmoid</strong> en sortie — probabilite 0-1. Softmax est pour le multi-classes.</li>"
            "<li><strong>Dropout</strong> — regularisation par extinction aleatoire de neurones.</li>"
            "<li><strong>Adam</strong> — optimiseur adaptatif. <strong>Early stopping</strong> patience=15.</li>"
            "</ul>"
        )

    with tabs[1]:
        mrow([
            ("Epochs realises","28 / 100",""),("Meilleur epoch","13","hi"),
            ("Meilleure val_loss","0.1850","best"),("Patience","15",""),
        ])
        show_img("courbes_apprentissage_nn.png","Loss et Accuracy par epoch")
        c1, c2 = st.columns(2)
        with c1:
            card("Graphe Loss",
                "<ul><li>Loss Train : 0.27 → 0.16</li>"
                "<li>Loss Validation : 0.198 → 0.185 puis stable</li>"
                "<li>Arret epoch 28, restauration epoch 13</li>"
                "<li>Pas de surapprentissage violent</li></ul>"
            )
        with c2:
            card("Graphe Accuracy",
                "<ul><li>Accuracy Train : 89 % → 92.5 %</li>"
                "<li>Accuracy Validation : stable ~91 % des l'epoch 5</li>"
                "<li>Ecart faible — bon equilibre</li></ul>"
            )

    with tabs[2]:
        mrow([
            ("Accuracy","0.9187",""),("F1-score","0.6158","hi"),
            ("Precision","0.6581",""),("Recall","0.5787",""),("AUC-ROC","0.9486","best"),
        ])
        htable(
            ["Metrique","47 variables","10 variables","Difference"],
            [["Accuracy","0.9150","0.9187","+0.0037"],
             ["F1-score","0.6102","0.6158","+0.0056"],
             ["Precision","0.6313","0.6581","+0.0268"],
             ["Recall","0.5905","0.5787","-0.0118"],
             ["AUC-ROC","0.9464","0.9486","+0.0022"]]
        )
        card("Conclusion",
            "<p>Le modele a <strong>10 variables surpasse le modele complet</strong> — "
            "les 37 autres apportaient du bruit. Fleau de la dimensionnalite.</p>"
            "<p>Top 5 : <strong>duration, euribor3m, nr.employed, age, campaign</strong>.</p>"
        )

    with tabs[3]:
        if not tf_available():
            st.warning(
                "TensorFlow n'est pas disponible sur ce systeme. "
                "Cause probable : DLL manquante (msvcp140_1.dll). "
                "Solution : installez Microsoft C++ Redistributable depuis "
                "https://aka.ms/vs/17/release/vc_redist.x64.exe puis redemarrez."
            )
        else:
            model_nn, scaler_nn, feats_nn = load_nn()
            demo_nn = model_nn is None
            if demo_nn:
                st.info("Fichiers ANN non trouves dans modeles/ — mode demonstration actif.")

        c_form, c_res = st.columns([3, 2])
        with c_form:
            c1, c2 = st.columns(2)
            with c1:
                p_dur   = st.number_input("Duree appel (s)", 0, 5000, 300, key="p2_dur")
                p_eur   = st.number_input("Euribor 3m", -2.0, 6.0, 1.3, key="p2_eur")
                p_nremp = st.number_input("Nr employed", 4900.0, 5300.0, 5099.1, key="p2_nremp")
                p_age   = st.number_input("Age", 18, 95, 40, key="p2_age")
                p_camp  = st.number_input("Nb contacts", 1, 50, 2, key="p2_camp")
            with c2:
                p_pdays = st.number_input("pdays", 0, 999, 999, key="p2_pdays")
                p_consc = st.number_input("Cons conf idx", -55.0, -25.0, -41.8, key="p2_consc")
                p_consp = st.number_input("Cons price idx", 92.0, 95.0, 93.2, key="p2_consp")
                p_pout  = st.selectbox("Campagne precedente reussie", [0, 1], key="p2_pout")
                p_hous  = st.selectbox("Pret immobilier", [0, 1], key="p2_hous")
            go_nn = st.button("Predire avec le reseau de neurones", use_container_width=True, key="p2_go")

        with c_res:
            if go_nn:
                if not tf_available() or demo_nn:
                    import random
                    proba_nn = round(random.uniform(0.05, 0.95), 3)
                    pred_nn  = 1 if proba_nn > 0.5 else 0
                else:
                    try:
                        vals   = [p_dur, p_eur, p_nremp, p_age, p_camp,
                                  p_pdays, p_consc, p_consp, p_pout, p_hous]
                        row    = np.array(vals).reshape(1, -1)
                        row_s  = scaler_nn.transform(row)
                        proba_nn = float(model_nn.predict(row_s, verbose=0)[0][0])
                        pred_nn  = 1 if proba_nn > 0.5 else 0
                    except Exception as e:
                        st.error(f"Erreur : {e}")
                        proba_nn, pred_nn = 0.0, 0
                lbl = "Souscription probable" if pred_nn == 1 else "Souscription improbable"
                pred_box(lbl, proba_nn, positive=(pred_nn == 1))
            else:
                empty("Remplissez le formulaire et lancez la prediction")


# ══════════════════════════════════════════════════════════════════════
# PARTIE 3 — DEEP LEARNING CNN
# ══════════════════════════════════════════════════════════════════════
else:
    title("Apprentissage Profond — CNN",
          "Fashion MNIST — Classification d'images de vetements — 10 categories")

    tabs = st.tabs(["Dataset","Architectures","Resultats","Prediction"])

    with tabs[0]:
        mrow([
            ("Images totales","70 000","best"),("Entrainement","60 000","hi"),
            ("Test","10 000","hi"),("Classes","10",""),
            ("Resolution","28 x 28",""),("Equilibre","6 000/classe",""),
        ])
        show_img("fashion_mnist_exemples.png","Exemples — une image par classe")
        c1, c2 = st.columns(2)
        with c1:
            card("Specificites du dataset",
                "<ul>"
                "<li><strong>Parfaitement equilibre</strong> : 6 000 images/classe.</li>"
                "<li><strong>Difficulte reelle</strong> : T-shirt vs Chemise, Sandale vs Sneaker.</li>"
                "<li>Cree par <strong>Zalando Research</strong>.</li>"
                "</ul>"
            )
        with c2:
            card("Pretraitement",
                "<ul>"
                "<li><strong>Normalisation</strong> : pixels / 255 → [0,1].</li>"
                "<li><strong>Encodage</strong> : <code>to_categorical()</code> → vecteurs dim=10.</li>"
                "<li><strong>Reshape</strong> : (28,28,1) — canal unique niveaux de gris.</li>"
                "<li><strong>Metriques</strong> : Categorical Crossentropy + Accuracy.</li>"
                "</ul>"
            )
        card("Pourquoi des CNN et non des reseaux Dense ?",
            "<p>Un reseau Dense ne comprend pas la <strong>structure spatiale</strong> d'une image. "
            "Un CNN applique des filtres qui detectent formes, contours et textures.</p>"
            "<ul>"
            "<li><strong>Conv2D</strong> : filtres de detection de patterns visuels.</li>"
            "<li><strong>MaxPooling2D</strong> : reduction spatiale.</li>"
            "<li><strong>Softmax</strong> en sortie : 10 probabilites sumant a 1.</li>"
            "</ul>"
        )

    with tabs[1]:
        archi_sel = st.selectbox("Selectionner une architecture", [
            "CNN de base","LeNet-5 (1998)","Mini-VGGNet (2014)","ResNet simplifie (2015)"
        ])
        div()
        if archi_sel == "CNN de base":
            card("CNN de base — 225 034 parametres","")
            codeblock(
                "Conv2D(32, (3,3), relu) → MaxPooling(2x2)\n"
                "Conv2D(64, (3,3), relu) → MaxPooling(2x2)\n"
                "Flatten → Dense(128, relu) → Dropout(0.3) → Dense(10, softmax)"
            )
            card("Analyse","<p>Bon rapport performance/complexite. Filtres 3x3 et ReLU modernes. "
                "<strong>Accuracy : 90.71 %</strong></p>")
        elif archi_sel == "LeNet-5 (1998)":
            card("LeNet-5 — 61 706 parametres (Yann LeCun, 1998)","")
            codeblock(
                "Conv2D(6,  (5,5), tanh, padding='same') → MaxPooling(2x2)\n"
                "Conv2D(16, (5,5), tanh) → MaxPooling(2x2)\n"
                "Flatten → Dense(120, tanh) → Dense(84, tanh) → Dense(10, softmax)"
            )
            card("Analyse","<p>Architecture historique pour chiffres. Filtres 5x5 et tanh penalisants. "
                "Avantage : tres leger. <strong>Accuracy : 89.17 %</strong></p>")
        elif archi_sel == "Mini-VGGNet (2014)":
            card("Mini-VGGNet — 872 042 parametres","")
            codeblock(
                "[Conv2D(32) + BatchNorm + Conv2D(32) + BN] → MaxPooling → Dropout(0.25)\n"
                "[Conv2D(64) + BatchNorm + Conv2D(64) + BN] → MaxPooling → Dropout(0.25)\n"
                "Flatten → Dense(256) + BN → Dropout(0.5) → Dense(10, softmax)"
            )
            card("Analyse","<p>Doubles couches de convolution + BatchNormalization. "
                "Le plus lourd (872K) mais tres performant. <strong>Accuracy : 93.21 %</strong></p>")
        else:
            card("ResNet simplifie — 515 146 parametres (He et al., 2015)","")
            codeblock(
                "def residual_block(x, filters):\n"
                "    shortcut = x\n"
                "    x = Conv2D(filters, (3,3), padding='same')(x)\n"
                "    x = BatchNorm()(x)\n"
                "    x = Conv2D(filters, (3,3), padding='same')(x)\n"
                "    x = BatchNorm()(x)\n"
                "    x = Add()([x, shortcut])   # skip connection\n"
                "    return Activation('relu')(x)\n\n"
                "Conv2D(32)+BN → ResidualBlock(32) → MaxPooling → Dropout\n"
                "Conv2D(64)+BN → ResidualBlock(64) → MaxPooling → Dropout\n"
                "Dense(128) → Dropout(0.4) → Dense(10, softmax)"
            )
            card("Analyse","<p>Connexions residuelles resolvent le vanishing gradient. "
                "Plus performant que VGGNet avec moins de parametres. "
                "<strong>Accuracy : 93.36 %</strong></p>")

    with tabs[2]:
        mrow([
            ("LeNet-5","89.17 %",""),("CNN de base","90.71 %",""),
            ("VGGNet","93.21 %","hi"),("ResNet","93.36 %","best"),
        ])
        show_img("comparaison_fashion_mnist.png","Accuracy et Loss — 4 architectures")
        div()
        htable(
            ["Modele","Accuracy","Loss","Parametres","Epochs"],
            [["LeNet-5","0.8917","0.2954","61 706","13"],
             ["CNN de base","0.9071","0.2514","225 034","13"],
             ["Mini-VGGNet","0.9321","0.1870","872 042","12"],
             ["ResNet simplifie","0.9336","0.1802","515 146","16"]],
            best="Accuracy"
        )
        card("Analyse comparative",
            "<ul>"
            "<li><strong>LeNet (89.17 %)</strong> — architecture 1998, limitee pour les vetements.</li>"
            "<li><strong>CNN de base (90.71 %)</strong> — bon rapport performance/complexite.</li>"
            "<li><strong>VGGNet (93.21 %)</strong> — saut significatif, mais 872K parametres.</li>"
            "<li><strong>ResNet (93.36 %)</strong> — meilleur modele avec moins de parametres "
            "grace aux connexions residuelles.</li>"
            "</ul>"
        )

    with tabs[3]:
        # Gestion DLL manquante
        if not tf_available():
            st.warning(
                "TensorFlow non disponible — DLL manquante probable (msvcp140_1.dll). "
                "Installez Microsoft C++ Redistributable : https://aka.ms/vs/17/release/vc_redist.x64.exe "
                "puis redemarrez le terminal et relancez streamlit."
            )
            st.stop()

        model_cnn = load_cnn()

        c1, c2 = st.columns([1.1, 1])
        with c1:
            if model_cnn is None:
                st.warning(
                    "Modele CNN non trouve. Verifiez que "
                    "resnet_fashion.keras ou bank-tel.keras "
                    "est dans le dossier modeles/."
                )
            card("Charger une image",
                "<p>Deposez une image PNG ou JPG. Elle sera convertie en "
                "niveaux de gris et redimensionnee a 28x28 automatiquement.</p>"
            )
            uploaded = st.file_uploader(
                "Deposer une image", type=["png","jpg","jpeg"],
                label_visibility="collapsed", key="p3_upload"
            )
            div()
            sec("Ou tester avec une image du dataset")
            rand_btn = st.button("Image aleatoire Fashion MNIST",
                                 use_container_width=True, key="p3_rand")
            if rand_btn:
                try:
                    import tensorflow as tf
                    (_, _), (Xt, yt) = tf.keras.datasets.fashion_mnist.load_data()
                    ri = np.random.randint(0, len(Xt))
                    st.session_state["rand_img"]   = Xt[ri]
                    st.session_state["rand_label"] = int(yt[ri])
                except Exception as e:
                    st.error(f"Impossible de charger Fashion MNIST : {e}")

        with c2:
            if uploaded is not None:
                pil_img = PILImage.open(uploaded).convert("L").resize((28, 28))
                arr = np.array(pil_img, dtype=np.float32) / 255.0
                if arr.mean() > 0.5:
                    arr = 1.0 - arr
                st.image(pil_img, caption="Image chargee (28x28)", width=160)
                if st.button("Predire cette image", key="p3_pred_up", use_container_width=True):
                    if model_cnn is not None:
                        preds = model_cnn.predict(arr.reshape(1, 28, 28, 1), verbose=0)[0]
                        idx = int(np.argmax(preds))
                        pred_box(f"Prediction : {CLASS_NAMES[idx]}", float(preds[idx]), positive=True)
                        htable(
                            ["Classe", "Probabilite"],
                            sorted([[CLASS_NAMES[i], f"{preds[i]:.2%}"] for i in range(10)],
                                   key=lambda x: float(x[1].strip("%")), reverse=True)
                        )
                    else:
                        st.warning("Modele non charge.")

            elif "rand_img" in st.session_state:
                rand_img = st.session_state["rand_img"]
                true_lbl = st.session_state["rand_label"]
                true_name = CLASS_NAMES[true_lbl]
                arr_rand = rand_img.astype(np.float32) / 255.0
                st.image(rand_img, caption=f"Vraie classe : {true_name}", width=160)
                if st.button("Predire cette image", key="p3_pred_rand", use_container_width=True):
                    if model_cnn is not None:
                        preds = model_cnn.predict(arr_rand.reshape(1, 28, 28, 1), verbose=0)[0]
                        idx = int(np.argmax(preds))
                        ok = CLASS_NAMES[idx] == true_name
                        suf = "Correct" if ok else f"Incorrect — vraie classe : {true_name}"
                        pred_box(f"{CLASS_NAMES[idx]} — {suf}", float(preds[idx]), positive=ok)
                    else:
                        st.warning("Modele non charge.")
            else:
                empty("Deposez une image ou generez-en une aleatoire")