"""Demo Streamlit: jalankan dengan `streamlit run app.py`"""
import html
import os
import re
from urllib.parse import urlparse

import joblib
import pandas as pd
import streamlit as st

from features import FEATURE_NAMES, SUSPICIOUS_WORDS, extract_features, normalize

st.set_page_config(page_title="PhishGuard – Pemeriksa URL", page_icon="🛡️", layout="wide")

MODEL_NAMES = ["LogisticRegression", "RandomForest", "XGBoost"]
EXAMPLES = {
    "Situs berita": "https://www.kompas.com/read/2026/11/19/artikel-pendidikan",
    "Login bank palsu": "http://secure-bca-login.verify-akun.xyz/update/confirm.php?id=88213",
    "Alamat IP": "http://192.168.44.7/paypal/webscr/login.php?token=a8s7d6f5",
}
SECOND_LEVEL = {"co.uk", "ac.uk", "co.id", "ac.id", "go.id", "or.id", "web.id", "com.au", "co.jp", "com.br"}

# ---------------------------------------------------------------- gaya
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@400;600&display=swap');

/* ── Variabel Warna ── */
:root {
  --primary: #6C63FF;
  --primary-dark: #4B44CC;
  --secondary: #FF4D8D;
  --accent: #A78BFA;
  --bg-gradient-start: #6C63FF;
  --bg-gradient-end: #A78BFA;
  --card: #FFFFFF;
  --card-glass: rgba(255,255,255,0.92);
  --ink: #1A1A2E;
  --muted: #64748B;
  --line: rgba(108,99,255,0.15);
  --safe: #059669;   --safe-bg: #ECFDF5;   --safe-border: #6EE7B7;
  --warn: #D97706;   --warn-bg: #FFFBEB;   --warn-border: #FCD34D;
  --bad:  #DC2626;   --bad-bg:  #FEF2F2;   --bad-border:  #FCA5A5;
}

/* ── Reset & Base ── */
html, body, [class*="css"], .stApp {
  font-family: 'Montserrat', system-ui, sans-serif !important;
  color: var(--ink);
}
#MainMenu, footer { visibility: hidden; }

/* ── Background Halaman ── */
.stApp {
  background: linear-gradient(135deg, #F0EFFF 0%, #E8E6FF 40%, #F5F0FF 100%) !important;
  min-height: 100vh;
}

/* ── Layout ── */
.block-container {
  max-width: 1200px;
  padding-top: 0 !important;
  padding-bottom: 4rem;
}

/* ── Header / Hero ── */
.hero-wrapper {
  background: linear-gradient(135deg, var(--primary) 0%, #8B7CF8 50%, var(--accent) 100%);
  margin: -1rem -1rem 2rem -1rem;
  padding: 3rem 2.5rem 2.5rem 2.5rem;
  border-radius: 0 0 32px 32px;
  position: relative;
  overflow: hidden;
  box-shadow: 0 20px 60px rgba(108,99,255,0.35);
}
.hero-wrapper::before {
  content: '';
  position: absolute; inset: 0;
  background: radial-gradient(circle at 70% 50%, rgba(255,255,255,0.08) 0%, transparent 60%),
              radial-gradient(circle at 20% 80%, rgba(255,77,141,0.15) 0%, transparent 50%);
  pointer-events: none;
}
.hero-badge {
  display: inline-flex; align-items: center; gap: 8px;
  background: rgba(255,255,255,0.2);
  backdrop-filter: blur(10px);
  border: 1px solid rgba(255,255,255,0.3);
  color: #fff; font-size: .78rem; font-weight: 700;
  letter-spacing: .08em; text-transform: uppercase;
  padding: 6px 14px; border-radius: 999px; margin-bottom: 1.2rem;
}
.hero h1 {
  font-size: 2.9rem; font-weight: 900;
  line-height: 1.08; margin: 0 0 .9rem 0;
  color: #fff; letter-spacing: -0.03em;
}
.hero h1 span { color: #FFD6EC; }
.hero p {
  font-size: 1.05rem; color: rgba(255,255,255,0.82);
  max-width: 580px; line-height: 1.65; margin: 0;
}
.hero-stats {
  display: flex; gap: 2rem; margin-top: 2rem;
  flex-wrap: wrap;
}
.hero-stat {
  background: rgba(255,255,255,0.15);
  backdrop-filter: blur(8px);
  border: 1px solid rgba(255,255,255,0.25);
  border-radius: 14px; padding: .8rem 1.2rem;
  color: #fff; min-width: 110px;
}
.hero-stat .num { font-size: 1.5rem; font-weight: 800; line-height: 1; }
.hero-stat .lbl { font-size: .72rem; opacity: .8; margin-top: 3px; text-transform: uppercase; letter-spacing: .06em; }

/* ── Kartu ── */
.card {
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 20px;
  padding: 1.4rem 1.6rem;
  margin-bottom: 1rem;
  box-shadow: 0 4px 24px rgba(108,99,255,0.08);
  transition: box-shadow .2s;
}
.card:hover { box-shadow: 0 8px 32px rgba(108,99,255,0.14); }
.card h4 {
  font-family: 'Montserrat', sans-serif;
  font-size: 1rem; font-weight: 700;
  margin: 0 0 .9rem 0; color: var(--ink);
  display: flex; align-items: center; gap: .5rem;
}
.card-icon {
  display: inline-flex; align-items: center; justify-content: center;
  width: 28px; height: 28px; border-radius: 8px;
  background: linear-gradient(135deg, var(--primary), var(--accent));
  font-size: .85rem;
}

/* ── Input Area ── */
.input-wrapper {
  background: var(--card);
  border: 2px solid var(--line);
  border-radius: 20px;
  padding: 1.6rem 1.8rem;
  margin-bottom: 1.2rem;
  box-shadow: 0 8px 32px rgba(108,99,255,0.1);
}
.input-wrapper:focus-within {
  border-color: var(--primary);
  box-shadow: 0 8px 32px rgba(108,99,255,0.2);
}
.input-label {
  font-size: .78rem; font-weight: 700; letter-spacing: .07em;
  text-transform: uppercase; color: var(--primary);
  margin-bottom: .5rem; display: block;
}

/* ── Tombol ── */
.stButton > button {
  font-family: 'Montserrat', sans-serif !important;
  font-weight: 700 !important;
  border-radius: 12px !important;
  letter-spacing: .03em;
  transition: all .2s !important;
}
.stButton > button[kind="primary"] {
  background: linear-gradient(135deg, var(--primary) 0%, #8B7CF8 100%) !important;
  border: none !important;
  box-shadow: 0 4px 16px rgba(108,99,255,0.4) !important;
  color: #fff !important;
}
.stButton > button[kind="primary"]:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 8px 24px rgba(108,99,255,0.5) !important;
}
.stButton > button:not([kind="primary"]) {
  background: rgba(108,99,255,0.07) !important;
  border: 1.5px solid rgba(108,99,255,0.25) !important;
  color: var(--primary) !important;
}
.stButton > button:not([kind="primary"]):hover {
  background: rgba(108,99,255,0.14) !important;
  border-color: var(--primary) !important;
}

/* ── Input Teks ── */
.stTextInput input {
  font-family: 'JetBrains Mono', monospace !important;
  font-size: .95rem !important;
  border-radius: 12px !important;
  border: 2px solid rgba(108,99,255,0.2) !important;
  padding: .75rem 1rem !important;
  background: #FAFAFE !important;
  transition: border .2s !important;
}
.stTextInput input:focus {
  border-color: var(--primary) !important;
  box-shadow: 0 0 0 3px rgba(108,99,255,0.12) !important;
}

/* ── Tab ── */
.stTabs [data-baseweb="tab-list"] {
  gap: .4rem;
  background: rgba(108,99,255,0.07) !important;
  padding: 5px 6px; border-radius: 14px;
  border-bottom: none !important;
}
.stTabs [data-baseweb="tab"] {
  border-radius: 10px !important;
  font-weight: 600 !important;
  font-family: 'Montserrat', sans-serif !important;
  font-size: .88rem !important;
  padding: .5rem 1.1rem !important;
  color: var(--muted) !important;
  background: transparent !important;
  border: none !important;
}
.stTabs [aria-selected="true"] {
  background: #fff !important;
  color: var(--primary) !important;
  box-shadow: 0 2px 10px rgba(108,99,255,0.18) !important;
}

/* ── Vonis / Verdict ── */
.verdict {
  border-radius: 20px; padding: 1.6rem 1.8rem;
  margin-bottom: 1rem; border: 1.5px solid;
  box-shadow: 0 4px 20px rgba(0,0,0,0.06);
}
.verdict.bad  { background: var(--bad-bg);  border-color: var(--bad-border);  color: var(--bad); }
.verdict.warn { background: var(--warn-bg); border-color: var(--warn-border); color: var(--warn); }
.verdict.safe { background: var(--safe-bg); border-color: var(--safe-border); color: var(--safe); }
.verdict .big {
  font-size: 2rem; font-weight: 800; line-height: 1.1;
  font-family: 'Montserrat', sans-serif;
}
.verdict .sub { margin-top: .4rem; font-size: .97rem; color: var(--ink); line-height: 1.5; }
.verdict-icon { font-size: 2.4rem; margin-bottom: .6rem; display: block; }

/* ── Meter ── */
.meter {
  position: relative; height: 12px; border-radius: 999px;
  margin: 1.3rem 0 .4rem 0;
  background: linear-gradient(90deg,#059669 0%,#34D399 25%,#FCD34D 50%,#F87171 75%,#DC2626 100%);
  box-shadow: inset 0 1px 3px rgba(0,0,0,0.1);
}
.meter .dot {
  position: absolute; top: -6px; width: 24px; height: 24px;
  border-radius: 50%; background: #fff;
  border: 3px solid var(--primary);
  transform: translateX(-50%);
  box-shadow: 0 2px 8px rgba(108,99,255,0.3);
  transition: left .4s ease;
}
.meter .thr {
  position: absolute; top: -10px; width: 2px; height: 32px;
  background: var(--ink); opacity: .4; border-radius: 2px;
}
.meter-lbl {
  display: flex; justify-content: space-between;
  font-size: .75rem; color: var(--muted); font-weight: 600;
}

/* ── Anatomi URL ── */
.anat {
  font-family: 'JetBrains Mono', monospace;
  font-size: .95rem; line-height: 2.1; word-break: break-all;
  background: linear-gradient(135deg, #0F172A 0%, #1E1B4B 100%);
  color: #C4B5FD; border-radius: 16px; padding: 1.1rem 1.3rem;
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.05);
}
.anat span { padding: 3px 6px; border-radius: 6px; }
.a-scheme { color: #94A3B8; }
.a-sub    { background: rgba(251,191,36,0.2); color: #FCD34D; }
.a-dom    { background: rgba(52,211,153,0.2); color: #6EE7B7; font-weight: 700; }
.a-path   { color: #A5B4FC; }
.a-query  { color: #DDD6FE; }
.a-flag   { background: rgba(248,113,113,0.2); color: #FCA5A5; font-weight: 700; }
.legend {
  display: flex; flex-wrap: wrap; gap: .8rem;
  margin-top: .8rem; font-size: .79rem; color: var(--muted);
}
.legend i {
  display: inline-block; width: 10px; height: 10px;
  border-radius: 3px; margin-right: 5px; vertical-align: middle;
}

/* ── Indikator / Flag ── */
.flag {
  display: flex; gap: .8rem; align-items: flex-start;
  padding: .6rem 0; border-bottom: 1px solid var(--line);
}
.flag:last-child { border-bottom: none; }
.flag b {
  display: inline-block; min-width: 76px;
  font-size: .73rem; padding: 3px 10px;
  border-radius: 999px; text-align: center; font-weight: 700;
}
.flag.r b { background: var(--bad-bg);  color: var(--bad);  border: 1px solid var(--bad-border); }
.flag.g b { background: var(--safe-bg); color: var(--safe); border: 1px solid var(--safe-border); }

/* ── Baris Model ── */
.row {
  display: flex; align-items: center; gap: .8rem;
  margin: .6rem 0; font-size: .92rem;
}
.row .nm { width: 148px; font-weight: 700; color: var(--ink); }
.row .tr {
  flex: 1; height: 10px; background: #EDE9FE;
  border-radius: 999px; overflow: hidden;
}
.row .fl { height: 100%; border-radius: 999px; }
.row .pc {
  width: 54px; text-align: right;
  font-family: 'JetBrains Mono', monospace; font-size: .88rem; font-weight: 600;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #F5F3FF 0%, #EDE9FE 100%) !important;
  border-right: 1px solid rgba(108,99,255,0.15) !important;
}
[data-testid="stSidebar"] .stMarkdown h3 {
  color: var(--primary); font-weight: 800;
}

/* ── Divider ── */
.section-divider {
  display: flex; align-items: center; gap: 1rem;
  margin: 1.5rem 0 1rem 0;
}
.section-divider .line { flex: 1; height: 1px; background: var(--line); }
.section-divider .label {
  font-size: .75rem; font-weight: 700; color: var(--muted);
  letter-spacing: .07em; text-transform: uppercase;
}

/* ── Info/Warning/Success Box ── */
.stAlert {
  border-radius: 14px !important;
  border-left-width: 4px !important;
}

/* ── Responsive ── */
@media (max-width: 700px) {
  .hero h1 { font-size: 2rem; }
  .hero-wrapper { padding: 2rem 1.2rem 2rem 1.2rem; }
  .row .nm { width: 90px; }
  .hero-stats { gap: .8rem; }
}
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------- fungsi
@st.cache_resource
def load_model(path):
    try:
        return joblib.load(path)
    except FileNotFoundError:
        return None


def predict(bundle, urls):
    X = pd.DataFrame([extract_features(u) for u in urls], columns=FEATURE_NAMES)
    return bundle["model"].predict_proba(X)[:, 1], X


def split_url(url):
    """Pecah URL menjadi skema, subdomain, domain, path, query (perkiraan)."""
    u = normalize(url)
    p = urlparse(u)
    host = (p.hostname or "")
    if re.match(r"^\d{1,3}(\.\d{1,3}){3}$", host):
        sub, dom = "", host
    else:
        parts = host.split(".")
        k = 3 if len(parts) >= 3 and ".".join(parts[-2:]) in SECOND_LEVEL else 2
        dom = ".".join(parts[-k:])
        sub = ".".join(parts[:-k])
    return p.scheme, sub, dom, p.path, p.query, p.port


def flag_words(text):
    out = text
    for w in sorted(SUSPICIOUS_WORDS, key=len, reverse=True):
        out = re.sub(f"({re.escape(w)})", "\x00\\1\x01", out, flags=re.I)
    safe = html.escape(out)
    return safe.replace("\x00", '<span class="a-flag">').replace("\x01", "</span>")


def anatomy_html(url):
    scheme, sub, dom, path, query, port = split_url(url)
    h = f'<span class="a-scheme">{html.escape(scheme)}://</span>'
    if sub:
        h += f'<span class="a-sub">{flag_words(sub)}</span>.'
    h += f'<span class="a-dom">{flag_words(dom)}</span>'
    if port:
        h += f'<span class="a-scheme">:{port}</span>'
    if path:
        h += f'<span class="a-path">{flag_words(path)}</span>'
    if query:
        h += f'<span class="a-query">?{flag_words(query)}</span>'
    legend = (
        '<div class="legend"><span><i style="background:#F2C46B"></i>subdomain</span>'
        '<span><i style="background:#6BE0B0"></i>domain utama</span>'
        '<span><i style="background:#9DB7FF"></i>path</span>'
        '<span><i style="background:#C8B2FF"></i>query</span>'
        '<span><i style="background:#FFB4AA"></i>kata yang sering dipakai phishing</span></div>'
    )
    return f'<div class="anat">{h}</div>{legend}'


def indicators(f):
    """Indikator sederhana berbasis aturan (penjelasan tambahan, bukan alasan internal model)."""
    items = []
    add = lambda bad, bad_txt, ok_txt: items.append(("r", "Waspada", bad_txt) if bad else ("g", "Normal", ok_txt))
    add(f["has_ip"], "Memakai alamat IP, bukan nama domain", "Memakai nama domain")
    add(not f["is_https"], "Tidak memakai HTTPS", "Memakai HTTPS")
    add(f["has_at"], "Mengandung simbol @ (bisa menyembunyikan tujuan asli)", "Tidak ada simbol @")
    add(f["suspicious_words"] >= 2, f"Mengandung {f['suspicious_words']} kata sensitif (login, verify, dst.)",
        "Sedikit kata sensitif" if f["suspicious_words"] else "Tidak ada kata sensitif")
    add(f["num_subdomains"] >= 3, f"Subdomain berlapis ({f['num_subdomains']} tingkat)", "Subdomain wajar")
    add(f["host_hyphens"] >= 2, f"Banyak tanda hubung di domain ({f['host_hyphens']})", "Tanda hubung di domain wajar")
    add(f["url_length"] > 75, f"URL panjang ({f['url_length']} karakter)", f"Panjang URL wajar ({f['url_length']} karakter)")
    add(f["has_punycode"] or f["is_shortener"], "Memakai punycode atau pemendek URL", "Tanpa punycode atau pemendek URL")
    return items


def level(p, thr):
    if p >= max(thr, 0.9):
        return "bad", "Sangat mungkin phishing", "Jangan buka tautan ini dan jangan isi data apa pun."
    if p >= thr:
        return "bad", "Terindikasi phishing", "Hati-hati. Verifikasi alamat resmi situsnya lewat sumber lain."
    if p >= thr * 0.6:
        return "warn", "Mencurigakan, tapi di bawah ambang", "Model ragu-ragu. Periksa baik-baik sebelum membuka."
    return "safe", "Terlihat aman", "Tidak ada pola phishing yang kuat pada teks URL."


def submit():
    st.session_state["checked"] = st.session_state.get("url_input", "").strip()


def use_example(u):
    st.session_state["url_input"] = u
    st.session_state["checked"] = u


# ---------------------------------------------------------------- sidebar
bundles = {n: load_model(f"models/{n}.joblib") for n in MODEL_NAMES}
available = [n for n, b in bundles.items() if b is not None]

with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center;padding:1.2rem 0 1rem 0;">
          <div style="display:inline-flex;align-items:center;justify-content:center;
                      width:52px;height:52px;border-radius:16px;
                      background:linear-gradient(135deg,#6C63FF,#A78BFA);
                      font-size:1.6rem;margin-bottom:.7rem;
                      box-shadow:0 6px 20px rgba(108,99,255,0.35);"></div>
          <div style="font-weight:800;font-size:1.05rem;color:#1A1A2E;letter-spacing:-.01em;">PhishGuard</div>
          <div style="font-size:.73rem;color:#64748B;margin-top:2px;letter-spacing:.05em;text-transform:uppercase;">URL Security Analyzer</div>
        </div>
        <hr style="border:none;border-top:1px solid rgba(108,99,255,0.15);margin:.5rem 0 1.2rem 0;">
        """,
        unsafe_allow_html=True,
    )
    st.markdown("**⚙️ Pengaturan Model**")
    if not available:
        st.error("Belum ada model. Jalankan `python train.py --data <dataset.csv>` dulu.")
        st.stop()
    choice = st.selectbox("Model utama", available, index=available.index("XGBoost") if "XGBoost" in available else 0)
    threshold = st.slider("Ambang batas phishing", 0.1, 0.9, 0.5, 0.05,
                          help="URL dianggap phishing jika probabilitas ≥ ambang. Turunkan untuk menangkap lebih banyak phishing (recall naik, false positive ikut naik).")
    st.markdown(
        """
        <div style="margin-top:1.2rem;padding:.85rem 1rem;background:rgba(108,99,255,0.07);
                    border-radius:12px;border:1px solid rgba(108,99,255,0.15);">
          <div style="font-size:.75rem;color:#6C63FF;font-weight:700;text-transform:uppercase;
                      letter-spacing:.06em;margin-bottom:.4rem;">ℹ️ Catatan</div>
          <div style="font-size:.8rem;color:#64748B;line-height:1.55;">
            Model hanya membaca <b>teks URL</b> (fitur leksikal). Situs yang diperiksa <b>tidak dibuka</b>.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

bundle = bundles[choice]

# ---------------------------------------------------------------- hero
st.markdown(
    """
    <div class="hero-wrapper">
      <div class="hero">
        <div class="hero-badge"> &nbsp; Powered by Machine Learning</div>
        <h1>Cek tautan sebelum<br><span>kamu klik.</span></h1>
        <p>Tempel sebuah URL. Sistem membaca pola penulisannya dan memperkirakan seberapa besar
        kemungkinan URL itu phishing, lengkap dengan bagian mana yang mencurigakan.</p>
        <div class="hero-stats">
          <div class="hero-stat"><div class="num">3</div><div class="lbl">Model ML</div></div>
          <div class="hero-stat"><div class="num">30+</div><div class="lbl">Fitur URL</div></div>
          <div class="hero-stat"><div class="num">100%</div><div class="lbl">Privasi</div></div>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab1, tab2, tab3 = st.tabs(["🔍  Periksa satu URL", "📋  Periksa banyak URL", "📊  Performa model"])

# ---------------------------------------------------------------- tab 1
with tab1:
    st.markdown('<div class="input-wrapper">', unsafe_allow_html=True)
    st.markdown('<span class="input-label">🔗 Masukkan URL untuk diperiksa</span>', unsafe_allow_html=True)
    c1, c2 = st.columns([5, 1])
    c1.text_input("URL", key="url_input", placeholder="https://contoh-domain.com/login",
                  label_visibility="collapsed", on_change=submit)
    c2.button("Periksa →", type="primary", width="stretch", on_click=submit)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="section-divider"><div class="line"></div>'
        '<div class="label">✨ coba contoh URL</div>'
        '<div class="line"></div></div>',
        unsafe_allow_html=True,
    )
    ec = st.columns(len(EXAMPLES))
    for col, (label, u) in zip(ec, EXAMPLES.items()):
        col.button(label, key=f"ex_{label}", width="stretch", on_click=use_example, args=(u,))

    url = st.session_state.get("checked", "")
    if not url:
        st.markdown(
            """
            <div style="text-align:center;padding:3rem 2rem;
                        background:rgba(108,99,255,0.04);border-radius:20px;
                        border:2px dashed rgba(108,99,255,0.2);margin-top:1.5rem;">
              <div style="font-size:3rem;margin-bottom:.8rem;">🔎</div>
              <div style="font-weight:700;font-size:1.1rem;color:#1A1A2E;margin-bottom:.4rem;">Belum ada URL yang diperiksa</div>
              <div style="color:#64748B;font-size:.92rem;">Tempel URL di kolom atas lalu tekan Enter, atau pilih salah satu contoh di atas.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        proba, X = predict(bundle, [url])
        p = float(proba[0])
        kind, title, advice = level(p, threshold)

        ICONS = {"bad": "🚨", "warn": "⚠️", "safe": "✅"}
        icon = ICONS[kind]

        left, right = st.columns([3, 2], gap="large")

        with left:
            st.markdown(
                f'<div class="verdict {kind}">'
                f'<span class="verdict-icon">{icon}</span>'
                f'<div class="big">{title}</div>'
                f'<div class="sub">{advice}</div>'
                f'<div class="meter">'
                f'<div class="thr" style="left:{threshold*100:.0f}%"></div>'
                f'<div class="dot" style="left:{min(max(p,0.01),0.99)*100:.1f}%"></div>'
                f'</div>'
                f'<div class="meter-lbl"><span>🟢 Aman</span>'
                f'<span>Probabilitas phishing: <b>{p:.1%}</b> (garis = ambang {threshold:.0%})</span>'
                f'<span>🔴 Phishing</span></div></div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="card"><h4><span class="card-icon">🔬</span> Bedah URL</h4>{anatomy_html(url)}</div>',
                unsafe_allow_html=True,
            )

        with right:
            flags = "".join(
                f'<div class="flag {c}"><b>{t}</b><span>{html.escape(m)}</span></div>'
                for c, t, m in indicators(X.iloc[0])
            )
            st.markdown(
                f'<div class="card"><h4><span class="card-icon">📌</span> Indikator pada URL</h4>{flags}</div>',
                unsafe_allow_html=True,
            )

            rows = ""
            for n in available:
                pn = float(predict(bundles[n], [url])[0][0])
                if pn >= threshold:
                    color = "linear-gradient(90deg,#F87171,#DC2626)"
                else:
                    color = "linear-gradient(90deg,#34D399,#059669)"
                rows += (
                    f'<div class="row"><span class="nm">{n}</span>'
                    f'<span class="tr"><span class="fl" '
                    f'style="display:block;width:{pn*100:.1f}%;background:{color}"></span></span>'
                    f'<span class="pc">{pn:.0%}</span></div>'
                )
            st.markdown(
                f'<div class="card"><h4><span class="card-icon">🤖</span> Pendapat tiap model</h4>{rows}</div>',
                unsafe_allow_html=True,
            )

        with st.expander("🔧 Lihat semua fitur yang diekstrak"):
            st.dataframe(X.T.rename(columns={0: "nilai"}), width="stretch")

# ---------------------------------------------------------------- tab 2
with tab2:
    st.markdown(
        '<div class="card"><h4><span class="card-icon">📂</span> Periksa banyak URL sekaligus</h4>'
        '<p style="color:#64748B;font-size:.92rem;margin:0 0 1rem 0;">Unggah file CSV yang memiliki kolom <code>url</code>. Hasilnya dapat langsung diunduh.</p>',
        unsafe_allow_html=True,
    )
    f = st.file_uploader("File CSV", type="csv", label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

    if f is not None:
        df = pd.read_csv(f)
        col = next((c for c in df.columns if c.lower() == "url"), None)
        if col is None:
            st.warning("Kolom `url` tidak ditemukan di file ini. Ganti nama kolom URL menjadi `url`.")
        else:
            proba, _ = predict(bundle, df[col].astype(str).tolist())
            df["proba_phishing"] = proba.round(4)
            df["prediksi"] = ["phishing" if q >= threshold else "aman" for q in proba]
            n_bad = int((df["prediksi"] == "phishing").sum())
            m1, m2, m3 = st.columns(3)
            m1.metric("📊 Total URL", len(df))
            m2.metric("🚨 Terindikasi phishing", n_bad)
            m3.metric("✅ Dianggap aman", len(df) - n_bad)
            st.dataframe(
                df.style.apply(
                    lambda r: ["background-color:#FEF2F2" if r["prediksi"] == "phishing" else "background-color:#ECFDF5" for _ in r],
                    axis=1,
                ),
                width="stretch",
            )
            st.download_button("⬇️ Unduh hasil (CSV)", df.to_csv(index=False), "hasil_prediksi.csv", "text/csv")

# ---------------------------------------------------------------- tab 3
with tab3:
    st.markdown(
        '<div class="card"><h4><span class="card-icon">📈</span> Performa & Evaluasi Model</h4>'
        '<p style="color:#64748B;font-size:.92rem;margin:0;">Hasil evaluasi pada data uji, dibuat oleh <code>train.py</code>.</p></div>',
        unsafe_allow_html=True,
    )
    if os.path.exists("reports/model_comparison.csv"):
        comp = pd.read_csv("reports/model_comparison.csv", index_col=0)
        st.dataframe(
            comp.style.format("{:.4f}").highlight_max(axis=0, color="#EDE9FE"),
            width="stretch",
        )
    else:
        st.info("Belum ada laporan. Jalankan `train.py` untuk membuatnya.")
    g1, g2 = st.columns(2)
    if os.path.exists("reports/confusion_matrix.png"):
        g1.image("reports/confusion_matrix.png", caption="Confusion matrix model terbaik")
    if os.path.exists("reports/feature_importance.png"):
        g2.image("reports/feature_importance.png", caption="15 fitur paling berpengaruh")
    e1, e2 = st.columns(2)
    for c, fn, ttl in ((e1, "false_positives", "False positive: situs aman yang dikira phishing"),
                       (e2, "false_negatives", "False negative: phishing yang lolos")):
        path = f"reports/{fn}.csv"
        if os.path.exists(path):
            d = pd.read_csv(path)
            c.markdown(f"**{ttl}** ({len(d)} URL)")
            c.dataframe(d.head(15), width="stretch")