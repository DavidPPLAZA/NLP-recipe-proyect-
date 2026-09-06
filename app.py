#!/usr/bin/env python3
"""
🍳 Recetario Inteligente
Motor de recuperación: intfloat/multilingual-e5-large + ChromaDB (coseno)
Normalización:         Ollama (OpenAI SDK compatible)
Transcripción:         Whisper (small)
Síntesis de voz:       Kokoro-ONNX
"""

import os, re, json, time, tempfile, io, wave
import numpy as np
from pathlib import Path
from typing import Optional

import streamlit as st

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Recetario Inteligente",
    page_icon="🍳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CONSTANTS ───────────────────────────────────────────────────────────────
BASE_DIR         = Path(__file__).parent
CHROMA_PATH      = str(BASE_DIR / "chroma_e5")   # colección ya indexada desde el notebook
COLLECTION_NAME  = "recetas_e5"
E5_MODEL_ID      = "intfloat/multilingual-e5-large"
KOKORO_ONNX_PATH = str(BASE_DIR / "kokoro-v1.0.onnx")
VOICES_BIN_PATH  = str(BASE_DIR / "voices.bin")

CATEGORIAS = [
    "Aperitivos y tapas","Arroces y cereales","Aves y caza","Carne",
    "Cócteles y bebidas","Ensaladas","Guisos y Potajes","Huevos y lácteos",
    "Legumbres","Mariscos","Pan y bollería","Pasta","Pescado","Postres",
    "Salsas","Sopas y cremas","Verduras",
]

DIFICULTAD_MAP = {
    "muy baja":{"color":"#22c55e","emoji":"🟢"},
    "baja":    {"color":"#84cc16","emoji":"🟡"},
    "media":   {"color":"#eab308","emoji":"🟠"},
    "alta":    {"color":"#f97316","emoji":"🔴"},
    "muy alta":{"color":"#ef4444","emoji":"🔥"},
}
TIEMPO_EMOJI = {"rapido":"⚡","normal":"🕐","moderado":"⏳","largo":"🕰️","muy largo":"📅"}

SYSTEM_PROMPT = """
Eres un normalizador de consultas para un buscador de recetas de cocina español.
Devuelve SIEMPRE un JSON y nada más, con exactamente este formato:
{"valida": true, "query": "texto limpio y normalizado para búsqueda semántica"}
o
{"valida": false, "motivo": "explicación breve en español"}

REGLAS:
1. Ingredientes escritos bien → devuélvelos limpios, sin cantidades ni unidades, en minúsculas.
   Ejemplo: "2 pechugas de pollo, 3 patatas y ajo" → {"valida": true, "query": "pollo patatas ajo"}
2. Errores ortográficos → corrígelos.
   Ejemplo: "tomattes, jamob, quesoo" → {"valida": true, "query": "tomates jamón queso"}
3. Consulta abstracta (sabor, ocasión, emoción) → tradúcela a ingredientes o conceptos buscables.
   Ejemplo: "quiero algo dulce para el postre" → {"valida": true, "query": "postre dulce chocolate azúcar"}
   Ejemplo: "cena ligera de verano" → {"valida": true, "query": "ensalada pescado verduras ligero verano"}
4. Sin relación con comida → invalida.
   Ejemplo: "cuánto mide la Torre Eiffel" → {"valida": false, "motivo": "No está relacionado con recetas"}

IMPORTANTE: responde SOLO con el JSON. Sin markdown, sin texto extra.
""".strip()

# ─── GLOBAL CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --cream:    #fdf8f0;
    --warm-100: #fef3e2;
    --warm-200: #fde8c8;
    --amber:    #d97706;
    --amber-lt: #fbbf24;
    --brown:    #78350f;
    --green:    #15803d;
    --red:      #dc2626;
    --ink:      #1c1917;
    --muted:    #78716c;
    --border:   #e7ddd0;
    --card-bg:  #fffcf7;
    --shadow:   0 2px 16px rgba(120,80,30,.09);
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--cream) !important;
    font-family: 'DM Sans', sans-serif;
}

[data-testid="stSidebar"] {
    background: var(--brown) !important;
    border-right: 2px solid #5c2d0a;
}
[data-testid="stSidebar"] * { color: #fde8c8 !important; }
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 { color: #fbbf24 !important; font-family: 'Playfair Display', serif !important; }

.main-header {
    text-align: center;
    padding: 2.5rem 0 1.5rem;
}
.main-header h1 {
    font-family: 'Playfair Display', serif;
    font-size: clamp(2.2rem, 5vw, 3.4rem);
    font-weight: 900;
    color: var(--brown);
    letter-spacing: -1px;
    margin: 0;
    line-height: 1.1;
}
.main-header .subtitle {
    color: var(--muted);
    font-size: 1.05rem;
    font-weight: 300;
    margin-top: .4rem;
    letter-spacing: .3px;
}

.mode-selector {
    display: flex;
    gap: .75rem;
    margin: 1.5rem 0 1rem;
    justify-content: center;
}
.mode-btn {
    padding: .55rem 1.6rem;
    border-radius: 50px;
    border: 2px solid var(--border);
    background: white;
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
    font-size: .92rem;
    cursor: pointer;
    transition: all .18s;
    color: var(--muted);
}
.mode-btn.active {
    background: var(--amber);
    border-color: var(--amber);
    color: white;
}

.input-card {
    background: var(--card-bg);
    border: 1.5px solid var(--border);
    border-radius: 16px;
    padding: 1.6rem;
    box-shadow: var(--shadow);
    margin-bottom: 1.5rem;
}
.input-label {
    font-family: 'Playfair Display', serif;
    font-size: 1.15rem;
    color: var(--brown);
    margin-bottom: .6rem;
    font-weight: 700;
}

.recipe-card {
    background: var(--card-bg);
    border: 1.5px solid var(--border);
    border-radius: 14px;
    padding: 1.4rem 1.5rem;
    margin-bottom: 1rem;
    box-shadow: var(--shadow);
    transition: transform .15s, box-shadow .15s;
    position: relative;
    overflow: hidden;
}
.recipe-card::before {
    content: '';
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 4px;
    background: linear-gradient(180deg, var(--amber), var(--amber-lt));
    border-radius: 4px 0 0 4px;
}
.recipe-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 24px rgba(120,80,30,.15);
}
.card-header {
    display: flex;
    align-items: center;
    gap: .6rem;
    margin-bottom: .5rem;
}
.rank-badge {
    font-family: 'Playfair Display', serif;
    font-size: 1.1rem;
    font-weight: 900;
    color: var(--amber);
    min-width: 2rem;
}
.score-pill {
    font-size: .75rem;
    font-weight: 600;
    background: var(--warm-200);
    color: var(--amber);
    border-radius: 20px;
    padding: .15rem .65rem;
    letter-spacing: .3px;
}
.recipe-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.18rem;
    font-weight: 700;
    color: var(--ink);
    margin: .3rem 0 .7rem;
    line-height: 1.3;
}
.meta-chips {
    display: flex;
    flex-wrap: wrap;
    gap: .45rem;
    margin-bottom: .85rem;
}
.chip {
    font-size: .78rem;
    font-weight: 500;
    padding: .2rem .7rem;
    border-radius: 20px;
    white-space: nowrap;
}
.chip-cat  { background:#fdf4e3; color:var(--amber); border:1px solid #fde8c8; }
.chip-dif  { border-width: 1px; border-style: solid; }
.chip-time { background:#f0fdf4; color:var(--green); border:1px solid #bbf7d0; }

.ing-section { margin-top: .5rem; }
.ing-label {
    font-size: .73rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: .8px;
    color: var(--muted);
    margin-bottom: .45rem;
}
.ing-tags { display: flex; flex-wrap: wrap; gap: .35rem; }
.ing-tag {
    font-size: .78rem;
    background: var(--warm-100);
    color: var(--brown);
    border: 1px solid var(--warm-200);
    border-radius: 8px;
    padding: .15rem .55rem;
}
.ing-more { color: var(--muted); font-style: italic; }

.status-box {
    padding: .8rem 1.1rem;
    border-radius: 10px;
    margin: .7rem 0;
    font-size: .9rem;
    font-weight: 500;
}
.status-ok    { background:#f0fdf4; color:#15803d; border:1px solid #bbf7d0; }
.status-warn  { background:#fffbeb; color:#92400e; border:1px solid #fde68a; }
.status-error { background:#fef2f2; color:#dc2626; border:1px solid #fecaca; }
.status-info  { background:#eff6ff; color:#1d4ed8; border:1px solid #bfdbfe; }

.results-header {
    font-family: 'Playfair Display', serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--brown);
    margin: 1.5rem 0 .8rem;
    border-bottom: 2px solid var(--warm-200);
    padding-bottom: .5rem;
}
.results-meta {
    font-size: .85rem;
    color: var(--muted);
    margin-bottom: 1rem;
}

.normalized-query {
    background: var(--warm-100);
    border-left: 3px solid var(--amber);
    padding: .6rem 1rem;
    border-radius: 0 8px 8px 0;
    margin: .7rem 0;
    font-size: .9rem;
    color: var(--brown);
}
.normalized-query strong { font-weight: 600; }

/* Streamlit widget overrides */
.stTextArea textarea {
    background: white !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: .95rem !important;
    color: var(--ink) !important;
}
.stTextArea textarea:focus {
    border-color: var(--amber) !important;
    box-shadow: 0 0 0 2px rgba(217,119,6,.15) !important;
}
.stButton > button {
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
    transition: all .15s !important;
}
.stButton > button[kind="primary"] {
    background: var(--amber) !important;
    border-color: var(--amber) !important;
    color: white !important;
    font-size: 1rem !important;
    padding: .55rem 2rem !important;
}
.stButton > button[kind="primary"]:hover {
    background: var(--brown) !important;
    border-color: var(--brown) !important;
}
.stMultiSelect [data-baseweb="tag"] {
    background: var(--amber) !important;
    border-radius: 6px !important;
}
.stSlider [data-baseweb="slider"] { color: var(--amber) !important; }
div[data-testid="stSlider"] > div > div > div > div {
    background: var(--amber) !important;
}

/* divider */
hr { border-color: var(--border) !important; margin: 1.2rem 0 !important; }

/* index progress bar */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, var(--amber), var(--amber-lt)) !important;
}
</style>
""", unsafe_allow_html=True)


# ─── CACHED LOADERS ──────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Cargando modelo E5-large…")
def load_embedder():
    from sentence_transformers import SentenceTransformer
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer(E5_MODEL_ID, device=device)

@st.cache_resource(show_spinner=False)
def load_chroma():
    import chromadb
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    col = client.get_collection(name=COLLECTION_NAME)   # ya existe, no crear
    return client, col

@st.cache_resource(show_spinner="Cargando Whisper (small)…")
def load_whisper():
    import whisper
    return whisper.load_model("small")

@st.cache_resource(show_spinner="Cargando Kokoro TTS…")

def load_kokoro():
    try:
        from kokoro_onnx import Kokoro
        return Kokoro(KOKORO_ONNX_PATH, VOICES_BIN_PATH)
    except Exception as e:
        st.error(f"Error Kokoro: {e}")  # ← así ves qué falla
        return None



# ─── QUERY NORMALIZATION ─────────────────────────────────────────────────────

def normalize_query(query: str, base_url: str, model: str) -> dict:
    try:
        from openai import OpenAI
        client = OpenAI(base_url=base_url, api_key="ollama")
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": query},
            ],
            temperature=0.1,
            timeout=15,
        )
        raw = resp.choices[0].message.content.strip()
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return {"valida": True, "query": query}  # fallback passthrough
        parsed = json.loads(match.group())
        return parsed
    except Exception as e:
        # Si Ollama no responde, pasamos la query tal cual
        return {"valida": True, "query": query, "_ollama_error": str(e)}


# ─── RETRIEVAL ───────────────────────────────────────────────────────────────

def retrieve(
    query: str,
    col,
    embedder,
    k: int = 10,
    categorias: Optional[list] = None,
    dificultad_max: Optional[int] = None,
    tiempo_max: Optional[int] = None,
) -> list:
    q_emb = embedder.encode(
        ["query: " + query], normalize_embeddings=True
    )[0].tolist()

    filtros = []
    if categorias:
        filtros.append({"categoria": {"$in": categorias}})
    if dificultad_max:
        filtros.append({"dificultad_num": {"$lte": dificultad_max}})
        filtros.append({"dificultad_num": {"$gt": 0}})
    if tiempo_max:
        filtros.append({"tiempo_minutos": {"$lte": tiempo_max}})
        filtros.append({"tiempo_minutos": {"$gt": 0}})

    where = None
    if len(filtros) == 1:
        where = filtros[0]
    elif len(filtros) > 1:
        where = {"$and": filtros}

    kwargs = {
        "query_embeddings": [q_emb],
        "n_results": min(k, col.count()),
        "include":   ["distances", "metadatas", "documents"],
    }
    if where:
        kwargs["where"] = where

    results = col.query(**kwargs)
    out = []
    for rid, dist, meta, doc in zip(
        results["ids"][0], results["distances"][0],
        results["metadatas"][0], results["documents"][0],
    ):
        out.append({
            "id":    rid,
            "score": round(1.0 - dist, 4),
            "_doc":  doc,          # texto_semantico completo, para parsear ingredientes
            **meta,
        })
    return out


# ─── AUDIO HELPERS ───────────────────────────────────────────────────────────

def transcribe_audio(audio_bytes: bytes, whisper_model) -> str:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    try:
        result = whisper_model.transcribe(tmp_path, language="es")
        return result["text"].strip()
    finally:
        os.unlink(tmp_path)


def synth_to_wav(text: str, kokoro, voice: str = "ef_dora") -> bytes:
    samples, sr = kokoro.create(text, voice=voice, speed=1.0, lang="es")
    samples_int = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(samples_int.tobytes())
    return buf.getvalue()


def build_tts_response(results: list, n_speak: int = 3) -> str:
    """Genera texto hablado con las top-n recetas."""
    lines = [f"Encontré {len(results)} recetas para ti."]
    for i, r in enumerate(results[:n_speak], 1):
        titulo = r["titulo"].replace("Receta de ", "").replace("Receta ", "")
        dif  = r.get("dificultad") or "dificultad desconocida"
        tmin = int(r.get("tiempo_minutos") or 0)
        # Extraer ingredientes del texto_semantico almacenado en _doc
        doc  = r.get("_doc", "")
        m    = re.search(r"Sus ingredientes son: (.+?)\.", doc)
        ings_str = m.group(1) if m else "varios ingredientes"
        lines.append(
            f"Número {i}: {titulo}. Dificultad {dif}, "
            f"tiempo de preparación {tmin} minutos. "
            f"Algunos ingredientes: {ings_str}."
        )
    return " ".join(lines)


# ─── CARD RENDERER ───────────────────────────────────────────────────────────

def render_recipe_card(r: dict, idx: int) -> None:
    dif     = r.get("dificultad") or "desconocida"
    dif_cfg = DIFICULTAD_MAP.get(dif, {"color": "#94a3b8", "emoji": "⚪"})
    tcat    = r.get("tiempo_categoria") or "desconocido"
    tmin    = int(r.get("tiempo_minutos") or 0)
    t_emoji = TIEMPO_EMOJI.get(tcat, "⏱️")
    score   = int(r["score"] * 100)
    cat     = r.get("categoria", "Sin categoría")

    # Ingredientes: parsear del texto_semantico guardado en _doc
    doc = r.get("_doc", "")
    m   = re.search(r"Sus ingredientes son: (.+?)\.", doc)
    if m:
        ings_raw = [i.strip() for i in m.group(1).split(",") if i.strip()]
    else:
        ings_raw = []
    ings_show = ings_raw[:7]
    ings_html = "".join(f'<span class="ing-tag">{i}</span>' for i in ings_show)
    if len(ings_raw) > 7:
        ings_html += f'<span class="ing-tag ing-more">+{len(ings_raw)-7} más</span>'

    dif_style = (
        f'background:{dif_cfg["color"]}22;'
        f'color:{dif_cfg["color"]};'
        f'border:1px solid {dif_cfg["color"]}55;'
    )

    st.markdown(f"""
    <div class="recipe-card">
        <div class="card-header">
            <span class="rank-badge">#{idx+1}</span>
            <span class="score-pill">✦ {score}% match</span>
        </div>
        <div class="recipe-title">{r['titulo']}</div>
        <div class="meta-chips">
            <span class="chip chip-cat">📂 {cat}</span>
            <span class="chip chip-dif" style="{dif_style}">{dif_cfg['emoji']} {dif}</span>
            <span class="chip chip-time">{t_emoji} {tmin} min</span>
        </div>
        <div class="ing-section">
            <div class="ing-label">Ingredientes</div>
            <div class="ing-tags">{ings_html}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─── SIDEBAR ─────────────────────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.markdown("## 🍳 Recetario\n### Inteligente")
        st.markdown("---")

        st.markdown("### 🔍 Filtros de búsqueda")

        top_k = st.slider(
            "Top-K resultados", min_value=1, max_value=20, value=5, step=1,
            help="Número máximo de recetas a recuperar",
        )

        cats_sel = st.multiselect(
            "Categorías",
            options=CATEGORIAS,
            default=[],
            placeholder="Todas las categorías…",
        )

        dif_labels = ["Cualquier dificultad","🟢 Muy baja","🟡 Baja","🟠 Media","🔴 Alta","🔥 Muy alta"]
        dif_sel = st.selectbox("Dificultad máxima", dif_labels, index=0)
        dif_max = dif_labels.index(dif_sel) if dif_sel != dif_labels[0] else None

        tiempo_max_enabled = st.checkbox("Filtrar por tiempo máximo", value=False)
        tiempo_max = None
        if tiempo_max_enabled:
            tiempo_max = st.slider("Tiempo máximo (min)", 10, 180, 60, step=5)

        st.markdown("---")
        st.markdown("### 🤖 Configuración LLM")

        ollama_base = st.text_input(
            "Ollama URL", value="http://localhost:11434/v1",
            help="URL base de tu instancia Ollama",
        )
        ollama_model = st.text_input(
            "Modelo Ollama", value="qwen2.5:3b",
            help="Modelo a usar para normalización de queries",
        )
        use_llm = st.checkbox("Usar LLM para normalizar", value=True)

        st.markdown("---")
        st.markdown("### 🔊 Voz (Kokoro)")
        voices_available = ["ef_dora","em_alex","af_heart","af_bella","bm_george"]
        voice_sel = st.selectbox("Voz TTS", voices_available, index=0)

        st.markdown("---")
        st.markdown(
            "<div style='font-size:.78rem;color:#fde8c899;line-height:1.6'>"
            "Motor: <b>E5-Large</b><br>"
            "DB: <b>ChromaDB</b> (coseno)<br>"
            "ASR: <b>Whisper small</b><br>"
            "TTS: <b>Kokoro ONNX v1</b>"
            "</div>",
            unsafe_allow_html=True,
        )

    return top_k, cats_sel, dif_max, tiempo_max, ollama_base, ollama_model, use_llm, voice_sel


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    # ── Header ──────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="main-header">
        <h1>🍳 Recetario Inteligente</h1>
        <p class="subtitle">
            Búsqueda semántica · E5-Large + ChromaDB · Normalización con LLM
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar ─────────────────────────────────────────────────────────────
    top_k, cats_sel, dif_max, tiempo_max, ollama_base, ollama_model, use_llm, voice_sel = render_sidebar()

    # ── Resource loading ────────────────────────────────────────────────────
    embedder = load_embedder()
    _, col   = load_chroma()

    n_recetas = col.count()
    st.markdown(
        f'<div class="status-box status-ok">'
        f'✅ Colección <strong>chroma_e5</strong> cargada · {n_recetas:,} recetas indexadas'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Mode selector ───────────────────────────────────────────────────────
    st.markdown("---")
    mode = st.radio(
        "Modo de entrada",
        options=["📝 Texto", "🎙️ Audio"],
        horizontal=True,
        label_visibility="collapsed",
    )

    # ═══════════════════════════════════════════════════════════════════════
    # MODO TEXTO
    # ═══════════════════════════════════════════════════════════════════════
    if mode == "📝 Texto":
        st.markdown("""
        <div class="input-label">¿Qué tienes en la nevera? ¿Qué te apetece?</div>
        """, unsafe_allow_html=True)

        query_text = st.text_area(
            label="query",
            placeholder=(
                "Ej: pollo, patatas, ajo, limón\n"
                "Ej: quiero algo dulce y rápido\n"
                "Ej: cena ligera de verano con pescado"
            ),
            height=110,
            label_visibility="collapsed",
        )

        col_btn1, col_btn2 = st.columns([1, 5])
        with col_btn1:
            search_btn = st.button("🔍 Buscar recetas", type="primary", use_container_width=True)

        if search_btn and query_text.strip():
            _run_search(
                raw_query=query_text.strip(),
                col=col,
                embedder=embedder,
                top_k=top_k,
                cats_sel=cats_sel,
                dif_max=dif_max,
                tiempo_max=tiempo_max,
                ollama_base=ollama_base,
                ollama_model=ollama_model,
                use_llm=use_llm,
                audio_mode=False,
                voice_sel=voice_sel,
            )
        elif search_btn and not query_text.strip():
            st.markdown('<div class="status-box status-warn">⚠️ Escribe algo para buscar.</div>', unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════════════
    # MODO AUDIO
    # ═══════════════════════════════════════════════════════════════════════
    else:
        st.markdown("""
        <div class="input-label">Graba tu consulta de voz</div>
        """, unsafe_allow_html=True)
        st.markdown(
            '<div class="status-box status-info">'
            '🎙️ Pulsa el botón para grabar. '
            'Di qué ingredientes tienes o qué tipo de plato quieres.'
            '</div>',
            unsafe_allow_html=True,
        )

        # audio_recorder_streamlit
        try:
            from audio_recorder_streamlit import audio_recorder
            audio_bytes = audio_recorder(
                text="  Grabar",
                recording_color="#d97706",
                neutral_color="#78350f",
                icon_size="2x",
                pause_threshold=3.0,
                sample_rate=16000,
            )
        except ImportError:
            st.error(
                "❌ Instala `audio-recorder-streamlit` para grabar audio:\n"
                "```\npip install audio-recorder-streamlit\n```"
            )
            audio_bytes = None

        if audio_bytes:
            st.audio(audio_bytes, format="audio/wav")

            whisper_model = load_whisper()
            with st.spinner("📝 Transcribiendo con Whisper…"):
                transcription = transcribe_audio(audio_bytes, whisper_model)

            st.markdown(
                f'<div class="normalized-query">'
                f'🎙️ <strong>Transcripción:</strong> {transcription}'
                f'</div>',
                unsafe_allow_html=True,
            )

            if transcription:
                _run_search(
                    raw_query=transcription,
                    col=col,
                    embedder=embedder,
                    top_k=top_k,
                    cats_sel=cats_sel,
                    dif_max=dif_max,
                    tiempo_max=tiempo_max,
                    ollama_base=ollama_base,
                    ollama_model=ollama_model,
                    use_llm=use_llm,
                    audio_mode=True,
                    voice_sel=voice_sel,
                )


# ─── SEARCH RUNNER ───────────────────────────────────────────────────────────

def _run_search(
    raw_query: str,
    col,
    embedder,
    top_k: int,
    cats_sel: list,
    dif_max: Optional[int],
    tiempo_max: Optional[int],
    ollama_base: str,
    ollama_model: str,
    use_llm: bool,
    audio_mode: bool,
    voice_sel: str,
) -> None:

    # 1. Normalización con LLM ────────────────────────────────────────────
    normalized_query = raw_query
    norm_result = None

    if use_llm:
        with st.spinner("🤖 Normalizando consulta con LLM…"):
            norm_result = normalize_query(raw_query, ollama_base, ollama_model)

        if not norm_result.get("valida", True):
            motivo = norm_result.get("motivo", "Consulta no válida para recetas.")
            st.markdown(
                f'<div class="status-box status-error">'
                f'❌ <strong>Consulta fuera de contexto:</strong> {motivo}'
                f'</div>',
                unsafe_allow_html=True,
            )
            if audio_mode:
                _speak_if_possible(f"Lo siento, {motivo}", voice_sel)
            return

        final_q = norm_result.get("query") or norm_result.get("ingredientes")
        if isinstance(final_q, list):
            normalized_query = " ".join(final_q)
        elif isinstance(final_q, str):
            normalized_query = final_q
        else:
            normalized_query = raw_query

        if "_ollama_error" in norm_result:
            st.markdown(
                f'<div class="status-box status-warn">'
                f'⚠️ Ollama no disponible — usando query sin normalizar. '
                f'<code>{norm_result["_ollama_error"]}</code>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="normalized-query">'
                f'🤖 <strong>Query normalizada:</strong> {normalized_query}'
                f'</div>',
                unsafe_allow_html=True,
            )

    # 2. Retrieval E5 ────────────────────────────────────────────────────
    with st.spinner("🔍 Buscando recetas similares…"):
        t0 = time.time()
        results = retrieve(
            query=normalized_query,
            col=col,
            embedder=embedder,
            k=top_k,
            categorias=cats_sel if cats_sel else None,
            dificultad_max=dif_max,
            tiempo_max=tiempo_max,
        )
        elapsed = (time.time() - t0) * 1000

    # 3. Mostrar resultados ───────────────────────────────────────────────
    if not results:
        st.markdown(
            '<div class="status-box status-warn">'
            '🔎 No se encontraron recetas con los filtros aplicados. '
            'Prueba a ampliar los filtros.'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    st.markdown(
        f'<div class="results-header">🍽️ Recetas encontradas</div>'
        f'<div class="results-meta">'
        f'{len(results)} resultado{"s" if len(results)!=1 else ""} · '
        f'{elapsed:.0f}ms de recuperación'
        f'</div>',
        unsafe_allow_html=True,
    )

    for i, r in enumerate(results):
        render_recipe_card(r, i)

    # 4. TTS (solo en modo audio) ─────────────────────────────────────────
    if audio_mode:
        tts_text = build_tts_response(results, n_speak=min(3, len(results)))
        _speak_if_possible(tts_text, voice_sel)


def _speak_if_possible(text: str, voice: str) -> None:
    kokoro = load_kokoro()
    if kokoro is None:
        st.markdown(
            '<div class="status-box status-warn">'
            '⚠️ Kokoro TTS no disponible. '
            'Instala <code>kokoro-onnx</code> y asegúrate de que los archivos '
            '<code>kokoro-v1_0.onnx</code> y <code>voices.bin</code> están en el directorio.'
            '</div>',
            unsafe_allow_html=True,
        )
        return
    with st.spinner("🔊 Sintetizando respuesta de voz…"):
        try:
            wav_bytes = synth_to_wav(text, kokoro, voice=voice)
            st.audio(wav_bytes, format="audio/wav", autoplay=True)
        except Exception as e:
            st.markdown(
                f'<div class="status-box status-warn">⚠️ Error TTS: {e}</div>',
                unsafe_allow_html=True,
            )


# ─── ENTRY POINT ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
