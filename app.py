"""
Interfaz de usuario para lectura y analisis de boletas y facturas peruanas (OCR + NLP).
Ejecutar: streamlit run app.py
"""
from __future__ import annotations

import html
import io
import json
import zipfile
from pathlib import Path

import pandas as pd
import streamlit as st

from src.nlp import CATEGORIAS, analyze, tfidf_keywords
from src.ocr import PSM_OPTIONS, available_langs, run_ocr
from src.preprocess import PreprocessConfig, load_file, preprocess, to_png_bytes

st.set_page_config(
    page_title="Lector de Boletas y Facturas · Peru",
    layout="wide",
)

SAMPLES_DIR = Path(__file__).parent / "samples"

LANG_LABELS = {
    "spa": "Español",
    "spa+eng": "Español + Inglés",
    "eng": "Inglés",
}


# ---------------------------------------------------------------------------
# Estilos CSS: Tema oscuro vanguardista (Negro y Morado) + Tipografia Segoe UI
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
/* 1. Tipografia global Segoe UI sin sobrescribir la fuente de iconos de Streamlit */
html, body, .stApp, p, label, li, td, th,
h1, h2, h3, h4, h5, h6, button, input, textarea, select,
[data-testid="stMarkdownContainer"],
[data-testid="stText"],
[data-testid="stCaptionContainer"] {
    font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* Proteger explicitamente los iconos Material de Streamlit (flechas, sidebar, upload) */
[data-testid="stIconMaterial"],
.material-symbols-rounded,
span[translate="no"] {
    font-family: 'Material Symbols Rounded' !important;
    font-weight: normal !important;
    font-style: normal !important;
    line-height: 1 !important;
    letter-spacing: normal !important;
    text-transform: none !important;
    white-space: nowrap !important;
    word-wrap: normal !important;
    direction: ltr !important;
}

/* 2. Fondo principal negro vanguardista con matices morados profundos */
.stApp {
    background: radial-gradient(circle at 15% 10%, #1a0b2e 0%, #0b0713 45%, #050408 100%);
    color: #EDE9FE;
}

/* Espaciado superior del contenedor principal para que el banner nunca se corte */
.block-container {
    padding-top: 3.4rem !important;
    padding-bottom: 3rem !important;
}

/* Encabezado superior transparente */
header[data-testid="stHeader"] {
    background: rgba(5, 4, 8, 0.78);
    backdrop-filter: blur(10px);
    border-bottom: 1px solid rgba(139, 92, 246, 0.18);
}

/* Barra lateral oscura */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d0818 0%, #07050c 100%);
    border-right: 1px solid rgba(147, 51, 234, 0.28);
}
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] {
    color: #E9D5FF;
}

/* Banner principal HTML */
.hero-banner {
    background: linear-gradient(135deg, rgba(88, 28, 135, 0.38) 0%, rgba(15, 10, 28, 0.92) 65%, rgba(8, 6, 14, 0.98) 100%);
    border: 1px solid rgba(168, 85, 247, 0.35);
    border-left: 4px solid #A855F7;
    border-radius: 14px;
    padding: 24px 28px;
    margin-top: 6px;
    margin-bottom: 22px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.65), 0 0 25px rgba(124, 58, 237, 0.12);
}
.hero-badge {
    display: inline-block;
    background: rgba(147, 51, 234, 0.22);
    color: #C084FC;
    border: 1px solid rgba(192, 132, 252, 0.35);
    border-radius: 6px;
    padding: 3px 10px;
    font-size: 0.74rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 8px;
}
.hero-title {
    font-size: 1.85rem;
    font-weight: 700;
    color: #FAF5FF !important;
    margin: 0 0 6px 0;
    letter-spacing: -0.02em;
}
.hero-subtitle {
    font-size: 0.95rem;
    color: #C4B5FD !important;
    margin: 0;
    line-height: 1.5;
}

/* Tarjetas KPI personalizadas */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 14px;
    margin: 12px 0 22px 0;
}
@media (max-width: 960px) {
    .kpi-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }
}
.kpi-card {
    background: linear-gradient(160deg, rgba(26, 16, 46, 0.92) 0%, rgba(11, 8, 20, 0.96) 100%);
    border: 1px solid rgba(139, 92, 246, 0.3);
    border-radius: 12px;
    padding: 16px 18px;
    box-shadow: 0 6px 18px rgba(0, 0, 0, 0.5);
    transition: border-color 0.2s ease, transform 0.2s ease;
}
.kpi-card:hover {
    border-color: #A855F7;
    transform: translateY(-2px);
}
.kpi-label {
    font-size: 0.76rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #A78BFA;
    margin-bottom: 6px;
}
.kpi-value {
    font-size: 1.4rem;
    font-weight: 700;
    color: #F5F3FF;
    margin-bottom: 6px;
    word-break: break-word;
}
.kpi-sub {
    font-size: 0.78rem;
    background: rgba(147, 51, 234, 0.16);
    border: 1px solid rgba(168, 85, 247, 0.22);
    display: inline-block;
    padding: 2px 8px;
    border-radius: 5px;
    color: #D8B4FE;
}

/* Contenedores y encabezados de seccion */
.section-panel {
    background: rgba(15, 11, 26, 0.85);
    border: 1px solid rgba(139, 92, 246, 0.24);
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 16px;
}
.section-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #F3E8FF;
    border-bottom: 1px solid rgba(139, 92, 246, 0.25);
    padding-bottom: 8px;
    margin-bottom: 12px;
    letter-spacing: 0.01em;
}

/* Alertas de validacion en HTML */
.status-ok {
    background: rgba(88, 28, 135, 0.25);
    border: 1px solid #9333EA;
    border-left: 4px solid #C084FC;
    color: #F3E8FF;
    padding: 12px 16px;
    border-radius: 8px;
    font-size: 0.9rem;
    font-weight: 600;
    margin-bottom: 14px;
}
.status-warn {
    background: rgba(59, 15, 76, 0.4);
    border: 1px solid rgba(216, 180, 254, 0.4);
    border-left: 4px solid #E879F9;
    color: #FAE8FF;
    padding: 10px 14px;
    border-radius: 8px;
    font-size: 0.88rem;
    margin-bottom: 10px;
}

/* Tarjetas de exportacion */
.export-card {
    background: linear-gradient(160deg, #130d21 0%, #090610 100%);
    border: 1px solid rgba(147, 51, 234, 0.3);
    border-radius: 12px;
    padding: 18px;
    margin-bottom: 10px;
}
.export-title {
    font-size: 1rem;
    font-weight: 700;
    color: #E9D5FF;
    margin-bottom: 4px;
}
.export-desc {
    font-size: 0.84rem;
    color: #A78BFA;
    margin-bottom: 4px;
}

/* Pestañas (Tabs) en morado sin subraya roja */
div[data-testid="stTabs"] button[role="tab"] {
    background: rgba(18, 12, 31, 0.75);
    color: #A78BFA !important;
    border: 1px solid rgba(139, 92, 246, 0.22);
    border-radius: 8px 8px 0 0;
    padding: 8px 18px;
    font-weight: 600;
    margin-right: 6px;
}
div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    background: linear-gradient(90deg, #6D28D9 0%, #9333EA 100%);
    color: #FFFFFF !important;
    border-color: #A855F7;
}
div[data-testid="stTabHighlight"] {
    background-color: #A855F7 !important;
}

/* Cargador de archivos oscuro y limpio */
section[data-testid="stFileUploader"] {
    background: rgba(17, 12, 29, 0.85);
    border: 1px dashed rgba(168, 85, 247, 0.48);
    border-radius: 12px;
    padding: 14px;
}
[data-testid="stFileUploaderDropzone"] {
    background-color: rgba(12, 8, 22, 0.9) !important;
    border: 1px dashed rgba(147, 51, 234, 0.45) !important;
    border-radius: 10px !important;
}
[data-testid="stFileUploaderDropzone"] button {
    background: linear-gradient(90deg, #5B21B6 0%, #7E22CE 100%) !important;
    color: #FAF5FF !important;
    border: 1px solid #A855F7 !important;
    border-radius: 8px !important;
}

/* Botones y Botones de descarga */
.stButton > button, .stDownloadButton > button {
    background: linear-gradient(90deg, #5B21B6 0%, #7E22CE 100%) !important;
    color: #FAF5FF !important;
    border: 1px solid #A855F7 !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 0.55rem 1.1rem !important;
    box-shadow: 0 4px 14px rgba(124, 58, 237, 0.35);
    transition: all 0.2s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    background: linear-gradient(90deg, #6D28D9 0%, #9333EA 100%) !important;
    border-color: #C084FC !important;
    box-shadow: 0 6px 20px rgba(168, 85, 247, 0.55);
}

/* Areas de texto, selectores y expanders */
.stTextArea textarea, div[data-baseweb="select"] > div {
    background-color: #0e0a18 !important;
    color: #F3E8FF !important;
    border: 1px solid rgba(139, 92, 246, 0.38) !important;
    border-radius: 8px !important;
}
details[data-testid="stExpander"] {
    background: rgba(15, 10, 26, 0.85);
    border: 1px solid rgba(139, 92, 246, 0.28);
    border-radius: 10px;
}
details[data-testid="stExpander"] summary {
    padding: 10px 14px;
}
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Funciones auxiliares y procesamiento en cache
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False, max_entries=64)
def process_file(data: bytes, name: str, lang: str, psm: int) -> list[dict]:
    """Procesa un archivo (imagen o PDF) internamente y devuelve los datos listos para el usuario."""
    cfg = PreprocessConfig()
    is_pdf = name.lower().endswith(".pdf")
    if psm == 0:  # Auto: PDF A4 usa segmentacion 3; fotos de tickets usan 4
        psm = 3 if is_pdf else 4
        if is_pdf:
            cfg.target_width = max(cfg.target_width, 2000)

    pages = load_file(data, name)
    previews: dict[str, bytes] = {}
    texts: list[str] = []
    confs: list[float] = []
    words = 0

    for p, img in enumerate(pages, start=1):
        pp = preprocess(img, cfg)
        ocr = run_ocr(pp.final, lang=lang, psm=psm)
        page_label = "Documento original" if len(pages) == 1 else f"Página {p}"
        previews[page_label] = to_png_bytes(img)
        texts.append(ocr.text)
        confs.append(ocr.confidence)
        words += ocr.n_words

    text = "\n".join(texts)
    return [
        {
            "archivo": name if len(pages) == 1 else f"{name} ({len(pages)} págs.)",
            "previews": previews,
            "ocr_text": text,
            "ocr_conf": round(sum(confs) / len(confs), 1) if confs else 0.0,
            "ocr_words": words,
            "analysis": analyze(text),
        }
    ]


def wordcloud_png(tokens: list[str]) -> bytes | None:
    if not tokens:
        return None
    from wordcloud import WordCloud

    wc = WordCloud(
        width=900,
        height=420,
        background_color="#0B0713",
        colormap="Purples",
        collocations=False,
        random_state=1,
    ).generate(" ".join(tokens))
    buf = io.BytesIO()
    wc.to_image().save(buf, format="PNG")
    return buf.getvalue()


def fmt_money(v: float | None, moneda: str | None = None) -> str:
    if v is None:
        return "—"
    simbolo = "$" if moneda and "USD" in moneda.upper() else "S/"
    return f"{simbolo} {v:,.2f}"


def yes_no(v: bool | None) -> str:
    return "—" if v is None else ("Sí" if v else "No")


def render_kpi_row(items: list[tuple[str, str, str]]) -> None:
    """Renderiza tarjetas KPI en una sola linea HTML sin sangria de 4 espacios."""
    cards_html = "".join(
        f'<div class="kpi-card">'
        f'<div class="kpi-label">{html.escape(str(label))}</div>'
        f'<div class="kpi-value">{html.escape(str(value))}</div>'
        f'<div class="kpi-sub">{html.escape(str(sub))}</div>'
        f"</div>"
        for label, value, sub in items
    )
    st.markdown(f'<div class="kpi-grid">{cards_html}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Barra lateral: carga rapida y ajustes de lectura
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        '<div class="section-panel">'
        '<div class="section-title">Panel de Control</div>'
        '<p style="font-size: 0.86rem; color: #C4B5FD; margin: 0;">'
        "Carga tus comprobantes de pago para extraer automaticamente los datos "
        "SUNAT, validar montos y clasificar tus gastos."
        "</p></div>",
        unsafe_allow_html=True,
    )

    use_samples = st.toggle(
        "Usar boletas de ejemplo",
        value=False,
        help="Carga comprobantes de prueba para explorar la interfaz.",
    )

    with st.expander("Ajustes de lectura (opcional)", expanded=False):
        langs = available_langs()
        lang_options = [
            l for l in ["spa", "spa+eng", "eng"] if l.split("+")[0] in langs
        ] or ["eng"]
        lang = st.selectbox(
            "Idioma del documento",
            options=lang_options,
            format_func=lambda code: LANG_LABELS.get(code, code),
        )
        psm = st.selectbox(
            "Tipo de diseño del documento",
            options=list(PSM_OPTIONS),
            format_func=PSM_OPTIONS.get,
        )


# ---------------------------------------------------------------------------
# Encabezado HTML y carga de archivos
# ---------------------------------------------------------------------------
st.markdown(
    '<div class="hero-banner">'
    '<div class="hero-badge">Sistema Inteligente OCR + NLP</div>'
    '<h1 class="hero-title">Lector y Analizador de Comprobantes de Pago</h1>'
    '<p class="hero-subtitle">'
    "Sube fotos o archivos PDF de tus boletas y facturas. El sistema identifica RUC, "
    "serie, fecha, IGV e importe total, verifica la validez tributaria y clasifica "
    "automaticamente cada gasto."
    "</p></div>",
    unsafe_allow_html=True,
)

uploads = st.file_uploader(
    "Arrastra o selecciona tus imágenes (JPG, PNG) o documentos PDF",
    type=["jpg", "jpeg", "png", "pdf"],
    accept_multiple_files=True,
)

files: list[tuple[str, bytes]] = [(u.name, u.getvalue()) for u in (uploads or [])]
if (use_samples or not files) and SAMPLES_DIR.exists():
    if use_samples:
        files += [(p.name, p.read_bytes()) for p in sorted(SAMPLES_DIR.glob("*.png"))]

if not files:
    st.markdown(
        '<div class="section-panel">'
        '<div class="section-title">Esperando documentos</div>'
        '<p style="margin: 0; color: #D8B4FE; font-size: 0.92rem;">'
        "Sube al menos una boleta o factura (JPG, PNG o PDF) en el recuadro superior, "
        "o activa la opción <strong>Usar boletas de ejemplo</strong> en el panel lateral para comenzar."
        "</p></div>",
        unsafe_allow_html=True,
    )
    st.stop()

docs: list[dict] = []
progress = st.progress(0.0, text="Analizando comprobantes...")
for i, (name, data) in enumerate(files):
    progress.progress(i / len(files), text=f"Analizando {name}...")
    docs.extend(process_file(data, name, lang, psm))
progress.empty()

# Palabras caracteristicas por comprobante (TF-IDF)
tfidf = tfidf_keywords([d["analysis"]["tokens_filtrados"] for d in docs], k=5)
for d, kw in zip(docs, tfidf):
    d["tfidf"] = kw

# Tabla resumen sin columnas tecnicas de preprocesamiento ni comparacion
rows = []
for d in docs:
    c = d["analysis"]["comprobante"]
    cl = d["analysis"]["clasificacion"]
    rows.append(
        {
            "Archivo": d["archivo"],
            "Tipo": c.tipo or "No identificado",
            "Serie - Número": f"{c.serie}-{c.numero}" if c.serie else "—",
            "RUC Emisor": c.ruc_emisor or "—",
            "RUC Válido": yes_no(c.ruc_emisor_valido),
            "Razón Social": c.razon_social or "—",
            "Fecha": c.fecha_emision or "—",
            "Moneda": c.moneda or "PEN",
            "Op. Gravada": c.op_gravada,
            "IGV": c.igv,
            "Total": c.total,
            "IGV Correcto (18%)": yes_no(c.igv_cuadra),
            "Categoría": cl["categoria"],
            "Confianza Categoría": cl["confianza"],
            "Confianza Lectura (%)": d["ocr_conf"],
            "Campos Detectados": c.campos_encontrados,
            "Palabras Clave": ", ".join(w for w, _ in d["tfidf"]),
            "Observaciones": "; ".join(c.alertas) if c.alertas else "Sin observaciones",
        }
    )
df = pd.DataFrame(rows)

tab_doc, tab_lote, tab_desc = st.tabs(
    [
        "Detalle por comprobante",
        "Resumen de gastos",
        "Exportar datos",
    ]
)

# ---------------------------------------------------------------------------
# Pestaña 1: vista de usuario para un comprobante individual
# ---------------------------------------------------------------------------
with tab_doc:
    sel = st.selectbox(
        "Selecciona un comprobante para revisar sus detalles",
        range(len(docs)),
        format_func=lambda i: docs[i]["archivo"],
    )
    d = docs[sel]
    a = d["analysis"]
    c = a["comprobante"]
    cl = a["clasificacion"]

    render_kpi_row(
        [
            (
                "Importe Total",
                fmt_money(c.total, c.moneda),
                f"Moneda: {c.moneda or 'PEN'}",
            ),
            (
                "Categoría de Gasto",
                cl["categoria"],
                f"Afinidad: {cl['confianza']:.0%}",
            ),
            (
                "Comprobante",
                c.tipo or "Comprobante",
                f"{c.serie}-{c.numero}" if c.serie else "Sin serie detectada",
            ),
            (
                "Campos Identificados",
                f"{c.campos_encontrados} / 7",
                f"Precisión de lectura: {d['ocr_conf']:.1f}%",
            ),
        ]
    )

    col_preview, col_info = st.columns([1, 1.35], gap="large")

    # Columna izquierda: vista previa del documento subido
    with col_preview:
        st.markdown(
            '<div class="section-title">Vista del Documento</div>',
            unsafe_allow_html=True,
        )
        preview_items = list(d["previews"].items())
        if len(preview_items) == 1:
            _, png_bytes = preview_items[0]
            st.image(png_bytes, width="stretch")
        else:
            page_tabs = st.tabs([label for label, _ in preview_items])
            for p_tab, (_, png_bytes) in zip(page_tabs, preview_items):
                with p_tab:
                    st.image(png_bytes, width="stretch")

    # Columna derecha: informacion estructurada y pestañas de detalle
    with col_info:
        sub_datos, sub_nlp, sub_texto = st.tabs(
            [
                "Datos SUNAT y Validación",
                "Clasificación y Palabras Clave",
                "Texto Reconocido",
            ]
        )

        with sub_datos:
            if not c.alertas:
                st.markdown(
                    '<div class="status-ok">Comprobante validado sin observaciones en RUC ni IGV.</div>',
                    unsafe_allow_html=True,
                )
            else:
                for al in c.alertas:
                    st.markdown(
                        f'<div class="status-warn">Observación: {html.escape(str(al))}</div>',
                        unsafe_allow_html=True,
                    )

            ent = pd.DataFrame(
                [
                    ("Tipo de comprobante", c.tipo),
                    ("Serie y número", f"{c.serie}-{c.numero}" if c.serie else None),
                    ("RUC del emisor", c.ruc_emisor),
                    ("RUC válido (SUNAT)", yes_no(c.ruc_emisor_valido)),
                    ("Razón social", c.razon_social),
                    ("RUC del cliente", c.ruc_cliente),
                    ("DNI del cliente", c.dni_cliente),
                    ("Fecha de emisión", c.fecha_emision),
                    ("Moneda", c.moneda),
                    ("Operación gravada", fmt_money(c.op_gravada, c.moneda)),
                    ("IGV (18%)", fmt_money(c.igv, c.moneda)),
                    ("Importe total", fmt_money(c.total, c.moneda)),
                    ("IGV cuadra con el 18%", yes_no(c.igv_cuadra)),
                ],
                columns=["Campo", "Valor"],
            )
            ent["Valor"] = ent["Valor"].fillna("—").astype(str)
            st.dataframe(ent, hide_index=True, width="stretch")

        with sub_nlp:
            st.markdown(
                '<div class="section-title">Afinidad por Categoría de Gasto</div>',
                unsafe_allow_html=True,
            )
            sc = pd.DataFrame(
                {
                    "Categoría": list(cl["puntajes"]),
                    "Coincidencias": list(cl["puntajes"].values()),
                }
            )
            sc = sc[sc["Coincidencias"] > 0].sort_values("Coincidencias")
            if len(sc):
                st.bar_chart(
                    sc,
                    x="Categoría",
                    y="Coincidencias",
                    horizontal=True,
                    height=220,
                    color="#9333EA",
                )
                st.caption(
                    "Términos detectados en el comprobante: "
                    + ", ".join(cl["evidencia"])
                )
            else:
                st.caption(
                    "No se encontraron términos específicos; clasificado como «Otros»."
                )

            st.markdown(
                '<div class="section-title">Términos Destacados del Documento</div>',
                unsafe_allow_html=True,
            )
            kc1, kc2 = st.columns(2)
            kc1.dataframe(
                pd.DataFrame(
                    a["keywords"], columns=["Palabra frecuente", "Repeticiones"]
                ),
                hide_index=True,
                width="stretch",
            )
            kc2.dataframe(
                pd.DataFrame(d["tfidf"], columns=["Término relevante", "Peso TF-IDF"]),
                hide_index=True,
                width="stretch",
            )

            png = wordcloud_png(a["tokens_filtrados"])
            if png:
                st.markdown(
                    '<div class="section-title">Nube de Palabras</div>',
                    unsafe_allow_html=True,
                )
                st.image(png, width="stretch")

        with sub_texto:
            st.text_area(
                "Texto completo leído del comprobante",
                value=d["ocr_text"] or "(sin texto detectado)",
                height=340,
            )
            with st.expander("Ver texto limpio y tokens procesados"):
                st.markdown("**Texto normalizado**")
                st.code(a["normalizado"] or "—", language=None)
                st.markdown("**Texto limpio (sin tildes ni símbolos)**")
                st.code(a["limpio"] or "—", language=None)
                st.markdown(
                    f"**Tokens totales:** {len(a['tokens'])} · "
                    f"**Palabras útiles (sin stopwords):** {len(a['tokens_filtrados'])}"
                )
                st.write(" · ".join(a["tokens_filtrados"]) or "—")

# ---------------------------------------------------------------------------
# Pestaña 2: resumen general de todos los comprobantes cargados
# ---------------------------------------------------------------------------
with tab_lote:
    valid_count = sum(1 for d in docs if not d["analysis"]["comprobante"].alertas)
    render_kpi_row(
        [
            (
                "Comprobantes Procesados",
                str(len(df)),
                "Lote actual",
            ),
            (
                "Gasto Total Acumulado",
                fmt_money(df["Total"].sum(min_count=1)),
                "Suma de importes detectados",
            ),
            (
                "Campos SUNAT Detectados",
                f"{df['Campos Detectados'].mean():.1f} / 7",
                "Promedio por documento",
            ),
            (
                "Comprobantes Validados",
                f"{valid_count} de {len(docs)}",
                "Sin observaciones tributarias",
            ),
        ]
    )

    g1, g2 = st.columns(2)
    with g1:
        st.markdown(
            '<div class="section-title">Gasto Acumulado por Categoría (S/)</div>',
            unsafe_allow_html=True,
        )
        by_cat = (
            df.groupby("Categoría", as_index=False)["Total"]
            .sum()
            .sort_values("Total")
        )
        st.bar_chart(
            by_cat,
            x="Categoría",
            y="Total",
            horizontal=True,
            height=260,
            color="#A855F7",
        )

    with g2:
        st.markdown(
            '<div class="section-title">Importe Total por Comprobante (S/)</div>',
            unsafe_allow_html=True,
        )
        by_doc = df[["Archivo", "Total"]].fillna(0).sort_values("Total")
        st.bar_chart(
            by_doc,
            x="Archivo",
            y="Total",
            horizontal=True,
            height=260,
            color="#7C3AED",
        )

    st.markdown(
        '<div class="section-title">Listado Consolidado de Comprobantes</div>',
        unsafe_allow_html=True,
    )
    st.dataframe(df, hide_index=True, width="stretch")

    all_tokens = [t for d in docs for t in d["analysis"]["tokens_filtrados"]]
    png = wordcloud_png(all_tokens)
    if png:
        st.markdown(
            '<div class="section-title">Nube de Palabras del Lote</div>',
            unsafe_allow_html=True,
        )
        st.image(png, width="stretch")

# ---------------------------------------------------------------------------
# Pestaña 3: exportacion y descargas
# ---------------------------------------------------------------------------
with tab_desc:
    st.markdown(
        '<div class="section-panel">'
        '<div class="section-title">Descargar Reportes y Resultados</div>'
        '<p style="margin: 0; color: #C4B5FD; font-size: 0.9rem;">'
        "Exporta la información extraída en el formato que necesites para tu contabilidad o registro de gastos."
        "</p></div>",
        unsafe_allow_html=True,
    )

    dl1, dl2, dl3 = st.columns(3)

    with dl1:
        st.markdown(
            '<div class="export-card">'
            '<div class="export-title">Hoja de Cálculo (CSV)</div>'
            '<div class="export-desc">Tabla compatible con Excel con todos los campos SUNAT y montos.</div>'
            "</div>",
            unsafe_allow_html=True,
        )
        st.download_button(
            "Descargar CSV",
            df.to_csv(index=False).encode("utf-8-sig"),
            "comprobantes.csv",
            "text/csv",
            width="stretch",
        )

    with dl2:
        st.markdown(
            '<div class="export-card">'
            '<div class="export-title">Datos Estructurados (JSON)</div>'
            '<div class="export-desc">Detalle completo de entidades, clasificación y palabras clave.</div>'
            "</div>",
            unsafe_allow_html=True,
        )
        full = []
        for d in docs:
            a = d["analysis"]
            full.append(
                {
                    "archivo": d["archivo"],
                    "texto_ocr": d["ocr_text"],
                    "texto_limpio": a["limpio"],
                    "tokens_filtrados": a["tokens_filtrados"],
                    "entidades": a["comprobante"].to_dict(),
                    "clasificacion": {k: v for k, v in a["clasificacion"].items()},
                    "palabras_clave": a["keywords"],
                    "tfidf": d["tfidf"],
                    "confianza_ocr": d["ocr_conf"],
                }
            )
        st.download_button(
            "Descargar JSON",
            json.dumps(full, ensure_ascii=False, indent=2).encode("utf-8"),
            "resultados.json",
            "application/json",
            width="stretch",
        )

    with dl3:
        st.markdown(
            '<div class="export-card">'
            '<div class="export-title">Textos Extraídos (ZIP)</div>'
            '<div class="export-desc">Archivos .txt individuales con el texto original y limpio.</div>'
            "</div>",
            unsafe_allow_html=True,
        )
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for d in docs:
                stem = Path(d["archivo"]).stem.replace(" ", "_")
                zf.writestr(f"{stem}_ocr.txt", d["ocr_text"])
                zf.writestr(f"{stem}_limpio.txt", d["analysis"]["limpio"])
        st.download_button(
            "Descargar ZIP",
            buf.getvalue(),
            "textos.zip",
            "application/zip",
            width="stretch",
        )

    st.write("")
    with st.expander("Ver diccionario de categorías y palabras clave"):
        st.json(CATEGORIAS)