"""
Evalúa el pipeline sobre una carpeta de imágenes y guarda una tabla con la
confianza del OCR (sin y con preprocesamiento), los campos extraídos y el tiempo.

Uso:
    python tools/evaluar.py data/raw            # tus fotos reales
    python tools/evaluar.py samples             # boletas sintéticas
Salida: resultados/evaluacion.csv
"""
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.nlp import analyze  # noqa: E402
from src.ocr import run_ocr  # noqa: E402
from src.preprocess import PreprocessConfig, load_file, preprocess  # noqa: E402

carpeta = Path(sys.argv[1] if len(sys.argv) > 1 else "data/raw")
archivos = sorted(p for p in carpeta.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".pdf"})
if not archivos:
    sys.exit(f"No hay imágenes en {carpeta}")

filas = []
for f in archivos:
    t0 = time.time()
    pages = load_file(f.read_bytes(), f.name)
    pdf = f.suffix.lower() == ".pdf"
    cfg = PreprocessConfig(target_width=2000) if pdf else PreprocessConfig()
    psm = 3 if pdf else 4  # mismo criterio que la opción "Auto" de la app
    raw_txt, txt, raw_c, conf, ang = [], [], [], [], 0.0
    for img in pages:  # un comprobante de varias páginas se analiza junto
        r = run_ocr(img, psm=psm)
        pp = preprocess(img, cfg)
        o = run_ocr(pp.final, psm=psm)
        raw_txt.append(r.text); txt.append(o.text)
        raw_c.append(r.confidence); conf.append(o.confidence)
        ang = ang or pp.angle
    a_raw = analyze("\n".join(raw_txt))
    a = analyze("\n".join(txt))
    c = a["comprobante"]
    filas.append({
        "archivo": f.name, "formato": "PDF" if pdf else "imagen", "paginas": len(pages),
        "conf_sin_prep": round(sum(raw_c) / len(raw_c), 1), "conf_con_prep": round(sum(conf) / len(conf), 1),
        "campos_sin_prep": a_raw["comprobante"].campos_encontrados,
        "campos_con_prep": c.campos_encontrados,
        "tipo": c.tipo, "serie_numero": f"{c.serie}-{c.numero}" if c.serie else None,
        "ruc_emisor": c.ruc_emisor, "ruc_valido": c.ruc_emisor_valido, "razon_social": c.razon_social,
        "fecha": c.fecha_emision, "op_gravada": c.op_gravada, "igv": c.igv, "total": c.total,
        "igv_cuadra": c.igv_cuadra, "categoria": a["clasificacion"]["categoria"],
        "alertas": "; ".join(c.alertas),
        "angulo_corregido": round(ang, 2), "segundos": round(time.time() - t0, 1),
    })
    print(f"{f.name}: conf {filas[-1]['conf_sin_prep']} -> {filas[-1]['conf_con_prep']} | "
          f"campos {filas[-1]['campos_sin_prep']} -> {c.campos_encontrados} | {filas[-1]['categoria']}")

df = pd.DataFrame(filas)
Path("resultados").mkdir(exist_ok=True)
df.to_csv("resultados/evaluacion.csv", index=False, encoding="utf-8-sig")
print("\nPromedios")
print(df[["conf_sin_prep", "conf_con_prep", "campos_sin_prep", "campos_con_prep"]].mean().round(2).to_string())
print(f"RUC válido: {df['ruc_valido'].mean():.0%} | IGV cuadra: {df['igv_cuadra'].dropna().mean():.0%}")
print("\nPor formato")
print(df.groupby("formato")[["conf_sin_prep", "conf_con_prep", "campos_sin_prep", "campos_con_prep"]].mean().round(2).to_string())
print("Guardado en resultados/evaluacion.csv")
