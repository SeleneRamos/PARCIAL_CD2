# OCR y NLP de boletas y facturas peruanas

Aplicación en Streamlit que lee fotos o PDF de comprobantes de pago peruanos (boletas, facturas, tickets), extrae el texto con OCR y lo analiza con técnicas de NLP para obtener los datos que pide la SUNAT y clasificar el tipo de gasto.

- **App desplegada:** https://ocr-boletas.streamlit.app
- **Notebook de desarrollo (Colab):** `OCR_NLP_boletas_peru.ipynb`
- **Presentación:** `OCR y NLP de boletas peruanas.pptx`

**Problema:** en el Perú muchas personas y pymes registran sus gastos a mano a partir de boletas impresas en papel térmico, que se borran con el tiempo, o de PDF con formatos distintos según la empresa. Copiar el RUC, la fecha, el IGV y el total de cada comprobante toma tiempo y genera errores al deducir gastos o al hacer la conciliación contable.

## Qué hace

| Etapa | Técnica |
|---|---|
| Carga | JPG, PNG y PDF (convertidos a 300 dpi con PyMuPDF). Un PDF de varias páginas se analiza como un solo comprobante |
| Preprocesamiento | Redimensionamiento, conversión a escala de grises, eliminación de ruido (non-local means, mediana o gaussiano), mejora de contraste (CLAHE), corrección de inclinación por perfil de proyección y binarización (adaptativa u Otsu) |
| OCR | Tesseract 5 con pytesseract, modelo en español y motor LSTM. Modo de segmentación automático: PSM 4 para fotos de tickets y PSM 3 para PDF A4. Confianza promedio por palabra |
| Limpieza del texto | Corrección de espacios y símbolos del OCR (O→0, 8001→B001, `S /`→`S/`), conversión a minúsculas, eliminación de tildes y caracteres especiales, tokenización y eliminación de stopwords (92 del español y 113 propias de comprobantes) |
| NLP | Extracción de entidades por reglas (tipo, serie-número, RUC, razón social, DNI, fecha, moneda, op. gravada, IGV y total), también en formato tabla A4. Validación del RUC (dígito verificador módulo 11) y del IGV (18 %). Clasificación temática en 13 categorías de gasto, con doble peso para las líneas de productos y corrección aproximada (difflib). Palabras clave por frecuencia y TF-IDF. Nube de palabras |
| App | Carga múltiple, vista de cada paso, texto sin y con preprocesamiento, texto limpio, resultados NLP con alertas, resumen del lote y descargas en CSV, JSON y ZIP |

## Estructura

```
├── app.py                         # Aplicación Streamlit
├── OCR_NLP_boletas_peru.ipynb     # Notebook de desarrollo (Google Colab)
├── src/
│   ├── preprocess.py              # Preprocesamiento de imágenes
│   ├── ocr.py                     # OCR con Tesseract
│   └── nlp.py                     # Limpieza de texto, entidades, clasificación, palabras clave
├── tools/
│   ├── evaluar.py                 # Evaluación por lotes -> resultados/evaluacion.csv
│   └── generar_muestras.py        # Genera boletas sintéticas de prueba
├── data/raw/                      # Boletas reales del dataset (no se suben, tienen datos personales)
├── samples/                       # 6 boletas sintéticas (datos ficticios)
├── requirements.txt               # Librerías de Python
└── packages.txt                   # Tesseract para Streamlit Cloud
```

## Ejecutar en local

1. Instalar **Tesseract** con el idioma español:
   - Windows: https://github.com/UB-Mannheim/tesseract/wiki (en *Additional language data*, marcar **Spanish**)
   - Ubuntu: `sudo apt install tesseract-ocr tesseract-ocr-spa`
   - macOS: `brew install tesseract tesseract-lang`
2. Crear el entorno e instalar las librerías (se recomienda Python 3.12):

```bash
python -m venv .venv
# Windows (PowerShell):
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

En Windows, `src/ocr.py` busca Tesseract solo en las rutas de instalación habituales. Si aun así sale `TesseractNotFoundError`, revisa que exista `C:\Program Files\Tesseract-OCR\tesseract.exe`.

Evita comas y caracteres especiales en la ruta de la carpeta del proyecto (por ejemplo `10,mo ciclo`), porque PowerShell no activa bien el entorno.

## Notebook en Google Colab

Abre `OCR_NLP_boletas_peru.ipynb` en https://colab.research.google.com y usa **Entorno de ejecución → Ejecutar todas**. La primera celda instala Tesseract y clona este repositorio. Para analizar tus propias boletas, cambia `SUBIR_ARCHIVOS = True`.

## Desplegar en Streamlit Community Cloud

1. Sube esta carpeta a un repositorio público de GitHub.
2. Entra a https://share.streamlit.io → **Create app** → elige el repositorio, la rama `main` y el archivo `app.py`. En *Advanced settings* elige Python 3.12.
3. `packages.txt` instala Tesseract y el paquete de español automáticamente.

## Dataset

| Fuente | Cantidad | Formato |
|---|---|---|
| Boletas electrónicas reales (Tiendas Aruma, Yape Market, Mifarma, Promotick, Fundación para el Desarrollo Agrario) | 9 | PDF |
| Boletas sintéticas con ruido, sombra, bajo contraste e inclinación | 6 | PNG |

Las boletas reales tienen nombre y DNI, por eso no están en el repositorio. Para evaluar las tuyas, ponlas en `data/raw/` y corre:

```bash
python tools/evaluar.py data/raw
```

## Resultados

| Medida | Fotos sintéticas (6) | PDF reales (9) |
|---|---|---|
| Confianza OCR sin / con preprocesamiento | 85.9 % / 91.5 % | 90.5 % / 89.6 % |
| Campos extraídos sin / con preprocesamiento (de 7) | 3.7 / 7.0 | 7.0 / 7.0 |
| RUC con dígito verificador válido | 6 de 6 | 9 de 9 |
| Categoría de gasto correcta | 6 de 6 | 8 de 9 |

El preprocesamiento es necesario en las fotos y no cambia el resultado en los PDF, que ya vienen limpios. La configuración por defecto (binarización adaptativa, bloque 41, C = 10) salió de comparar seis configuraciones sobre las boletas sintéticas; es la única que extrajo los 7 campos en las 6.

## Limitaciones y mejoras

- La extracción usa reglas: cada formato nuevo de PDF (tabla A4, varias páginas) obligó a ajustar patrones.
- La clasificación depende de que la boleta describa el producto. Un cupón cuya descripción es un correo electrónico queda en «Otros».
- Tesseract confunde letras parecidas («LG.V» por «I.G.V.») y no lee letra a mano.
- Mejoras: leer directamente el texto que traen los PDF electrónicos, leer el código QR de la boleta, comparar con EasyOCR o PaddleOCR y entrenar un clasificador con más boletas etiquetadas.
