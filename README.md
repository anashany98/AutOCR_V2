# AutOCR v2 - Cascaded OCR + Layout Pipeline

AutOCR automatiza el procesamiento posterior al escaneo en Windows combinando deteccion de layout (PaddleOCR), un OCR hibrido PaddleOCR + EasyOCR, extraccion de tablas, embeddings visuales CLIP y un panel web.

## Caracteristicas principales

- **Pipeline modular GPU/CPU**: deteccion de bloques, OCR primario PaddleOCR y fallback EasyOCR con fusion configurable (confianza, cascada o Levenshtein).
- **Tablas estructuradas**: exportacion automatica de CSV y JSON por tabla detectada.
- **Embeddings visuales**: CLIP + FAISS para busqueda de imagenes similares.
- **Dashboard Flask**: metricas de procesamiento, descarga de tablas, busqueda visual y configuracion en caliente.
- **Batch y Web**: motor CLI `postbatch_processor.py` y subida directa desde la interfaz.
- **Modelos PP-OCRv4 listos**: el pipeline descarga y usa automaticamente los pesos PP-OCRv4 (deteccion/rec) para idiomas latinos.

## Requisitos

1. Python 3.10+
2. Dependencias Python: `pip install -r requirements.txt`
3. (Opcional) CUDA para `paddlepaddle-gpu` y PyTorch GPU.

## Configuracion rapida

1. Copia `config.yaml` y actualiza rutas en la seccion `postbatch`.
2. Ajusta el bloque `ocr_pipeline` para definir motores, idiomas y estrategia de fusion.
3. Ejecuta un lote manual:
   ```bash
   python postbatch_processor.py --immediate
   ```
4. Inicia la web:
   ```bash
   python web_app/app.py
   ```
   Luego abre `http://localhost:5000`.

## GPU Setup

- Instala `paddlepaddle-gpu` compatible con tu version de CUDA (`https://www.paddlepaddle.org.cn/install/quick`).
- Instala PyTorch GPU con el indice oficial (`https://download.pytorch.org/whl/cu118`).
- En `/settings` activa la casilla GPU o edita `config.yaml` para definir `ocr_pipeline.engines[].gpu: true`.
- Si no hay GPU disponible, ambos motores se ejecutaran en CPU automaticamente.

### PP-OCRv4

- Los pesos PP-OCRv4 (det/rec/cls) se almacenan en `models/paddle/ppocrv4`. Por defecto se descargan la primera vez que corras el pipeline.
- Personaliza rutas o perfiles (p. ej. `model_profile`) en el bloque `ocr_pipeline.engines[].model_*` del `config.yaml` si necesitas otras variantes.

## Demo rapida

Genera muestras sinteticas (1 PDF con tabla + 1 imagen factura) y procesa el flujo completo:
```bash
make demo
```
Los resultados se ubican en `samples/demo_processed`.

## Tests

Los tests unitarios cubren fusion de OCR, fallback de bloques, exportacion de tablas y la construccion de indice visual:
```bash
make test
```

## Dashboard

- **Tablas**: pestana dedicada con enlaces a CSV/JSON por documento.
- **Busqueda de imagenes**: sube una foto de producto y obtiene las mas similares (requiere indice FAISS construido desde la configuracion).
- **Configuracion**: modifica directorios, idiomas, GPU y reconstruye el indice visual sin salir de la web.

## Estructura relevante

```
modules/
  fusion_manager.py
  layout_manager.py
  ocr_manager.py
  table_manager.py
  vision_manager.py
postbatch_processor.py
web_app/
  app.py
  templates/
scripts/run_demo.py
samples/
  README.md
```

## Resolucion de problemas

- **"EasyOCR not found"**: instala EasyOCR (`pip install easyocr`) y asegura que PyTorch este disponible.
- **Problemas con PaddleOCR GPU**: verifica version de CUDA o instala `paddlepaddle` (CPU) como alternativa.
- **Busqueda visual sin resultados**: reconstruye el indice desde la configuracion y rellena el directorio de galeria.

## Licencia

MIT. Contribuciones via Pull Requests son bienvenidas.
