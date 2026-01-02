# AutOCR - Interfaz Web

## DescripciÃ³n

La interfaz web de AutOCR proporciona una forma intuitiva de gestionar documentos OCR, visualizar estadÃ­sticas y controlar el procesamiento batch desde un navegador web.

## CaracterÃ­sticas

### Dashboard
- **EstadÃ­sticas generales**: Total de documentos, procesados, fallidos y duplicados
- **GrÃ¡ficos de tipos**: VisualizaciÃ³n de distribuciÃ³n de tipos de documento
- **Documentos recientes**: Lista de los Ãºltimos documentos procesados
- **MÃ©tricas de rendimiento**: Historial de tiempos de procesamiento y fiabilidad

### GestiÃ³n de Documentos
- **Lista paginada**: NavegaciÃ³n eficiente por todos los documentos
- **Filtros avanzados**: Por estado, tipo y bÃºsqueda de texto
- **Vista detallada**: InformaciÃ³n completa de cada documento incluyendo texto OCR
- **Funcionalidad de bÃºsqueda**: BÃºsqueda en tiempo real

### Subida de Archivos
- **Subida mÃºltiple**: Procesar varios archivos simultÃ¡neamente
- **ValidaciÃ³n**: Control de tipos y tamaÃ±os de archivo
- **Procesamiento automÃ¡tico**: OCR y clasificaciÃ³n inmediata
- **Feedback visual**: Indicadores de progreso y resultados

### Procesamiento Batch
- **EjecuciÃ³n manual**: Trigger para procesar archivos en cola
- **IntegraciÃ³n completa**: Usa la misma lÃ³gica que el procesador CLI

## InstalaciÃ³n

1. **Instalar dependencias**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Verificar configuraciÃ³n**:
   AsegÃºrate de que `config.yaml` estÃ© correctamente configurado, especialmente las rutas de base de datos y carpetas.

## Uso

### Iniciar la aplicaciÃ³n web

```bash
python run_web.py
```

O directamente:

```bash
python -m web_app.app
```

La aplicaciÃ³n estarÃ¡ disponible en: `http://localhost:5000`

### NavegaciÃ³n

- **Dashboard** (`/`): Vista general y estadÃ­sticas
- **Documentos** (`/documents`): Lista y bÃºsqueda de documentos
- **Subir** (`/upload`): Subida y procesamiento de nuevos archivos
- **Procesamiento Batch** (`/batch_process`): Ejecutar procesamiento automÃ¡tico

## Estructura de Archivos

```
web_app/
â”œâ”€â”€ app.py                 # AplicaciÃ³n Flask principal
â”œâ”€â”€ templates/             # Plantillas HTML
â”‚   â”œâ”€â”€ base.html         # Plantilla base
â”‚   â”œâ”€â”€ dashboard.html    # Dashboard principal
â”‚   â”œâ”€â”€ documents.html    # Lista de documentos
â”‚   â”œâ”€â”€ document_detail.html # Vista detallada
â”‚   â””â”€â”€ upload.html       # Formulario de subida
â”œâ”€â”€ static/               # Archivos estÃ¡ticos
â”‚   â”œâ”€â”€ css/
â”‚   â”‚   â””â”€â”€ style.css     # Estilos personalizados
â”‚   â”œâ”€â”€ js/
â”‚   â”‚   â””â”€â”€ main.js       # JavaScript del cliente
â”‚   â””â”€â”€ uploads/          # Archivos subidos temporalmente
â””â”€â”€ __init__.py
```

## API Endpoints

### GET Routes
- `GET /` - Dashboard
- `GET /documents` - Lista de documentos (con filtros)
- `GET /document/<id>` - Detalle de documento
- `GET /upload` - Formulario de subida
- `GET /batch_process` - Trigger procesamiento batch

### POST Routes
- `POST /upload` - Procesar archivos subidos

## CaracterÃ­sticas TÃ©cnicas

### TecnologÃ­as
- **Backend**: Flask (Python)
- **Frontend**: Bootstrap 5, JavaScript vanilla
- **Base de datos**: SQLite (por defecto) o SQL Server
- **OCR**: PaddleOCR/EasyOCR
- **GrÃ¡ficos**: Chart.js (planeado)

### Seguridad
- ValidaciÃ³n de tipos de archivo
- LÃ­mites de tamaÃ±o de archivo (50MB)
- SanitizaciÃ³n de nombres de archivo
- ProtecciÃ³n CSRF bÃ¡sica

### Rendimiento
- Procesamiento paralelo de archivos
- PaginaciÃ³n de resultados
- OptimizaciÃ³n de consultas a BD
- CachÃ© de plantillas

## ConfiguraciÃ³n

La aplicaciÃ³n web usa la misma configuraciÃ³n que el procesador CLI (`config.yaml`). ParÃ¡metros relevantes:

```yaml
postbatch:
  max_workers: 4          # Workers para procesamiento paralelo
  ocr_enabled: true       # Habilitar OCR
  classification_enabled: true # Habilitar clasificaciÃ³n

app:
  db_path: "data/digitalizerai.db" # Ruta a la base de datos
  log_level: "INFO"       # Nivel de logging
```

## Desarrollo

### Ejecutar en modo debug

```bash
export FLASK_ENV=development
python run_web.py
```

### AÃ±adir nuevas funcionalidades

1. Crear nueva ruta en `app.py`
2. Crear plantilla HTML en `templates/`
3. AÃ±adir estilos en `static/css/style.css`
4. AÃ±adir JavaScript en `static/js/main.js`

## SoluciÃ³n de Problemas

### Error de conexiÃ³n a base de datos
- Verificar que la ruta en `config.yaml` sea correcta
- Asegurarse de que el directorio `data/` existe

### Error de OCR
- Verificar que EasyOCR estÃ© instalado
- Comprobar que las dependencias de `requirements.txt` estÃ©n instaladas

### Archivos no se procesan
- Revisar logs en `data/reports/web_app.log`
- Verificar permisos de escritura en carpetas de destino

## ContribuciÃ³n

Para contribuir a la interfaz web:

1. Seguir la estructura de archivos existente
2. Mantener consistencia con el estilo del cÃ³digo
3. AÃ±adir documentaciÃ³n para nuevas funcionalidades
4. Probar en diferentes navegadores

## Licencia

Este proyecto mantiene la misma licencia que AutOCR.

