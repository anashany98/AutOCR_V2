# Plan de Implementación - AutoOCR Mejoras

## 1. Bot Telegram para 20 Gestores

### 1.1 Tabla PostgreSQL - telegram_gestores

```sql
CREATE TABLE telegram_gestores (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    telegram_id     BIGINT UNIQUE NOT NULL,
    username        TEXT,
    first_name      TEXT NOT NULL,
    last_name       TEXT,
    user_id         UUID REFERENCES users(id) ON DELETE SET NULL,
    tenant_id       UUID REFERENCES tenants(id) ON DELETE CASCADE,
    hotel_id        UUID REFERENCES hotels(id) ON DELETE SET NULL,
    is_active       BOOLEAN DEFAULT TRUE,
    is_verified     BOOLEAN DEFAULT FALSE,
    verified_at     TIMESTAMPTZ,
    notify_invoices BOOLEAN DEFAULT TRUE,
    notify_expiry   BOOLEAN DEFAULT TRUE,
    notify_alerts   BOOLEAN DEFAULT TRUE,
    language        TEXT DEFAULT 'es',
    last_command    TEXT,
    last_seen_at    TIMESTAMPTZ,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_telegram_gestores_telegram_id ON telegram_gestores(telegram_id);
CREATE INDEX idx_telegram_gestores_user ON telegram_gestores(user_id);
CREATE INDEX idx_telegram_gestores_tenant ON telegram_gestores(tenant_id);
```

### 1.2 Autenticación por telegram_user_id

El sistema debe:
1. Buscar telegram_id en tabla `telegram_gestores`
2. Verificar que `is_active = TRUE`
3. Obtener el `user_id`, `tenant_id`, `hotel_id` vinculados
4. Crear sesión de usuario en AutoOCR con esos datos

### 1.3 Integración con Chat V2 API

El bot de Telegram debe llamar a `POST /api/v2/chat/query` con:
- `tenant_id`: del gestor vinculado
- `hotel_ids`: del gestor vinculado  
- `query`: el texto del usuario

### 1.4 Menús con Botones Inline

Usar `InlineKeyboardMarkup` de python-telegram-bot para:
- Menú principal con opciones
- Botones de confirmación
- Navegación por documentos

---

## 2. Email Outlook - Microsoft Graph API

### 2.1 Autenticación OAuth2

Configurar en Azure AD:
- Registrar aplicación
- Permisos: `Mail.Read`, `Mail.ReadWrite`, `Mail.Send`
- Secret de cliente

### 2.2 Lectura de Emails

```python
# endpoints de Microsoft Graph
GET https://graph.microsoft.com/v1.0/me/messages
GET https://graph.microsoft.com/v1.0/users/{id}/mailFolders/{id}/messages
```

### 2.3 Anti-duplicados en 3 Capas

1. **Message-ID**: Tabla `email_message_ids`
```sql
CREATE TABLE email_message_ids (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id       UUID REFERENCES tenants(id) ON DELETE CASCADE,
    message_id      TEXT NOT NULL,
    subject         TEXT,
    received_at     TIMESTAMPTZ NOT NULL,
    document_id     UUID REFERENCES documents(id) ON DELETE SET NULL,
    processed       BOOLEAN DEFAULT FALSE,
    UNIQUE (tenant_id, message_id)
);
```

2. **SHA256**: Campo en tabla documents
```sql
ALTER TABLE documents ADD COLUMN sha256_hash TEXT;
CREATE INDEX idx_docs_sha256_hash ON documents(sha256_hash);
```

3. **Similitud pgvector**: Comparar embeddings
- Usar campo vector existente en tabla `chunks`
- Comparar con `cosine_distance < 0.05` (95% similitud)

### 2.4 Carpeta "AutOCR Procesados"

Mover email después de procesar:
```
POST /me/messages/{id}/move
{
  "destinationId": "AutOCR Procesados"
}
```

---

## 3. Control de Presupuestos por Proyecto

### 3.1 Campos en Tabla Projects

```sql
ALTER TABLE projects ADD COLUMN budget_amount NUMERIC(15,2);
ALTER TABLE projects ADD COLUMN budget_currency TEXT DEFAULT 'EUR';
ALTER TABLE projects ADD COLUMN start_date DATE;
ALTER TABLE projects ADD COLUMN end_date DATE;
ALTER TABLE projects ADD COLUMN alert_threshold_percent INTEGER DEFAULT 80;
```

### 3.2 Gasto Real vs Presupuesto

```sql
SELECT 
    p.name as proyecto,
    p.budget_amount as presupuesto,
    COALESCE(SUM(d.total_amount), 0) as gastado,
    p.budget_amount - COALESCE(SUM(d.total_amount), 0) as restante,
    (COALESCE(SUM(d.total_amount), 0) / p.budget_amount * 100) as porcentaje
FROM projects p
LEFT JOIN documents d ON d.project_id = p.id 
    AND d.status = 'completed'
    AND d.doc_type = 'invoice'
WHERE p.id = :project_id
GROUP BY p.id;
```

### 3.3 Alertas por Porcentaje

Notificar cuando `gastado / presupuesto * 100 >= alert_threshold_percent`

---

## 4. Vencimientos de Pago

### 4.1 Extraer Fecha Vencimiento del OCR

En el pipeline de extracción, buscar patrones:
- "Fecha de vencimiento: DD/MM/YYYY"
- "Vence: DD/MM/YYYY"
- "Payment due: DD/MM/YYYY"
- Fechas en formato español: "30 días", "60 días"

### 4.2 Celery Beat - Revisión Matutina

```python
# tasks.py
@celery_app.task
def check_expiring_documents():
    """Revisa documentos que vencen en 7 y 1 día."""
    seven_days = date.today() + timedelta(days=7)
    one_day = date.today() + timedelta(days=1)
    
    # Docs que vencen en 7 días
    docs_7 = Documents.query.filter(
        Documents.due_date == seven_days,
        Documents.payment_status == 'pending'
    ).all()
    
    # Docs que vencen en 1 día
    docs_1 = Documents.query.filter(
        Documents.due_date == one_day,
        Documents.payment_status == 'pending'
    ).all()
    
    # Enviar notificaciones por Telegram
```

---

## 5. Detección de Anomalías

### 5.1 Tabla vendor_statistics

```sql
CREATE TABLE vendor_statistics (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id       UUID REFERENCES tenants(id) ON DELETE CASCADE,
    vendor_nif       TEXT NOT NULL,
    vendor_name      TEXT NOT NULL,
    avg_amount      NUMERIC(15,2),
    min_amount      NUMERIC(15,2),
    max_amount      NUMERIC(15,2),
    std_deviation   NUMERIC(15,2),
    invoice_count   INTEGER DEFAULT 0,
    first_invoice_date DATE,
    last_invoice_date DATE,
    is_approved     BOOLEAN DEFAULT FALSE,
    UNIQUE (tenant_id, vendor_nif)
);
```

### 5.2 Detectar Facturas Fuera de Rango

```python
def check_anomaly(document):
    vendor_stats = get_vendor_stats(document.vendor_nif)
    if vendor_stats:
        # Verificar si está fuera del rango histórico
        if document.total_amount > vendor_stats.max_amount * 1.2:
            return "ALERTA: Factura 20% mayor al máximo histórico"
```

### 5.3 Proveedores No Homologados

```python
def check_unapproved_vendor(document):
    vendor_stats = get_vendor_stats(document.vendor_nif)
    if vendor_stats and not vendor_stats.is_approved:
        return "ALERTA: Proveedor no homologado"
```

---

## 6. Comparación Albarán vs Pedido

### 6.1 Tabla document_matches

```sql
CREATE TABLE document_matches (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id       UUID REFERENCES tenants(id) ON DELETE CASCADE,
    document_a_id   UUID REFERENCES documents(id) ON DELETE CASCADE,
    document_b_id   UUID REFERENCES documents(id) ON DELETE CASCADE,
    match_type      TEXT NOT NULL,
    match_confidence REAL,
    differences     JSONB DEFAULT '{}',
    discrepancies    JSONB DEFAULT '[]',
    status          TEXT DEFAULT 'pending',
    reviewed_by     UUID REFERENCES users(id) ON DELETE SET NULL,
    reviewed_at     TIMESTAMPTZ,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);
```

### 6.2 Detectar Discrepancias

```python
def compare_albaran_pedido(albaran, pedido):
    discrepancies = []
    
    # Comparar importes
    if albaran.total_amount != pedido.total_amount:
        discrepancies.append({
            'type': 'amount_mismatch',
            'albaran': albaran.total_amount,
            'pedido': pedido.total_amount,
            'difference': albaran.total_amount - pedido.total_amount
        })
    
    # Comparar cantidades
    # ... similar lógica
    
    return discrepancies
```

---

## 7. Dashboard Métricas Flask

### 7.1 Endpoints API

- `GET /api/metrics/summary` - Resumen general
- `GET /api/metrics/by-vendor` - Gasto por proveedor
- `GET /api/metrics/by-month` - Gasto por mes
- `GET /api/metrics/by-category` - Gasto por categoría
- `GET /api/metrics/pending` - Documentos pendientes

### 7.2 Exportar Excel

```python
from openpyxl import Workbook

def export_to_excel(data):
    wb = Workbook()
    ws = wb.active
    
    for row in data:
        ws.append(row)
    
    return wb
```

### 7.3 Gráficas Chart.js

```javascript
// En el template HTML
new Chart(ctx, {
    type: 'bar',
    data: {
        labels: [...],
        datasets: [{
            label: 'Gasto por Mes',
            data: [...],
            backgroundColor: 'rgba(54, 162, 235, 0.5)'
        }]
    }
});
```

---

## Orden de Implementación Recomendado

1. **Telegram Bot con PostgreSQL** (Prioridad alta)
   - Crear tabla telegram_gestores
   - Modificar telegram_bot.py para usar DB
   - Integrar con Chat V2

2. **Anti-duplicados Email** (Prioridad alta)
   - Añadir campos a documents
   - Crear tabla email_message_ids
   - Implementar las 3 capas de detección

3. **Control de Presupuestos** (Prioridad media)
   - Añadir campos a projects
   - Consultas SQL
   - Alertas

4. **Vencimientos** (Prioridad media)
   - Extraer fecha del OCR
   - Celery Beat
   - Notificaciones Telegram

5. **Dashboard y Métricas** (Prioridad baja)
   - Endpoints API
   - Gráficas Chart.js
   - Export Excel
