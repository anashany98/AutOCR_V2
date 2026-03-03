-- ============================================================================
-- Telegram Bot Integration - User Management Table
-- Migration: 002_telegram_and_budget
-- ============================================================================

-- Tabla para vincular usuarios de Telegram con AutoOCR (20 gestores)
CREATE TABLE IF NOT EXISTS telegram_gestores (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Telegram user identification
    telegram_id     BIGINT UNIQUE NOT NULL,
    username        TEXT,
    first_name      TEXT NOT NULL,
    last_name       TEXT,
    
    -- AutoOCR user link
    user_id         UUID REFERENCES users(id) ON DELETE SET NULL,
    tenant_id       UUID REFERENCES tenants(id) ON DELETE CASCADE,
    hotel_id        UUID REFERENCES hotels(id) ON DELETE SET NULL,
    
    -- Authentication status
    is_active       BOOLEAN DEFAULT TRUE,
    is_verified     BOOLEAN DEFAULT FALSE,
    verified_at     TIMESTAMPTZ,
    
    -- Notification preferences
    notify_invoices BOOLEAN DEFAULT TRUE,
    notify_expiry   BOOLEAN DEFAULT TRUE,
    notify_alerts   BOOLEAN DEFAULT TRUE,
    
    -- Metadata
    language        TEXT DEFAULT 'es',
    last_command    TEXT,
    last_seen_at    TIMESTAMPTZ,
    
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Index for fast lookups by telegram_id
CREATE INDEX IF NOT EXISTS idx_telegram_gestores_telegram_id ON telegram_gestores(telegram_id);
CREATE INDEX IF NOT EXISTS idx_telegram_gestores_user ON telegram_gestores(user_id);
CREATE INDEX IF NOT EXISTS idx_telegram_gestores_tenant ON telegram_gestores(tenant_id);
CREATE INDEX IF NOT EXISTS idx_telegram_gestores_active ON telegram_gestores(is_active) WHERE is_active = TRUE;

-- ============================================================================
-- Telegram Bot Commands Log - Audit Trail
-- ============================================================================

CREATE TABLE IF NOT EXISTS telegram_command_log (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    telegram_id     BIGINT NOT NULL,
    command         TEXT NOT NULL,
    args            TEXT,
    result          TEXT,
    success         BOOLEAN DEFAULT TRUE,
    
    -- Context
    tenant_id       UUID REFERENCES tenants(id) ON DELETE SET NULL,
    user_id         UUID REFERENCES users(id) ON DELETE SET NULL,
    
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_telegram_command_log_telegram ON telegram_command_log(telegram_id);
CREATE INDEX IF NOT EXISTS idx_telegram_command_log_created ON telegram_command_log(created_at DESC);

-- ============================================================================
-- Add SHA256 hash field to documents table for better deduplication
-- ============================================================================

ALTER TABLE documents ADD COLUMN IF NOT EXISTS sha256_hash TEXT;
CREATE INDEX IF NOT EXISTS idx_docs_sha256_hash ON documents(sha256_hash);

-- Add extracted financial fields for budget tracking
ALTER TABLE documents ADD COLUMN IF NOT EXISTS total_amount NUMERIC(15,2);
ALTER TABLE documents ADD COLUMN IF NOT EXISTS currency TEXT DEFAULT 'EUR';
ALTER TABLE documents ADD COLUMN IF NOT EXISTS vendor_name TEXT;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS vendor_nif TEXT;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS invoice_number TEXT;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS invoice_date DATE;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS due_date DATE;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS payment_status TEXT DEFAULT 'pending';

-- Indexes for financial queries
CREATE INDEX IF NOT EXISTS idx_docs_total_amount ON documents(total_amount);
CREATE INDEX IF NOT EXISTS idx_docs_due_date ON documents(due_date);
CREATE INDEX IF NOT EXISTS idx_docs_vendor ON documents(vendor_name);
CREATE INDEX IF NOT EXISTS idx_docs_payment_status ON documents(payment_status);

-- ============================================================================
-- Email deduplication tracking
-- ============================================================================

CREATE TABLE IF NOT EXISTS email_message_ids (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id       UUID REFERENCES tenants(id) ON DELETE CASCADE,
    
    -- Email identification
    message_id      TEXT NOT NULL,
    subject         TEXT,
    from_address    TEXT,
    received_at     TIMESTAMPTZ NOT NULL,
    
    -- Document link
    document_id     UUID REFERENCES documents(id) ON DELETE SET NULL,
    processed       BOOLEAN DEFAULT FALSE,
    
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (tenant_id, message_id)
);

CREATE INDEX IF NOT EXISTS idx_email_message_ids_message ON email_message_ids(message_id);
CREATE INDEX IF NOT EXISTS idx_email_message_ids_processed ON email_message_ids(processed) WHERE processed = FALSE;

-- ============================================================================
-- Project budget tracking
-- ============================================================================

ALTER TABLE projects ADD COLUMN IF NOT EXISTS budget_amount NUMERIC(15,2);
ALTER TABLE projects ADD COLUMN IF NOT EXISTS budget_currency TEXT DEFAULT 'EUR';
ALTER TABLE projects ADD COLUMN IF NOT EXISTS start_date DATE;
ALTER TABLE projects ADD COLUMN IF NOT EXISTS end_date DATE;
ALTER TABLE projects ADD COLUMN IF NOT EXISTS alert_threshold_percent INTEGER DEFAULT 80;

-- ============================================================================
-- Anomaly detection configuration
-- ============================================================================

CREATE TABLE IF NOT EXISTS vendor_statistics (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id       UUID REFERENCES tenants(id) ON DELETE CASCADE,
    vendor_nif      TEXT NOT NULL,
    vendor_name     TEXT NOT NULL,
    
    -- Historical data
    avg_amount      NUMERIC(15,2),
    min_amount     NUMERIC(15,2),
    max_amount     NUMERIC(15,2),
    std_deviation  NUMERIC(15,2),
    invoice_count  INTEGER DEFAULT 0,
    
    -- Date range
    first_invoice_date DATE,
    last_invoice_date DATE,
    
    -- Status
    is_approved     BOOLEAN DEFAULT FALSE,
    approved_by    UUID REFERENCES users(id) ON DELETE SET NULL,
    approved_at    TIMESTAMPTZ,
    
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW(),
    
    UNIQUE (tenant_id, vendor_nif)
);

CREATE INDEX IF NOT EXISTS idx_vendor_statistics_tenant ON vendor_statistics(tenant_id);
CREATE INDEX IF NOT EXISTS idx_vendor_statistics_nif ON vendor_statistics(vendor_nif);

-- ============================================================================
-- Document comparison / matching (for albarán vs pedido)
-- ============================================================================

CREATE TABLE IF NOT EXISTS document_matches (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id       UUID REFERENCES tenants(id) ON DELETE CASCADE,
    
    -- Documents being compared
    document_a_id   UUID REFERENCES documents(id) ON DELETE CASCADE,
    document_b_id   UUID REFERENCES documents(id) ON DELETE CASCADE,
    
    -- Match type
    match_type      TEXT NOT NULL,  -- 'albaran_pedido', 'duplicate', 'related'
    match_confidence REAL,
    
    -- Comparison results
    differences     JSONB DEFAULT '{}',
    discrepancies    JSONB DEFAULT '[]',
    
    -- Status
    status          TEXT DEFAULT 'pending',  -- 'pending', 'confirmed', 'rejected'
    reviewed_by     UUID REFERENCES users(id) ON DELETE SET NULL,
    reviewed_at     TIMESTAMPTZ,
    
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_document_matches_doc_a ON document_matches(document_a_id);
CREATE INDEX IF NOT EXISTS idx_document_matches_doc_b ON document_matches(document_b_id);
CREATE INDEX IF NOT EXISTS idx_document_matches_status ON document_matches(status);
