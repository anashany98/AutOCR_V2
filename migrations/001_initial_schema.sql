-- ============================================================================
-- AutoOCR Document AI Platform — PostgreSQL + pgvector Schema
-- Migration: 001_initial_schema
-- Date: 2026-02-13
-- Description: Complete schema for multi-tenant Document AI with RAG
-- ============================================================================

-- ============================================================================
-- A) EXTENSIONS & TYPES
-- ============================================================================

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector";        -- pgvector
CREATE EXTENSION IF NOT EXISTS "pg_trgm";       -- trigram similarity

-- Document processing status
CREATE TYPE doc_status AS ENUM (
    'uploaded',
    'queued',
    'processing',
    'ocr_complete',
    'layout_complete',
    'embedding_complete',
    'completed',
    'failed',
    'archived'
);

-- Processing job status
CREATE TYPE job_status AS ENUM (
    'pending',
    'running',
    'completed',
    'failed',
    'cancelled',
    'retrying'
);

-- Job step types
CREATE TYPE job_step AS ENUM (
    'ingestion',
    'ocr',
    'layout',
    'table_extraction',
    'image_extraction',
    'visual_analysis',
    'chunking',
    'embedding',
    'classification'
);

-- Block types from layout detection
CREATE TYPE block_type AS ENUM (
    'text',
    'title',
    'table',
    'figure',
    'seal',
    'signature',
    'header',
    'footer',
    'caption',
    'equation',
    'list',
    'other'
);

-- Workflow states
CREATE TYPE workflow_state AS ENUM (
    'new',
    'pending_review',
    'verified',
    'rejected'
);

-- User roles
CREATE TYPE user_role AS ENUM (
    'CLIENTE',
    'GESTOR',
    'DIRECCION',
    'ADMIN'
);

-- Document visibility
CREATE TYPE doc_visibility AS ENUM (
    'private',
    'hotel',
    'tenant',
    'public'
);

-- Financial sensitivity level
CREATE TYPE financial_level AS ENUM (
    'none',
    'low',
    'medium',
    'high',
    'confidential'
);

-- Chat message role
CREATE TYPE chat_role AS ENUM (
    'user',
    'assistant',
    'system',
    'tool'
);

-- ============================================================================
-- B) TENANTS / HOTELS / PROJECTS
-- ============================================================================

CREATE TABLE tenants (
    id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name        TEXT NOT NULL,
    slug        TEXT UNIQUE NOT NULL,       -- URL-friendly identifier
    settings    JSONB DEFAULT '{}',         -- tenant-level config overrides
    is_active   BOOLEAN DEFAULT TRUE,
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    updated_at  TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX idx_tenants_slug ON tenants(slug);

CREATE TABLE hotels (
    id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id   UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    name        TEXT NOT NULL,
    code        TEXT NOT NULL,              -- short code (e.g., "HTL-001")
    description TEXT DEFAULT '',
    settings    JSONB DEFAULT '{}',         -- hotel-level config
    is_active   BOOLEAN DEFAULT TRUE,
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    updated_at  TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (tenant_id, code)
);
CREATE INDEX idx_hotels_tenant ON hotels(tenant_id);

CREATE TABLE projects (
    id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    hotel_id    UUID NOT NULL REFERENCES hotels(id) ON DELETE CASCADE,
    tenant_id   UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    name        TEXT NOT NULL,
    code        TEXT,                       -- optional project code (e.g., "PRJ-001")
    description TEXT DEFAULT '',
    settings    JSONB DEFAULT '{}',
    is_active   BOOLEAN DEFAULT TRUE,
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    updated_at  TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (hotel_id, code)
);
CREATE INDEX idx_projects_hotel ON projects(hotel_id);
CREATE INDEX idx_projects_tenant ON projects(tenant_id);

-- ============================================================================
-- C) USERS & MEMBERSHIPS
-- ============================================================================

CREATE TABLE users (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username        TEXT UNIQUE NOT NULL,
    email           TEXT UNIQUE,
    password_hash   TEXT NOT NULL,
    role            user_role DEFAULT 'CLIENTE',
    is_active       BOOLEAN DEFAULT TRUE,
    is_verified     BOOLEAN DEFAULT FALSE,
    verification_token TEXT,
    reset_token     TEXT,
    token_expiry    TIMESTAMPTZ,
    preferences     JSONB DEFAULT '{}',
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX idx_users_email ON users(email);

-- Associates users to hotels with specific roles
CREATE TABLE user_memberships (
    id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id     UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    tenant_id   UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    hotel_id    UUID REFERENCES hotels(id) ON DELETE CASCADE, -- NULL = tenant-wide access
    role        user_role NOT NULL DEFAULT 'CLIENTE',
    is_active   BOOLEAN DEFAULT TRUE,
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (user_id, tenant_id, hotel_id)
);
CREATE INDEX idx_memberships_user ON user_memberships(user_id);
CREATE INDEX idx_memberships_tenant_hotel ON user_memberships(tenant_id, hotel_id);

-- ============================================================================
-- D) DOCUMENTS & METADATA
-- ============================================================================

CREATE TABLE documents (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id       UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    hotel_id        UUID REFERENCES hotels(id) ON DELETE SET NULL,
    project_id      UUID REFERENCES projects(id) ON DELETE SET NULL,
    owner_id        UUID REFERENCES users(id) ON DELETE SET NULL,

    -- File information
    filename        TEXT NOT NULL,
    original_filename TEXT,
    file_path       TEXT NOT NULL,           -- relative storage path
    file_size       BIGINT,
    mime_type       TEXT,
    md5_hash        TEXT NOT NULL,
    page_count      INTEGER DEFAULT 0,

    -- Processing state
    status          doc_status DEFAULT 'uploaded',
    workflow_state  workflow_state DEFAULT 'new',
    error_message   TEXT,

    -- Classification & metadata
    doc_type        TEXT DEFAULT 'other',    -- Invoice, Contract, Blueprint...
    visibility      doc_visibility DEFAULT 'private',
    financial_level financial_level DEFAULT 'none',
    tags            JSONB DEFAULT '[]',
    metadata        JSONB DEFAULT '{}',      -- flexible key-value store

    -- Content (denormalized for quick access)
    text_content    TEXT,                    -- concatenated plain text
    markdown_content TEXT,                   -- markdown version
    language        TEXT,
    confidence      REAL,

    -- Timestamps
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW(),
    processed_at    TIMESTAMPTZ
);

-- Composite indexes for tenant-scoped queries
CREATE INDEX idx_docs_tenant ON documents(tenant_id);
CREATE INDEX idx_docs_tenant_hotel ON documents(tenant_id, hotel_id);
CREATE INDEX idx_docs_tenant_status ON documents(tenant_id, status);
CREATE INDEX idx_docs_owner ON documents(owner_id);
CREATE INDEX idx_docs_hash ON documents(md5_hash);
CREATE INDEX idx_docs_type ON documents(doc_type);
CREATE INDEX idx_docs_created ON documents(created_at DESC);
CREATE INDEX idx_docs_project ON documents(project_id);

-- GIN index on JSONB tags for tag-based filtering
CREATE INDEX idx_docs_tags ON documents USING GIN (tags);
CREATE INDEX idx_docs_metadata ON documents USING GIN (metadata);

-- Full-text search on document content
ALTER TABLE documents ADD COLUMN fts_vector tsvector
    GENERATED ALWAYS AS (
        to_tsvector('spanish',
            COALESCE(filename, '') || ' ' ||
            COALESCE(text_content, '') || ' ' ||
            COALESCE(doc_type, '')
        )
    ) STORED;
CREATE INDEX idx_docs_fts ON documents USING GIN (fts_vector);

-- Unique constraint: same file hash within a hotel
CREATE UNIQUE INDEX idx_docs_unique_hash_hotel ON documents(md5_hash, hotel_id)
    WHERE hotel_id IS NOT NULL;

-- ============================================================================
-- E) DOCUMENT PAGES
-- ============================================================================

CREATE TABLE document_pages (
    id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page_number INTEGER NOT NULL,            -- 1-indexed
    width       INTEGER,
    height      INTEGER,
    dpi         INTEGER,
    text_content TEXT,                       -- page-level text
    confidence  REAL,
    image_path  TEXT,                        -- path to rendered page image
    metadata    JSONB DEFAULT '{}',
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (document_id, page_number)
);
CREATE INDEX idx_pages_document ON document_pages(document_id);

-- ============================================================================
-- F) DOCUMENT BLOCKS (Layout Output)
-- ============================================================================

CREATE TABLE document_blocks (
    id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page_id     UUID REFERENCES document_pages(id) ON DELETE CASCADE,
    page_number INTEGER NOT NULL,

    -- Layout detection output
    block_type  block_type NOT NULL DEFAULT 'text',
    bbox        INTEGER[4],                  -- [x1, y1, x2, y2]
    rotation    REAL DEFAULT 0,
    confidence  REAL,

    -- Content
    text_content TEXT,
    markdown_content TEXT,
    table_data  JSONB,                       -- structured table data (rows/cols)
    language    TEXT,

    -- Ordering
    reading_order INTEGER,                   -- reading order within page
    metadata    JSONB DEFAULT '{}',
    created_at  TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX idx_blocks_document ON document_blocks(document_id);
CREATE INDEX idx_blocks_page ON document_blocks(page_id);
CREATE INDEX idx_blocks_type ON document_blocks(block_type);

-- ============================================================================
-- G) DOCUMENT ASSETS (Cropped Images)
-- ============================================================================

CREATE TABLE document_assets (
    id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    block_id    UUID REFERENCES document_blocks(id) ON DELETE SET NULL,
    page_number INTEGER,

    -- Asset information
    asset_type  TEXT NOT NULL DEFAULT 'figure', -- figure, seal, signature, chart, photo
    file_path   TEXT NOT NULL,                  -- relative path to cropped image
    file_size   BIGINT,
    mime_type   TEXT DEFAULT 'image/png',
    width       INTEGER,
    height      INTEGER,

    -- Source coordinates (from parent page)
    source_bbox INTEGER[4],

    -- Optional caption/description
    caption     TEXT,
    alt_text    TEXT,
    metadata    JSONB DEFAULT '{}',
    created_at  TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX idx_assets_document ON document_assets(document_id);
CREATE INDEX idx_assets_block ON document_assets(block_id);

-- ============================================================================
-- H) VISUAL ANALYSIS (VL Model Results)
-- ============================================================================

CREATE TABLE visual_analysis (
    id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    asset_id    UUID REFERENCES document_assets(id) ON DELETE CASCADE,
    page_number INTEGER,

    -- VL model output
    model_name  TEXT NOT NULL DEFAULT 'PaddleOCR-VL-1.5',
    description TEXT,                        -- natural language description
    caption     TEXT,                        -- concise caption for RAG
    labels      JSONB DEFAULT '[]',          -- detected labels/categories
    structured_output JSONB DEFAULT '{}',    -- any structured analysis
    confidence  REAL,

    -- Processing metadata
    processing_time_ms INTEGER,
    metadata    JSONB DEFAULT '{}',
    created_at  TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX idx_visual_document ON visual_analysis(document_id);
CREATE INDEX idx_visual_asset ON visual_analysis(asset_id);

-- ============================================================================
-- I) CHUNKS (RAG Units)
-- ============================================================================

CREATE TABLE chunks (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id     UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    tenant_id       UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    hotel_id        UUID REFERENCES hotels(id) ON DELETE SET NULL,

    -- Source tracking
    page_number     INTEGER,
    block_id        UUID REFERENCES document_blocks(id) ON DELETE SET NULL,
    asset_id        UUID REFERENCES document_assets(id) ON DELETE SET NULL,

    -- Chunk content
    chunk_index     INTEGER NOT NULL,        -- order within document
    content         TEXT NOT NULL,
    content_type    TEXT DEFAULT 'text',      -- text, table, caption, visual_description
    token_count     INTEGER,
    char_count      INTEGER,

    -- Deduplication
    content_hash    TEXT,                    -- SHA-256 hash of normalized content

    -- Metadata for retrieval
    metadata        JSONB DEFAULT '{}',      -- source filename, page, block_type, etc.
    created_at      TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX idx_chunks_document ON chunks(document_id);
CREATE INDEX idx_chunks_tenant ON chunks(tenant_id);
CREATE INDEX idx_chunks_tenant_hotel ON chunks(tenant_id, hotel_id);
CREATE INDEX idx_chunks_hash ON chunks(content_hash);

-- Full-text search on chunk content
ALTER TABLE chunks ADD COLUMN fts_vector tsvector
    GENERATED ALWAYS AS (
        to_tsvector('spanish', COALESCE(content, ''))
    ) STORED;
CREATE INDEX idx_chunks_fts ON chunks USING GIN (fts_vector);

-- ============================================================================
-- J) EMBEDDINGS (pgvector)
-- ============================================================================

CREATE TABLE embeddings (
    id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    chunk_id    UUID NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    tenant_id   UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    hotel_id    UUID REFERENCES hotels(id) ON DELETE SET NULL,

    -- Vector data
    embedding   vector(384),                 -- paraphrase-multilingual-MiniLM-L12-v2
    model_name  TEXT NOT NULL DEFAULT 'paraphrase-multilingual-MiniLM-L12-v2',
    model_version TEXT DEFAULT 'v1',

    -- Metadata for filtering
    content_type TEXT DEFAULT 'text',
    metadata    JSONB DEFAULT '{}',
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

-- HNSW index for fast approximate nearest neighbor search
-- m=16, ef_construction=200 provides good recall/speed trade-off
CREATE INDEX idx_embeddings_hnsw ON embeddings
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 200);

-- Composite indexes for tenant-scoped vector search
CREATE INDEX idx_embeddings_tenant ON embeddings(tenant_id);
CREATE INDEX idx_embeddings_tenant_hotel ON embeddings(tenant_id, hotel_id);
CREATE INDEX idx_embeddings_document ON embeddings(document_id);
CREATE INDEX idx_embeddings_chunk ON embeddings(chunk_id);

-- ============================================================================
-- K) CHAT SESSIONS & MESSAGES
-- ============================================================================

CREATE TABLE chat_sessions (
    id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id   UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    user_id     UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    hotel_id    UUID REFERENCES hotels(id) ON DELETE SET NULL,
    title       TEXT,                        -- auto-generated or user-defined
    is_active   BOOLEAN DEFAULT TRUE,
    metadata    JSONB DEFAULT '{}',
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    updated_at  TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX idx_sessions_user ON chat_sessions(user_id);
CREATE INDEX idx_sessions_tenant ON chat_sessions(tenant_id);

CREATE TABLE chat_messages (
    id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id  UUID NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
    role        chat_role NOT NULL,
    content     TEXT NOT NULL,

    -- Source citations
    sources     JSONB DEFAULT '[]',          -- [{chunk_id, document_id, filename, page, score}]

    -- Tool calls
    tool_calls  JSONB,                       -- tool call data if role='assistant'
    tool_result JSONB,                       -- tool result if role='tool'

    -- Token usage
    prompt_tokens   INTEGER,
    completion_tokens INTEGER,

    metadata    JSONB DEFAULT '{}',
    created_at  TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX idx_messages_session ON chat_messages(session_id);
CREATE INDEX idx_messages_created ON chat_messages(created_at);

-- ============================================================================
-- L) AUDIT LOGS
-- ============================================================================

CREATE TABLE audit_logs (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id       UUID REFERENCES tenants(id) ON DELETE SET NULL,
    user_id         UUID REFERENCES users(id) ON DELETE SET NULL,
    action          TEXT NOT NULL,            -- e.g., 'document.upload', 'chat.query'
    resource_type   TEXT,                     -- e.g., 'document', 'chat_session'
    resource_id     TEXT,
    details         JSONB DEFAULT '{}',
    ip_address      INET,
    user_agent      TEXT,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX idx_audit_tenant ON audit_logs(tenant_id);
CREATE INDEX idx_audit_user ON audit_logs(user_id);
CREATE INDEX idx_audit_action ON audit_logs(action);
CREATE INDEX idx_audit_created ON audit_logs(created_at DESC);

-- Partitioning hint: for large deployments, partition by month
-- CREATE TABLE audit_logs (...) PARTITION BY RANGE (created_at);

-- ============================================================================
-- M) PROCESSING JOBS
-- ============================================================================

CREATE TABLE processing_jobs (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id       UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    document_id     UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,

    -- Job specification
    job_type        job_step NOT NULL,
    priority        INTEGER DEFAULT 0,       -- higher = more urgent
    status          job_status DEFAULT 'pending',

    -- Scheduling
    scheduled_at    TIMESTAMPTZ DEFAULT NOW(),
    started_at      TIMESTAMPTZ,
    completed_at    TIMESTAMPTZ,

    -- Retry control
    attempt         INTEGER DEFAULT 0,
    max_attempts    INTEGER DEFAULT 3,
    next_retry_at   TIMESTAMPTZ,

    -- Idempotency
    idempotency_key TEXT,                    -- doc_id + job_type + content_hash
    worker_id       TEXT,                    -- which worker picked this up

    -- Results
    result          JSONB DEFAULT '{}',
    error_message   TEXT,
    error_traceback TEXT,
    processing_time_ms INTEGER,

    metadata        JSONB DEFAULT '{}',
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_jobs_tenant ON processing_jobs(tenant_id);
CREATE INDEX idx_jobs_document ON processing_jobs(document_id);
CREATE INDEX idx_jobs_status ON processing_jobs(status);
CREATE INDEX idx_jobs_queue ON processing_jobs(status, priority DESC, scheduled_at)
    WHERE status IN ('pending', 'retrying');
CREATE UNIQUE INDEX idx_jobs_idempotency ON processing_jobs(idempotency_key)
    WHERE idempotency_key IS NOT NULL;

-- ============================================================================
-- N) SUPPLEMENTARY TABLES (Products, Templates)
-- ============================================================================

-- Product catalog (ERP integration, preserved from existing schema)
CREATE TABLE products (
    id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id   UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    sku         TEXT NOT NULL,
    name        TEXT NOT NULL,
    description TEXT,
    price       NUMERIC(12,2),
    stock       INTEGER DEFAULT 0,
    image_url   TEXT,
    category    TEXT,
    tags        JSONB DEFAULT '[]',
    attributes  JSONB DEFAULT '{}',
    embedding   vector(384),                 -- CLIP or text embedding for product search
    is_active   BOOLEAN DEFAULT TRUE,
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    updated_at  TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (tenant_id, sku)
);
CREATE INDEX idx_products_tenant ON products(tenant_id);
CREATE INDEX idx_products_category ON products(category);

-- OCR extraction templates
CREATE TABLE templates (
    id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id   UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    name        TEXT NOT NULL,
    description TEXT,
    zones_json  JSONB DEFAULT '[]',          -- extraction zones
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    updated_at  TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX idx_templates_tenant ON templates(tenant_id);

-- ============================================================================
-- O) HELPER FUNCTIONS
-- ============================================================================

-- Auto-update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply trigger to tables with updated_at
CREATE TRIGGER trg_tenants_updated BEFORE UPDATE ON tenants
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();
CREATE TRIGGER trg_hotels_updated BEFORE UPDATE ON hotels
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();
CREATE TRIGGER trg_projects_updated BEFORE UPDATE ON projects
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();
CREATE TRIGGER trg_users_updated BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();
CREATE TRIGGER trg_documents_updated BEFORE UPDATE ON documents
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();
CREATE TRIGGER trg_sessions_updated BEFORE UPDATE ON chat_sessions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();
CREATE TRIGGER trg_jobs_updated BEFORE UPDATE ON processing_jobs
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();
CREATE TRIGGER trg_products_updated BEFORE UPDATE ON products
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();
CREATE TRIGGER trg_templates_updated BEFORE UPDATE ON templates
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- ============================================================================
-- P) OPTIONAL: ROW LEVEL SECURITY (RLS)
-- ============================================================================
-- Recommendation: Use server-side ACL filtering in the application layer first.
-- Enable RLS as a defense-in-depth measure for production.
--
-- Pros of server-side ACL:
--   + Simpler to debug and test
--   + More flexible (can change without schema changes)
--   + Works with any connection pooler
--
-- Pros of RLS:
--   + Defense-in-depth (even if app has bug, data is isolated)
--   + Enforced at database level
--
-- To enable RLS later, uncomment and run:
--
-- ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
-- CREATE POLICY tenant_isolation_docs ON documents
--     USING (tenant_id = current_setting('app.current_tenant_id')::UUID);
--
-- ALTER TABLE chunks ENABLE ROW LEVEL SECURITY;
-- CREATE POLICY tenant_isolation_chunks ON chunks
--     USING (tenant_id = current_setting('app.current_tenant_id')::UUID);
--
-- ALTER TABLE embeddings ENABLE ROW LEVEL SECURITY;
-- CREATE POLICY tenant_isolation_embeddings ON embeddings
--     USING (tenant_id = current_setting('app.current_tenant_id')::UUID);

-- ============================================================================
-- Q) SEED DATA (Default Tenant for Migration)
-- ============================================================================

-- Create a default tenant for migrating existing data
INSERT INTO tenants (id, name, slug, settings) VALUES
    ('00000000-0000-0000-0000-000000000001', 'Default', 'default', '{}')
ON CONFLICT (slug) DO NOTHING;
