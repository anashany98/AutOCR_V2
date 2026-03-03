from huey import SqliteHuey, crontab
from pathlib import Path
import os
import sys
import hmac
import hashlib
import json

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Huey instance using SQLite for persistence on Windows
# Huey instance: Redis (Production/Docker) vs SQLite (Local Windows)
redis_url = os.environ.get('REDIS_URL')
if redis_url:
    from huey import RedisHuey
    huey = RedisHuey('autoocr', url=redis_url)
else:
    huey = SqliteHuey(filename=str(PROJECT_ROOT / 'data' / 'huey_db.db'))



def use_celery_backend() -> bool:
    """
    Decide whether task dispatch should use Celery or Huey.

    Priority order:
    1) `AUTOOCR_QUEUE_BACKEND=celery|huey` explicit override
    2) production env (`AUTOOCR_ENV=production`)
    3) presence of a Celery broker URL
    """
    forced = (os.environ.get("AUTOOCR_QUEUE_BACKEND") or "").strip().lower()
    if forced == "celery":
        return True
    if forced == "huey":
        return False

    if (os.environ.get("AUTOOCR_ENV") or "").strip().lower() == "production":
        return True

    return bool((os.environ.get("CELERY_BROKER_URL") or "").strip())


def _process_document_logic(file_path, options):
    """
    Core logic for processing a single document.
    """
    from postbatch_processor import process_single_file
    from web_app.services import get_db, get_logger, resolve_path, load_configuration, get_pipeline, get_classifier, get_llm_client
    
    logger = get_logger()
    db = get_db()
    
    # Reload config to get latest paths etc
    config = load_configuration(reload=True)
    post_conf = config.get("postbatch", {})
    
    # Inject Phase 4 Metadata into config for process_single_file
    for key in ["owner_id", "hotel_id", "doc_type", "visibility", "financial_level"]:
        if options.get(key):
            config[key] = options[key]
            post_conf[key] = options[key]
    
    processed_folder = resolve_path(post_conf.get("processed_folder"), "data/scans_processed")
    failed_folder = resolve_path(post_conf.get("failed_folder"), "data/scans_failed")
    
    pipeline = get_pipeline()
    classifier = get_classifier()
    
    try:
        result = process_single_file(
            file_path,
            pipeline,
            classifier,
            db,
            processed_folder,
            failed_folder,
            delete_original=options.get("delete_original", True),
            ocr_enabled=options.get("ocr_enabled", True),
            classification_enabled=options.get("classification_enabled", True),
            logger=logger,
            input_root=options.get("input_root", "input"),
            handwriting_mode=options.get("handwriting_mode", False),
            pipeline_conf=config,
        )

        
        # --- AI Summarization Step ---
        if result.get("status") == "OK":
            try:
                llm_client = get_llm_client()
                if llm_client and getattr(llm_client, "enabled", False):
                    doc_id = result.get("doc_id")
                    if doc_id:
                        try:
                            doc_id_int = int(doc_id)
                        except Exception:
                            doc_id_int = None
                    else:
                        doc_id_int = None

                    text = ""
                    if doc_id_int:
                        doc = db.get_document(doc_id_int) or {}
                        text = (doc.get("text") or "").strip()

                    if text:
                        logger.info("Generating AI Summary for %s (doc_id=%s)...", file_path, doc_id_int)
                        sum_res = llm_client.summarize_document(text)
                        if sum_res.get("success"):
                            summary = sum_res.get("summary")
                            if summary and doc_id_int:
                                with db.get_connection() as conn:
                                    cursor = db.get_cursor(conn)
                                    cursor.execute(
                                        f"SELECT structured_data FROM ocr_texts WHERE id_doc = {db.placeholder}",
                                        (doc_id_int,),
                                    )
                                    row = cursor.fetchone()
                                    s_data = None
                                    if row:
                                        if isinstance(row, (tuple, list)):
                                            s_data = row[0]
                                        else:
                                            try:
                                                s_data = row["structured_data"]
                                            except Exception:
                                                try:
                                                    s_data = row[0]
                                                except Exception:
                                                    s_data = None
                                    try:
                                        data_json = json.loads(s_data) if s_data else {}
                                    except Exception:
                                        data_json = {}

                                    data_json["ai_summary"] = summary
                                    cursor.execute(
                                        f"UPDATE ocr_texts SET structured_data = {db.placeholder} WHERE id_doc = {db.placeholder}",
                                        (json.dumps(data_json, ensure_ascii=False), doc_id_int),
                                    )
                                    conn.commit()
                                    logger.info("AI Summary saved.")
            except Exception as e:
                logger.error(f"AI Summary failed: {e}")

        # --- OUTGOING WEBHOOK ---
        webhook_url = config.get("app", {}).get("webhook_url")
        if webhook_url and result.get("status") == "OK":
            try:
                import requests
                logger.info(f"Sending webhook to {webhook_url}...")
                # Prepare payload (exclude massive base64 or internal paths if needed, 
                # but sending full result is usually fine for internal use)
                payload = {
                    "event": "document_processed",
                    "doc_id": result.get("doc_id"), # Might need to ensure this is present
                    "filename": os.path.basename(file_path),
                    "data": result
                }
                webhook_secret = (
                    os.environ.get("WEBHOOK_SIGNING_SECRET")
                    or os.environ.get("AUTOOCR_WEBHOOK_SECRET")
                    or config.get("app", {}).get("webhook_secret")
                    or ""
                )

                headers = {}
                if webhook_secret:
                    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
                    signature = hmac.new(
                        webhook_secret.encode("utf-8"),
                        body.encode("utf-8"),
                        hashlib.sha256,
                    ).hexdigest()
                    headers["X-AutoOCR-Signature"] = f"sha256={signature}"

                requests.post(webhook_url, json=payload, timeout=5, headers=headers or None)
            except Exception as e:
                logger.error(f"Webhook failed: {e}")

        return result
    except Exception as e:
        logger.error(f"Task failed for {file_path}: {e}", exc_info=True)
        return {"status": "FAILED", "error": str(e)}


# --------------------------------------------------------------------------- #
# Task Definitions (Huey vs Celery)
# --------------------------------------------------------------------------- #

# 1. Huey Task (Development / Windows)
@huey.task()
def _process_document_task_huey(file_path, options):
    return _process_document_logic(file_path, options)

# 2. Celery Task (Production / Docker)
try:
    from celery_app import celery_app
    @celery_app.task(queue="ocr_default")
    def _process_document_task_celery(file_path, options):
        return _process_document_logic(file_path, options)
except (ImportError, ModuleNotFoundError):
    _process_document_task_celery = None


# 3. Dispatcher
def process_document_task(file_path, options):
    """
    Dispatch task to appropriate queue backend based on environment.
    """
    if use_celery_backend() and _process_document_task_celery:
        # Determine priority based on file size or options
        queue_name = "ocr_batch" # Default
        
        try:
            # Check file size (e.g., < 5MB goes to fast lane)
            if os.path.exists(file_path) and os.path.getsize(file_path) < 5 * 1024 * 1024:
                 queue_name = "ocr_fast"
            
            # Forced priority
            if options.get("priority") == "high":
                 queue_name = "ocr_fast"
                 
        except Exception:
            pass # Fallback to batch

        # Celery .delay() handles async enqueueing
        # We use apply_async to specify queue dynamically
        return _process_document_task_celery.apply_async(
            args=[file_path, options],
            queue=queue_name
        )
    else:
        # Huey task instance is callable to enqueue
        return _process_document_task_huey(file_path, options)

# --------------------------------------------------------------------------- #
# Vision Index Rebuild Task
# --------------------------------------------------------------------------- #

def _rebuild_index_logic():
    """
    Logic to rebuild the vision index.
    """
    from web_app.services import get_db, get_llm_client, load_configuration, get_logger, PROJECT_ROOT
    from modules.vision_manager import VisionManager, VisionManagerConfig
    
    logger = get_logger()
    logger.info("Starting background vision index rebuild...")
    
    try:
        config = load_configuration(reload=True)
        vision_conf = config.get("vision", {})
        
        # Initialize VisionManager with config
        vm_config = VisionManagerConfig(
            enabled=vision_conf.get("enabled", True),
            index_path=str(PROJECT_ROOT / "data" / "vision_index.faiss"),
            embeddings_dir=str(PROJECT_ROOT / "data" / "vision_embeddings"),
            model_name=vision_conf.get("model_name", "ViT-B-32"),
            pretrained=vision_conf.get("pretrained", "laion2b_s34b_b79k"),
            use_gpu=config.get("app", {}).get("gpu_enabled", False)
        )
        
        vm = VisionManager(config=vm_config, logger=logger)
        gallery_dir = str(PROJECT_ROOT / vision_conf.get("gallery_dir", "data/vision_gallery"))
        
        vm.build_index(gallery_dir)
        logger.info("Vision index rebuild completed successfully.")
        return True
    except Exception as e:
        logger.error(f"Vision index rebuild failed: {e}", exc_info=True)
        return False

@huey.task()
def _rebuild_index_task_huey():
    return _rebuild_index_logic()

try:
    from celery_app import celery_app
    @celery_app.task(queue="ocr_batch")
    def _rebuild_index_task_celery():
        return _rebuild_index_logic()
except (ImportError, ModuleNotFoundError):
    _rebuild_index_task_celery = None

def rebuild_vision_index_task():
    """
    Dispatch rebuild index task.
    """
    if use_celery_backend() and _rebuild_index_task_celery:
        return _rebuild_index_task_celery.delay()
    else:
        return _rebuild_index_task_huey()


# --------------------------------------------------------------------------- #
# Async Chat Task
# --------------------------------------------------------------------------- #

def _chat_inference_logic(query, session_id, hotel_id, doc_id, user_context, image_path=None):
    """
    Background logic for processing chat.
    """
    from web_app.services import get_orchestrator, get_db, get_rag_manager, get_prompt_manager, get_logger
    
    logger = get_logger()
    db = get_db()
    role = str((user_context or {}).get("role", "")).upper()
    user_id = (user_context or {}).get("user_id")

    scope_list = []
    for h in ((user_context or {}).get("hotel_scope") or []):
        try:
            scope_list.append(int(h))
        except Exception:
            continue

    if role != "ADMIN" and not scope_list:
        return {"answer": "Hotel access denied", "results": []}

    requested_hotel_id = None
    if hotel_id:
        try:
            requested_hotel_id = int(hotel_id)
        except Exception:
            return {"answer": "Invalid hotel_id", "results": []}
        if role != "ADMIN" and requested_hotel_id not in set(scope_list):
            return {"answer": "Hotel access denied", "results": []}

    effective_hotel_ids = None if role == "ADMIN" else ([requested_hotel_id] if requested_hotel_id is not None else scope_list)
    effective_owner_id = int(user_id) if (user_id is not None and role in {"CLIENTE", "CLIENT"}) else None
    
    try:
        # --- Vision Flow ---
        if image_path:
             # Logic similar to original synchronous flow but adapting to path
             pass # For now, let's focus on Text/RAG as usually Vision is separate or we handle image_path
             
        # --- Text / Orchestrator Flow ---
        orchestrator = get_orchestrator()
        
        # We need to reconstruct user context if needed by orchestrator
        # But orchestrator.route_request uses it.
        route = orchestrator.route_request(query, user_context)
        
        if route["action"] == "DENIED":
            return {"answer": route["message"], "results": []}

        target_tool = route["tool"]
        results = []
        tool_output = None
        answer = ""
        
        # ... (Include RAG Logic) ...
        rag = get_rag_manager()
        
        # Check permissions logic (simplified for background task, assume passed context is valid or double check DB?)
        # For speed, trust the passed context or check simple scope
        
        if target_tool == "TOOL_CALL" and route.get("tool_name"):
            res_tool = orchestrator.execute_tool(
                route["tool_name"], route["params"], user_context=user_context
            )
            tool_output = res_tool.get("output", "")
            answer = f"Acción ejecutada: {route['tool_name']}. \n\nResultado: {tool_output}"
        else:
            if target_tool in ["RAG_TEXT", "RAG_FINANCIAL", "CHAT_GENERAL"]:
                if doc_id:
                     # Contextual Chat
                     with db.get_connection() as conn:
                        cursor = db.get_cursor(conn)
                        try:
                            doc_id_int = int(doc_id)
                        except Exception:
                            return {"answer": "Documento no encontrado.", "results": []}

                        cursor.execute(
                            f"""
                            SELECT d.owner_id, d.hotel_id, o.text, d.filename
                            FROM documents d
                            LEFT JOIN ocr_texts o ON d.id = o.id_doc
                            WHERE d.id = {db.placeholder}
                            """,
                            (doc_id_int,),
                        )
                        row = cursor.fetchone()
                        if not row:
                            return {"answer": "Documento no encontrado.", "results": []}

                        owner_id = row[0]
                        doc_hotel_id = row[1]
                        text_val = row[2] or ""
                        filename_val = row[3] or ""

                        if effective_owner_id is not None and str(owner_id) != str(effective_owner_id):
                            return {"answer": "Document access denied", "results": []}
                        if role != "ADMIN":
                            if doc_hotel_id is None or int(doc_hotel_id) not in set(scope_list):
                                return {"answer": "Hotel access denied", "results": []}

                        if text_val:
                            results = [{"doc_id": doc_id_int, "text": text_val, "filename": filename_val, "score": 1.0}]
                        else:
                            return {"answer": "Documento no encontrado o sin texto.", "results": []}
                else:
                    # Normal RAG
                    if rag:
                        results = rag.search(
                            query,
                            k=5,
                            db_manager=db,
                            owner_id=effective_owner_id,
                            hotel_ids=effective_hotel_ids,
                        )

            context_str = ""
            for item in results:
                context_str += f"[Doc ID: {item.get('doc_id')}] Contenido: {item.get('text')}\n\n"

            role_prompt_key = str((user_context or {}).get("role") or "CLIENTE").upper()
            prompt_manager = get_prompt_manager()
            system_prompt = prompt_manager.get_prompt(role_prompt_key)
            if not system_prompt:
                system_prompt = prompt_manager.get_prompt("CHAT_GENERAL")
            if not system_prompt:
                system_prompt = "Eres un asistente inteligente de documentos. Responde en espanol."

            instruction = f"Contexto encontrado:\n{context_str}\n\nUsuario: {query}"
            
            # Use LLM via Orchestrator or Client directly?
            # Creating new client or using existing
            llm = orchestrator.llm # This might be the global one
            
            # Determine profile? (Text vs Vision) - handled in LLM class now
            # We can use llm.chat() wrapper which uses default profile
            res = llm.chat(user_prompt=instruction, system_prompt=system_prompt, profile="default")
            
            if res.get("success"):
                answer = res.get("analysis")
            else:
                answer = f"Error generando respuesta: {res.get('error')}"

        # Save to DB
        if user_id is not None:
            db.insert_chat_message(session_id, "user", query, user_id=str(user_id))
            db.insert_chat_message(session_id, "assistant", answer, user_id=str(user_id))
        else:
            db.insert_chat_message(session_id, "user", query)
            db.insert_chat_message(session_id, "assistant", answer)
        
        return {
            "answer": answer,
            "results": results,
            "tool_output": tool_output
        }
        
    except Exception as e:
        logger.error(f"Async Chat Failed: {e}", exc_info=True)
        return {"answer": f"Error interno: {str(e)}", "error": True}

@huey.task()
def _chat_task_huey(query, session_id, hotel_id, doc_id, user_context):
    return _chat_inference_logic(query, session_id, hotel_id, doc_id, user_context)

try:
    from celery_app import celery_app
    @celery_app.task(queue="ocr_fast") # Use fast queue for chat
    def _chat_task_celery(query, session_id, hotel_id, doc_id, user_context):
        return _chat_inference_logic(query, session_id, hotel_id, doc_id, user_context)
except:
    _chat_task_celery = None

def process_chat_async(query, session_id, hotel_id, doc_id, user_context):
    """Dispatch chat task"""
    if use_celery_backend() and _chat_task_celery:
        return _chat_task_celery.delay(query, session_id, hotel_id, doc_id, user_context)
    else:
        return _chat_task_huey(query, session_id, hotel_id, doc_id, user_context)


# --------------------------------------------------------------------------- #
# Email Import Task
# --------------------------------------------------------------------------- #

def _email_check_logic():
    from web_app.services import load_configuration, get_logger, resolve_path
    from modules.email_importer import EmailImporter
    from modules.outlook_importer import get_outlook_importer
    
    config = load_configuration()
    email_conf = config.get("email_importer", {})
    outlook_conf = config.get("outlook_importer", {})
    
    # Check if Outlook importer is enabled
    if outlook_conf.get("enabled"):
        logger = get_logger()
        logger.info("Checking Outlook emails via Graph API...")
        
        outlook_importer = get_outlook_importer(outlook_conf)
        if outlook_importer:
            outlook_importer.check_now()
            logger.info("Outlook email check completed")
    
    # Also check traditional IMAP if enabled
    if not email_conf.get("enabled"):
        return "Email importers disabled"
        
    post_conf = config.get("postbatch", {})
    input_folder = resolve_path(post_conf.get("input_folder", "input"))
    
    importer = EmailImporter(email_conf, input_folder)
    importer.check_now()
    return "Email check completed"

@huey.task()
def _email_task_huey():
    return _email_check_logic()

try:
    from celery_app import celery_app
    @celery_app.task(queue="ocr_default")
    def _email_task_celery():
        return _email_check_logic()
except:
    _email_task_celery = None

def trigger_email_check_task():
    if use_celery_backend() and _email_task_celery:
        return _email_task_celery.delay()
    else:
        return _email_task_huey()


# --------------------------------------------------------------------------- #
# Ephemeral Data Retention (Exports / Vision Artifacts / Upload Temp)
# --------------------------------------------------------------------------- #

def _purge_ephemeral_data_logic():
    """
    Best-effort purge of ephemeral artifact directories to prevent unbounded disk growth.

    This intentionally does NOT touch processed documents or OCR outputs (those are
    business records). Only temporary/derivative artifacts are purged.
    """

    from web_app.services import load_configuration, get_logger, PROJECT_ROOT, get_db
    from modules.retention import purge_directory

    logger = get_logger()
    config = load_configuration(reload=True)
    retention_conf = (config.get("app", {}) or {}).get("retention", {}) or {}

    enabled = bool(retention_conf.get("enabled", True))
    if not enabled:
        return {"enabled": False}

    # Defaults tuned for safety: keep a short window of ephemeral data.
    exports_days = float(retention_conf.get("exports_days", 7))
    vision_days = float(retention_conf.get("vision_days", 7))
    uploads_days = float(retention_conf.get("uploads_days", 2))
    dry_run = bool(retention_conf.get("dry_run", False))
    max_deletions = int(retention_conf.get("max_deletions", 50_000))

    data_dir = Path(PROJECT_ROOT) / "data"
    targets = [
        ("exports", data_dir / "exports", exports_days),
        ("vision", data_dir / "vision", vision_days),
        ("uploads", data_dir / "uploads", uploads_days),
    ]

    results = {"enabled": True, "dry_run": dry_run, "targets": {}}
    for name, path, days in targets:
        try:
            stats = purge_directory(
                str(path),
                max_age_days=days,
                logger=logger,
                dry_run=dry_run,
                max_deletions=max_deletions,
            )
            results["targets"][name] = {
                "base_dir": stats.base_dir,
                "max_age_days": stats.max_age_days,
                "scanned_files": stats.scanned_files,
                "deleted_files": stats.deleted_files,
                "deleted_dirs": stats.deleted_dirs,
                "bytes_freed": stats.bytes_freed,
                "errors": stats.errors,
            }
        except Exception as exc:
            logger.error("Retention purge failed for %s: %s", str(path), exc, exc_info=True)
            results["targets"][name] = {"error": str(exc)}

    # Chat retention (DB rows) to avoid unbounded growth under heavy chat usage.
    try:
        db = get_db()
        chat_conf = (config.get("chat", {}) or {})
        history_days = float(chat_conf.get("history_ttl_days", 30))
        task_days = float(chat_conf.get("task_ttl_days", 7))
        deleted_history = db.purge_chat_history_older_than(days=history_days)
        deleted_tasks = db.purge_chat_tasks_older_than(days=task_days, only_terminal=True)
        results["chat"] = {
            "history_ttl_days": history_days,
            "task_ttl_days": task_days,
            "deleted_history_rows": deleted_history,
            "deleted_task_rows": deleted_tasks,
        }
    except Exception as exc:
        logger.error("Chat retention purge failed: %s", exc, exc_info=True)
        results["chat"] = {"error": str(exc)}

    return results


@huey.periodic_task(crontab(minute=15, hour=3))
def purge_ephemeral_data_daily():
    return _purge_ephemeral_data_logic()


# Email check every 15 minutes (for Outlook Graph API and IMAP)
@huey.periodic_task(crontab(minute='*/15'))
def check_emails_periodic():
    """Check emails every 15 minutes."""
    return _email_check_logic()


# Due date alerts every morning at 8 AM
@huey.periodic_task(crontab(minute=0, hour=8))
def check_due_dates_daily():
    """Check and send due date alerts every morning."""
    from modules.logger_manager import get_logger
    logger = get_logger(__name__)
    from modules.payment_due_dates import send_due_date_alerts
    from web_app.services import load_configuration
    
    config = load_configuration()
    tenants = config.get("tenants", {})
    
    for tenant_id in tenants.keys():
        try:
            # Run async function
            import asyncio
            asyncio.run(send_due_date_alerts(tenant_id))
        except Exception as e:
            logger.error(f"Error sending due date alerts for tenant {tenant_id}: {e}")
    
    return "Due date alerts sent"


# Celery equivalent (production / Docker). Huey periodic tasks do not run under Celery Beat.
try:
    from celery_app import celery_app

    @celery_app.task(queue="ocr_default")
    def purge_ephemeral_data_daily_celery():
        return _purge_ephemeral_data_logic()
except (ImportError, ModuleNotFoundError):
    purge_ephemeral_data_daily_celery = None
