import os
import sys
import threading
import yaml
from pathlib import Path
from typing import Any, Dict, Optional

# Shared path helper (used by background tasks).
from web_app.utils import resolve_path

# Dependencies
from modules.db_manager import DBManager
from modules.logger_manager import setup_logger
from modules.file_utils import ensure_directories
from modules.config_normalizer import normalize_config
from postbatch_processor import PipelineComponents, initialise_pipeline
from modules.classifier import DocumentClassifier
from modules.rag_manager import RAGManager
from modules.tool_manager import ToolManager

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Configuration Path
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

# Thread-local storage for config caching within request context if needed
local = threading.local()

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

def load_configuration(reload: bool = False) -> Dict[str, Any]:
    """Load configuration from YAML. If reload=True or not cached, reads from disk."""
    if reload or getattr(local, "config", None) is None:
        if CONFIG_PATH.exists():
            with open(CONFIG_PATH, "r", encoding="utf-8") as handle:
                raw = yaml.safe_load(handle) or {}
                local.config = normalize_config(raw if isinstance(raw, dict) else {})
        else:
            local.config = {}
    return local.config

def save_configuration(config: Dict[str, Any]) -> None:
    global _pipeline_instance
    global _classifier_instance
    global _rag_instance
    global _tool_instance
    global _vision_instance
    global _voice_instance
    global _llm_instance
    global _prompt_instance
    global _orchestrator_instance
    global _product_manager_instance

    config = normalize_config(config if isinstance(config, dict) else {})
    with open(CONFIG_PATH, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)
    local.config = config
    
    # Invalidate heavy singletons so updated config applies without restart.
    with _pipeline_lock:
        _pipeline_instance = None
    with _classifier_lock:
        _classifier_instance = None
    with _rag_lock:
        _rag_instance = None
    with _tool_lock:
        _tool_instance = None
    with _vision_lock:
        _vision_instance = None
    with _voice_lock:
        _voice_instance = None
    with _llm_lock:
        _llm_instance = None
    with _prompt_lock:
        _prompt_instance = None
    with _orchestrator_lock:
        _orchestrator_instance = None
    with _product_manager_lock:
        _product_manager_instance = None

# --------------------------------------------------------------------------- #
# Singletons
# --------------------------------------------------------------------------- #

_db_instance: Optional[DBManager] = None
_db_lock = threading.Lock()

_pipeline_instance: Optional[PipelineComponents] = None
_pipeline_lock = threading.Lock()

_classifier_instance: Optional[DocumentClassifier] = None
_classifier_lock = threading.Lock()

_rag_instance: Optional[RAGManager] = None
_rag_lock = threading.Lock()

_tool_instance: Optional[ToolManager] = None
_tool_lock = threading.Lock()

_voice_instance: Optional["VoiceManager"] = None
_voice_lock = threading.Lock()

def get_db() -> DBManager:
    """Get database manager singleton with thread-safe initialization."""
    global _db_instance
    if _db_instance is None:
        with _db_lock:
            if _db_instance is None:
                config = load_configuration()
                _db_instance = DBManager(config)
    return _db_instance

def get_logger():
    # Keep logger in local or simplify
    if getattr(local, "logger", None) is None:
        config = load_configuration()
        app_conf = config.get("app", {})
        log_level = app_conf.get("log_level", "INFO")
        log_dir = PROJECT_ROOT / "web_app" / "logs"
        ensure_directories(str(log_dir))
        log_path = log_dir / "web_app.log"
        local.logger = setup_logger(str(log_path), level=log_level, db_manager=get_db())
    return local.logger

def get_pipeline() -> PipelineComponents:
    """Get pipeline components singleton with thread-safe initialization."""
    global _pipeline_instance
    if _pipeline_instance is None:
        with _pipeline_lock:
            if _pipeline_instance is None:
                config = load_configuration()
                _pipeline_instance = initialise_pipeline(config, str(PROJECT_ROOT), get_logger())
    return _pipeline_instance

def get_classifier() -> Optional[DocumentClassifier]:
    """Get document classifier singleton with thread-safe initialization."""
    global _classifier_instance
    if _classifier_instance is None:
        with _classifier_lock:
            if _classifier_instance is None:
                config = load_configuration()
                post_conf = config.get("postbatch", {})
                if post_conf.get("classification_enabled", True):
                    model_path = PROJECT_ROOT / "data" / "models" / "classifier.pkl"
                    _classifier_instance = DocumentClassifier(model_path=str(model_path))
                else:
                    _classifier_instance = None
    return _classifier_instance

def get_rag_manager() -> Optional[RAGManager]:
    """Get RAG manager singleton with thread-safe initialization."""
    global _rag_instance
    if _rag_instance is None:
        with _rag_lock:
            if _rag_instance is None:
                try:
                    rag_dir = PROJECT_ROOT / "data" / "rag_index"
                    _rag_instance = RAGManager(str(rag_dir))
                except Exception as e:
                    get_logger().error(f"Failed to load RAG Manager: {e}")
                    _rag_instance = None
    return _rag_instance

def get_tool_manager() -> ToolManager:
    """Get tool manager singleton with thread-safe initialization."""
    global _tool_instance
    if _tool_instance is None:
        with _tool_lock:
            if _tool_instance is None:
                config = load_configuration()
                post_conf = config.get("postbatch", {})
                allowed_doc_roots = [
                    resolve_path(post_conf.get("processed_folder"), "processed"),
                    resolve_path(post_conf.get("failed_folder"), "errors"),
                    resolve_path(post_conf.get("input_folder"), "input"),
                    str(PROJECT_ROOT / "data" / "uploads"),
                ]
                pipeline = get_pipeline()
                vision = pipeline.vision_manager if pipeline else None
                products = get_product_manager()
                _tool_instance = ToolManager(
                    get_db(),
                    str(PROJECT_ROOT),
                    vision_manager=vision,
                    product_manager=products,
                    allowed_doc_roots=allowed_doc_roots,
                )
    return _tool_instance

_vision_instance: Optional["VisionManager"] = None
_vision_lock = threading.Lock()

def get_vision_manager():
    """Get vision manager singleton with thread-safe initialization."""
    global _vision_instance
    if _vision_instance is None:
        with _vision_lock:
            if _vision_instance is None:
                try:
                    from modules.vision_manager import VisionManager
                    config = load_configuration()
                    from modules.vision_manager import VisionManagerConfig

                    v_conf = config.get("vision", {}) or {}
                    model_name = (
                        v_conf.get("model_name")
                        or v_conf.get("model")
                        or "ViT-B-32"
                    )
                    vm_conf = VisionManagerConfig(
                        enabled=v_conf.get("enabled", True),
                        model_name=model_name,
                        pretrained=v_conf.get("pretrained", "laion2b_s34b_b79k"),
                        index_path=resolve_path(v_conf.get("index_path"), "data/vision_index.faiss"),
                        embeddings_dir=resolve_path(v_conf.get("embeddings_dir"), "data/vision_embeddings"),
                        use_gpu=bool(config.get("app", {}).get("gpu_enabled", False)),
                    )
                    
                    _vision_instance = VisionManager(config=vm_conf, logger=get_logger())
                except Exception as e:
                    get_logger().error(f"Failed to load Vision Manager: {e}")
                    _vision_instance = None
    return _vision_instance

def get_voice_manager() -> Optional["VoiceManager"]:
    """Get voice manager singleton with thread-safe initialization."""
    global _voice_instance
    if _voice_instance is None:
        with _voice_lock:
            if _voice_instance is None:
                try:
                    from modules.voice_manager import VoiceManager
                    config = load_configuration()
                    v_conf = config.get("app", {}).get("voice", {})
                    model_size = v_conf.get("model_size", "base")
                    # Auto-detect device (use GPU if available)
                    gpu = config.get("app", {}).get("gpu_enabled", True)
                    device = "cuda" if gpu else "cpu"
                    _voice_instance = VoiceManager(model_size=model_size, device=device)
                except Exception as e:
                    get_logger().error(f"Failed to load Voice Manager: {e}")
                    _voice_instance = None
    return _voice_instance
def reload_classifier():
    """Force reload of the classifier instance."""
    global _classifier_instance
    with _classifier_lock:
        _classifier_instance = None
        get_classifier() # Re-initialize immediately

_llm_instance: Optional["LLMClient"] = None
_llm_lock = threading.Lock()

_prompt_instance: Optional["PromptManager"] = None
_prompt_lock = threading.Lock()

_orchestrator_instance: Optional["AIOrchestrator"] = None
_orchestrator_lock = threading.Lock()

def get_llm_client():
    """Get LLM Client singleton with thread-safe initialization."""
    global _llm_instance
    if _llm_instance is None:
        with _llm_lock:
            if _llm_instance is None:
                try:
                    from modules.llm_client import LLMClient
                    config = load_configuration()
                    llm_conf = config.get("llm", {})
                    _llm_instance = LLMClient(llm_conf, get_logger())
                except Exception as e:
                    get_logger().error(f"Failed to load LLM Client: {e}")
                    _llm_instance = None
    return _llm_instance

def get_prompt_manager():
    """Get Prompt Manager singleton."""
    global _prompt_instance
    if _prompt_instance is None:
        with _prompt_lock:
            if _prompt_instance is None:
                from modules.prompt_manager import PromptManager
                _prompt_instance = PromptManager(str(PROJECT_ROOT / "data" / "prompts"))
    return _prompt_instance

_product_manager_instance: Optional["ProductManager"] = None
_product_manager_lock = threading.Lock()

def get_product_manager() -> Optional["ProductManager"]:
    """Get Product Manager singleton."""
    global _product_manager_instance
    if _product_manager_instance is None:
        with _product_manager_lock:
            if _product_manager_instance is None:
                try:
                    from modules.product_manager import ProductManager
                    # Try to get VisionManager for multimodal support
                    vision = get_vision_manager()
                    _product_manager_instance = ProductManager(get_db(), vision_manager=vision)
                except Exception as e:
                    get_logger().warning(
                        f"Product Manager unavailable; continuing without product tools: {e}"
                    )
                    _product_manager_instance = None
    return _product_manager_instance

def get_orchestrator():
    """Get AI Orchestrator singleton."""
    global _orchestrator_instance
    if _orchestrator_instance is None:
        with _orchestrator_lock:
            if _orchestrator_instance is None:
                from modules.ai_orchestrator import AIOrchestrator
                llm = get_llm_client()
                prompts = get_prompt_manager()
                tools = get_tool_manager()
                products = get_product_manager()
                if llm and prompts:
                    _orchestrator_instance = AIOrchestrator(llm, prompts, tool_manager=tools, product_manager=products)
                else:
                    get_logger().error("Failed to initialize AI Orchestrator: LLM or Prompts missing.")
    return _orchestrator_instance
